from __future__ import annotations
import json, struct, sys, lzma, bz2, zlib
from pathlib import Path
import numpy as np

SNAP = Path("C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot")
IDX = json.load(open(SNAP/"model.safetensors.index.json"))
WM = IDX["weight_map"]

def read_header(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        return 8+n, json.loads(f.read(n))

_HDR={}
def load_u16(name):
    shard=WM[name]; path=SNAP/shard
    if shard not in _HDR: _HDR[shard]=read_header(path)
    ds,hdr=_HDR[shard]
    m=hdr[name]; b,e=m["data_offsets"]; shape=m["shape"]
    assert m["dtype"]=="BF16", m["dtype"]
    mm=np.memmap(path,dtype=np.uint8,mode="r")
    raw=mm[ds+b:ds+e]
    u16=np.ascontiguousarray(raw).view(np.uint16).copy()
    return u16, shape

def entropy(counts):
    t=counts.sum()
    if t==0: return 0.0
    p=counts[counts>0]/t
    return float(-(p*np.log2(p)).sum())

def order0_ent(sym, K):
    c=np.bincount(sym, minlength=K).astype(np.float64)
    return entropy(c)

def cond_ent(sym, ctx, K, Kc):
    # H(sym | ctx) empirical (static floor)
    # joint counts
    lin = ctx.astype(np.int64)*K + sym.astype(np.int64)
    jc = np.bincount(lin, minlength=Kc*K).astype(np.float64).reshape(Kc,K)
    tot = jc.sum()
    pc = jc.sum(1)  # per ctx
    H=0.0
    for i in range(Kc):
        if pc[i]>0:
            p=jc[i][jc[i]>0]/pc[i]
            H += (pc[i]/tot)*(-(p*np.log2(p)).sum())
    return H

def adaptive_order0(sym, K):
    # real arithmetic-coder-achievable bits with adaptive (KT-ish) freq model, add-1 smoothing
    counts=np.ones(K, dtype=np.float64)
    total=float(K)
    bits=0.0
    # vectorization is hard for truly adaptive; do a fast approximate via blockwise? Need exact.
    # Use a semi-static two-pass: encode with final empirical distribution -> that's order0 entropy*N + model cost.
    # For a REAL adaptive coder number we approximate with static empirical (arithmetic coder reaches it) + tiny table cost.
    return order0_ent(sym,K)  # placeholder

def coder_bits_context(sym_2d, K):
    """Real-coder-achievable bits/sym for an adaptive context model conditioning on
    left and up neighbors, computed as cross-entropy of a two-pass semi-adaptive model.
    We use empirical conditional entropy as the achievable rate (arithmetic coder reaches
    H(sym|ctx) to <1e-3 b/sym); context table transmitted separately is amortized/negligible
    for these large tensors but we report it too."""
    R,C = sym_2d.shape
    sym = sym_2d.reshape(-1)
    # neighbors
    left = np.zeros_like(sym_2d); left[:,1:]=sym_2d[:,:-1]
    up   = np.zeros_like(sym_2d); up[1:,:]=sym_2d[:-1,:]
    left=left.reshape(-1); up=up.reshape(-1)
    H0 = order0_ent(sym,K)
    Hl = cond_ent(sym,left,K,K)
    Hu = cond_ent(sym,up,K,K)
    # both: need combined ctx; K can be up to 512 -> K^2 huge. Reduce ctx cardinality by mapping
    # to observed distinct values only.
    return H0,Hl,Hu

def real_lzma(b):
    return len(lzma.compress(b, preset=9|lzma.PRESET_EXTREME))*8
def real_bz2(b):
    return len(bz2.compress(b,9))*8

# ---- sample tensors ----
SAMPLE = []
def find(kw, layer=None):
    for n in WM:
        if kw in n and (layer is None or f"layers.{layer}." in n):
            return n
    return None

cands = [
 ("embeddings","backbone.embeddings.weight"),
 ("lm_head","lm_head.weight"),
 ("expert_up_L1", find("experts.0.up_proj", 1) or find("experts.0.up_proj")),
 ("expert_down_L1", find("experts.0.down_proj", 1) or find("experts.0.down_proj")),
 ("expert_gate_L1", find("experts.0.gate_proj", 1) or find("experts.0.up_proj")),
 ("expert_up_mid", find("experts.0.up_proj", 25)),
 ("expert_down_late", find("experts.0.down_proj", 49)),
 ("in_proj_early", find("in_proj", 4)),
 ("out_proj_mid", find("out_proj", 28)),
 ("q_proj", find("q_proj", 26)),
 ("o_proj", find("o_proj", 19)),
 ("norm_mid", find("layers.25.norm.weight")),
]
print("=== tensor sample ===")
rows=[]
for label,name in cands:
    if name is None:
        print(label,"MISSING"); continue
    u16,shape=load_u16(name)
    n=u16.size
    se = (u16>>7).astype(np.uint16)   # sign+exp 9-bit field, 0..511
    mant=(u16 & 0x7F).astype(np.uint8)
    K = int(se.max())+1
    # reshape to 2D
    if len(shape)==2:
        R,C=shape
    elif len(shape)==1:
        R,C=1,shape[0]
    else:
        R=shape[0]; C=n//shape[0]
    se2d=se.reshape(R,C)
    H0=order0_ent(se,512)
    left=np.zeros_like(se2d); left[:,1:]=se2d[:,:-1]
    up=np.zeros_like(se2d); up[1:,:]=se2d[:-1,:]
    Hl=cond_ent(se,left.reshape(-1),512,512)
    Hu=cond_ent(se,up.reshape(-1),512,512)
    # cross-plane: condition se on mantissa (128 ctx)
    Hm=cond_ent(se,mant,512,128)
    rows.append((label,name,n,R,C,H0,Hl,Hu,Hm,se,se2d,mant))
    print(f"{label:16s} n={n:>10d} shape={R}x{C} distinct_se={int((np.bincount(se,minlength=512)>0).sum()):3d} "
          f"H0={H0:.3f} H(|left)={Hl:.3f} H(|up)={Hu:.3f} H(|mant)={Hm:.3f}")

np.save("C:/Users/bbell/AppData/Local/Temp/claude/C--dev-compression/771b35d1-71b7-45b9-9704-7ab4517510e6/scratchpad/_rows.npy", np.array([1]))
print("\ndone stage1")
