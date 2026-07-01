from __future__ import annotations
import json, struct, lzma, bz2
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
    ds,hdr=_HDR[shard]; m=hdr[name]; b,e=m["data_offsets"]; shape=m["shape"]
    mm=np.memmap(path,dtype=np.uint8,mode="r")
    u16=np.ascontiguousarray(mm[ds+b:ds+e]).view(np.uint16).copy()
    return u16, shape

def ent(counts):
    t=counts.sum()
    if t==0: return 0.0
    p=counts[counts>0]/t
    return float(-(p*np.log2(p)).sum())

def H0(sym,K): return ent(np.bincount(sym,minlength=K).astype(np.float64))

def Hcond(sym, ctx, K, Kc):
    lin=ctx.astype(np.int64)*K+sym.astype(np.int64)
    jc=np.bincount(lin,minlength=Kc*K).astype(np.float64).reshape(Kc,K)
    tot=jc.sum(); pc=jc.sum(1)
    # vectorized conditional entropy
    with np.errstate(divide='ignore',invalid='ignore'):
        P=jc/pc[:,None]
        term=np.where(jc>0, -P*np.log2(P), 0.0).sum(1)
    return float((pc/tot*term).sum())

def remap(sym):
    u=np.unique(sym); lut=np.zeros(sym.max()+1,dtype=np.int64); lut[u]=np.arange(u.size)
    return lut[sym].astype(np.int64), u.size

def find(kw, layer=None):
    for n in WM:
        if kw in n and (layer is None or f"layers.{layer}." in n): return n
    return None

targets=[
 ("embeddings","backbone.embeddings.weight"),
 ("expert_up_L1", find("experts.0.up_proj",1)),
 ("expert_down_L1", find("experts.0.down_proj",1)),
 ("in_proj_early", find("in_proj",4)),
 ("q_proj", find("q_proj",26)),
 ("out_proj_mid", find("out_proj",28)),
]

def realcoder_bits_per_sym(sym2d, D):
    """Semi-static per-column arithmetic coder: transmit per-column histogram then
    code column values -> achievable bits = per-column empirical entropy + table cost.
    Also do adaptive order-0 (single pass) as a real coder sanity number."""
    R,C=sym2d.shape
    # per-column static entropy (arithmetic coder reaches this given per-col table)
    cols=sym2d  # [R,C]
    # bincount per column
    Hcol=0.0
    for c in range(C):
        Hcol += ent(np.bincount(cols[:,c],minlength=D).astype(np.float64))*R
    Hcol/= (R*C)
    return Hcol

print("label            H0    Hcol  H|up  H|left H|u+l  H|col+up  lzma_raw bz2_raw lzma_col")
for label,name in targets:
    if name is None:
        print(label,"MISSING"); continue
    u16,shape=load_u16(name)
    se=(u16>>7).astype(np.int64)
    d,D=remap(se)          # dense symbols
    n=d.size
    if len(shape)==2: R,C=shape
    else: R,C=1,shape[0]
    d2=d.reshape(R,C)
    h0=H0(d,D)
    # column context
    colidx=np.tile(np.arange(C),R)
    hcol=Hcond(d,colidx,D,C)
    # neighbors
    up=np.zeros_like(d2); up[1:,:]=d2[:-1,:]; up=up.reshape(-1)
    left=np.zeros_like(d2); left[:,1:]=d2[:,:-1]; left=left.reshape(-1)
    hup=Hcond(d,up,D,D)
    hleft=Hcond(d,left,D,D)
    # combined up+left
    ul=up*D+left
    ul_d,ULc=remap(ul.astype(np.int64))
    hul=Hcond(d,ul_d,D,ULc)
    # column + up (does neighbor add beyond column?)
    cu=colidx*D+up
    cu_d,CUc=remap(cu.astype(np.int64))
    hcu=Hcond(d,cu_d,D,CUc)
    # real coders on the sign+exp9 plane. Pack as bytes: sym fits in 9 bits -> store 2 bytes,
    # but better feed the raw high-info: use uint16 of se. Cap size for speed.
    cap=min(n, 6_000_000)
    seb=se[:cap].astype(np.uint16).tobytes()
    lz=len(lzma.compress(seb,preset=6))*8/cap
    bz=len(bz2.compress(seb,9))*8/cap
    # lzma on column-delta residual (predict from up neighbor) to expose 2D structure
    resid=((se.reshape(R,C).astype(np.int32) - up.reshape(R,C).astype(np.int32)) & 0x1FF).astype(np.uint16)
    lzc=len(lzma.compress(resid.reshape(-1)[:cap].tobytes(),preset=6))*8/cap
    print(f"{label:16s} {h0:.3f} {hcol:.3f} {hup:.3f} {hleft:.3f} {hul:.3f}  {hcu:.3f}    {lz:.3f}  {bz:.3f}  {lzc:.3f}")
