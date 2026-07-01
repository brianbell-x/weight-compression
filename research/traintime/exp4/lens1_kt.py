from __future__ import annotations
import json, struct, lzma
from pathlib import Path
import numpy as np
from scipy.special import gammaln

SNAP = Path("C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot")
IDX = json.load(open(SNAP/"model.safetensors.index.json")); WM=IDX["weight_map"]
def read_header(p):
    with open(p,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]; return 8+n,json.loads(f.read(n))
_H={}
def load_u16(name):
    sh=WM[name]; p=SNAP/sh
    if sh not in _H: _H[sh]=read_header(p)
    ds,hdr=_H[sh]; m=hdr[name]; b,e=m["data_offsets"]
    mm=np.memmap(p,dtype=np.uint8,mode="r")
    return np.ascontiguousarray(mm[ds+b:ds+e]).view(np.uint16).copy(), m["shape"]

LOG2=np.log(2.0)
def kt_bits(ctx, sym, D, alpha=0.5):
    """Exact total code length (bits) of an adaptive Dirichlet(alpha) coder over a
    partition into contexts. Order-independent; includes model-learning cost.
    ctx: int64 context id per symbol, sym: dense int64 symbol 0..D-1."""
    N=sym.size
    Kc=int(ctx.max())+1
    lin=ctx*D+sym
    jc=np.bincount(lin, minlength=Kc*D).astype(np.float64).reshape(Kc,D)
    Nc=jc.sum(1)
    act=Nc>0
    # bits = -sum_ctx [ gammaln(alpha*D) - gammaln(Nc+alpha*D) + sum_k(gammaln(n_k+alpha)-gammaln(alpha)) ] / ln2
    # per-context symbol term (only nonzero counts contribute beyond gammaln(alpha) baseline)
    # sum_k gammaln(n_k+alpha) over all D symbols:
    sym_term = gammaln(jc+alpha).sum(1) - D*gammaln(alpha)
    ctx_term = gammaln(alpha*D) - gammaln(Nc+alpha*D)
    total = -(ctx_term[act] + sym_term[act]).sum()/LOG2
    return total/N, Kc

def remap(x):
    u=np.unique(x); lut=np.zeros(int(x.max())+1,dtype=np.int64); lut[u]=np.arange(u.size)
    return lut[x].astype(np.int64), u.size

def find(kw, layer=None):
    for n in WM:
        if kw in n and (layer is None or f"layers.{layer}." in n): return n
    return None

targets=[
 ("embeddings","backbone.embeddings.weight"),
 ("expert_up_L1", find("experts.0.up_proj",1)),
 ("expert_down_L1", find("experts.0.down_proj",1)),
 ("expert_up_L25", find("experts.0.up_proj",25)),
 ("expert_down_L49", find("experts.0.down_proj",49)),
 ("in_proj_early", find("in_proj",4)),
 ("q_proj", find("q_proj",26)),
 ("out_proj_mid", find("out_proj",28)),
 ("o_proj", find("o_proj",19)),
]

print(f"{'label':16s} {'D':>4s} {'kt0':>6s} {'ktCol':>6s} {'ktUp':>6s} {'ktC+U':>6s} {'ktU+L':>6s}  savings_vs_kt0")
res={}
for label,name in targets:
    if name is None: print(label,"MISSING"); continue
    u16,shape=load_u16(name)
    se=(u16>>7).astype(np.int64)
    d,D=remap(se)
    if len(shape)==2: R,C=shape
    else: R,C=1,shape[0]
    d2=d.reshape(R,C)
    up=np.zeros_like(d2); up[1:,:]=d2[:-1,:]; up=up.reshape(-1)
    left=np.zeros_like(d2); left[:,1:]=d2[:,:-1]; left=left.reshape(-1)
    col=np.tile(np.arange(C),R).astype(np.int64)
    kt0,_=kt_bits(np.zeros(d.size,dtype=np.int64), d, D)
    ktcol,_=kt_bits(col, d, D)
    ktup,_=kt_bits(up.astype(np.int64), d, D)
    cu,_=remap(col*D+up); ktcu,_=kt_bits(cu, d, D)
    ul,_=remap(up*D+left); ktul,_=kt_bits(ul, d, D)
    best=min(ktcol,ktup,ktcu,ktul)
    res[label]=(name, d.size, kt0, ktcol, ktup, ktcu, ktul)
    print(f"{label:16s} {D:>4d} {kt0:6.3f} {ktcol:6.3f} {ktup:6.3f} {ktcu:6.3f} {ktul:6.3f}  {kt0-best:+.3f}")

import pickle
pickle.dump(res, open("C:/Users/bbell/AppData/Local/Temp/claude/C--dev-compression/771b35d1-71b7-45b9-9704-7ab4517510e6/scratchpad/kt_res.pkl","wb"))
