from __future__ import annotations
import json, struct
from pathlib import Path
import numpy as np
from scipy.special import gammaln
SNAP=Path("C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot")
IDX=json.load(open(SNAP/"model.safetensors.index.json")); WM=IDX["weight_map"]
def rh(p):
    with open(p,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]; return 8+n,json.loads(f.read(n))
_H={}
def load(name):
    sh=WM[name]; p=SNAP/sh
    if sh not in _H:_H[sh]=rh(p)
    ds,hd=_H[sh]; m=hd[name]; b,e=m["data_offsets"]
    mm=np.memmap(p,dtype=np.uint8,mode="r")
    return np.ascontiguousarray(mm[ds+b:ds+e]).view(np.uint16).copy(),m["shape"]
LOG2=np.log(2)
def kt(ctx,sym,D,a=0.5):
    N=sym.size;Kc=int(ctx.max())+1
    jc=np.bincount(ctx*D+sym,minlength=Kc*D).astype(np.float64).reshape(Kc,D)
    Nc=jc.sum(1);act=Nc>0
    st=gammaln(jc+a).sum(1)-D*gammaln(a)
    ct=gammaln(a*D)-gammaln(Nc+a*D)
    return -(ct[act]+st[act]).sum()/LOG2/N
def remap(x):
    u=np.unique(x);lut=np.zeros(int(x.max())+1,dtype=np.int64);lut[u]=np.arange(u.size);return lut[x].astype(np.int64),u.size
def find(kw,layer):
    for n in WM:
        if kw in n and f"layers.{layer}." in n:return n
    return None

# exp LSB entropy check (is it a coin flip?)
u16,_=load("backbone.embeddings.weight")
lsb=((u16>>7)&1)
p=lsb.mean(); Hl=-(p*np.log2(p)+(1-p)*np.log2(1-p))
print(f"exp-LSB(bit7) entropy embeddings: p1={p:.4f} H={Hl:.4f} b")

print("\n--- expert up_proj across layers (kt0 vs best neighbor ktU+L) ---")
for L in [0,1,2,3,5,10,20,30,40,49]:
    nm=find("experts.0.up_proj",L)
    if nm is None:
        print(L,"none");continue
    u16,shape=load(nm); se=(u16>>7).astype(np.int64); d,D=remap(se)
    R,C=shape; d2=d.reshape(R,C)
    up=np.zeros_like(d2);up[1:,:]=d2[:-1,:];up=up.reshape(-1)
    left=np.zeros_like(d2);left[:,1:]=d2[:,:-1];left=left.reshape(-1)
    k0=kt(np.zeros(d.size,dtype=np.int64),d,D)
    ul,_=remap(up*D+left); kul=kt(ul,d,D)
    print(f"L{L:2d} up_proj  D={D:3d} kt0={k0:.3f} ktU+L={kul:.3f} save={k0-kul:+.3f}")

print("\n--- shared (up,left) model across 8 experts, layer 1 up_proj (amortize model cost) ---")
allse=[];allul=[]
for ei in range(8):
    nm=find(f"experts.{ei}.up_proj",1)
    u16,shape=load(nm); se=(u16>>7).astype(np.int64); d,D0=remap(se)
    R,C=shape; d2=se.reshape(R,C)  # use raw se for shared symbol space
    up=np.zeros_like(d2);up[1:,:]=d2[:-1,:];up=up.reshape(-1)
    left=np.zeros_like(d2);left[:,1:]=d2[:,:-1];left=left.reshape(-1)
    allse.append(se); allul.append(up*512+left)
se=np.concatenate(allse); d,D=remap(se); ul,_=remap(np.concatenate(allul))
# per-expert independent total: sum kt0 each; shared: one model
k0_pool=kt(np.zeros(d.size,dtype=np.int64),d,D)
kul_shared=kt(ul,d,D)
print(f"8-expert pool: kt0={k0_pool:.3f} shared-ktU+L={kul_shared:.3f} save={k0_pool-kul_shared:+.3f}")
