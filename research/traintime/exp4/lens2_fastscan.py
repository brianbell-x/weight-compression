from __future__ import annotations
import json, struct
from pathlib import Path
import numpy as np

SNAP = Path('C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot')

def read_header(path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        return 8+n, json.loads(f.read(n))

def H(counts):
    c=counts[counts>0].astype(np.float64); t=c.sum(); p=c/t
    return float(-(p*np.log2(p)).sum())

def cond_H(sym,cond,ksym,kcond):
    joint=np.bincount(cond.astype(np.int64)*ksym+sym.astype(np.int64),minlength=ksym*kcond).reshape(kcond,ksym)
    tot=joint.sum(); hc=0.0
    nz=joint.sum(axis=1)
    for c in np.nonzero(nz)[0]:
        row=joint[c]; hc+=(nz[c]/tot)*H(row)
    return hc

shards=sorted(SNAP.glob('model-*.safetensors'))
rows=[]  # (H_mant, H_mant_given_exp, name, numel)
tot_w=0
wsum_marg=0.0; wsum_cond=0.0
for shard in shards:
    ds,h=read_header(shard)
    mm=np.memmap(shard,dtype=np.uint8,mode='r')
    for nm,meta in h.items():
        if nm=='__metadata__' or meta['dtype']!='BF16': continue
        b,e=meta['data_offsets']
        u16=mm[ds+b:ds+e].view(np.uint16)
        if u16.size<1000: continue
        s=u16 if u16.size<=4_000_000 else u16[:4_000_000]
        mant=(s&0x7F).astype(np.uint8)
        exp=((s>>7)&0xFF).astype(np.uint8)
        Hm=H(np.bincount(mant,minlength=128))
        Hme=cond_H(mant,exp,128,256)
        rows.append((Hm,Hme,nm,int(u16.size)))
        tot_w+=u16.size
        wsum_marg+=Hm*u16.size
        wsum_cond+=Hme*u16.size
rows.sort()
print('=== lowest MARGINAL mantissa entropy tensors ===')
for Hm,Hme,nm,ne in rows[:20]:
    print(f'H_mant={Hm:.4f} H_mant|exp={Hme:.4f} n={ne} {nm}')
a=np.array([r[0] for r in rows]); ac=np.array([r[1] for r in rows])
print(f'\nBF16 tensors: {len(rows)}  total weights(≈): {tot_w}')
print(f'H_mant       : min={a.min():.4f} med={np.median(a):.4f} max={a.max():.4f}')
print(f'H_mant|exp   : min={ac.min():.4f} med={np.median(ac):.4f} max={ac.max():.4f}')
print(f'count H_mant<6.5: {(a<6.5).sum()}  <6.0: {(a<6.0).sum()}  <5.0: {(a<5.0).sum()}')
print(f'weighted marginal mantissa entropy: {wsum_marg/tot_w:.4f} b/w')
print(f'weighted cond|exp mantissa entropy: {wsum_cond/tot_w:.4f} b/w')
print(f'whole-model mantissa saving vs 7.0 raw: marginal {7.0-wsum_marg/tot_w:.4f}  cond {7.0-wsum_cond/tot_w:.4f} b/w')
print(f'=> additional whole-MODEL % vs bf16(16b): marginal {(7.0-wsum_marg/tot_w)/16*100:.3f}%  cond {(7.0-wsum_cond/tot_w)/16*100:.3f}%')
