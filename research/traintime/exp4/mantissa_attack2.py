from __future__ import annotations
import json, struct, lzma, bz2
from pathlib import Path
import numpy as np

SNAP = Path('C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot')

def read_header(path):
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return 8+n, json.loads(f.read(n))

def get_u16(shard, name):
    ds, h = read_header(shard)
    b,e = h[name]['data_offsets']
    mm = np.memmap(shard, dtype=np.uint8, mode='r')
    return mm[ds+b:ds+e].view(np.uint16), h[name]['shape']

def H(counts):
    counts = counts[counts>0].astype(np.float64); t=counts.sum(); p=counts/t
    return float(-(p*np.log2(p)).sum())

def cond_H(sym, cond, ksym, kcond):
    joint = np.bincount(cond.astype(np.int64)*ksym+sym.astype(np.int64), minlength=ksym*kcond).reshape(kcond,ksym)
    tot=joint.sum(); hc=0.0
    for c in range(kcond):
        row=joint[c]; nn=row.sum()
        if nn: hc += (nn/tot)*H(row)
    return hc

def real_cond_coder(mant, exp):
    # realize H(mant|exp) with a real coder: bz2 each exponent-group separately
    tot_bytes = 0
    order = np.argsort(exp, kind='stable')
    exp_s = exp[order]; mant_s = mant[order]
    # boundaries
    uniq, starts = np.unique(exp_s, return_index=True)
    starts = list(starts)+[len(exp_s)]
    for i in range(len(uniq)):
        seg = mant_s[starts[i]:starts[i+1]].tobytes()
        tot_bytes += min(len(bz2.compress(seg,9)), len(lzma.compress(seg,preset=6)))
    return tot_bytes*8/len(mant)

# collect many expert tensors + others
shard = SNAP/'model-00001-of-00013.safetensors'
ds,h = read_header(shard)
names = [k for k in h if k!='__metadata__' and h[k]['dtype']=='BF16']
expert_up = [k for k in names if '.experts.' in k and 'up_proj' in k][:12]
expert_dn = [k for k in names if '.experts.' in k and 'down_proj' in k][:12]

def summarize(namelist, label, cap=None):
    rows=[]
    for nm in namelist:
        u16,shape = get_u16(shard,nm)
        if cap and u16.size>cap: u16=u16[:cap]
        mant=(u16&0x7F).astype(np.uint8); exp=((u16>>7)&0xFF).astype(np.uint8)
        Hm=H(np.bincount(mant,minlength=128))
        Hme=cond_H(mant,exp,128,256)
        bz=len(bz2.compress(mant.tobytes(),9))*8/mant.size
        rc=real_cond_coder(mant,exp)
        rows.append((Hm,Hme,bz,rc))
    a=np.array(rows)
    print(f'{label}: n={len(rows)} H_mant={a[:,0].mean():.4f} H_mant|exp={a[:,1].mean():.4f} bz2={a[:,2].mean():.4f} real_cond_coder={a[:,3].mean():.4f}')
    return a

summarize(expert_up,'expert.up_proj', cap=None)
summarize(expert_dn,'expert.down_proj', cap=None)
