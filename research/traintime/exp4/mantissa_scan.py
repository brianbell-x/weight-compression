from __future__ import annotations
import json, struct, bz2
from pathlib import Path
import numpy as np

SNAP = Path('C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot')

def read_header(path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        return 8+n, json.loads(f.read(n))

def H(counts):
    counts=counts[counts>0].astype(np.float64); t=counts.sum(); p=counts/t
    return float(-(p*np.log2(p)).sum())

shards = sorted(SNAP.glob('model-*.safetensors'))
best=[]  # (H_mant, bz2, name)
allH=[]
for shard in shards:
    ds,h = read_header(shard)
    mm = np.memmap(shard, dtype=np.uint8, mode='r')
    for nm,meta in h.items():
        if nm=='__metadata__' or meta['dtype']!='BF16': continue
        b,e = meta['data_offsets']
        u16 = mm[ds+b:ds+e].view(np.uint16)
        if u16.size < 1000: continue
        s = u16 if u16.size<=3_000_000 else u16[:3_000_000]
        mant=(s&0x7F).astype(np.uint8)
        Hm=H(np.bincount(mant,minlength=128))
        bz=len(bz2.compress(mant.tobytes(),9))*8/mant.size
        allH.append(Hm)
        best.append((Hm,bz,nm,meta['dtype']))
best.sort()
print('=== lowest marginal mantissa entropy tensors ===')
for Hm,bz,nm,dt in best[:15]:
    print(f'H_mant={Hm:.4f} bz2={bz:.4f} {nm}')
a=np.array([x[0] for x in best]); bzs=np.array([x[1] for x in best])
print(f'\nBF16 tensors scanned: {len(best)}')
print(f'H_mant: min={a.min():.4f} median={np.median(a):.4f} max={a.max():.4f} mean={a.mean():.4f}')
print(f'bz2   : min={bzs.min():.4f} median={np.median(bzs):.4f} frac<7.0={np.mean(bzs<7.0):.3f}')
print(f'count H_mant<6.9: {(a<6.9).sum()}  <6.8: {(a<6.8).sum()}  <6.5: {(a<6.5).sum()}')
