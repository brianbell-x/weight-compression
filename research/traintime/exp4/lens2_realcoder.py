from __future__ import annotations
import json, struct, lzma, bz2
from pathlib import Path
import numpy as np

SNAP = Path('C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot')

def read_header(path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        return 8+n, json.loads(f.read(n))

def get_u16(shard,name):
    ds,h=read_header(shard); b,e=h[name]['data_offsets']
    mm=np.memmap(shard,dtype=np.uint8,mode='r')
    return mm[ds+b:ds+e].view(np.uint16), h[name]['shape']

def H(counts):
    c=counts[counts>0].astype(np.float64); t=c.sum(); p=c/t
    return float(-(p*np.log2(p)).sum())

def cond_H(sym,cond,ksym,kcond):
    joint=np.bincount(cond.astype(np.int64)*ksym+sym.astype(np.int64),minlength=ksym*kcond).reshape(kcond,ksym)
    tot=joint.sum(); hc=0.0
    for c in range(kcond):
        row=joint[c]; n=row.sum()
        if n==0: continue
        hc+=(n/tot)*H(row)
    return hc

def real_group_by_exp(mant, exp, n):
    """Real bit-exact coder: partition mantissa bytes by exponent value, compress each
    contiguous group with lzma. Decoder needs group sizes (tiny) + the per-exp code.
    Round-trip is exact by construction (concatenation of exact groups, reordered by a
    stable argsort whose permutation is recoverable from exp which is stored anyway in 0009)."""
    order = np.argsort(exp, kind='stable')  # group same-exp together; exp known to decoder
    mant_sorted = mant[order]
    total = 0
    # compress the whole reordered stream (a real coder adapts to the now-homogeneous runs)
    total_lzma = len(lzma.compress(mant_sorted.tobytes(), preset=9|lzma.PRESET_EXTREME))*8/n
    total_bz2  = len(bz2.compress(mant_sorted.tobytes(),9))*8/n
    return total_lzma, total_bz2

def analyze(shard,name,sample=6_000_000):
    u16,shape=get_u16(shard,name)
    if u16.size>sample: u16=u16[:sample]
    n=u16.size
    mant=(u16&0x7F).astype(np.uint8)
    exp=((u16>>7)&0xFF).astype(np.uint8)
    r={'name':name,'n':int(n)}
    r['H_mant']=H(np.bincount(mant,minlength=128))
    r['H_mant_given_exp']=cond_H(mant,exp,128,256)
    # combined context: exp (quantized to distinct) + prev mant bucket(hi 3 bits)
    prevbucket=np.empty(n,dtype=np.int64); prevbucket[0]=0; prevbucket[1:]=(mant[:-1]>>4)
    ctx = exp.astype(np.int64)*8 + prevbucket
    r['H_mant_given_exp_prevhi']=cond_H(mant,ctx.astype(np.int64),128,256*8)
    # raw real coders on unsorted mantissa
    r['mant_lzma_raw']=len(lzma.compress(mant.tobytes(),preset=9|lzma.PRESET_EXTREME))*8/n
    r['mant_bz2_raw']=len(bz2.compress(mant.tobytes(),9))*8/n
    # real coder: reorder by exp then compress
    lz,bz=real_group_by_exp(mant,exp,n)
    r['mant_lzma_expsorted']=lz
    r['mant_bz2_expsorted']=bz
    return r

if __name__=='__main__':
    shard=SNAP/'model-00001-of-00013.safetensors'
    for t in ['backbone.layers.1.mixer.experts.0.up_proj.weight',
              'backbone.layers.1.mixer.experts.0.down_proj.weight',
              'backbone.embeddings.weight']:
        print(json.dumps(analyze(shard,t)))
