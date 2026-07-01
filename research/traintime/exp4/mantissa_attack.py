from __future__ import annotations
import json, struct, sys, lzma, bz2, zlib
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
    counts = counts[counts>0].astype(np.float64)
    t = counts.sum()
    p = counts/t
    return float(-(p*np.log2(p)).sum())

def cond_H(sym, cond, ksym, kcond):
    # H(sym|cond) = sum_c p(c) H(sym|c)
    joint = np.bincount(cond.astype(np.int64)*ksym + sym.astype(np.int64), minlength=ksym*kcond).reshape(kcond, ksym)
    tot = joint.sum()
    hc = 0.0
    for c in range(kcond):
        row = joint[c]
        n = row.sum()
        if n==0: continue
        hc += (n/tot)*H(row)
    return hc

def comp_bits(data_bytes, n_weights):
    out = {}
    out['lzma9e'] = len(lzma.compress(data_bytes, preset=9|lzma.PRESET_EXTREME))*8/n_weights
    out['bz2'] = len(bz2.compress(data_bytes, 9))*8/n_weights
    return out

def bitplane_bits(mant, n):
    # mant: uint8 values 0..127, 7 bits. Compress each bitplane packed.
    total = 0
    per = []
    for b in range(7):
        bit = ((mant>>b)&1).astype(np.uint8)
        packed = np.packbits(bit).tobytes()
        c = min(len(lzma.compress(packed, preset=6)), len(bz2.compress(packed,9)))
        per.append(c*8/n)
        total += c*8/n
    return total, per

def analyze(shard, name, sample=None):
    u16, shape = get_u16(shard, name)
    if sample and u16.size > sample:
        u16 = u16[:sample]
    n = u16.size
    mant = (u16 & 0x7F).astype(np.uint8)
    exp = ((u16>>7)&0xFF).astype(np.uint8)
    sign = (u16>>15).astype(np.uint8)
    ncol = shape[-1] if len(shape)>=1 else 1
    col = (np.arange(n) % ncol).astype(np.int64)
    r = {'name':name,'n':int(n),'shape':shape}
    r['H_mant'] = H(np.bincount(mant, minlength=128))
    r['H_mant_given_exp'] = cond_H(mant, exp, 128, 256)
    r['H_mant_given_sign'] = cond_H(mant, sign, 128, 2)
    # neighbor: previous mantissa (bounded cond space 128)
    r['H_mant_given_prevmant'] = cond_H(mant[1:], mant[:-1], 128, 128)
    # column position: bin columns into 64 groups to keep tractable
    colb = (col % 64).astype(np.int64)
    r['H_mant_given_colbin64'] = cond_H(mant, colb, 128, 64)
    # exp field distinct
    r['exp_distinct'] = int((np.bincount(exp)>0).sum())
    r['exp_entropy'] = H(np.bincount(exp, minlength=256))
    # compressors on mantissa bytes
    cb = comp_bits(mant.tobytes(), n)
    r.update({'mant_'+k:v for k,v in cb.items()})
    # bitplane
    bt, per = bitplane_bits(mant, n)
    r['mant_bitplane_lzma'] = bt
    r['mant_bitplane_per'] = [round(x,4) for x in per]
    return r

if __name__ == '__main__':
    shard = SNAP/'model-00001-of-00013.safetensors'
    targets = [
        'backbone.embeddings.weight',
        'backbone.layers.1.mixer.experts.0.up_proj.weight',
        'backbone.layers.1.mixer.experts.0.down_proj.weight',
        'backbone.layers.0.mixer.in_proj.weight',
        'backbone.layers.0.mixer.out_proj.weight',
    ]
    SAMPLE = 4_000_000
    for t in targets:
        r = analyze(shard, t, sample=SAMPLE)
        print(json.dumps(r))
