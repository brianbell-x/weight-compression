from __future__ import annotations
"""Lens 4 attack (b): does |W| or W align between tensor pairs so a byte-delta
compresses better than raw? Real coder (lzma). Also RLE gain on repeats.
Tests adjacent experts + a magnitude-sorted alignment (multiset match)."""
import json, struct, lzma, zlib
from pathlib import Path
import numpy as np

MODEL = Path("C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot")

def read_header(path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))

def prod(xs):
    o=1
    for x in xs: o*=x
    return o

def load(shard, name):
    data_start, header = read_header(shard)
    mm = np.memmap(shard, dtype=np.uint8, mode="r")
    b,e = header[name]["data_offsets"]
    return np.array(mm[data_start+b:data_start+e].view(np.uint16))

def lz(b):
    return len(lzma.compress(b, preset=6))

def bits_per_w(nbytes, n):
    return 8.0*nbytes/n

def main():
    shard = MODEL/"model-00001-of-00013.safetensors"
    _, header = read_header(shard)
    exps = sorted([n for n in header if "experts." in n and "up_proj" in n and "backbone.layers.1." in n])
    out={}
    # pick two experts in same layer
    n0,n1 = exps[0], exps[1]
    w0,w1 = load(shard,n0), load(shard,n1)
    n=w0.size
    # raw mantissa+hi baselines
    raw0 = w0.tobytes()
    out["expertA"]=n0; out["expertB"]=n1; out["numel"]=int(n)
    out["lzma_raw_A_bpw"]=bits_per_w(lz(raw0),n)
    # byte delta between two experts (u16 subtract mod 2^16)
    d = (w0.astype(np.uint16)-w1.astype(np.uint16)).astype(np.uint16)
    out["lzma_delta_AB_bpw"]=bits_per_w(lz(d.tobytes()),n)
    # magnitude |W| : mask sign, compare
    absA=(w0 & 0x7FFF); absB=(w1 & 0x7FFF)
    dabs=(absA-absB).astype(np.uint16)
    out["lzma_absdelta_AB_bpw"]=bits_per_w(lz(dabs.tobytes()),n)
    # sorted multiset delta (upper bound if we ALSO stored the permutation -> not free)
    sA=np.sort(w0); sB=np.sort(w1)
    out["sorted_delta_lzma_bpw_NOPERM"]=bits_per_w(lz((sA-sB).astype(np.uint16).tobytes()),n)
    out["frac_identical_sorted"]=float((sA==sB).mean())
    # exponent-plane delta (hi byte) between experts
    hiA=(w0>>8).astype(np.uint8); hiB=(w1>>8).astype(np.uint8)
    out["lzma_hi_raw_A_bpw"]=8.0*lz(hiA.tobytes())/n
    out["lzma_hi_delta_AB_bpw"]=8.0*lz(((hiA-hiB).astype(np.uint8)).tobytes())/n
    print(json.dumps(out,indent=2))
    Path("C:/dev/compression/research/traintime/exp4/lens4_delta_result.json").write_text(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
