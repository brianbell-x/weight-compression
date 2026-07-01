"""Extract a small REAL-weight sample + its fixed-width encoding for the GPU benchmark.

Emits one up_proj and one down_proj layer-1 expert tensor, encoded in the clean
2-plane byte-split K=15 layout (the simplest faithful prototype, ~25% = 12 b/w):
  - codebook: top-15 high-byte (sign+exp7) values, uint8
  - idx_packed: 4-bit index per weight (code 0..14, or 15=ESCAPE), two per byte
  - low: raw low byte (exp_lsb + 7 mantissa), uint8, one per weight
  - escape_vals: high bytes of escaped weights, row-major order, uint8
  - row_escape_offset: cumulative escape count before each row, int32
  - raw_u16: original bytes (for a bit-exact lossless check on the GPU)
Reconstruction: high = idx<15 ? codebook[idx] : next-escape ; w = (high<<8)|low -> bf16.

Self-checks reconstruction == original on CPU so the handed-off sample is proven good.
"""
from __future__ import annotations
import struct, json, re, mmap, hashlib
from pathlib import Path
import numpy as np

SHARD = Path("models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot/model-00001-of-00013.safetensors")
OUT = Path(__file__).resolve().parent / "gpu_sample.npz"
K = 15  # codebook size; 15 codes + 1 escape -> 4-bit index


def read_header(p):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return 8 + n, h


def encode(raw, shape):
    a = np.frombuffer(raw, dtype=np.uint8)
    low = a[0::2].copy()
    high = a[1::2].copy()
    R, C = shape
    hist = np.bincount(high, minlength=256)
    codebook = np.argsort(hist)[::-1][:K].astype(np.uint8)
    lut = {int(v): i for i, v in enumerate(codebook)}
    ESC = K
    idx = np.fromiter((lut.get(int(v), ESC) for v in high), dtype=np.uint8, count=high.size)
    high2d = high.reshape(R, C)
    idx2d = idx.reshape(R, C)
    escape_vals = high2d[idx2d == ESC].astype(np.uint8)            # row-major order
    row_counts = (idx2d == ESC).sum(axis=1)
    row_escape_offset = np.zeros(R, dtype=np.int32)
    row_escape_offset[1:] = np.cumsum(row_counts)[:-1]
    # pack idx to 4-bit, two per byte (low nibble = even col)
    assert idx.size % 2 == 0
    idx_packed = (idx[0::2] | (idx[1::2] << 4)).astype(np.uint8)
    return dict(codebook=codebook, idx_packed=idx_packed, low=low,
                escape_vals=escape_vals, row_escape_offset=row_escape_offset,
                shape=np.array(shape, dtype=np.int64))


def decode(enc):
    R, C = [int(x) for x in enc["shape"]]
    codebook = enc["codebook"]; ESC = K
    idx_packed = enc["idx_packed"]
    idx = np.empty(R * C, dtype=np.uint8)
    idx[0::2] = idx_packed & 0x0F
    idx[1::2] = idx_packed >> 4
    idx2d = idx.reshape(R, C)
    high = np.empty(R * C, dtype=np.uint8).reshape(R, C)
    esc = enc["escape_vals"]
    off = enc["row_escape_offset"]
    for r in range(R):
        row = idx2d[r]
        is_esc = row == ESC
        out = np.where(is_esc, 0, codebook[np.minimum(row, K - 1)]).astype(np.uint8)
        nesc = int(is_esc.sum())
        if nesc:
            out[is_esc] = esc[off[r]:off[r] + nesc]
        high[r] = out
    high = high.reshape(-1)
    u16 = (high.astype(np.uint16) << 8) | enc["low"].astype(np.uint16)
    return u16


def main():
    ds, h = read_header(SHARD)
    def first(proj):
        ks = [k for k in h if re.search(rf"layers\.1\.mixer\.experts\.\d+\.{proj}_proj\.weight$", k)]
        return sorted(ks, key=lambda k: int(re.search(r"experts\.(\d+)\.", k).group(1)))[0]
    f = open(SHARD, "rb"); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    bundle = {}
    meta = {}
    for proj in ("up", "down"):
        k = first(proj)
        b, e = h[k]["data_offsets"]
        raw = bytes(mm[ds + b: ds + e])
        shape = tuple(h[k]["shape"])
        enc = encode(raw, shape)
        # CPU lossless self-check
        u16 = decode(enc)
        ok = hashlib.sha256(u16.tobytes()).hexdigest() == hashlib.sha256(raw).hexdigest()
        assert ok, f"{proj} reconstruction not exact!"
        bits_per_w = (enc["idx_packed"].size * 8 + enc["low"].size * 8
                      + enc["escape_vals"].size * 8 + enc["row_escape_offset"].size * 32
                      + enc["codebook"].size * 8) / (shape[0] * shape[1])
        for kk, vv in enc.items():
            bundle[f"{proj}__{kk}"] = vv
        bundle[f"{proj}__raw_u16"] = np.frombuffer(raw, dtype=np.uint16).copy()
        meta[proj] = {"tensor": k, "shape": list(shape),
                      "n_escape": int(enc["escape_vals"].size),
                      "bits_per_weight": round(float(bits_per_w), 3),
                      "reduction": round(1 - float(bits_per_w) / 16, 4),
                      "lossless_exact": ok}
    mm.close(); f.close()
    bundle["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(OUT, **bundle)
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
