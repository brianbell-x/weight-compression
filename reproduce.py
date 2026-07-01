#!/usr/bin/env python
"""
reproduce.py -- independently verify the lossless BF16 weight-compression claim.

For every BF16 tensor in a .safetensors file (or a whole model directory) this:
  1. encodes it with a fixed-width sign+exponent codebook + a sparse escape stream,
  2. DECODES it back using ONLY (codebook, fixed-width index, in-order escape
     stream, raw mantissa) -- no entropy decoder, every weight at a known offset,
  3. checks the reconstructed bytes are bit-for-bit identical (SHA-256), and
  4. reports the size in bits/weight and the percent reduction vs 16-bit BF16.

Two layouts, both exactly lossless (same codec as the published probes):
  - byte-split (K=15, ~12.0 b/w, ~25 percent): 4-bit index over the high byte
    (sign+exp7) + the raw 8-bit low byte. This is the GPU-validated layout.
  - regroup    (K=15, ~11.3 b/w, ~30 percent): 4-bit index over the 9-bit
    sign+exponent field + the raw 7-bit mantissa. This is the headline number.

numpy only. No GPU, no torch, no download to run the codec -- point it at any
BF16 .safetensors you already have, or at one Nemotron shard.

Usage:
  uv run python reproduce.py --model PATH [--layout both|bytesplit|regroup] [--limit N]

PATH may be a single .safetensors file, a directory of shards, or an hf_snapshot
directory. Exit code is 0 if every tensor round-tripped bit-exact, else 1.
"""
from __future__ import annotations
import argparse, json, struct, mmap, hashlib, math, sys
from pathlib import Path
import numpy as np

K = 15  # codebook entries (+1 escape) -> 4-bit index


def read_header(p: Path):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def roundtrip_bytesplit(raw: bytes, rows: int):
    """Codebook the high byte (sign+exp7); store the low byte raw. Decode from
    ONLY (codebook, index, in-order escape stream, low byte); compare bit-exact."""
    a = np.frombuffer(raw, np.uint8)
    low, high = a[0::2], a[1::2]
    n = high.size
    hist = np.bincount(high, minlength=256)
    top = np.argsort(hist)[::-1][:K].astype(np.uint8)
    code_map = np.full(256, K, np.uint8); code_map[top] = np.arange(K, dtype=np.uint8)
    idx = code_map[high]                              # 0..K-1, or K = escape
    esc = idx == K
    escape_stream = high[esc]                         # stored verbatim, in order
    # ---- decode using only (top, idx, escape_stream, low) ----
    cb = np.zeros(K + 1, np.uint8); cb[:K] = top
    high_rec = cb[idx]
    high_rec[esc] = escape_stream
    rec = np.empty_like(a); rec[0::2] = low; rec[1::2] = high_rec
    ok = hashlib.sha256(rec.tobytes()).digest() == hashlib.sha256(raw).digest()
    n_esc = int(esc.sum())
    bits = n * 4 + n_esc * 8 + rows * max(1, math.ceil(math.log2(n_esc + 1))) + K * 8 + n * 8
    return ok, bits, n, n_esc


def roundtrip_regroup(raw: bytes, rows: int):
    """Codebook the 9-bit sign+exp field; store the 7-bit mantissa raw. Decode
    from ONLY (codebook, index, in-order escape stream, mantissa); compare bit-exact."""
    u = np.frombuffer(raw, np.uint16)
    sign = (u >> 15).astype(np.uint32)
    exp = ((u >> 7) & 0xFF).astype(np.uint32)
    mant = (u & 0x7F).astype(np.uint16)
    sym = ((sign << 8) | exp).astype(np.int64)        # 0..511
    n = sym.size
    hist = np.bincount(sym, minlength=512)
    top = np.argsort(hist)[::-1][:K].astype(np.int64)
    code_map = np.full(512, K, np.int64); code_map[top] = np.arange(K)
    idx = code_map[sym]                               # 0..K-1, or K = escape
    esc = idx == K
    escape_stream = sym[esc]
    # ---- decode using only (top, idx, escape_stream, mantissa) ----
    cb = np.zeros(K + 1, np.int64); cb[:K] = top
    sym_rec = cb[idx]
    sym_rec[esc] = escape_stream
    sign_rt = (sym_rec >> 8).astype(np.uint16)
    exp_rt = (sym_rec & 0xFF).astype(np.uint16)
    u_rt = ((sign_rt << 15) | (exp_rt << 7) | mant).astype(np.uint16)
    ok = hashlib.sha256(u_rt.tobytes()).digest() == hashlib.sha256(raw).digest()
    n_esc = int(esc.sum())
    idx_w = max(1, math.ceil(math.log2(K + 1)))
    bits = n * idx_w + n_esc * 9 + rows * max(1, math.ceil(math.log2(n_esc + 1))) + K * 9 + n * 7
    return ok, bits, n, n_esc


def iter_files(model: Path):
    if model.is_file():
        return [model]
    if not model.exists():
        sys.exit(f"path not found: {model}")
    files = sorted(model.glob("*.safetensors"))
    if not files:
        sys.exit(f"no .safetensors found in {model}")
    return files


def main():
    ap = argparse.ArgumentParser(description="Verify lossless BF16 compression on any model.")
    ap.add_argument("--model", required=True,
                    help=".safetensors file, a directory of shards, or an hf_snapshot dir")
    ap.add_argument("--layout", choices=["both", "bytesplit", "regroup"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="stop after N BF16 tensors (0 = all)")
    args = ap.parse_args()

    do_bs = args.layout in ("both", "bytesplit")
    do_rg = args.layout in ("both", "regroup")
    t = dict(n=0, weights=0, bs_bits=0, rg_bits=0, esc_bs=0, esc_rg=0, all_ok=True)

    files = iter_files(Path(args.model))
    print(f"scanning {len(files)} file(s) under {args.model}")
    for fp in files:
        ds, h = read_header(fp)
        with open(fp, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for name, meta in h.items():
                if name == "__metadata__" or meta.get("dtype") != "BF16":
                    continue
                b, e = meta["data_offsets"]
                if e - b < 2:
                    continue
                raw = mm[ds + b: ds + e]
                shape = meta["shape"]; rows = shape[0] if shape else 1
                if do_bs:
                    ok, bits, n, ne = roundtrip_bytesplit(raw, rows)
                    t["all_ok"] &= ok; t["bs_bits"] += bits; t["esc_bs"] += ne
                if do_rg:
                    ok, bits, n, ne = roundtrip_regroup(raw, rows)
                    t["all_ok"] &= ok; t["rg_bits"] += bits; t["esc_rg"] += ne
                t["n"] += 1; t["weights"] += (e - b) // 2
                if args.limit and t["n"] >= args.limit:
                    break
            mm.close()
        if args.limit and t["n"] >= args.limit:
            break

    if t["weights"] == 0:
        sys.exit("no BF16 tensors found")
    w = t["weights"]
    print()
    print(f"  BF16 tensors checked : {t['n']:,}")
    print(f"  weights              : {w:,}")
    print(f"  bit-exact round-trip : {'ALL PASS' if t['all_ok'] else 'FAIL'}")
    if do_bs:
        bpw = t["bs_bits"] / w
        print(f"  byte-split           : {bpw:6.3f} b/w   -{100*(1-bpw/16):4.1f}%   escapes {100*t['esc_bs']/w:.3f}%")
    if do_rg:
        bpw = t["rg_bits"] / w
        print(f"  regroup (headline)   : {bpw:6.3f} b/w   -{100*(1-bpw/16):4.1f}%   escapes {100*t['esc_rg']/w:.3f}%")
    print()
    sys.exit(0 if t["all_ok"] else 1)


if __name__ == "__main__":
    main()
