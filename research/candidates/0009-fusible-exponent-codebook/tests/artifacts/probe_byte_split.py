"""Probe: can the 32% lossless BF16 exponent-plane scheme be made FIXED-WIDTH
(random-access / fusible into a matmul) instead of variable-length entropy-coded?

Idea: the high byte (sign+exp7) of BF16 takes only a handful of distinct values.
Replace the variable-length rANS code with a small fixed-width CODEBOOK INDEX
(K common values) + a rare-value ESCAPE side stream + raw low byte (mantissa).
Fixed-width => known bit offset per weight => random access => fusible (Regime D),
and still EXACTLY lossless.

We measure, on real layer-1 experts (shard 1):
  1. high-byte / mag7 support (distinct values) and top-K cumulative mass (escape rate)
  2. realized bits/weight for fixed-width variants incl. escape side channel + per-row
     escape offsets + codebook storage
  3. EXACT lossless round-trip (reconstruct original tensor bytes via SHA256) for the
     chosen variant, decoded using ONLY (codebook, fixed-width index, in-order escape
     stream, per-row escape offsets) -- no entropy decode anywhere.
Reference points: raw BF16 (16 b/w), rANS order-0 floor (H_high + 8), INT8 (8 b/w).
"""
from __future__ import annotations
import struct, json, re, mmap, hashlib, math
from pathlib import Path
import numpy as np

SHARD = Path("C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot/model-00001-of-00013.safetensors")
N_EXPERTS = 8


def read_header(p):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return 8 + n, h


def order0_bits(arr, alphabet):
    c = np.bincount(arr.astype(np.int64), minlength=alphabet).astype(np.float64)
    n = c.sum()
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def topk_mass(hist, n, ks):
    order = np.sort(hist)[::-1]
    cum = np.cumsum(order)
    return {k: float(cum[min(k, len(cum)) - 1] / n) for k in ks}


def fixed_width_bits(high, low, shape, K):
    """Bits/weight for: codebook top-K high-byte values (idx_width=ceil(log2(K+1))
    incl. one ESCAPE code) + escape stream (8b each) + per-row escape offsets +
    codebook table + raw low byte (8b). shape=(R,C) for the per-row offset table."""
    n = high.size
    R, Cc = shape
    hist = np.bincount(high, minlength=256)
    order = np.argsort(hist)[::-1]
    top_vals = order[:K]
    in_top = np.isin(high, top_vals)
    n_escape = int((~in_top).sum())
    idx_width = max(1, math.ceil(math.log2(K + 1)))      # K codes + ESCAPE
    off_bits = R * max(1, math.ceil(math.log2(n_escape + 1)))  # per-row cumulative escape count
    cb_bits = K * 8                                       # codebook (amortizable across experts)
    total = n * idx_width + n_escape * 8 + off_bits + cb_bits + n * 8
    return {
        "K": K, "idx_width": idx_width, "n_escape": n_escape,
        "escape_rate": n_escape / n, "bits_per_weight": total / n,
    }


def roundtrip_fixed_width(raw, high, low, shape, K):
    """EXACT lossless encode+decode using only fixed-width index + escape stream +
    per-row escape offsets + codebook. Returns (ok, rebuilt_bytes)."""
    n = high.size
    R, Cc = shape
    hist = np.bincount(high, minlength=256)
    order = np.argsort(hist)[::-1]
    top_vals = order[:K].astype(np.uint8)
    code_of = {int(v): i for i, v in enumerate(top_vals)}   # value -> index (0..K-1)
    ESC = K                                                 # escape code
    high2d = high.reshape(R, Cc)
    idx = np.empty((R, Cc), dtype=np.int32)
    escape_stream = []          # high-byte values, in row-major order
    row_escape_offset = np.zeros(R, dtype=np.int64)
    running = 0
    for r in range(R):
        row_escape_offset[r] = running
        row = high2d[r]
        for c in range(Cc):
            v = int(row[c])
            if v in code_of:
                idx[r, c] = code_of[v]
            else:
                idx[r, c] = ESC
                escape_stream.append(v)
                running += 1
    escape_stream = np.array(escape_stream, dtype=np.uint8)

    # ---- DECODE using only (top_vals codebook, idx, escape_stream, row_escape_offset) ----
    high_rt = np.empty((R, Cc), dtype=np.uint8)
    for r in range(R):
        esc_ptr = int(row_escape_offset[r])
        row_idx = idx[r]
        for c in range(Cc):
            code = int(row_idx[c])
            if code == ESC:
                high_rt[r, c] = escape_stream[esc_ptr]
                esc_ptr += 1
            else:
                high_rt[r, c] = top_vals[code]
    high_rt = high_rt.reshape(-1)
    # reinterleave with raw low byte
    out = np.empty(n * 2, dtype=np.uint8)
    out[0::2] = low
    out[1::2] = high_rt
    rebuilt = out.tobytes()
    ok = hashlib.sha256(rebuilt).hexdigest() == hashlib.sha256(raw).hexdigest()
    return ok, len(escape_stream)


def main():
    ds, h = read_header(SHARD)
    def pick(proj):
        ks = [k for k in h if re.search(rf"layers\.1\.mixer\.experts\.\d+\.{proj}_proj\.weight$", k)]
        return sorted(ks, key=lambda k: int(re.search(r"experts\.(\d+)\.", k).group(1)))[:N_EXPERTS]
    ups, dns = pick("up"), pick("down")

    f = open(SHARD, "rb"); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    def slice_raw(meta):
        b, e = meta["data_offsets"]; return bytes(mm[ds + b: ds + e])

    KS = [3, 7, 15, 31, 63]
    report = {}
    for kind, keys in (("up", ups), ("down", dns)):
        per = {"support_high": [], "support_mag7": [], "H_high": [],
               "topk_high": {k: [] for k in [4, 8, 16, 32, 64]},
               "fw": {K: [] for K in KS}}
        rebuilt_check = None
        for ki, k in enumerate(keys):
            raw = slice_raw(h[k])
            shape = tuple(h[k]["shape"])
            a = np.frombuffer(raw, dtype=np.uint8)
            low = a[0::2].copy(); high = a[1::2].copy()
            mag7 = (high & 0x7F)
            n = high.size
            hist_high = np.bincount(high, minlength=256)
            per["support_high"].append(int((hist_high > 0).sum()))
            per["support_mag7"].append(int((np.bincount(mag7, minlength=128) > 0).sum()))
            per["H_high"].append(order0_bits(high, 256))
            tk = topk_mass(hist_high, n, [4, 8, 16, 32, 64])
            for kk in tk: per["topk_high"][kk].append(tk[kk])
            for K in KS:
                per["fw"][K].append(fixed_width_bits(high, low, shape, K))
            # exact round-trip on FIRST tensor of each kind, K=15 (4-bit index)
            if ki == 0:
                ok, n_esc = roundtrip_fixed_width(raw, high, low, shape, 15)
                rebuilt_check = {"tensor": k, "K": 15, "exact_roundtrip": ok,
                                 "n_escape": n_esc, "shape": shape, "numel": n}
        def meanfw(K):
            rows = per["fw"][K]
            return {"K": K, "idx_width": rows[0]["idx_width"],
                    "mean_escape_rate": round(float(np.mean([r["escape_rate"] for r in rows])), 6),
                    "mean_bits_per_weight": round(float(np.mean([r["bits_per_weight"] for r in rows])), 4),
                    "reduction_vs_bf16": round(1 - float(np.mean([r["bits_per_weight"] for r in rows])) / 16, 4)}
        report[kind] = {
            "support_high_mean": round(float(np.mean(per["support_high"])), 1),
            "support_high_max": int(np.max(per["support_high"])),
            "support_mag7_mean": round(float(np.mean(per["support_mag7"])), 1),
            "H_high_mean": round(float(np.mean(per["H_high"])), 4),
            "rANS_floor_bits_per_weight": round(float(np.mean(per["H_high"])) + 8, 4),
            "topk_high_cum_mass": {f"top{kk}": round(float(np.mean(per["topk_high"][kk])), 6)
                                   for kk in [4, 8, 16, 32, 64]},
            "fixed_width_variants": {f"K{K}": meanfw(K) for K in KS},
            "exact_roundtrip_check": rebuilt_check,
        }
    mm.close(); f.close()

    out = {"shard": SHARD.name, "n_experts": N_EXPERTS,
           "reference_bits_per_weight": {"raw_bf16": 16, "int8_lossy": 8},
           "results": report}
    print(json.dumps(out, indent=2))
    (Path(__file__).resolve().parent / "byte_split_result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
