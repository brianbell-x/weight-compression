"""How much lossless structure is actually left, and where compounding stops.

For representative real tensors, run a cascade of lossless views and measure both the
order-0 entropy AND what a real strong compressor (lzma) achieves, to separate
"random at order-0" from "truly random". The gap between order-0 entropy and lzma tells
us whether higher-order / spatial structure exists for a compounding stage to exploit.
"""
from __future__ import annotations
import lzma, json, sys
from pathlib import Path
import numpy as np
from safetensors import safe_open

SHARD1 = (r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
          r"\hf_snapshot\model-00001-of-00013.safetensors")


def h0(counts):
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def lzma_bits_per_elem(b: bytes, n_elem: int, cap_bytes: int = 32 * 1024 * 1024) -> float:
    # compress a contiguous chunk (stream ~stationary) so lzma stays fast on huge tensors
    bytes_per_elem = max(1, len(b) // max(1, n_elem))
    chunk = b[:cap_bytes]
    n_chunk_elem = max(1, len(chunk) // bytes_per_elem)
    comp = lzma.compress(chunk, preset=6)
    return len(comp) * 8.0 / n_chunk_elem


def analyze(name, u16: np.ndarray):
    n = u16.size
    hi = (u16 >> 8).astype(np.uint8)
    lo = (u16 & 0xFF).astype(np.uint8)
    # order-0 entropies
    H_val = h0(np.bincount(u16, minlength=65536))
    H_hi = h0(np.bincount(hi, minlength=256))
    H_lo = h0(np.bincount(lo, minlength=256))
    # conditional H(lo|hi) from joint
    joint = np.zeros((256, 256), dtype=np.int64)
    np.add.at(joint, (hi, lo), 1)
    phi = joint.sum(1) / n
    H_lo_given_hi = 0.0
    for i in range(256):
        if joint[i].sum() > 0:
            H_lo_given_hi += phi[i] * h0(joint[i])
    # real-compressor probes (lzma) on planes + a spatial-delta view of the mantissa
    lo_lzma = lzma_bits_per_elem(lo.tobytes(), n)
    hi_lzma = lzma_bits_per_elem(hi.tobytes(), n)
    # horizontal byte-delta of mantissa (detect autocorrelation along stored order)
    lo_delta = (lo.astype(np.int16) - np.roll(lo.astype(np.int16), 1)).astype(np.uint8)
    lo_delta_lzma = lzma_bits_per_elem(lo_delta.tobytes(), n)
    raw_lzma = lzma_bits_per_elem(u16.tobytes(), n)
    return {
        "name": name, "n": int(n),
        "H_value16": round(H_val, 4),
        "H_hi": round(H_hi, 4), "H_lo": round(H_lo, 4),
        "H_lo_given_hi": round(H_lo_given_hi, 4),
        "mutual_info_hi_lo": round(H_lo - H_lo_given_hi, 4),
        "plane_sum": round(H_hi + H_lo, 4),
        "lzma_bpw_raw16": round(raw_lzma, 4),
        "lzma_bpw_hi": round(hi_lzma, 4),
        "lzma_bpw_lo": round(lo_lzma, 4),
        "lzma_bpw_lo_delta": round(lo_delta_lzma, 4),
        # honest floor estimate: exponent by strong coder + mantissa at its lzma cost
        "compounded_floor_bpw": round(hi_lzma + min(lo_lzma, 8.0), 4),
        "pct_vs_16": round(100 * (1 - (hi_lzma + min(lo_lzma, 8.0)) / 16), 2),
    }


def load_u16(path, name):
    import json as _json, struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = _json.loads(f.read(n))
        data_start = 8 + n
        meta = header[name]
        b, e = meta["data_offsets"]
        f.seek(data_start + b)
        raw = f.read(e - b)
    return np.frombuffer(raw, dtype=np.uint16)


if __name__ == "__main__":
    targets = [
        ("expert_up", "backbone.layers.1.mixer.experts.0.up_proj.weight"),
        ("expert_down", "backbone.layers.1.mixer.experts.0.down_proj.weight"),
        ("embeddings", "backbone.embeddings.weight"),
    ]
    out = []
    for tag, name in targets:
        try:
            u16 = load_u16(SHARD1, name)
        except Exception as e:
            print(f"skip {name}: {e}")
            continue
        r = analyze(tag, u16)
        out.append(r)
        print(json.dumps(r))
    Path("lossless_ceiling_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote lossless_ceiling_result.json")
