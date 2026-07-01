"""Lossless try-harder — the exponent-plane context lever, and re-confirming the walls.

0009 exploited only the ORDER-0 concentration of the sign+exponent field (~2.7 b). If weight
magnitudes vary smoothly across a matrix, the exponent has 2-D spatial structure a context
model can exploit BELOW 2.7 b — pure lossless, no lossy, no combination. This measures:
  - exponent field (8b) order-0 vs order-1(row) vs 2-D conditional entropy
  - strong real compressors (zlib/bz2/lzma) on the 2-D exponent plane vs raw
  - same battery on sign and mantissa planes to re-confirm they are random walls
Reports the best achievable lossless bits/weight and % vs 16, per tensor and combined.
"""
from __future__ import annotations
import json, struct, zlib, bz2, lzma, sys
from pathlib import Path
import numpy as np

SHARD1 = (r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
          r"\hf_snapshot\model-00001-of-00013.safetensors")


def load_u16(path, name):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n)); ds = 8 + n
        b, e = header[name]["data_offsets"]; shape = header[name]["shape"]
        f.seek(ds + b); raw = f.read(e - b)
    return np.frombuffer(raw, dtype=np.uint16), shape


def h0(sym, base_bits):
    c = np.bincount(sym.reshape(-1), minlength=1 << base_bits).astype(np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def cond_entropy(sym, ctx):
    """H(sym | ctx) via joint counts. sym, ctx flat uint arrays (small alphabets)."""
    s = sym.reshape(-1).astype(np.int64); c = ctx.reshape(-1).astype(np.int64)
    key = c * 256 + s
    jc = np.bincount(key)
    # p(c): marginal
    total = s.size
    H = 0.0
    # iterate contexts present
    cc = np.bincount(c)
    for cval in np.nonzero(cc)[0]:
        block = jc[cval * 256:(cval + 1) * 256]
        n = block.sum()
        if n == 0:
            continue
        p = block[block > 0] / n
        H += (n / total) * float(-(p * np.log2(p)).sum())
    return H


def comp_bpw(arr_bytes, n):
    out = {}
    out["zlib"] = len(zlib.compress(arr_bytes, 9)) * 8 / n
    out["bz2"] = len(bz2.compress(arr_bytes, 9)) * 8 / n
    out["lzma"] = len(lzma.compress(arr_bytes, preset=6)) * 8 / n
    return {k: round(v, 4) for k, v in out.items()}


def analyze(name, u16, shape):
    n = u16.size
    exp = ((u16 >> 7) & 0xFF).astype(np.uint8)     # 8-bit exponent field
    sign = (u16 >> 15).astype(np.uint8)
    mant = (u16 & 0x7F).astype(np.uint8)           # 7-bit mantissa
    rows = shape[0] if len(shape) >= 2 else 1
    cols = n // rows
    exp2d = exp[: rows * cols].reshape(rows, cols)

    # exponent entropies: order-0, order-1 (left neighbor), 2-D (left+up quantized)
    H0 = h0(exp, 8)
    left = np.zeros_like(exp2d); left[:, 1:] = exp2d[:, :-1]
    H1 = cond_entropy(exp2d, left)
    up = np.zeros_like(exp2d); up[1:, :] = exp2d[:-1, :]
    # 2-D context = combine left+up into one context id (coarsen to keep table small)
    ctx2d = ((left.astype(np.int64)) ^ (up.astype(np.int64) * 131)) & 0xFF
    H2 = cond_entropy(exp2d, ctx2d.astype(np.uint8))

    exp_comp = comp_bpw(exp2d.tobytes(), n)
    sign_comp_lzma = len(lzma.compress(np.packbits(sign).tobytes(), preset=6)) * 8 / n
    mant_comp_lzma = len(lzma.compress(mant.tobytes(), preset=6)) * 8 / n

    # best lossless bits/weight: best exponent code + sign(1) + mantissa(7, random)
    best_exp = min(H0, H1, H2, exp_comp["lzma"], exp_comp["bz2"])
    best_bpw = best_exp + min(1.0, sign_comp_lzma) + min(7.0, mant_comp_lzma)
    return {
        "name": name, "shape": shape, "n": int(n),
        "exp_H0": round(H0, 4), "exp_H1_row": round(H1, 4), "exp_H2_2d": round(H2, 4),
        "exp_comp": exp_comp,
        "sign_lzma_bpw": round(sign_comp_lzma, 4), "mant_lzma_bpw": round(mant_comp_lzma, 4),
        "best_lossless_bpw": round(best_bpw, 4), "pct_vs_16": round(100 * (1 - best_bpw / 16), 2),
    }


if __name__ == "__main__":
    targets = [
        ("expert_up", "backbone.layers.1.mixer.experts.0.up_proj.weight"),
        ("expert_down", "backbone.layers.1.mixer.experts.0.down_proj.weight"),
        ("attn_qkv", "backbone.layers.0.mixer.in_proj.weight"),
    ]
    out = []
    for tag, name in targets:
        try:
            u16, shape = load_u16(SHARD1, name)
        except Exception as e:
            print(f"skip {name}: {e}", flush=True); continue
        r = analyze(tag, u16, shape); out.append(r); print(json.dumps(r), flush=True)
    Path("lossless_battery_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote lossless_battery_result.json")
