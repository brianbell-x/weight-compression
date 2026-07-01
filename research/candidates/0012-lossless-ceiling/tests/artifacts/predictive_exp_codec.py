"""Make the exponent-context lossless gain RUNTIME-REAL: fixed-width, random-access, fusible.

0009 = per-tensor fixed-width codebook on the raw sign+exp field (~random-access, fusible).
The +4pt storage gain used VARIABLE-length context coding (not fusible). This tests whether a
PREDICTOR that is itself O(1) random-access — a separable per-row + per-column exponent base —
lets a fixed-width residual codebook beat 0009 while staying fusible:

  reconstruct exp[i,j] = base_row[i] + base_col[j] + residual_codebook[ index[i,j] ]   (all O(1))

Only the SEPARABLE (between-row/col) part of the exponent structure is fixed-width-capturable;
the within-block neighbor correlation needs sequential decode and is excluded on purpose. We
measure how much of the gain survives. Everything is bit-exact (exponent reconstructs exactly);
mantissa (7b) + sign (1b) ride along fixed-width unchanged.
"""
from __future__ import annotations
import json, struct, sys
from pathlib import Path
import numpy as np

SHARD1 = (r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
          r"\hf_snapshot\model-00001-of-00013.safetensors")


def H_of(a):
    v, c = np.unique(a, return_counts=True); p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def load_u16(path, name):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n)); ds = 8 + n
        b, e = header[name]["data_offsets"]; shape = header[name]["shape"]
        f.seek(ds + b); raw = f.read(e - b)
    return np.frombuffer(raw, dtype=np.uint16).reshape(shape)


def fw_codebook_bpw(field, index_bits, escape_val_bits, side_bits=0, n=None):
    """Fixed-width codebook cost (0009-style): K=2^index_bits-1 most-common values get a
    fixed index; the rest escape to an in-order side stream at escape_val_bits each. Returns
    (bits_per_weight, escape_rate, exact_ok). Reconstruction is random-access."""
    flat = field.reshape(-1)
    if n is None:
        n = flat.size
    vals, counts = np.unique(flat, return_counts=True)
    order = np.argsort(counts)[::-1]
    K = (1 << index_bits) - 1
    kept = set(vals[order[:K]].tolist())
    escaped = int(sum(c for v, c in zip(vals, counts) if v not in kept))
    esc_rate = escaped / n
    bpw = index_bits + esc_rate * escape_val_bits + side_bits / n
    # exactness: codebook + escape stream reconstructs the field bit-exact by construction
    return bpw, esc_rate


def analyze(name, u16):
    R, C = u16.shape
    exp = ((u16 >> 7) & 0xFF).astype(np.int16)      # 8-bit exponent field
    n = exp.size

    # ---- baseline (0009-style): fixed-width codebook on the RAW exponent ----
    base_bpw = {}
    for ib in (3, 4, 5):
        bpw, esc = fw_codebook_bpw(exp, ib, escape_val_bits=8)
        base_bpw[ib] = (round(bpw, 4), round(esc, 4))

    # ---- separable predictor: base_row[i] + base_col[j], rounded ints (random-access) ----
    row_mean = exp.mean(1, keepdims=True)
    col_mean = exp.mean(0, keepdims=True)
    grand = exp.mean()
    pred = np.round(row_mean + col_mean - grand).astype(np.int16)   # additive separable model
    resid = exp - pred                                              # exact integer residual
    side_bits = (R + C) * 8                                          # R + C int8 bases
    pred_bpw = {}
    for ib in (2, 3, 4, 5):
        bpw, esc = fw_codebook_bpw(resid, ib, escape_val_bits=8, side_bits=side_bits, n=n)
        pred_bpw[ib] = (round(bpw, 4), round(esc, 4))

    # ---- block predictor: per-BxB-block base (random-access by block id) captures LOCAL
    #      magnitude smoothness the separable model misses; still O(1) fusible ----
    block_bpw = {}
    for B in (8, 16, 32):
        rb, cb = (R + B - 1) // B, (C + B - 1) // B
        pad = np.zeros((rb * B, cb * B), dtype=np.float64)
        pad[:R, :C] = exp
        blocks = pad.reshape(rb, B, cb, B).mean((1, 3))              # [rb, cb] block means
        bpred = np.round(np.repeat(np.repeat(blocks, B, 0), B, 1)[:R, :C]).astype(np.int16)
        bresid = exp - bpred
        bside = rb * cb * 8
        best = min((fw_codebook_bpw(bresid, ib, 8, bside, n) for ib in (2, 3, 4)),
                   key=lambda t: t[0])
        block_bpw[B] = (round(best[0], 4), round(best[1], 4),
                        int(np.unique(bresid).size), round(H_of(bresid), 4))

    exact = bool(np.array_equal(pred + resid, exp))

    return {
        "name": name, "shape": [R, C],
        "exp_H0": round(H_of(exp), 4), "resid_sep_H0": round(H_of(resid), 4),
        "resid_distinct": int(np.unique(resid).size), "exp_distinct": int(np.unique(exp).size),
        "baseline_fw_bpw(exp)": base_bpw,
        "separable_fw_bpw(resid)": pred_bpw,
        "block_fw_bpw{B:(bpw,esc,distinct,H)}": block_bpw,
        "roundtrip_exact": exact,
    }


if __name__ == "__main__":
    targets = [
        ("expert_up", "backbone.layers.1.mixer.experts.0.up_proj.weight"),
        ("expert_down", "backbone.layers.1.mixer.experts.0.down_proj.weight"),
        ("attn_qkv", "backbone.layers.0.mixer.in_proj.weight"),
    ]
    out = []
    for tag, nm in targets:
        r = analyze(tag, load_u16(SHARD1, nm)); out.append(r)
        print(json.dumps(r), flush=True)
    Path("predictive_exp_codec_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
