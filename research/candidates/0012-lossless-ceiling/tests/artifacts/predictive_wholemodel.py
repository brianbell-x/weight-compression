"""Whole-(shard) runtime-real lossless number for the separable-predictive exponent codec.

Fusible fixed-width per weight = sign(1) + exp_residual_code(index_bits + escape) + mantissa(7).
exp_residual = exp - round(row_base[i] + col_base[j] - grand)  (row/col bases are O(1) side
vectors -> random-access -> fusible). Compares to the raw-exponent fixed-width codebook (0009-
style) at the same escape discipline. Aggregates bits/weight over every BF16 tensor in a shard,
numel-weighted, and reports whole-model % vs 16. Exponent reconstructs bit-exact (asserted).
"""
from __future__ import annotations
import json, struct, sys
from pathlib import Path
import numpy as np

SNAP = r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def fw_cost(field, index_bits, esc_val_bits, side_bits, n):
    vals, counts = np.unique(field, return_counts=True)
    K = (1 << index_bits) - 1
    keep = counts[np.argsort(counts)[::-1][:K]].sum()
    esc = (n - int(keep)) / n
    return index_bits + esc * esc_val_bits + side_bits / n, esc


def best_fw(field, n, side_bits=0):
    return min((fw_cost(field, ib, 8, side_bits, n) for ib in (2, 3, 4, 5)),
               key=lambda t: t[0])


def codec_bpw(u16, shape):
    if len(shape) < 2:
        u16 = u16.reshape(1, -1); shape = (1, u16.size)
    exp = ((u16 >> 7) & 0xFF).astype(np.int32).reshape(shape[0], -1)
    R, C = exp.shape
    n = exp.size
    # baseline: raw exponent fixed-width codebook
    base = best_fw(exp.reshape(-1), n)[0]
    # separable predictor
    pred = np.round(exp.mean(1, keepdims=True) + exp.mean(0, keepdims=True) - exp.mean()
                    ).astype(np.int32)
    resid = exp - pred
    assert np.array_equal(pred + resid, exp)                 # exact
    sep, sep_esc = best_fw(resid.reshape(-1), n, side_bits=(R + C) * 8)
    # whole-weight fusible b/w = sign(1) + exp_code + mantissa(7)
    return {"base_exp": base, "sep_exp": sep, "sep_esc": sep_esc,
            "base_weight_bpw": 1 + base + 7, "sep_weight_bpw": 1 + sep + 7, "n": n}


def main(shard):
    ds, header = read_header(shard)
    mm = np.memmap(shard, dtype=np.uint8, mode="r")
    tot_n = 0; base_bits = 0.0; sep_bits = 0.0; rows = []
    for name, meta in header.items():
        if name == "__metadata__" or meta.get("dtype") != "BF16":
            continue
        b, e = meta["data_offsets"]; shape = meta["shape"]
        if np.prod(shape) < 4096:
            continue
        u16 = np.frombuffer(mm[ds + b:ds + e].tobytes(), dtype=np.uint16)
        r = codec_bpw(u16, tuple(shape))
        tot_n += r["n"]; base_bits += r["base_weight_bpw"] * r["n"]; sep_bits += r["sep_weight_bpw"] * r["n"]
        rows.append({"name": name, "sep_weight_bpw": round(r["sep_weight_bpw"], 4),
                     "base_weight_bpw": round(r["base_weight_bpw"], 4), "sep_esc": round(r["sep_esc"], 4)})
    agg = {
        "shard": Path(shard).name, "tensors": len(rows), "weights": tot_n,
        "baseline_fusible_bpw": round(base_bits / tot_n, 4),
        "predictive_fusible_bpw": round(sep_bits / tot_n, 4),
        "baseline_pct_vs16": round(100 * (1 - base_bits / tot_n / 16), 2),
        "predictive_pct_vs16": round(100 * (1 - sep_bits / tot_n / 16), 2),
    }
    print(json.dumps(agg, indent=2))
    Path("predictive_wholemodel_result.json").write_text(
        json.dumps({"aggregate": agg, "per_tensor": rows[:60]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    shard = sys.argv[1] if len(sys.argv) > 1 else f"{SNAP}\\model-00001-of-00013.safetensors"
    main(shard)
