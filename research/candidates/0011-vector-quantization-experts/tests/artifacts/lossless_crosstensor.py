"""Lossless cross-tensor exponent structure: do experts share a per-column magnitude
(exponent) profile even though their VALUES are uncorrelated (survey 0010)? If a column is
systematically high/low-magnitude across all 128 experts, a SHARED per-column exponent
predictor + per-expert residual codes the exponent below its per-tensor order-0 entropy —
a lossless slice 0009 (per-tensor) does not capture. Bit-exact, no lossy.
"""
from __future__ import annotations
import json, struct
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
    return np.frombuffer(raw, dtype=np.uint16).reshape(shape)


def h0(sym, k=256):
    c = np.bincount(sym.reshape(-1), minlength=k).astype(np.float64)
    p = c[c > 0] / c.sum(); return float(-(p * np.log2(p)).sum())


def cond_on_ref(sym, ref):
    """H(sym | ref) with both uint8 (ref = predictor code)."""
    s = sym.reshape(-1).astype(np.int64); r = ref.reshape(-1).astype(np.int64)
    total = s.size; H = 0.0
    key = r * 256 + s; jc = np.bincount(key, minlength=256 * 256)
    rc = np.bincount(r, minlength=256)
    for rv in np.nonzero(rc)[0]:
        blk = jc[rv * 256:(rv + 1) * 256]; nn = blk.sum()
        if nn: p = blk[blk > 0] / nn; H += (nn / total) * float(-(p * np.log2(p)).sum())
    return H


if __name__ == "__main__":
    N = 32
    names = [f"backbone.layers.1.mixer.experts.{i}.up_proj.weight" for i in range(N)]
    exps = []
    for nm in names:
        W = load_u16(SHARD1, nm)                      # [1856, 2688] = [out, in]
        exps.append(((W >> 7) & 0xFF).astype(np.uint8))
    E = np.stack(exps, 0)                              # [N, out, in]
    # shared per-column (in-axis) mean exponent profile across experts+rows
    col_mean = E.mean(axis=(0, 1)).round().astype(np.uint8)      # [in]
    row_mean = E.mean(axis=(0, 2)).round().astype(np.uint8)      # [out]
    per_tensor_H0 = np.mean([h0(e) for e in E])
    # predictor 1: shared column profile (broadcast over rows+experts)
    ref_col = np.broadcast_to(col_mean[None, None, :], E.shape)
    H_given_col = cond_on_ref(E, ref_col)
    # predictor 2: shared (row,col) additive profile, coarsened to a uint8 code
    rc = (row_mean[None, :, None].astype(np.int64) + col_mean[None, None, :].astype(np.int64)) // 2
    ref_rc = np.broadcast_to(rc, E.shape).astype(np.uint8)
    H_given_rowcol = cond_on_ref(E, ref_rc)
    # how much of exponent variance is the shared profile? (cross-expert corr of column means)
    per_expert_colmean = E.mean(axis=1)               # [N, in]
    cc = np.corrcoef(per_expert_colmean)
    off = cc[np.triu_indices(N, 1)]
    res = {
        "n_experts": N,
        "exp_per_tensor_H0": round(float(per_tensor_H0), 4),
        "exp_H_given_shared_col": round(H_given_col, 4),
        "exp_H_given_shared_rowcol": round(H_given_rowcol, 4),
        "col_profile_saving_bits": round(float(per_tensor_H0) - H_given_col, 4),
        "cross_expert_colmean_corr_mean": round(float(off.mean()), 4),
        "cross_expert_colmean_corr_max": round(float(off.max()), 4),
    }
    print(json.dumps(res, indent=2))
    Path("lossless_crosstensor_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
