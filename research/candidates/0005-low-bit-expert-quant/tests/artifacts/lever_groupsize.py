"""Lever sweep: INT4 per-group symmetric RTN group-size sweep on real layer-1 experts.

Question: at 4 payload bits, does shrinking the quant group size (128 -> 64 -> 32 -> 16)
along the INPUT dim rescue INT4 output fidelity toward the <=1-2% band, and what does the
extra scale overhead cost in effective bits/weight?

Method:
  - Load N real layer-1 routed experts (up_proj + down_proj) via the prebuilt harness.
  - For each expert tensor and each group size, quantize with symmetric per-group max-abs
    RTN to signed 4-bit levels [-7, 7] (INT4) along axis=0 = the input/contraction dim
    (so each row of W that X multiplies is split into groups; matches make_inputs in_features).
  - Score with harness fidelity() on the SAME seeded X. Average rel_err / cosine over experts.
  - Effective bits/weight = 4 payload + one fp16 (16-bit) scale per group, via harness
    bits_per_weight(). Report implied full-model resident VRAM via harness implied_vram_gb().

We score along axis=0 because X is [batch, in_features] and W is [in_features, out_features];
grouping along the contraction dim (axis=0) is what per-group activation-aware quant does.
We also report the INT8 gs=128 baseline and an INT4 gs=128 along axis=1 sanity row is NOT
needed -- the lever is purely group size at 4 bits.
"""

from __future__ import annotations

import csv
import os

import torch

import stage1_probe as sp

SHARD = (
    r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    r"\hf_snapshot\model-00001-of-00013.safetensors"
)
OUT_CSV = os.path.join(os.path.dirname(__file__), "lever_groupsize_results.csv")

N_EXPERTS = 16
GROUP_SIZES = [128, 64, 32, 16]
INT4_MAX = 7  # signed 4-bit symmetric levels [-7, 7]

# Tensors per expert. axis=0 is the input/contraction dim (= in_features that X hits).
TENSORS = ["up_proj.weight", "down_proj.weight"]


def int4_per_group_rtn(W: torch.Tensor, group_size: int, axis: int = 0):
    """Symmetric per-group max-abs RTN to signed 4-bit [-7,7], dequantized to float32.

    Groups of `group_size` weights along `axis` share one fp16 max-abs scale.
    Returns (W_prime, effective_bits_per_weight).
    """
    W = W.to(torch.float32)
    orig_shape = W.shape
    Wt = W.movedim(axis, -1).contiguous()
    moved_shape = Wt.shape
    n_along = moved_shape[-1]

    # Handle a non-divisible axis with a final partial group via zero-padding.
    # Padded zeros don't inflate a group's max_abs and dequantize to 0, then we crop
    # them off. n_groups uses ceil so the scale overhead counts the partial group too.
    n_groups = (n_along + group_size - 1) // group_size
    n_padded = n_groups * group_size
    if n_padded != n_along:
        pad = torch.zeros(*moved_shape[:-1], n_padded - n_along, dtype=Wt.dtype)
        Wt_p = torch.cat([Wt, pad], dim=-1)
    else:
        Wt_p = Wt

    Wg = Wt_p.reshape(*moved_shape[:-1], n_groups, group_size)
    max_abs = Wg.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / INT4_MAX).clamp_min(1e-12)
    q = torch.clamp(torch.round(Wg / scale), -INT4_MAX, INT4_MAX)
    Wg_hat = q * scale
    Wt_hat = Wg_hat.reshape(*moved_shape[:-1], n_padded)[..., :n_along]
    W_prime = Wt_hat.movedim(-1, axis).reshape(orig_shape)

    num_weights = W.numel()
    # one fp16 scale per group, per "column" along the other axis
    other = num_weights // n_along
    total_groups = other * n_groups
    bpw = sp.bits_per_weight(
        num_weights,
        payload_bits=4 * num_weights,
        scale_bits=16 * total_groups,  # one fp16 scale per group
    )
    return W_prime, bpw


def run():
    # Cache loaded experts so each tensor is read once.
    loaded = {}  # (idx, tname) -> (W, X)
    for idx in range(N_EXPERTS):
        for tname in TENSORS:
            name = f"backbone.layers.1.mixer.experts.{idx}.{tname}"
            W = sp.load_expert(SHARD, name)
            in_features = W.shape[0]
            X = sp.make_inputs(in_features, batch=256, seed=0)
            loaded[(idx, tname)] = (W, X)
        print(f"[load] expert {idx} loaded ({len(TENSORS)} tensors)")

    rows = []

    # INT8 gs=128 reference (axis=1, the harness baseline) for context.
    for gs in GROUP_SIZES:
        # accumulate over all experts and both tensors
        rel_errs = []
        cosines = []
        bpws = []
        for idx in range(N_EXPERTS):
            for tname in TENSORS:
                W, X = loaded[(idx, tname)]
                W_prime, bpw = int4_per_group_rtn(W, group_size=gs, axis=0)
                f = sp.fidelity(W, W_prime, X)
                rel_errs.append(f["rel_err"])
                cosines.append(f["mean_cosine"])
                bpws.append(bpw)
        mean_rel = sum(rel_errs) / len(rel_errs)
        mean_cos = sum(cosines) / len(cosines)
        mean_bpw = sum(bpws) / len(bpws)
        vram = sp.implied_vram_gb(mean_bpw)
        config = f"INT4_gs{gs}_axis0"
        rows.append({
            "config": config,
            "bits_per_weight": round(mean_bpw, 4),
            "rel_err_pct": round(mean_rel * 100, 4),
            "mean_cosine": round(mean_cos, 6),
            "implied_vram_gb": round(vram, 2),
        })
        print(f"[{config}] rel_err={mean_rel*100:.3f}%  cos={mean_cos:.6f}  "
              f"bpw={mean_bpw:.4f}  vram={vram:.2f}GB  (n={len(rel_errs)} tensor-evals)")

    # INT8 reference row (harness helper, gs=64 axis=0 -- divides both 1856 and 2688).
    rel_errs, cosines, bpws = [], [], []
    for idx in range(N_EXPERTS):
        for tname in TENSORS:
            W, X = loaded[(idx, tname)]
            W_prime, meta = sp.int8_per_group_rtn(W, group_size=64, axis=0)
            f = sp.fidelity(W, W_prime, X)
            rel_errs.append(f["rel_err"])
            cosines.append(f["mean_cosine"])
            bpws.append(meta["bits_per_weight"])
    mean_rel = sum(rel_errs) / len(rel_errs)
    mean_cos = sum(cosines) / len(cosines)
    mean_bpw = sum(bpws) / len(bpws)
    rows.append({
        "config": "INT8_gs64_axis0_ref",
        "bits_per_weight": round(mean_bpw, 4),
        "rel_err_pct": round(mean_rel * 100, 4),
        "mean_cosine": round(mean_cos, 6),
        "implied_vram_gb": round(sp.implied_vram_gb(mean_bpw), 2),
    })
    print(f"[INT8_gs64_axis0_ref] rel_err={mean_rel*100:.3f}%  cos={mean_cos:.6f}  "
          f"bpw={mean_bpw:.4f}")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[done] wrote {OUT_CSV}")
    return rows


if __name__ == "__main__":
    run()
