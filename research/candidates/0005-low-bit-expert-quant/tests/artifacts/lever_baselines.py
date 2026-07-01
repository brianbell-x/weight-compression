"""Baseline lever: INT8 vs INT4 per-group RTN on real layer-1 routed experts.

Imports the prebuilt Stage-1 validator (stage1_probe.py) -- does NOT rewrite the
metric. Sweeps INT8 and INT4 symmetric per-group round-to-nearest (group_size=128)
over N>=16 sampled layer-1 experts (mix of up_proj and down_proj), reports mean and
spread of matmul output error + cosine, effective bits/weight (incl. fp16 scale
overhead), and implied full-model resident VRAM.

Grouping axis note: group_size=128 divides the 2688 dimension but not 1856, so each
tensor is grouped along its length-2688 axis (up_proj -> axis 1 out_features,
down_proj -> axis 0 in_features). X always matches the tensor's in_features (axis 0).
"""
from __future__ import annotations

import csv
import statistics as stats

import torch

import stage1_probe as sp

SHARD = (
    r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    r"\hf_snapshot\model-00001-of-00013.safetensors"
)
OUT_CSV = (
    r"C:\dev\compression\research\candidates\0005-low-bit-expert-quant"
    r"\tests\artifacts\lever_baselines_results.csv"
)


def intN_per_group_rtn(W: torch.Tensor, bits: int, group_size: int = 128, axis: int = 1):
    """Symmetric per-group max-abs RTN to signed `bits`-bit, dequantized to float32.

    Generalizes stage1_probe.int8_per_group_rtn to any bit width. One fp16 scale per
    group; levels span [-(2^(bits-1)-1), 2^(bits-1)-1] (symmetric, no zero-point).
    Returns (W_prime, effective_bits_per_weight).
    """
    W = W.to(torch.float32)
    qmax = (1 << (bits - 1)) - 1  # INT8 -> 127, INT4 -> 7

    orig_shape = W.shape
    Wt = W.movedim(axis, -1)
    moved_shape = Wt.shape
    n_along = moved_shape[-1]
    if n_along % group_size != 0:
        raise ValueError(f"axis length {n_along} not divisible by group_size {group_size}")
    n_groups = n_along // group_size

    Wg = Wt.reshape(*moved_shape[:-1], n_groups, group_size)
    max_abs = Wg.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(Wg / scale), -qmax, qmax)
    Wg_hat = q * scale
    W_prime = Wg_hat.reshape(moved_shape).movedim(-1, axis).reshape(orig_shape)

    num_weights = W.numel()
    total_groups = num_weights // group_size
    bpw = sp.bits_per_weight(
        num_weights,
        payload_bits=bits * num_weights,
        scale_bits=16 * total_groups,  # one fp16 scale per group
    )
    return W_prime, bpw


def group_axis(W: torch.Tensor, group_size: int) -> int:
    """Pick the axis whose length is divisible by group_size (the length-2688 axis)."""
    if W.shape[1] % group_size == 0:
        return 1
    if W.shape[0] % group_size == 0:
        return 0
    raise ValueError(f"no axis of {tuple(W.shape)} divisible by {group_size}")


def summarize(vals):
    return (stats.mean(vals), min(vals), max(vals),
            stats.pstdev(vals) if len(vals) > 1 else 0.0)


def main():
    expert_ids = list(range(0, 128, 8))  # 16 experts spread across layer 1
    GS = 128
    configs = [("INT8_pg128_RTN", 8), ("INT4_pg128_RTN", 4)]

    # name -> per (config) lists
    acc = {name: {"rel": [], "cos": [], "bpw": []} for name, _ in configs}
    rows = []

    n_tensors = 0
    for eid in expert_ids:
        for proj in ("up_proj", "down_proj"):
            name = f"backbone.layers.1.mixer.experts.{eid}.{proj}.weight"
            W = sp.load_expert(SHARD, name)
            X = sp.make_inputs(W.shape[0], batch=256, seed=0)
            ax = group_axis(W, GS)
            n_tensors += 1
            for cfg_name, bits in configs:
                W_prime, bpw = intN_per_group_rtn(W, bits=bits, group_size=GS, axis=ax)
                f = sp.fidelity(W, W_prime, X)
                acc[cfg_name]["rel"].append(f["rel_err"])
                acc[cfg_name]["cos"].append(f["mean_cosine"])
                acc[cfg_name]["bpw"].append(bpw)
                rows.append({
                    "config": cfg_name, "expert": eid, "proj": proj,
                    "shape": "x".join(map(str, W.shape)), "axis": ax,
                    "rel_err_pct": f["rel_err"] * 100, "mean_cosine": f["mean_cosine"],
                    "bits_per_weight": bpw,
                    "implied_vram_gb": sp.implied_vram_gb(bpw),
                })

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"sampled {len(expert_ids)} experts x 2 proj = {n_tensors} tensors (layer 1)\n")
    summary = []
    for cfg_name, _ in configs:
        rel_m, rel_lo, rel_hi, rel_sd = summarize(acc[cfg_name]["rel"])
        cos_m, cos_lo, cos_hi, _ = summarize(acc[cfg_name]["cos"])
        bpw = acc[cfg_name]["bpw"][0]
        vram = sp.implied_vram_gb(bpw)
        print(f"{cfg_name}: bits/weight={bpw:.4f}  implied_vram={vram:.2f} GB")
        print(f"  rel_err  mean={rel_m*100:.3f}%  [{rel_lo*100:.3f}, {rel_hi*100:.3f}]  sd={rel_sd*100:.3f}")
        print(f"  cosine   mean={cos_m:.6f}  [{cos_lo:.6f}, {cos_hi:.6f}]\n")
        summary.append((cfg_name, bpw, rel_m * 100, cos_m, vram))
    return summary


if __name__ == "__main__":
    main()
