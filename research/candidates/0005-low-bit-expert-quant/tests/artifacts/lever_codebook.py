"""Lever: shape-shared non-uniform codebook (k-means / Lloyd-Max) vs uniform RTN.

Question (candidate 0005): does a non-uniform codebook fit to the POOLED normalized
routed-expert population beat uniform per-group RTN at 4 and 3 bits?

Method
------
- Load N>=16 real layer-1 routed experts (up_proj.weight [1856,2688], BF16->f32).
- Normalize each expert's weights by a scale (per-expert max-abs OR per-group max-abs,
  group_size=128 along axis=1) so values land in ~[-1,1].
- Fit a 1-D Lloyd-Max (k-means) codebook on a subsample of the POOLED normalized
  population: 16 levels (4-bit) and 8 levels (3-bit). One codebook, shared across all
  experts (shape-shared) -> amortized over the full 29.4e9 routed-expert population.
- Apply: for each expert, normalize by its scale(s), quantize each weight to the
  nearest codebook level, dequantize = level * scale.
- Compare against uniform symmetric per-group RTN at the same bit width (the baseline
  the codebook must beat).
- Score with the prebuilt Stage-1 matmul-fidelity harness (imported, not rewritten).

Effective bits/weight INCLUDES: index payload (4 or 3), per-group/per-expert fp16
scale, and the fp16 codebook amortized over the whole routed-expert population.
"""

from __future__ import annotations

import csv
import os

import torch

import stage1_probe as s1

SHARD = (
    r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    r"\hf_snapshot\model-00001-of-00013.safetensors"
)
LAYER = 1
N_EXPERTS = 16
PROJ = "up_proj"          # [1856, 2688]; in_features = 1856
GROUP_SIZE = 128
AXIS = 1                  # group along out_features (contiguous rows of W along dim 1)
FIT_SAMPLE = 3_000_000    # subsample size for codebook fitting (per scale mode)
ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ARTIFACT_DIR, "lever_codebook_results.csv")


# ----------------------------------------------------------------------------------
# scale helpers (per-expert and per-group max-abs)
# ----------------------------------------------------------------------------------
def per_expert_scale(W: torch.Tensor) -> torch.Tensor:
    """One max-abs scale for the whole tensor. Returns scalar tensor."""
    return W.abs().amax().clamp_min(1e-12)


def grouped(W: torch.Tensor, group_size: int, axis: int):
    """Reshape W so groups of `group_size` are contiguous along the last dim.
    Returns (Wg, restore_fn). Wg shape [..., n_groups, group_size]."""
    Wt = W.movedim(axis, -1)
    moved_shape = Wt.shape
    n_along = moved_shape[-1]
    if n_along % group_size != 0:
        raise ValueError(f"axis len {n_along} not divisible by group_size {group_size}")
    n_groups = n_along // group_size
    Wg = Wt.reshape(*moved_shape[:-1], n_groups, group_size)

    def restore(Wg_mod: torch.Tensor) -> torch.Tensor:
        return Wg_mod.reshape(moved_shape).movedim(-1, axis).reshape(W.shape)

    return Wg, restore, n_groups


def normalize_per_group(W: torch.Tensor, group_size: int, axis: int):
    """Return (normalized_values_flat, scales, Wg, restore) for per-group max-abs."""
    Wg, restore, n_groups = grouped(W, group_size, axis)
    scale = Wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    norm = Wg / scale
    return norm, scale, Wg, restore


# ----------------------------------------------------------------------------------
# 1-D Lloyd-Max / k-means codebook fitting
# ----------------------------------------------------------------------------------
def fit_codebook(values: torch.Tensor, n_levels: int, iters: int = 40,
                 seed: int = 0) -> torch.Tensor:
    """1-D k-means (Lloyd-Max) on a flat tensor of values. Returns sorted centers."""
    g = torch.Generator().manual_seed(seed)
    v = values.flatten()
    if v.numel() > FIT_SAMPLE:
        idx = torch.randperm(v.numel(), generator=g)[:FIT_SAMPLE]
        v = v[idx]
    v = v.to(torch.float32)
    # init centers at quantiles of the data (robust for skewed/peaked dists)
    qs = torch.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
    centers = torch.quantile(v, qs)
    for _ in range(iters):
        # assign each value to nearest center
        d = (v.unsqueeze(1) - centers.unsqueeze(0)).abs()
        a = d.argmin(dim=1)
        new = centers.clone()
        for k in range(n_levels):
            mask = a == k
            if mask.any():
                new[k] = v[mask].mean()
        if torch.allclose(new, centers, atol=1e-7):
            centers = new
            break
        centers = new
    return torch.sort(centers).values


def quantize_to_codebook(norm: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Map each value in `norm` to nearest center. Returns dequantized values."""
    d = (norm.unsqueeze(-1) - centers.view(*([1] * norm.dim()), -1)).abs()
    a = d.argmin(dim=-1)
    return centers[a]


# ----------------------------------------------------------------------------------
# codecs
# ----------------------------------------------------------------------------------
def codec_uniform_rtn(W, bits, group_size, axis):
    """Symmetric per-group uniform RTN at `bits`. Returns (W_prime, n_groups)."""
    qmax = (1 << (bits - 1)) - 1  # signed levels [-qmax, qmax]
    Wg, restore, n_groups = grouped(W, group_size, axis)
    scale = (Wg.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(Wg / scale), -qmax, qmax)
    return restore(q * scale), n_groups


def codec_codebook_pergroup(W, centers, group_size, axis):
    """Per-group max-abs scale + shared non-uniform codebook. Returns (W_prime, n_groups)."""
    norm, scale, Wg, restore = normalize_per_group(W, group_size, axis)
    deq = quantize_to_codebook(norm, centers) * scale
    return restore(deq), scale.numel()


def codec_codebook_perexpert(W, centers):
    """Per-expert max-abs scale + shared non-uniform codebook. Returns (W_prime, n_scales=1)."""
    scale = per_expert_scale(W)
    deq = quantize_to_codebook(W / scale, centers) * scale
    return deq, 1


# ----------------------------------------------------------------------------------
# bits/weight accounting (codebook amortized over full routed-expert population)
# ----------------------------------------------------------------------------------
def eff_bits(num_weights, payload_bpw, n_scales, n_levels):
    payload_bits = payload_bpw * num_weights
    scale_bits = 16 * n_scales
    # one fp16 codebook of n_levels, shared across ALL routed experts (29.4e9 weights)
    codebook_global_bpw = 16 * n_levels / s1.ROUTED_EXPERT_PARAMS
    bpw = s1.bits_per_weight(num_weights, payload_bits, scale_bits=scale_bits) + codebook_global_bpw
    return bpw


# ----------------------------------------------------------------------------------
# main sweep
# ----------------------------------------------------------------------------------
def main():
    names = [f"backbone.layers.{LAYER}.mixer.experts.{i}.{PROJ}.weight"
             for i in range(N_EXPERTS)]
    print(f"loading {N_EXPERTS} experts: {PROJ} layer {LAYER}")
    Ws = [s1.load_expert(SHARD, n) for n in names]
    in_features = Ws[0].shape[0]
    X = s1.make_inputs(in_features, batch=256, seed=0)
    numw = Ws[0].numel()
    print(f"  shape={tuple(Ws[0].shape)} in_features={in_features} numel/expert={numw}")

    # --- pooled normalized populations for codebook fitting ---
    print("pooling normalized weights for codebook fit...")
    pe_pool = torch.cat([(W / per_expert_scale(W)).flatten() for W in Ws])
    pg_pool = torch.cat([normalize_per_group(W, GROUP_SIZE, AXIS)[0].flatten() for W in Ws])

    # fit codebooks (per-expert-normalized pool and per-group-normalized pool)
    print("fitting codebooks (Lloyd-Max)...")
    cb = {
        ("perexpert", 16): fit_codebook(pe_pool, 16),
        ("perexpert", 8): fit_codebook(pe_pool, 8),
        ("pergroup", 16): fit_codebook(pg_pool, 16),
        ("pergroup", 8): fit_codebook(pg_pool, 8),
    }

    configs = []
    # baselines: uniform per-group RTN at 4 and 3 bits
    configs.append(("uniform_rtn_4b_g128", lambda W: codec_uniform_rtn(W, 4, GROUP_SIZE, AXIS), 4, 16))
    configs.append(("uniform_rtn_3b_g128", lambda W: codec_uniform_rtn(W, 3, GROUP_SIZE, AXIS), 3, 8))
    # codebook, per-group scale
    configs.append(("codebook16_pergroup_g128", lambda W: codec_codebook_pergroup(W, cb[("pergroup", 16)], GROUP_SIZE, AXIS), 4, 16))
    configs.append(("codebook8_pergroup_g128", lambda W: codec_codebook_pergroup(W, cb[("pergroup", 8)], GROUP_SIZE, AXIS), 3, 8))
    # codebook, per-expert scale
    configs.append(("codebook16_perexpert", lambda W: codec_codebook_perexpert(W, cb[("perexpert", 16)]), 4, 16))
    configs.append(("codebook8_perexpert", lambda W: codec_codebook_perexpert(W, cb[("perexpert", 8)]), 3, 8))
    # reference: INT8 RTN baseline floor
    configs.append(("int8_rtn_g128_ref", lambda W: codec_uniform_rtn(W, 8, GROUP_SIZE, AXIS), 8, 256))

    rows = []
    for cfg_name, codec, payload_bpw, n_levels in configs:
        rel_errs, cosines, n_scales_used = [], [], None
        for W in Ws:
            W_prime, n_scales = codec(W)
            f = s1.fidelity(W, W_prime, X)
            rel_errs.append(f["rel_err"])
            cosines.append(f["mean_cosine"])
            n_scales_used = n_scales
        rel = sum(rel_errs) / len(rel_errs)
        cos = sum(cosines) / len(cosines)
        # int8 ref has no shared codebook; treat codebook overhead 0 for uniform RTN configs
        if cfg_name.startswith("uniform_rtn") or cfg_name.startswith("int8"):
            bpw = s1.bits_per_weight(numw, payload_bpw * numw, scale_bits=16 * n_scales_used)
        else:
            bpw = eff_bits(numw, payload_bpw, n_scales_used, n_levels)
        vram = s1.implied_vram_gb(bpw)
        rows.append({
            "config": cfg_name,
            "bits_per_weight": round(bpw, 4),
            "rel_err_pct": round(rel * 100, 4),
            "mean_cosine": round(cos, 6),
            "implied_vram_gb": round(vram, 2),
        })
        print(f"{cfg_name:30s} bpw={bpw:7.4f}  rel_err={rel*100:7.3f}%  "
              f"cos={cos:.6f}  vram={vram:5.2f}GB")

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "bits_per_weight", "rel_err_pct",
                                           "mean_cosine", "implied_vram_gb"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {CSV_PATH}")


if __name__ == "__main__":
    main()
