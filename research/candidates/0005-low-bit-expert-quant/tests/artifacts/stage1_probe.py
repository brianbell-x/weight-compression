"""Stage-1 matmul-fidelity harness for low-bit routed-expert quantization (candidate 0005).

This is the reusable *validator* the runtime-track sweep agents import. It judges a
candidate codec not by byte round-trip but by how much it perturbs the matmul a routed
expert actually performs: for an expert weight W and a fixed input batch X, the metric
is the relative output error ||X@W - X@W'|| / ||X@W|| and the mean per-row cosine of the
two outputs, where W' is the codec's dequantized reconstruction.

Why per-tensor / per-matmul (not full inference): the model is 63 GB and this box has
33.7 GB RAM and no CUDA, so a full forward will not run. Every function here operates on
ONE expert matrix at a time, loaded via safetensors.safe_open, so the RAM blocker never
applies. See research/notes/capability-eval-path.md (Stage 1) and the candidate brief.

All math is done in float32 on CPU. bf16 expert weights are cast to float32 on load.

Public API
----------
load_expert(path, name) -> torch.Tensor [float32]
    Load a single named tensor from a .safetensors shard, cast to float32.

make_inputs(in_features, batch=256, seed=0) -> torch.Tensor [float32, batch x in_features]
    Deterministic unit-norm (per row) random input batch. Same seed => same X, so two
    codecs are compared on identical activations.

fidelity(W, W_prime, X) -> dict(rel_err, mean_cosine)
    rel_err     = ||X@W - X@W'||_F / ||X@W||_F   (Frobenius relative output error)
    mean_cosine = mean over rows of cosine(row of X@W, row of X@W')

bits_per_weight(num_weights, payload_bits, scale_bits, codebook_bits) -> float
    Effective bits/weight INCLUDING scale + codebook overhead. Use this for the x-axis
    of every size/fidelity tradeoff so overhead is never hidden.

implied_vram_gb(bits_per_weight) -> float
    Full-model resident size at `bits_per_weight` on the routed experts, using the
    measured size table: 4.4 GB non-expert floor + 29.4e9 expert params * b/8 bytes.

Helper (used by the self-test, handy for sweep baselines)
---------------------------------------------------------
int8_per_group_rtn(W, group_size=128, axis=1) -> (W_prime, meta)
    Symmetric per-group round-to-nearest INT8 quantize+dequantize baseline. Returns the
    float32 reconstruction and a meta dict with the effective bits/weight of this codec
    (8 payload bits + one fp16 scale per group).
"""

from __future__ import annotations

from typing import Optional

import torch
from safetensors import safe_open

# --- Measured model constants (from all 13 shard headers; see brief.md) -------------
ROUTED_EXPERT_PARAMS = 29.4e9   # routed-expert params = 93% of the BF16 model
NON_EXPERT_FLOOR_GB = 4.4       # mamba + attention + embeddings + shared experts, kept as-is
_BYTES_PER_GB = 1024 ** 3       # GiB (binary), matches how resident VRAM is usually reported


# =====================================================================================
# Loading
# =====================================================================================
def load_expert(path: str, name: str) -> torch.Tensor:
    """Load tensor `name` from the safetensors shard at `path`, returned as float32.

    Only the requested tensor is read into memory (safetensors does a sliced/zero-copy
    read), so this works on the 5 GB shard without loading the whole model.
    """
    with safe_open(path, framework="pt") as f:
        W = f.get_tensor(name)
    return W.to(torch.float32)


# =====================================================================================
# Fixed input batch
# =====================================================================================
def make_inputs(in_features: int, batch: int = 256, seed: int = 0) -> torch.Tensor:
    """Deterministic input batch X of shape [batch, in_features], float32.

    Rows are random Gaussian then L2-normalized to unit norm, so ||X@W|| reflects the
    operator norm of W along sampled directions without an arbitrary input-scale factor.
    Seeded so every codec in a sweep is scored on the *same* X.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(batch, in_features, generator=g, dtype=torch.float32)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return X


# =====================================================================================
# Fidelity metric (the validator)
# =====================================================================================
def fidelity(W: torch.Tensor, W_prime: torch.Tensor, X: torch.Tensor) -> dict:
    """Matmul-fidelity of reconstruction W' vs original W on input batch X.

    W, W' have shape [in_features, out_features]; X has shape [batch, in_features].

    Returns dict:
      rel_err     -- ||X@W - X@W'||_F / ||X@W||_F  (Frobenius relative output error)
      mean_cosine -- mean over rows of cosine(row of X@W, row of X@W')
    """
    W = W.to(torch.float32)
    W_prime = W_prime.to(torch.float32)
    X = X.to(torch.float32)

    Y = X @ W
    Y_prime = X @ W_prime

    denom = Y.norm()
    rel_err = ((Y - Y_prime).norm() / denom.clamp_min(1e-12)).item()

    cos = torch.nn.functional.cosine_similarity(Y, Y_prime, dim=1, eps=1e-12)
    mean_cosine = cos.mean().item()

    return {"rel_err": rel_err, "mean_cosine": mean_cosine}


# =====================================================================================
# Cost accounting
# =====================================================================================
def bits_per_weight(
    num_weights: int,
    payload_bits: float,
    scale_bits: float = 0.0,
    codebook_bits: float = 0.0,
) -> float:
    """Effective bits per weight INCLUDING scale and codebook overhead.

    payload_bits  -- total bits of quantized index payload for `num_weights` weights
                     (e.g. 4 bits/weight INT4 over 1M weights = 4_000_000).
    scale_bits    -- total bits spent on per-group/per-expert scales (and zero-points).
    codebook_bits -- total bits spent on shared codebook(s), amortized over these weights.

    Returns total_bits / num_weights.
    """
    total_bits = payload_bits + scale_bits + codebook_bits
    return total_bits / num_weights


def implied_vram_gb(bits_per_weight: float) -> float:
    """Full-model resident size (GB) with routed experts at `bits_per_weight`.

    = 4.4 GB non-expert floor + 29.4e9 expert params * bits/8 bytes.
    INT8 -> ~34 GB, INT4 -> ~19.5 GB, INT3 -> ~15.9 GB (matches brief.md).
    """
    expert_bytes = ROUTED_EXPERT_PARAMS * bits_per_weight / 8.0
    return NON_EXPERT_FLOOR_GB + expert_bytes / _BYTES_PER_GB


# =====================================================================================
# INT8 per-group RTN baseline (also the self-test codec)
# =====================================================================================
def int8_per_group_rtn(
    W: torch.Tensor,
    group_size: int = 128,
    axis: int = 1,
) -> tuple[torch.Tensor, dict]:
    """Symmetric per-group round-to-nearest INT8 quant+dequant of W.

    Groups of `group_size` weights along `axis` share one fp16 max-abs scale; each group
    is mapped to signed 8-bit levels [-127, 127] and dequantized back to float32. This is
    the safe ~2x resident-VRAM baseline (brief: ~0.68% output error).

    Returns (W_prime, meta) where meta has:
      group_size, axis,
      bits_per_weight -- effective bits/weight incl. one fp16 scale per group.
    """
    W = W.to(torch.float32)
    orig_shape = W.shape
    # Move the quantized axis to the end so groups are contiguous along dim -1.
    Wt = W.movedim(axis, -1)
    moved_shape = Wt.shape
    n_along = moved_shape[-1]
    if n_along % group_size != 0:
        raise ValueError(
            f"axis length {n_along} not divisible by group_size {group_size}"
        )
    n_groups = n_along // group_size

    Wg = Wt.reshape(*moved_shape[:-1], n_groups, group_size)
    max_abs = Wg.abs().amax(dim=-1, keepdim=True)            # one scale per group
    scale = (max_abs / 127.0).clamp_min(1e-12)              # stored as fp16 in practice
    q = torch.clamp(torch.round(Wg / scale), -127, 127)
    Wg_hat = q * scale

    W_prime = Wg_hat.reshape(moved_shape).movedim(-1, axis).reshape(orig_shape)

    num_weights = W.numel()
    total_groups = num_weights // group_size
    meta = {
        "group_size": group_size,
        "axis": axis,
        "bits_per_weight": bits_per_weight(
            num_weights,
            payload_bits=8 * num_weights,
            scale_bits=16 * total_groups,   # one fp16 scale per group
        ),
    }
    return W_prime, meta


# =====================================================================================
# Self-test: load ONE real layer-1 expert and run INT8 RTN through fidelity()
# =====================================================================================
if __name__ == "__main__":
    SHARD = (
        r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        r"\hf_snapshot\model-00001-of-00013.safetensors"
    )
    NAME = "backbone.layers.1.mixer.experts.0.up_proj.weight"

    print(f"[self-test] loading {NAME}")
    W = load_expert(SHARD, NAME)
    print(f"[self-test] shape={tuple(W.shape)} dtype={W.dtype}")

    # up_proj.weight is [in_features=1856, out_features=2688]; X must match in_features.
    in_features = W.shape[0]
    X = make_inputs(in_features, batch=256, seed=0)

    W_prime, meta = int8_per_group_rtn(W, group_size=128, axis=1)
    f = fidelity(W, W_prime, X)

    print(f"[self-test] INT8 per-group RTN (group_size=128)")
    print(f"[self-test]   rel_err      = {f['rel_err'] * 100:.3f}%")
    print(f"[self-test]   mean_cosine  = {f['mean_cosine']:.6f}")
    print(f"[self-test]   bits/weight  = {meta['bits_per_weight']:.4f}")
    print(f"[self-test]   implied_vram = {implied_vram_gb(meta['bits_per_weight']):.2f} GB")

    assert f["rel_err"] < 0.02, f"INT8 rel_err {f['rel_err']:.4f} unexpectedly high"
    print("[self-test] PASS")
