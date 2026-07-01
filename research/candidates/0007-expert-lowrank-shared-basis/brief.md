# Candidate: Low-Rank / Shared-Basis Expert Density

## Claim
The 128 same-layer MoE experts might share a low-dimensional basis (atoms stored
once) with small per-expert cores, or each expert might be individually low-rank —
giving structural density (fewer stored numbers for the same linear map), computed
natively as a factored matmul. (First concrete test of the density direction in
research/notes/density-and-native-compute.md, and of the user's "shared atoms"
intuition.)

## Why It Might Work
Experts share a near-identical marginal distribution (cross-expert KL ~0.027), and
trained weight matrices are often partly low-rank. If experts span a common
subspace smaller than the union, one shared basis + per-expert cores would beat
storing 128 full matrices.

## Tensor Group
`backbone.layers.1.mixer.experts.N.up_proj.weight` [1856,2688] (sampled).

## Measurement (run)
SVD of 8 experts. Stage-1 matmul-output error vs truncation rank (per-expert) and
vs shared-subspace dimension R (row-stacked SVD). Stored-number ratio at the rank/R
giving ~2% output error.

## Findings — REJECTED (full-rank, independent subspaces)
- **Per-expert NOT low-rank**: 99% Frobenius energy needs rank 0.79 of full; ~2%
  matmul-output error needs rank 0.96. Factored U+V at that rank stores **1.63x**
  the dense matrix — worse than dense.
- **No shared basis**: shared subspace reconstruction error 0.90/0.83/0.70/0.49 at
  R=128/256/512/1024; ~2% is never reached below the full ambient dim (~2688),
  where the stored ratio is ~1.01 (no compression). No R is both small and accurate.
- Experts require nearly independent, near-full-rank subspaces. Low-rank and
  shared-basis structural density are dead for these tensors.

## The convergence (why this matters)
This is the third independent way the experts resist compression, alongside
[[0001-bf16-exponent-plane]] (high-entropy mantissa, lossless capped ~32%) and
[[0003-cross-expert-base-delta]] (no position-wise cross-expert structure). The
trained expert matrices are statistically dense / random-like; their capacity is
used up. **No structural slack exists in the current weight representation.**

## Redirect (where density actually lives)
Two places, both away from post-hoc linear re-representation of finished weights:
1. **Activation structure, not weight structure.** The weights are full-rank, but
   the *activations* the model actually produces are low-dimensional / have outlier
   channels. That is the only exploitable structure left and is exactly what
   AWQ/GPTQ-class methods use — and what candidate [[0005-low-bit-expert-quant]]'s
   sub-4-bit path is blocked on. Needs real activations (see ledger keystone).
2. **Train-time density, not post-hoc.** "Weights that fundamentally look different
   but carry the same information tighter" cannot be extracted from an
   already-maximally-trained dense matrix — it must be built in: structured layers
   (Monarch/butterfly) or trained-in superposition, learned from scratch. This is
   an architecture/training direction, not a compression-of-existing-weights one.

## Status
Rejected
