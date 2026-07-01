# Candidate: BF16 Exponent-Plane Codec (shared table, sign-folded)

## Claim
Splitting BF16 weights into a high-byte plane (sign + top 7 exponent bits) and a
low-byte plane (1 exponent bit + 7 mantissa bits), then entropy-coding ONLY the
high-byte plane — with the sign bit folded out and a SINGLE static table shared
across all 128 same-layer experts — losslessly shrinks the MoE routed-expert
weights by roughly 25-30%, because the high byte is hyper-concentrated while the
low byte is near-uniform.

## Why It Might Work
Measured in `byte_hist_layer1_expert000_up.csv`, the stored BF16 high byte piles
almost all mass on ~8 symbols (bins 60 -> 1,285,165 and 188 -> 1,289,277 of
~4,988,928 elements; bins 56-62 and 184-190 hold nearly everything, ~240 other
bins near zero) -> measured high-byte entropy 2.99 bits/byte. The low byte is
flat (~13k-27k across all 256 values, lone spike at 32) -> measured 7.93
bits/byte, effectively incompressible.

Two model-specific levers:
- **Sign fold:** the sign-mirrored pairs (60 vs 188, 59 vs 187, 61 vs 189) carry
  near-identical counts, so the exponent magnitude distribution is independent of
  sign. Peeling the sign bit out collapses each pair to one symbol, dropping
  order-0 entropy toward ~2.9 bits.
- **Shared table:** `layer1_expert_stats.csv` shows all 128 experts have
  near-identical std (up 0.0171-0.0180, down 0.0200-0.0210) and ~0 mean, so the
  exponent distribution is statistically interchangeable across experts and one
  static table amortizes header cost to ~nothing across 128 experts (205
  same-shape tensors per `summary.json`).

Generic zstd over interleaved BF16 cannot see this because the low-entropy
exponent bits are smeared between high-entropy mantissa bytes every 2 bytes; the
de-interleave is what exposes the compressible plane.

## Tensor Group
Primary: `backbone.layers.*.mixer.experts.*.up_proj.weight` and `.down_proj.weight`
(BF16, layer-1 shapes [1856,2688] and [2688,1856], 128 experts each;
`same_shape_counts` = 205 of each; MoE block = 4,131,526,400 of the 4,991,205,008
file bytes). Secondary targets sharing the identical mechanism (validate after
the primary): `backbone.embeddings.weight` [131072,2688] and the Mamba
`in_proj`/`out_proj` projections.

## Measurement
1. De-interleave each BF16 tensor into high-byte and low-byte planes; compute
   Shannon entropy (bits/byte) of each plane. Confirm the asymmetry holds across
   several experts and both projections, not just expert000.
2. Fold the sign bit out of the high byte; recompute high-plane order-0 entropy.
3. Compute pairwise KL divergence of high-byte histograms across the 128 experts
   to confirm one shared table is valid (test up vs down separately — scale
   differs, so up and down may each need their own table).
4. Build one static rANS/Huffman table from a few experts, encode the folded
   high-byte plane of all experts with it, store the low-byte plane raw, and
   verify EXACT bit-for-bit round-trip (re-interleave == original safetensors
   bytes, hash check).
5. Report total compressed bytes vs the 4.13 GB MoE block, AND vs a zstd-max
   baseline on the raw interleaved tensors (the win must be net of what an
   off-the-shelf coder already achieves).

## Promising Result
Folded high-byte plane entropy <= ~3.5 bits/byte AND near-zero cross-expert KL
(validating the shared table), with an exact lossless round-trip landing total
near 11-12 of every 16 bits (~25-30% reduction, ~1.0-1.2 GB saved on the MoE
block) AND materially beating the zstd-max baseline. Drop the idea if high-byte
entropy is high, per-expert KL is large, or the gain does not exceed stock zstd.

## Test Target
Synthetic first (`models/synthetic/nemotron_tiny`) to prove the de-interleave +
sign-fold + re-interleave round-trip is exactly reversible and the codec plumbing
is correct, then shard 1 of the true weights for the real entropy numbers.

## Status
Passed True Weights

(Test 001: exact lossless ~32% reduction of layer-1 routed experts on true shard 1,
beats zstd-max-on-file by ~10 pts. Caveats: sign-fold yields no order-0 gain, and the
custom rANS ties zstd-on-de-interleaved-planes — the win is the de-interleave itself.
See tests/test-001.md.)

## Tested Refinements (negative — do not re-attempt)
- Sign-fold: no order-0 gain (fold total 36 B worse). Dropped.
- Cross-expert pooling: concatenating 32 experts' high-byte planes into one
  zstd-19 stream was 1.37% worse than per-tensor; a shared static rANS table
  0.38% worse. The tiny cross-expert KL (~0.027 bits) that makes a shared table
  lossless-safe also means there is ~nothing to amortize, while pooling costs
  per-tensor adaptivity. Keep per-tensor plane compression.
- Position-wise cross-expert base+delta falsified — see
  [[0003-cross-expert-base-delta]].

Strategic framing (per research/notes/compression-vs-compute-payoff): this is a
Regime-C lossless method. Honest payoff = storage / transfer / resident-VRAM, not
decode speed; keep it off the per-token critical path or expect a ~1.4-2x batch-1
slowdown.
