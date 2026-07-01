# Candidate: F32 Control Tensors Carry Dead Precision

## Claim
The model's small F32 control tensors store values that are EXACTLY
BF16-representable (their low 16 mantissa bits are all zero), so the bottom 2
bytes of every element are dead weight and can be dropped and reconstructed
exactly by zero-padding — a guaranteed lossless 50% on this family.

## Why It Might Work
The Mamba scout measured every sampled F32 control value and found the low 16
bits all zero: A_log [64] = 64/64 exactly BF16-representable, RMSNorm
`norm_weight` [2688] = 2688/2688 exactly BF16-representable (e.g. 0.890625,
0.89453125 — values with no low-mantissa content). If a value's low 16 bits are
zero, storing F32 spends 4 bytes to hold what 2 bytes (the BF16 form) already
holds exactly. Reconstruction is trivial and bit-exact: re-append 16 zero bits.

This is a DIFFERENT mechanism from the exponent-plane codec ([[0001-bf16-exponent-plane]]):
that idea entropy-codes a skewed distribution; this one removes provably-constant
zero bits with no coding at all. It is the rare case where the win is guaranteed,
not statistical.

## Tensor Group
All F32 tensors in the model: `backbone.layers.*.mixer.A_log` [64],
`backbone.layers.*.mixer.D` [64], `backbone.layers.*.mixer.gate.e_score_correction_bias`
[128], and all RMSNorm `*.norm.weight` / `norm_f.weight` tensors stored as F32.
Identify the full set via `manifest.csv` (dtype = F32).

## Measurement
For every F32 tensor in `manifest.csv`: load the raw bytes, reinterpret as
little-endian uint32, and test whether `(word & 0x0000FFFF) == 0` for ALL
elements (i.e. low 16 bits zero). Record the fraction of tensors that are 100%
clean and the worst-case nonzero-low-bit count for any that are not. For the
clean tensors, drop the low 2 bytes, then verify byte-exact round-trip by
zero-padding back and hash-comparing to the original. Report total F32 bytes and
the bytes saved.

## Promising Result
If a large majority of F32 tensors are 100% low-16-bits-zero, those tensors
compress exactly 50% with zero coding cost and a guaranteed exact round-trip.
Even partial coverage is a clean, free win. The idea fails only if most F32
tensors carry real low-mantissa content (low bits frequently nonzero), in which
case fall back to entropy coding instead of truncation.

## Test Target
Synthetic first to wire up the uint32 reinterpret + zero-pad round-trip check,
then run the scan across shard 1 of the true weights to measure real coverage
across the whole F32 family.

## Status
Rejected — dominated (mechanism confirmed, generalized, and mapped).

test-001: mechanism passed synthetic + true shard 1 bit-exact (4/5 F32 tensors
clean, 512 B saved of 1536 B). Lossless and correct, but the F32 family looked
KB-scale.

test-002 (try-harder): measured on the FULL model, not extrapolated. (1) The whole
F32 family is 23,552 B → ~11.6 KB free; immaterial confirmed at model scale. (2)
Generalized the mechanism to "any constant bit position across a tensor" and ran
it on all 13 shards incl. BF16: **9.72 GB (15.4%) of the BF16 mass is
provably-constant bits** — ~400,000× the F32 family. BUT every constant bit is a
top-of-exponent bit (masks 0x7000/0x6000/0x4000); **sign is never constant and the
mantissa has ZERO dead bits anywhere.** So constant-bit dropping is a weaker,
fusible-ceiling ~15–19% realization of exactly the exponent structure that
[[0009-fusible-exponent-codebook]] already gets at 25–29% — strictly dominated,
adds nothing. Direction closed.

Salvage: (a) keep `(word & 0xFFFF)==0` F32 clean-detect as a free exact-50%
sub-rule for the control-tensor family; (b) new project fact — the **BF16 mantissa
is the hard lossless frontier** (no dead bits; all realized gains are
sign+exponent). See tests/test-001.md, tests/test-002.md.
