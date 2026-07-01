# Candidate: Cross-Expert Shared Base + Small Deltas

## Claim
Same-layer MoE routed experts may be a shared position-wise base tensor plus
small per-expert deltas (`expert_i = base + delta_i`), so storing one base plus
128 low-entropy deltas would losslessly beat storing 128 full experts. (This is
AGENTS.md's flagship example hypothesis.)

## Why It Might Work (and why it didn't)
`layer1_expert_stats.csv` showed all 128 experts share near-identical std
(up ~0.017, down ~0.0206) and ~0 mean, and candidate
[[0001-bf16-exponent-plane]] confirmed their high-byte histograms are nearly
identical (cross-expert KL ~0.027 bits). That distributional similarity made a
shared position-wise base look plausible.

## Tensor Group
`backbone.layers.1.mixer.experts.N.up_proj.weight` (BF16 [1856,2688], N=0..127);
generalizes to all routed experts.

## Measurement (run — see Findings)
On 16 layer-1 up_proj experts (cast to float32): (1) position-wise Pearson
correlation of expert0 vs each other expert; (2) base = position-wise mean,
delta_i = expert_i - base; compare std(delta) vs std(raw) and BF16 high-byte
(exponent) plane entropy of delta vs raw.

## Promising Result (threshold)
Meaningful position-wise correlation AND delta exponent-plane entropy materially
below raw, such that base + 128 deltas stores smaller than 128 raw experts with
exact reconstruction.

## Findings — REJECTED (empirically falsified on true shard 1)
Measured on 16 experts:
- Position-wise correlation is **noise**: mean |corr| = 0.0284, max = 0.0313.
- delta std 0.01654 vs raw 0.01732 — only ~4.5% smaller.
- delta high-byte entropy 2.878 vs raw 2.932 bits — only −0.054 bit.

The tiny gain is pure distributional **centering**, not shared position-wise
structure. A lossless base+delta stores delta at the same per-value precision as
raw and gains ~nothing. **Experts share a marginal distribution, not a common
per-position base.** Do not pursue alignment- or base-subtraction-based
cross-expert schemes for this model. (Artifacts from the loop-back probe;
see scout run notes.)

Forward implication: cross-expert *structural* redundancy is exhausted. Lossless
gains on experts come only from the per-tensor BF16 plane split
([[0001-bf16-exponent-plane]]), which is near the BF16 lossless ceiling (~32%,
mantissa bits are essentially random). Further runtime/size wins on the expert
bulk likely require the lossy lighter-representation track, not lossless coding.

## Status
Rejected
