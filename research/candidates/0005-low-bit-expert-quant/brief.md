# Candidate: Low-Bit Routed-Expert Quantization (the runtime track)

## Claim
The routed MoE experts are 93% of the model (29.4B params / 58.75 GB of 63.2 GB),
so quantizing ONLY them to a narrow, read-into-compute form is the dominant lever
for making this model lighter to RUN. INT8 with per-group scales is a safe ~2x
resident-VRAM win (measured matmul output error 0.68%); the open research problem
is how far below 4 bits we can push while keeping per-layer output error in a
capability-safe band, by exploiting these experts' measured structure.

## Why It Might Work
Measured exactly from all 13 shard headers: routed experts = 29,374,808,064
params = 93.0% of the BF16 model; the non-expert floor (mamba, attention,
embeddings, shared experts) is only 4.4 GB. So the resident-VRAM prize lives
almost entirely in the experts: quantizing just them gives full-model resident of
~34 GB at INT8, ~19.5 GB at INT4, ~15.9 GB at INT3 (frees ~29 / ~44 / ~47 GB).
This is a Regime-D target (per research/notes/compression-vs-compute-payoff):
weights stay narrow into a fused matmul, never re-inflated to full width in VRAM —
the only track that reduces resident VRAM AND decode cost at once.

Measured structure to exploit (and its limits):
- Experts are Gaussian-ish, ~0 mean, near-identical distribution SHAPE across the
  128 experts (cross-expert high-byte KL ~0.027).
- BUT they differ in per-group MAGNITUDE/max-abs, so a single shared grid fails
  (shared-grid INT4 output error 38% vs per-expert 12%) — see
  [[0003-cross-expert-base-delta]] family of negative results. Per-expert (or
  per-group) max-abs scales are tiny (1 fp16 / 128 weights) and indispensable.
- Naive symmetric group-wise INT4 (RTN) gives 12% output error — unusable when
  compounded over 23 MoE layers. A Gaussian-fit 16-level codebook (shape-shared,
  per-expert scaled) trims it to 9.6%. So sub-4-bit needs more structure; that
  gap IS the research.

## Tensor Group
All routed-expert weights: `backbone.layers.{MoE layers}.mixer.experts.N.up_proj.weight`
[1856,2688] and `.down_proj.weight` [2688,1856], 128 experts x 23 MoE layers
(5,888 tensors). Start on layer 1, then sample across MoE layers.

## Measurement
Use the Stage-1 matmul-fidelity probe (research/notes/capability-eval-path.md):
for sampled experts, apply each candidate codec to get W', and on a fixed input
batch X compute relative output error ‖XW−XW′‖/‖XW‖ and cosine. Sweep and compare:
1. Baselines: INT8 per-group RTN, INT4 per-group RTN (expected ~0.7% and ~12%).
2. Group size sweep (128 → 64 → 32) at 4 bits — does finer granularity rescue INT4?
3. Shape-shared non-uniform codebook (k-means/Lloyd-Max fit to pooled population)
   WITH per-expert scale, at 4 and 3 bits.
4. Salient-channel mixed precision: keep the top-k highest-magnitude input
   channels (or highest activation-energy channels if real activations available)
   at INT8, rest at INT4/INT3; measure error vs extra bits spent.
5. Report bits/weight (incl. scale overhead) vs output error for each, and the
   resident-VRAM each implies from the size table above.

## Promising Result
A lever (or stack) that holds per-layer matmul output error under ~1-2% at <=4
effective bits/weight — i.e. better than the ~19.5 GB INT4 point without INT4's
12% error. INT8 already clears the bar as a safe floor (~2x, 0.68%); the win
worth escalating is anything that reaches INT4-class size at INT8-class fidelity.
Survivors escalate to Stage-2 single-forward KL once inference is runnable (the
proxy is necessary, not sufficient — it can't see 52-layer error compounding or
router top-6 flips).

## Test Target
True weights directly — quantization error depends on the trained value
distribution, which the synthetic snapshot (random values) cannot represent.
Layer-1 experts are in shard 1, already local; no full-model load needed (the
Stage-1 probe works per-tensor, sidestepping the RAM blocker).

## Status
INT8 Confirmed end-to-end (0008); INT4 (plain RTN) is functional-but-degraded at scale (+5.1% ppl, 8% top-1 flips on 41 prompts) — fits-vs-doesnt-fit only, not a clean second halving

(test-001 + test-002: INT8 per-group RTN = safe ~2x runtime floor, 0.28-0.63% error,
validated on true weights AND real captured layer-1 activations [187,2688]. Sub-4-bit
fails every lever — data-free AND activation-aware. AWQ's scale search is inert against
per-group RTN (best gain from its clip term, → 4.71% at 4.17 b/w); GPTQ overfits the 187
cached tokens (XᵀX rank 1939/2688) and does NOT beat RTN on held-out (5.45% vs 5.12%);
down_proj worse (~10-20%). Real activations also calibrate Stage-1: the random-X proxy
over-states error ~2.4x (real codec numbers ~0.42x of random). Best 4-bit ~4.7%, still
~2.4x outside the bar. INT4's ~5% per-layer is only reopenable if Stage-2 end-to-end eval
shows the model tolerates it → candidate 0008. Real-activation capture path now works and
is reusable: tests/artifacts/activation_capture.py, ~3.55 GB peak, ~2.3 s. See tests/.)

Scout note 2026-06-30: selective mixed precision is a 0005 follow-up, not a new
candidate. `tools/fused_eval_mixed.py` is already testable/in progress and uses the
0008 streamed Stage-2 harness to compare policies such as `u4d8`, `u8d4`,
`last6_int8`, and `even6_int8` against BF16/INT8/INT4 on KL, perplexity, top-1,
router overlap, drift, effective expert bits, and implied VRAM. Litmus: still
Regime-D/narrow-into-compute. Promising result would be INT8-class behavior
(KL≈1e-3, flat perplexity/top-1) at materially below all-INT8 resident VRAM.
Record completed results as another 0005 test report; do not split to 0009 unless
it becomes a principled learned sensitivity/bit-allocation method rather than a
hand-chosen policy sweep.
