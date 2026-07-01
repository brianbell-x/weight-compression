# Findings Ledger

Rolled-up record of what experiments have settled, so scouting builds on prior
results instead of re-deriving them. Update this when a candidate resolves.
See `research/notes/compression-vs-compute-payoff.md` for the cost-axis framing
that reorganized our priorities.

## Confirmed

- **BF16 de-interleave plane split is the lossless lever** (candidate 0001,
  Passed True Weights). Splitting BF16 into a high-byte (sign+exponent) plane and
  a low-byte (mantissa) plane, then entropy-coding the high plane, gives ~32%
  exact lossless reduction on layer-1 routed experts and beats zstd-on-whole-file
  by ~10 pts. Measured: high-byte entropy ~2.9 bits, low-byte ~7.95 bits.
- **Cross-expert distribution is near-identical** but only as a *marginal*
  distribution: high-byte histogram KL across 128 experts ~0.027 bits. A single
  shared static entropy table is therefore lossless-safe.
- **F32 control tensors (A_log, D) are exactly BF16-representable** → free exact
  50% truncation (candidate 0002). True but ~KB scale; salvage as a sub-rule.
- **The BF16 mantissa is the hard lossless frontier** (candidate 0002, test-002).
  Full-model scan for provably-constant bit positions: 9.72 GB / 15.4% of the BF16
  mass is constant bits, but **100% of them are top-of-exponent bits** (masks
  0x7000/0x6000/0x4000) — sign is never constant, and the 7-bit mantissa has
  **ZERO** dead bits in *any* tensor. All realized lossless gains (0001, 0009)
  come from sign+exponent; the mantissa is fully live. Corollary: any further
  lossless progress must attack mantissa *statistics*, not dead bits.

## Falsified — do not re-propose

- **Sign-fold** of the high byte: no order-0 entropy gain (slightly worse). (0001)
- **Cross-expert pooling**: concatenating expert high-byte planes into one stream
  is ~1.4% *worse* than per-tensor; shared rANS table ~0.4% worse. Tiny KL means
  nothing to amortize. (0001 refinements)
- **Cross-expert base+delta** (AGENTS.md flagship): experts are position-wise
  uncorrelated (mean |corr| ~0.03); delta is no cheaper than raw. Experts share a
  distribution, not aligned values. (candidate 0003)
- **RMSNorm-as-F32 premise**: norms are stored BF16 in this model, so there is no
  F32 norm family to harvest. (0002)
- **Constant-bit dropping (generalized F32 dead-precision)**: 15.4% of BF16 is
  provably-constant bits, but all exponent (fusible ceiling ~13–15 bits/elem =
  6–19%) — strictly dominated by 0009's exponent codebook (25–29%). Same field,
  less of it; adds nothing. (candidate 0002, test-002)
- **Embedding vocab-tail row redundancy**: no untrained high-ID tail exists —
  tokenizer defines a string for every ID in [0, 131071], 0 zero/constant rows.
  Embeddings have only 16 removable dup rows (net LOSS after index cost); lm_head
  has 710 (the reserved <SPECIAL_0..999> output rows, at LOW IDs) for +0.5%, and
  0001's entropy coder already captures those low-norm rows, so it is not additive.
  Storage-only and immaterial. (candidate 0004, Rejected)

## Key strategic lesson (the pivot)

Exact lossless compression only moves the Storage / Load / Resident-VRAM axes; it
does NOT make decode faster and on the per-token path is slower (0001 is a
Regime-C / DFloat11-family method). Decode is memory-bandwidth bound. To make the
model cheaper to **run**, weights must stay narrow into a fused matmul and never
be re-inflated to full width in VRAM (Regime D, lossy, capability-validated). For
this MoE the dominant lever is the **resident expert bulk** (~30B resident, ~3B
active/token). Primary scouting target is now that runtime track; lossless work
continues as a secondary storage product and structure-learning tool.

## Runtime track — sized and opened

- **The prize is the experts**: routed experts = 29.4B params / 58.75 GB = **93%
  of the 63.2 GB model**; non-expert floor is 4.4 GB. Quantizing only experts:
  INT8 → ~34 GB resident (frees ~29), INT4 → ~19.5 GB (frees ~44), INT3 → ~15.9 GB
  (frees ~47). (runtime-pivot probe A)
- **INT8 per-group is a safe ~2x** (matmul output error 0.68%). **Naive INT4 is
  unusable** (12% output error even per-expert; shared grid 38%). Shape-shared +
  per-expert-scaled k-means16 trims INT4 to 9.6% — still too high. Sub-4-bit needs
  more structure (finer groups, salient-channel mixed precision); that gap is the
  research. (runtime-pivot probe B) → candidate **0005 (Testing)**.
- **Shared quantization grid is dead**: experts share distribution *shape*, not
  *scale*; per-expert/per-group scales are tiny and indispensable.
- **Inference does not run here** (63 GB model, 33.7 GB RAM, no GPU; RAM is the
  blocker, not kernels). Capability-eval path fixed in
  `research/notes/capability-eval-path.md`: Stage-1 matmul-fidelity probe (runs
  now, no inference) → Stage-2 single-forward KL (needs RAM solved).

## Split-quant / precision-ladder — tested, rejected (candidate 0006)

User idea: low-bit base resident for all experts + higher-precision residual paged
on-demand for the ~6 active experts/token. Measured (24 experts): fidelity recovery
WORKS (INT4 base 12.2% → INT4 base+INT4 residual 0.885%, INT8-class), but the
residual is INCOMPRESSIBLE (entropy 3.87/4 bits, 9.9% zeros) — it's a second full
INT4 tensor, not a cheap top-up. So base+residual = 8 bits = just INT8, bits merely
relocated (4b VRAM + 4b host). Per-token residual paging ≈ 690 MB ≈ 28–43 ms/token —
dominates decode. Only a fits-vs-doesn't-fit play; resident INT8 (0005) is strictly
better when VRAM allows. Lesson: experts have NO cheap/sparse residual; sub-4-bit
must be SELECTIVE (salient-channel mixed precision), not a uniform top-up.

## Density / native-compute direction (new, exploratory)

A second axis orthogonal to quantization: instead of fewer bits per number, fewer
parameters carrying the same function, or multiple functions sharing the same
parameters (superposition) — with the math run directly on the compact form, never
unpacked. "Small by construction and runs that way," not compress+decompress. See
`research/notes/density-and-native-compute.md`. First concrete probe to design:
reconstruct a MoE layer's 128 experts from a shared learned dictionary of K atoms +
per-expert coefficients, measured by matmul-output fidelity vs K and stored-number
count, run natively. Composes with 0005.

## 0005 Stage-1 result + the critical path (real activations)

Stage-1 matmul-fidelity sweep on true layer-1 experts (test-001):
- **INT8 per-group RTN confirmed**: 0.67% output error, cos 0.99998, 32.2 GB
  resident. Bankable ~2x runtime win. (candidate 0005, status Needs Deep Analysis)
- **All data-free sub-4-bit levers fail**: best 4-bit = 9.66% (non-uniform per-group
  codebook); group-size and weight-magnitude salient mixing each shave only ~1 pt.
  The 4-bit grid error (~10-12%) is the wall; structure-only levers don't breach it.
- **Why it's blocked, not closed**: the levers that actually close the 4-bit gap
  (AWQ activation-energy saliency, GPTQ Hessian error-feedback) need REAL
  activations; Stage-1's random-X is blind to them. Salient test used *weight*
  max-abs, not *activation* energy — so activation-driven saliency is untested.

**THE BINDING CONSTRAINT is now real activations.** Both sub-4-bit quant and proper
capability eval need them. Key unlock: capturing layer-1 expert-input activations
only needs a forward through embeddings + layers 0-1 (~few GB), which FITS in
33.7 GB RAM — no full 63 GB load. Standing up a partial early-layer forward to cache
real activations is the keystone next infra step; it unblocks 0005's sub-4-bit path
AND lets density/fidelity probes use real X instead of Gaussian.

## Density probe result — structural density dead, convergence clear (0007)

Experts are full-rank (2% error needs rank 0.96; factoring stores 1.63x dense) with
independent subspaces (shared basis never usable below full ambient dim). Low-rank
and shared-basis density are dead. Third independent confirmation (with 0001's
high-entropy mantissa and 0003's no-position-wise-structure) that the trained
experts are statistically dense/random-like — NO structural slack in the current
weight representation. Density must come from (a) ACTIVATION structure, not weight
structure (the only post-hoc lever left; AWQ/GPTQ use it; = 0005's blocker), or
(b) TRAIN-TIME architecture (structured layers / trained-in superposition), not
post-hoc re-representation.

**Unifying conclusion**: every post-hoc lever now routes through REAL ACTIVATIONS.
The weights are dense; the activations are structured. Capture them via a partial
early-layer forward (embeddings + layers 0-1, fits in 33.7 GB RAM) — the keystone.

## Keystone built — real activations captured, NOW A REUSABLE TOOL (unblocks runtime track)

`research/candidates/0005-low-bit-expert-quant/tests/artifacts/activation_capture.py`
runs embeddings → layer-0 Mamba2 (pure-PyTorch CPU torch_forward) → layer-1 pre-mixer
RMSNorm on real prompts, capturing the routed-expert / gate input (post
`backbone.layers.1.norm`) WITHOUT loading the full model — only shard 1's upstream
weights touched, the 128 experts (58 GB) never loaded. **Peak RAM 3.55 GB, wall ~2.3 s**
(dominated by the 1.41 GB embedding table); ~9x under the 33.7 GB budget. Output:
`tests/artifacts/activations/real_X_layer1.npy` ([187,2688] float32, 12 prompts) +
`channel_energy_layer1.npy` [2688]. Engineering hurdles solved: mamba_ssm import stub,
gated-RMSNorm reimplemented in pure torch, CPU cuda.stream bypass. **This is now the
standing tool for any post-hoc activation-aware lever or future Stage-2 work.**

Activations show the structure random-X lacked: per-input-channel energy **max/mean =
4.30** (random unit-norm X ≈ 1.2) — the outlier subspace AWQ/GPTQ target.

## Post-hoc quantization concluded — INT8 is the floor (0005 test-002)

With REAL captured activations [187,2688], both previously-untestable levers are now
tested and REJECTED for clearing the bar:
- **INT4 RTN improves to 5.07%** on real X (vs 12.28% random) — real activations live in
  a structured outlier subspace that plain per-group RTN already captures.
- **AWQ's scale search is inert** (best α drives 5.069%→5.067%): per-group RTN already
  normalizes each 128-col group by its max-abs, so AWQ's rescale only steals resolution.
  Its clip term + 1% salient-INT8 reach 4.71% at 4.17 b/w — still ~2.4x outside the bar.
- **GPTQ does NOT beat RTN on held-out tokens** (5.45% vs 5.12%): XᵀX from 187 tokens is
  rank 1939/2688; a damping sweep shows it generalizes better the more it's regularized
  back toward RTN — textbook overfitting. Correct impl (in-sample 5.07%→2.88%) but data-
  starved; needs ~1e5 tokens. down_proj worse (10–20%, MoE routing starves second hop).
- **Stage-1 calibration**: real X over-states nothing — the random-X proxy OVER-states
  error ~2.4x (every codec ≈ 0.42x of its random number); ranking preserved, so treat
  test-001's absolute rel_err as a ~2.4x conservative upper bound.

**INT8 (up 0.28% / down 0.63%, 8.125 b/w, ~32 GB) is the confirmed runtime floor;
sub-4-bit is REJECTED at the proxy.** Consistent with experts being dense (full-rank,
high-entropy — 0003, 0007). INT4's ~4.7–5% per-layer is reopenable only by Stage-2
end-to-end eval (candidate 0008).

## The big picture (both tracks mapped to their floors)

- **Storage (lossless)**: ~32% via BF16 plane split (0001). Capped.
- **Runtime (lossy quant)**: INT8 ~2x (0005). Nothing below survives the proxy.
- **Structural density**: dead (full-rank, no shared basis — 0007).
- **The experts are fundamentally dense.** Post-hoc compression of this finished
  model is thoroughly mapped. The two remaining real frontiers:
  1. **Stage-2 end-to-end eval** (candidate 0008): does INT8 truly preserve
     capability, and is INT4's ~5%/layer actually catastrophic or tolerable? Needs a
     streamed full forward (extends the working activation harness; RAM-feasible).
  2. **Train-time density** (research/notes/density-and-native-compute.md): the
     "fundamentally different weights" win can't be extracted post-hoc; it must be
     trained in (structured layers / superposition from scratch). A from-scratch
     research program, not compression of existing weights.

## Train-time density exp1 — parity, with one redundancy-based win (research/traintime/exp1)

Capability-per-parameter of dense vs structured FFN layers on a small char-LM (CPU,
seeded). At EQUAL params, no structured family (low-rank, Monarch, shared-dict) beats
dense — no free lunch from factorizing a densely-trained matrix. The ONLY per-param
win: shared-dictionary at low K recovers ~99% of dense val at ~0.5-0.66x params, but
that is weight-TYING across redundant FFN sites (storage win; fails native-compute,
rebuilds the matrix each forward). Big caveat: task is capacity-SATURATED (dense val
flat ~1.17-1.19 across the sweep), so it can't strongly discriminate families.

Through-line (whole project): density wins come from REDUNDANCY; a model that needs
its capacity has little to give. Post-hoc experts = dense, no slack. Train-time win
appeared only where the small task left weights redundant. Exp2 must use a
CAPACITY-BOUND task (dense val keeps dropping with params) and add a superposition
probe, to test whether any per-param win survives capacity pressure.

## Train-time density exp2 — the synthesis (research/traintime/exp2)

On a CAPACITY-BOUND task (real prose, dense val drops 1.68→1.56):
- **Structural weight-sharing does NOT beat dense per-param.** shared_dict is
  strictly worse at every matched-param point (~0.05-0.08 nats). K=4 is expressively
  equivalent to dense yet trains ~0.05 nats worse — the coupled parameterization
  hurts OPTIMIZATION, not just capacity. exp1's low-K win was a redundancy artifact
  of the easy task. (Confirms structural density dead from the train side too.)
- **Superposition (occupancy density) IS real but gated by SPARSITY.** Toy model:
  dense inputs → d dims carry ~d features (F/d≈1-2); sparse inputs (1% active) →
  d dims carry ~10-16x more features. Faithful overcompleteness ~1/density.

### THE SYNTHESIS (both tracks agree)
The weights are dense and incompressible; the exploitable structure lives in the
ACTIVATIONS — their sparsity. Density that works exploits activation sparsity, not
weight redundancy. (Post-hoc weight compression capped; AWQ only slight traction;
structural sharing fails; superposition succeeds only when sparse.) Nemotron is
already a sparse MoE (6/128 active) — the regime where superposition works. The
"fundamentally different weights" win, if any, is **sparsity-gated superposition**:
more effective capacity in a shared/superposed parameter pool read out SPARSELY.

→ exp3: a wide-hidden FFN with top-k SPARSE activation built from a shared atom
pool — does sparsity make superposed weights non-interfering enough to beat dense
per-param? (The train-time version of finer sparse experts sharing a param pool.)

## Stage-2 end-to-end eval done — INT8 confirmed, INT4 REOPENED (0008, Passed True Weights)

Streamed full forward of all 52 layers within ~11 GB RAM (`tools/streamed_forward.py`
+ `tools/fused_eval.py`, one disk load for all three conditions, per-layer checkpoint
to survive the ~10-min background kill). BF16 sanity passed (Paris/oxygen/east/blue,
ppl geomean 11.30). End-to-end next-token distributions vs BF16 over 8 prompts:
- **INT8 experts: KL 3.5e-4, ppl 11.30 (unchanged), top-1 100%, router overlap 99.4%.**
  The ~2x VRAM win is confirmed on real behavior, not just proxy — ships as the floor.
- **INT4 experts: KL 0.076, ppl 11.36 (+0.5%), top-1 96%, router overlap 92.5%.**
  Despite ~5% per-layer Stage-1 error, end-to-end behavior barely moves — the per-layer
  error does NOT compound catastrophically. **The Stage-1 proxy was too pessimistic;
  the ~18.5 GB INT4 target is REOPENED.** Caveat: 8 prompts / ~50 positions is
  under-powered — shows INT4 is *not broken*, not yet that it's production-equivalent.
- New reusable tool: `streamed_forward.py` is the standing Stage-2 harness (any future
  lossy lever can be measured end-to-end on real prompts within RAM).

## Stage-2 end-to-end eval — INT4 REOPENED (candidate 0008, Passed True Weights)

Streamed full forward built and run (tools/streamed_forward.py + fused_eval.py,
peak RAM ~11 GB, one layer at a time, checkpointed). BF16 sanity passes (correct
factual completions). End-to-end next-token vs BF16 (8 prompts):
- **INT8 experts: KL 0.0003, perplexity unchanged (11.30), 100% top-1, 99.4% router
  overlap.** The ~2x VRAM win is confirmed on real behavior. Ship as floor.
- **INT4 experts: KL 0.076, perplexity 11.30→11.36 (+0.5%), 96% top-1, 92.5% router
  overlap.** The ~5% per-layer Stage-1 error does NOT compound catastrophically — it
  largely washes out. **The Stage-1 matmul proxy was too pessimistic.** This REOPENS
  the ~18.5 GB INT4 target: a potential ~3.4x runtime reduction at +0.5% perplexity.

Caveat: under-powered (8 prompts, ~50 positions; next-token only, no generation
drift). Shows INT4 is not broken, not that it's production-equivalent. Next: power-up
eval (100-200 diverse prompts + held-out perplexity + a generation task). If INT4 KL
stays small at scale, INT4 experts become the headline runtime deliverable.

LESSON: matmul-fidelity proxies over-penalize per-layer error; cross-layer error
cancels. Validate quant end-to-end, not just per-tensor. (Re-evaluate 0005's
"sub-4-bit rejected" — it was rejected at the proxy, which Stage-2 just overturned.)

## Train-time density exp3 — superposition: right trend, confounded test (research/traintime/exp3)

Sparse-superposed FFN did NOT beat dense per-param (best gap +0.076). BUT: (1)
sparsity helps directionally (k/H 1.0→0.12 improves val — mechanism real, partial,
optimum ~12%); (2) the test was CONFOUNDED — input-side atom sharing made W1 rank-K
(16 input dirs), so widening H trapped units in a 16-dim subspace (bottleneck artifact,
not a verdict). Fair test = exp4: full-rank W1, sharing/sparsity on a wide hidden→output
dictionary, bigger d_model (64 too small for superposition to pay).

Meta-read (exp1+2+3): no weight-structure/superposition scheme beats dense
capability-per-param at this scale. The only reliable lever is ACTIVATION SPARSITY — a
runtime/compute lever (= what MoE already does), not weight density. exp4 is
superposition's fairest shot; if it fails cleanly, train-time density concludes: dense
is ~optimal per-param at small scale; the real win is activation sparsity (MoE).

## Train-time density CONCLUDED — NEGATIVE (exp4, the fair test)

exp4 removed exp3's confound (full-rank W1, d_model 64→128, output-side sharing). The
corrected design finally crossed below dense (gap +0.076 → −0.032) — but the win is NOT
real: it only appears at the smallest budget, is a seed-noise tie at mid budget, and
VANISHES (trails dense) at the largest budget. It is not driven by sparsity (best config
uses sp=1.0; extreme sparsity regresses); the edge is a small-budget REGULARIZATION
effect (extra biases + low-rank output factor), not capacity density.

**FINAL train-time conclusion**: dense weights are ~optimal per-parameter at this scale.
No weight restructuring / sharing / superposition buys capability-per-parameter. The
only reliable density lever is ACTIVATION sparsity — a runtime/compute saving (MoE) — not
train-time weight density.

## PROJECT-LEVEL CONCLUSION (both tracks complete)

You cannot make the weights themselves carry more per parameter — they are already
dense and ~optimal (shown post-hoc: full-rank/high-entropy/no shared basis; and
train-time: dense beats every structured family). Models are made lighter by two levers,
both validated here:
1. **Fewer bits per weight (quantization)** — INT8 = 2x (KL~3e-4 end-to-end), INT4 = 3.4x
   (+0.5% perplexity end-to-end, candidate 0008; needs power-up eval to ship).
2. **Fewer weights active per token (sparsity / MoE)** — the model already does this; it
   is a runtime/compute win, and it is the only "density" that survives scrutiny.
Restructuring or superposing weights to be "tighter" does not work.

Remaining actionable item: confirm INT4 (0008 power-up) at scale → potential headline
3.4x runtime deliverable alongside the solid INT8 2x floor.

INT4 power-up — **DONE and DEFINITIVE** (0008 test-002, 41 diverse prompts via
tools/fused_eval.py + prompts_eval.txt, checkpointed across 5 windows →
fused_large_summary.json). The small-sample optimism did NOT hold:
- **INT8: KL 0.0008, ppl 12.45→12.42 (−0.3%), top-1 99.7%, router 98.9%** — rock-solid,
  ships as the floor.
- **INT4: KL 0.089, ppl 12.45→13.09 (+5.1%), top-1 91.8%, router 92.8%** — the 8-prompt
  +0.5% was a ~10x under-estimate. INT4 plain RTN is **functional-but-degraded, NOT a
  free 3.4x**: model stays coherent but pays +5% perplexity and 8% top-1 flips. It's a
  *fits-vs-doesn't-fit* option only. Closing to INT8-class at 4 bits needs properly-
  powered activation-aware quant (GPTQ ~1e5 tokens / AWQ), not plain RTN.
- **LESSON (corrects the "INT4 REOPENED, +0.5%" sections above): an 8-prompt Stage-2 read
  over-stated INT4 ~10x on perplexity. Tiny-sample KL is directional only; capability
  verdicts need the larger prompt set.**

## Lossless → RUNTIME bridge found: fixed-width exponent codebook (candidate 0009, Probe Confirmed)

The 32% lossless scheme (0001) was storage-only because its high-byte plane is
**variable-length** entropy-coded (rANS/zstd) — no fixed bit-offset per weight, so a
matmul can't random-access it and must inflate to full-width BF16 in VRAM (Regime C).
**Fix: replace the variable-length code with a fixed-width codebook index + sparse escape.**
The sign+exponent field has only ~31–56 distinct values (top-16 cover >98%), so a 4–5-bit
index into a per-tensor codebook + an in-order escape stream for rare values is exactly
lossless AND fixed-width → random-access → fusible like INT4/INT8 dequant (BF16 rebuilt in
registers, never written to VRAM = Regime D). Measured on true layer-1 experts (8 up + 8
down), exact SHA-256 round-trip verified:
- **Headline: bit-regrouped codebook K=15 (4-bit index) = ~11.3 b/w = ~29.4% lossless**,
  fixed-width, escape 3.25%. Gives up only ~3 pts vs 0001's 32% to become fusible.
- Conservative: K=31 (5-bit index) = 24.9%, escape 0.06% (inside SqueezeLLM's proven
  fusible 0.05–0.45% sparse range — cleanest to fuse today).
- Key tweak: **regroup bit-wise** (codebook sign+exp8, store mantissa7 raw) beats the
  byte-split (codebook high byte, store low byte raw) by ~2 pts — the byte split wastes a
  bit by trapping the exponent LSB in the raw mantissa byte.
- **Ceiling is honest:** lossless ⇒ the random mantissa (~7 b/w) moves verbatim, so
  ~29–31% is the lossless-fusible ceiling; below that needs lossy quant (INT8=50%, 0005).
  This is the best *runtime* win with ZERO quality change — the lossless runtime option.
- **Proven (CPU):** losslessness + bit budget + addressability (real weights).
- **Proven (GPU, RTX 4090, candidate 0008-style Triton kernel — test-002):** on-device exact
  reconstruction; a kernel computes the matvec DIRECTLY on the narrow form (BF16 rebuilt in
  registers, never written to VRAM = Regime D), numerically exact (rel ~1e-7); and in the
  **bandwidth-bound regime the fused kernel runs at 0.756× the BF16 time** (both baselines at
  the 4090's ~1 TB/s ceiling) — a **24% decode speedup matching the 25% byte reduction.** The
  per-token bandwidth win is MEASURED, not just argued. (Caveat: at single-expert ~18µs scale,
  fixed launch overhead hides the win — it needs the memory-bound regime, which is exactly
  where MoE decode lives, reading ~GBs of experts per token.)
- **Proven (whole real 30B model — test-003):** ALL 6,174 BF16 tensors round-trip BIT-EXACT;
  whole model −24.95% (byte-split 12 b/w) / −30.03% (regroup 11.3 b/w), 58.8→44.1/41.2 GiB.
  Model is ~100% BF16 so this is the whole model, not just experts; ~0.3% escapes. Bit-exactness
  ⇒ logits provably identical (KL=0 exactly) — no quality eval possible or needed. This is the
  categorical separation from lossy quant: quality preserved by definition, not by measurement.

- **DECODE-SPEED VERDICT (H100 — test-004/005, deep 10-variant kernel investigation):** the
  fused lossless kernel **TIES bf16 on H100, does not beat it.** Clean saturating SXM (bf16=3.15
  TB/s=94% peak): best fused 2.34 TB/s (ratio 1.01), interleaved+arithmetic-decode 2.15 (ratio
  1.09, bit-exact). Diagnosis nailed: narrow loads are fine (a single plane ties bf16 per-byte);
  the costs were a per-element LUT gather (−26%, killed via ARITHMETIC decode — codebook is
  contiguous exp ranges, `high=(sign<<7)|(BASE+off)`, verified lossless 0.53% esc) and strided-x
  (−25%, killed via Marlin-style interleaved layout). Despite both fixes, every variant plateaued
  at a **~2.3 TB/s fused ceiling**. Crisp rule: **fused beats bf16 on GPUs with HBM < ~2.3 TB/s
  (4090 ~1, A100 ~2), ties/loses above (H100 3.35).** NOT "4090-specific" in raw power — the
  fused kernel runs ~2× faster on H100 than 4090 — but the ratio flips because bf16 is near-peak
  on H100 and the **25% byte margin can't absorb any unpack overhead**. Silver lining: a tie =
  25% lossless VRAM at ZERO speed cost, strictly better than DFloat11 (lossless but 1.4–2×
  slower). Open lever (untested, thin margin): a production CUDA/Marlin kernel. `generate()` on
  the reference model is separately loop-bound (all 128 experts/token) — needs a fused-MoE stack.
- **SCOPE (2026-07-01): the runtime track is PARKED.** The 0009 lossless codec (25–30% exact,
  whole-model, provably identical) is the on-axis deliverable and it's done. The runtime/kernel/
  generate()/serving work is a documented side-finding — do NOT re-open (no CUDA-kernel or
  serving integration) unless scope deliberately shifts to runtime. Project stays on weights
  compression; new scouting targets weight structure/statistics, not decode speed.

LESSON: "lossless = storage-only" was a property of the *encoding* (variable-length), not of
losslessness itself — a fixed-width lossless code IS addressable. BUT converting fewer bytes
into faster decode is a separate, kernel-quality-bound problem: the dequant must run at ≥ the
GPU's memory bandwidth, or the narrow read is pointless. Solid deliverable = **whole-model
lossless 25–30% with provably identical outputs** (storage + resident VRAM). The decode
SPEEDUP is not yet general (naive kernel wins only on ~1 TB/s GPUs).

## Whole-model similarity survey — no new lossless structure (candidate 0010, verified)

Exhaustive mechanical sweep of all 6243 tensors (exact global BF16 histogram over 31.6B
weights; per-tensor SHA-256 + hi_hist), 5 lenses + adversarial re-derivation from raw
bytes (19-agent workflow). **Found no fusible lossless structure beyond 0009.**
- **Zero byte-identical tensors** model-wide. embeddings vs lm_head are NOT tied
  (true cos +0.031 ≈ orthogonal) — kills the "tied-embedding delta" idea.
- **No value/structural near-dups survive**: every block-mean "similar" pair collapses to
  true |cos| 0.003–0.05 (experts/projections independent across ALL families, extends
  0003/0007). 1D norm/bias high cosines are pure DC-offset. The one real cross-layer cluster
  (input norm.weight[2688], centered cos→0.996) is 0.0005% of the model → gain 0.
- **Byte layout confirms 0001/0009 whole-model**: hi-plane 2.72 b, mantissa 7.96 b;
  ~99.997% of mass shares ONE high-byte distribution; per-role sharing is 6× *worse* than
  0009's per-tensor. Global order-0 value entropy = **10.50 b/w**; lzma on the mantissa =
  **7.85–8.0 of 8 bits** (byte-delta doesn't help) → mantissa is a **hard random wall**,
  triple-confirmed (survey + entropy + real compressor). **Lossless floor ≈ 11 b/w fixed
  (~31%) / 10.5 entropy-coded (~34%), essentially all exponent.**

## Lossless improvement — exponent 2-D context lever, ~30% → ~34% (candidate 0012, in progress)

New genuinely-lossless (bit-exact) lever beyond 0009's order-0 exponent codebook: the
sign+exponent field has **2-D spatial structure** a context model exploits below its order-0
entropy. Measured on real tensors (`0011/tests/artifacts/lossless_battery.py`,
`lossless_crosstensor.py`):
- **Within-tensor 2-D context** (condition exp on left+up neighbors): expert_up 2.87→2.64 b,
  expert_down 2.66→2.50 b (~0.17–0.23 b saved). attn/embeddings near-flat (2.61→2.60, 2.63→2.63)
  — the structure is in the FFN experts, not attention/embeddings.
- **Cross-tensor**: per-column exponent profile is **99.65% correlated across 32 experts** (real
  salient-channel structure, NOVEL — survey showed VALUES uncorrelated, but magnitude profile is
  shared) — yet conditioning saves only ~0.20 b (the exponent entropy is within-column).
- **Best lossless bits/weight** (ideal sign 1.0 + context-exp ~2.5 + random mantissa ~7.0) ≈
  **10.5 b/w = ~34%**, uniform across roles (experts 33.9/34.4%, attn 33.8%, embeddings 33.6%).
  This ~34% == the global order-0 value-entropy floor: context modeling reaches the entropy
  floor, and the floor is 10.5 b because sign+mantissa (8 b) are random.
- **Verdict: lossless ceiling ≈ 34%** (a real +4 pts over 0009's ~30%, from exponent context).
  90% lossless is **information-theoretically impossible** — 8 of 16 bits/weight are random
  (sign 1.0; mantissa 7.96/8 order-0, lzma/bz2 7.0, zero dead mantissa bits — 0002/0010).
  Full adversarial whole-model verification: workflow `lossless_exhaustion` → candidate 0012.
- NOTE: the 2-D-context and cross-tensor exponent codes are VARIABLE-LENGTH (storage-only, not
  fusible/random-access); 0009's fixed-width codebook stays the runtime-real form (~29%).
- **RUNTIME-REAL slice verified whole-model (RUNTIME.md).** A separable exponent predictor
  `exp_residual = exp − round(row_base[i]+col_base[j]−grand)` (O(R+C) int8 side vectors) IS
  fixed-width/random-access/fusible and bit-exact: measured numel-weighted over all 13 shards =
  **11.1976 fusible b/w = 30.01%**, every shard round-trips bit-exact. Kernel-fusible (register-
  only reconstruct, 2 int adds + tile-cached base loads, stays bandwidth-bound ~4-6 ops/byte).
  BUT its edge over a well-tuned non-predictive fixed-width baseline (29.95%) is only **+0.06 pt
  whole-model** — concentrated in shard 1 (embeddings/early layers, +0.67 pt); expert shards 4-13
  are ties. Vs 0009's 29.4% the gain is +0.64 pt (mostly from best index selection, not the
  predictor). **Runtime-real lossless ceiling ≈ 30%; storage ceiling ~34%; the ~4 pt gap is
  variable-length (not fusible). Correction: the earlier "+0.67 pt" was shard-1 only.**
- **LOSSLESS DIRECTION COMPLETE (working codec + mantissa hunt).** `lossless_codec.py` is a real
  end-to-end codec: 32.9–33.7% bit-exact round-trip (`np.array_equal`), at/above SOTA (DFloat11
  ~30%). Strongest compressors (brotli-11, lzma-9e) on the mantissa = 6.84–7.00 b; `mantissa_hunt.py`
  across 11 tensors: model-wide mantissa 6.9875 b = **0.078% exploitable**, and the 6.84b sliver is a
  LOCALIZED anomaly (layer-1 expert-0 only; the exp=4 near-zero cluster), not systematic. The
  AGENTS.md "peel until random" diagnostic TERMINATES: mantissa 6.99/7 + sign 1.000/1 look random to
  the best tools that exist. **Final lossless frontier: ~34–35% storage / ~30% fusible, +5 pt over
  0009, working codec verified. 90% lossless impossible (8 random bits/weight); only lever beyond is
  lossy/QAT (out of lossless scope).**

## Sub-4-bit expert quant — VQ + incoherence, and the RATE-DISTORTION WALL (candidate 0011)

Output-aware VQ/GPTQ + randomized-Hadamard incoherence (QuIP#/AQLM family), the untested
sub-4-bit frontier, measured on held-out real activations (30k cal / 2.6k held-out tokens),
non-padding RHT (R=Q₂₁⊗H₁₂₈, exact 2688 dims):
- **4-bit: incoherence+GPTQ = 3.35% output error — beats the ledger's INT4-RTN (5.07%).**
  New best sub-INT8 post-hoc codec (Pareto win at INT4 size). 3-bit ≈ 7.8%; 2-bit VQ 16.9%.
- **The wall is rate-distortion, not tuning.** Incoherence makes the (structureless, per
  0010) weights near-Gaussian i.i.d.; measured errors track the Gaussian bound D(R)≈2⁻²ᴿ
  (2b→25%→VQ 17%, 3b→12.5%→8%, 4b→6.25%→3.35%). VQ recovers only the space-filling gain;
  output-aware weighting adds nothing (incoherence flattens diag(H) 5.42→2.11). **Post-hoc
  quality floor = ~3–4 bit.** Same "experts are dense" wall (0003/0005/0007), now as a law.
- **Compounded combined frontier (proven, post-hoc)**: 4-bit experts + lossless non-experts
  = **~71% reduction at near-INT4 quality**; 3-bit ≈ 78% (degraded); 2-bit ≈ 84% (broken).
  90% needs experts at ~1 b/w — 2× past the proven wall.
- **The only lever past ~78% is training** (QAT/distillation): the R-D wall bounds
  compression of *fixed* weights, but the function has a weight-manifold; QAT searches it for
  a low-bit-representable point downstream layers co-adapt to (why BitNet-1.58 works trained,
  not post-hoc). End-to-end → GPU. This is the honest path to the 90% target.
- **QAT-breaks-the-wall VALIDATED locally (qat_demo.py, CharTransformer on real text).** At
  2-bit, POST-HOC is catastrophic (val loss 1.388→2.451, +77%) but QAT recovers to 1.507 —
  **88.8% of the gap closed, near-FP** — by letting downstream layers co-adapt. 4-bit QAT even
  beats FP (regularization). Confirms the manifold argument: sub-2-bit at good quality is a
  TRAINING result, unreachable post-hoc. Scaling to the 30B (→ ~84–90% combined) is a GPU job.
- **Compounding-on-quantized slice (task 3)**: the 4-bit incoherent stream has bell-shaped
  codes (order-0 entropy 2.975/4, lzma 3.24) → entropy-coding compounds ~0.8 b/w at identical
  quality → combined STORAGE ~76% at 4-bit quality (variable-length, storage-only; a companded
  quantizer captures it fixed-width).

## Open / live

- **0005 — low-bit expert quant**: RESOLVED end-to-end. INT8 Confirmed (KL~8e-4, ppl
  flat). INT4 plain RTN harden-tested (0008 test-002, 41 prompts): functional-but-
  degraded (+5.1% ppl, 8% top-1 flips) — fits-vs-doesn't-fit only, NOT a clean second
  halving. Sub-4-bit at INT8-class quality remains open ONLY via properly-powered
  activation-aware quant (GPTQ ~1e5 tokens / AWQ).
- **0008 — Stage-2 streamed eval**: Passed True Weights, harden test done (test-002).
  INT8 = capability-safe runtime floor; INT4 = documented degraded option. The streamed
  forward + fused_eval (checkpointed) + prompts_eval.txt are the standing Stage-2 tools.
- **Live thread**: properly-powered GPTQ — cache ~1e5 tokens via the streamed forward,
  full-rank Hessian, re-measure INT4 end-to-end against the INT8-class bar (KL≈1e-3).
- **TOOL available — real-activation capture**:
  `0005/tests/artifacts/activation_capture.py` captures layer-1 expert/gate input
  ([N,2688] real X + per-channel energy) at ~3.5 GB RAM / ~2.3 s, no full-model load.
  Reuse for any activation-aware lever or as the Stage-2 streaming seed.
- **0004 — embedding vocab-tail**: Rejected (no untrained tail; immaterial).
- **0009 — fusible exponent codebook**: **Runtime win CONFIRMED on GPU** (RTX 4090). Lossless
  ~25–29% fixed-width, random-access, exact round-trip (CPU test-001); on-device exact + fused
  kernel at **0.756× BF16 time (24% faster) in the bandwidth-bound regime** (test-002). The
  lossless→runtime bridge, proven end-to-end. Artifacts: `bench_gpu.py`, `gpu_bench_result.json`,
  `extract_sample.py`, `probe_regroup.py`. Remaining = optimization only (fold escape into the
  kernel; 12→11.3 b/w for ~29%). Pod driven via `runpodctl` (removed after run).
