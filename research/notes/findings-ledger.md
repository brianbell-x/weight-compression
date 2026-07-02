# Findings Ledger

Rolled-up record of what experiments have settled, so scouting builds on prior
results instead of re-deriving them. Update this when a candidate resolves.
Scope is strictly lossless — exact bit-for-bit round-trip — per AGENTS.md.

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
  ~29–31% is the lossless-fusible ceiling; anything below that is lossy territory, out of
  scope. This is the best *runtime* win with ZERO quality change — the lossless runtime option.
- **Proven (CPU):** losslessness + bit budget + addressability (real weights).
- **Proven (GPU, RTX 4090, Triton kernel — test-002):** on-device exact
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
  0003). 1D norm/bias high cosines are pure DC-offset. The one real cross-layer cluster
  (input norm.weight[2688], centered cos→0.996) is 0.0005% of the model → gain 0.
- **Byte layout confirms 0001/0009 whole-model**: hi-plane 2.72 b, mantissa 7.96 b;
  ~99.997% of mass shares ONE high-byte distribution; per-role sharing is 6× *worse* than
  0009's per-tensor. Global order-0 value entropy = **10.50 b/w**; lzma on the mantissa =
  **7.85–8.0 of 8 bits** (byte-delta doesn't help) → mantissa is a **hard random wall**,
  triple-confirmed (survey + entropy + real compressor). **Lossless floor ≈ 11 b/w fixed
  (~31%) / 10.5 entropy-coded (~34%), essentially all exponent.**

## Lossless ceiling — exponent 2-D context lever, ~30% → ~34% (candidate 0012)

New genuinely-lossless (bit-exact) lever beyond 0009's order-0 exponent codebook: the
sign+exponent field has **2-D spatial structure** a context model exploits below its order-0
entropy. Measured on real tensors (scripts in
`research/candidates/0012-lossless-ceiling/tests/artifacts/`: `lossless_battery.py`,
`lossless_crosstensor.py`, `lossless_ceiling.py`):
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
  Anything much beyond ~34–35% storage / ~30% fusible is **information-theoretically
  impossible losslessly** — 8 of 16 bits/weight are provably random (sign 1.0 b; mantissa
  ~6.99/7 b: 7.96/8 order-0, lzma/bz2 ~7.0, zero dead mantissa bits — 0002/0010).
  Full adversarial whole-model verification: workflow `lossless_exhaustion` → candidate 0012.
- NOTE: the 2-D-context and cross-tensor exponent codes are VARIABLE-LENGTH (storage-only, not
  fusible/random-access); 0009's fixed-width codebook stays the runtime-real form (~29%).
- **RUNTIME-REAL slice verified whole-model**
  (`research/candidates/0012-lossless-ceiling/RUNTIME.md`). A separable exponent predictor
  `exp_residual = exp − round(row_base[i]+col_base[j]−grand)` (O(R+C) int8 side vectors) IS
  fixed-width/random-access/fusible and bit-exact: measured numel-weighted over all 13 shards =
  **11.1976 fusible b/w = 30.01%**, every shard round-trips bit-exact. Kernel-fusible (register-
  only reconstruct, 2 int adds + tile-cached base loads, stays bandwidth-bound ~4-6 ops/byte).
  BUT its edge over a well-tuned non-predictive fixed-width baseline (29.95%) is only **+0.06 pt
  whole-model** — concentrated in shard 1 (embeddings/early layers, +0.67 pt); expert shards 4-13
  are ties. Vs 0009's 29.4% the gain is +0.64 pt (mostly from best index selection, not the
  predictor). **Runtime-real lossless ceiling ≈ 30%; storage ceiling ~34%; the ~4 pt gap is
  variable-length (not fusible). Correction: the earlier "+0.67 pt" was shard-1 only.**
- **LOSSLESS DIRECTION COMPLETE (working codec + mantissa hunt).** `lossless_codec.py` (same
  artifacts dir) is a real end-to-end codec: 32.9–33.7% bit-exact round-trip (`np.array_equal`),
  at/above SOTA (DFloat11 ~30%). Strongest compressors (brotli-11, lzma-9e) on the mantissa =
  6.84–7.00 b; `mantissa_hunt.py` across 11 tensors: model-wide mantissa 6.9875 b = **0.078%
  exploitable**, and the 6.84b sliver is a LOCALIZED anomaly (layer-1 expert-0 only; the exp=4
  near-zero cluster), not systematic. The
  AGENTS.md "peel until random" diagnostic TERMINATES: mantissa 6.99/7 + sign 1.000/1 look random to
  the best tools that exist. **Final lossless frontier: ~34–35% storage / ~30% fusible, +5 pt over
  0009, working codec verified. 90% lossless impossible (8 random bits/weight); anything beyond
  the ceiling is out of lossless scope.**

## Container v2 shipped — .stz serialized artifact, whole model 31.89% SHA-256-verified (2026-07-01)

Direction E of `research/notes/NEXT_DIRECTIONS.md` is done: the flagship fusible number is
now a demonstrated artifact, not a bit-accounting estimate. `stz.py` (0009 tools) serializes
every shard to a real container — per-tensor min-envelope chooser over index_bits b∈{2..5} ×
second-level escape codebook (escapes coded in k bits against the next-L syms, raw-9 reserve),
raw16 fallback, per-row escape prefixes so every stream stays row-addressable (fusible) —
and `verify` decodes from the .stz alone:

- **Whole model: 63.16 GB → 43.02 GB = 31.89% smaller; all 13 shards reconstruct
  SHA-256-identical** end-to-end, safetensors headers included
  (`tests/artifacts/stz/stz_manifest.json`, `stz_verify.json`).
- **Realized BF16 cost: 10.8975 b/w numel-weighted (experts 10.8949)** — beats the 11.1976
  b/w whole-model accounting estimate by 0.30 b/w. The chooser + second-level escape
  codebook are real wins, and the raw16 fallback kills the 1-D-tensor 18-bpw pathology
  (NEXT_DIRECTIONS integrity items 2 and 3 closed).
- **The fusible-vs-storage gap narrowed to ~0.4 b/w** (realized 10.90 vs ~10.5 entropy-coded
  storage floor). This reprices the open directions: any new fusible lever (column-keyed
  codebooks, block-granular ANS) must now beat the *realized stz per-tensor costs*
  (`stz_tensor_stats.jsonl`, 6243 tensors), not the old 11.2 b/w baseline — part of the
  escape-side headroom those directions priced in has already been harvested.

## Column-keyed codebooks FALSIFIED — stz certified near-optimal at per-weight width (candidate 0014, 2026-07-01)

Direction D (NEXT_DIRECTIONS) probed decisively on real weights (shard 7, layer 27: all 128
experts × {up,down}_proj = 256 tensors, 1.28 B params) and **falsified**. Probe
(`0014/tools/probe_column_codebooks.py`, adversarially reviewed, parity gate vs
`stz_tensor_stats.jsonl` EXACT to 0.000000 on all 256 tensors; skeptic re-derived all
weighted numbers from raw per-tensor bits — not refuted, high confidence):

- **All 16 column-keyed variants lose to realized stz** (per-tensor and expert-shared ×
  g∈{1,4,16,64} × b∈{3,4}, every side cost exactly charged). Best (shared, g=64, b=3) is
  still −0.0283 b/w; an adoption-aware envelope letting every tensor freely adopt any
  variant chose the stz baseline **256/256 times**.
- **Mechanism (record this rule)**: after stz's second-level escape recoder, escape-rate
  reductions convert at only (k−b) recoded bits per converted escape — "fewer escapes"
  levers must be priced through the recoder first. That killed b=4 column tables (escape
  halving reproduced, but the extra index bit costs +0.21–0.31 b/w net) and b=3 (K=7 runs
  21–22% escapes).
- **Escape forensics: the emitted escape mask is near-random** (down_proj Fano ≈1.0,
  adjacency lifts ≈1.00, H(sign|col)≈H(sign)≈1.0) — a recursive peel-until-random pass on
  stz's own emission that certifies per-weight fixed-width column conditioning is done.
  The one residual signal: **up_proj row-wise escape overdispersion (Fano ~2.3 vs ~0.79
  binomial)** → a small per-row escape-k chooser lever, ceiling ~0.01–0.03 b/w.
- Consequence: the ~0.4 b/w fusible-vs-storage gap cannot be closed by finer weight-level
  keying — it moves to **block/tile granularity (direction A)** or storage realization
  (E/F). Per-column BASE re-centering survives only as a chooser option (ceiling
  ~0.02–0.05 b/w); column-keying the second-level table is ruled out (<0.01 b/w).
- **Cross-layer rider (same day, 7 layers swept)**: the certificate holds from mid-model
  on (layers ≥13: envelope ≈0), but **early layers keep a small real win** — the
  adoption-aware envelope gains +0.098 (L1), +0.070 (L3), decaying to 0 by L13; averaged
  over all 23 MoE layers ≈ **+0.011 b/w experts ≈ +0.065 pt whole-model**. The vetting
  numbers that motivated D (escape halving, H(sign|col)=0.978) were early-layer
  properties that don't transfer (L27 H(sign|col)=0.9997) — layer identity was the
  hidden variable. Actionable residue: add `colkey` as a variant in the .stz per-tensor
  chooser (E); free upside wherever it wins. Layer-1 expert-0's known anomaly resurfaces
  as extreme escape-mask spatial structure (Fano ~166–178 vs ~1.0 elsewhere); localized,
  not a lever.

## Block-granular coding, first probe — PARTIAL: storage fires, tile point is coder mechanics (candidate 0015, 2026-07-01)

Direction A probed on the canonical shard-7 layer-27 set (256 tensors; parity vs stz exact;
round-trip gate on 10,493 blocks + 512 rANS lanes SHA-256-exact; skeptic re-derived every
grid point from raw integer bits to <5e-6 — not refuted, high confidence):

- **The falsifier did NOT trigger: per-block code-length tails are THIN** (numel-weighted
  p99/p50 = 1.17 at W=32, 1.08 at W=128, → 1.01 at W=16K; worst block ever 5.10 b/w).
  Padding percentile choice moves results by only ~0.001–0.05 b/w. This was the probe's
  one genuine unknown and it came back favorable.
- **Tile-fusible grid (fixed stride, W≤128) FAILS both gates at this operating point**:
  best W128_P99 = 11.0349 b/w, 0.153 WORSE than stz (10.8822). Killer = fixed per-block
  overhead, exactly decomposed: coder excess 0.133 (12-bit flush 0.094 + renorm 0.039)
  + byte-ceil budget slack 0.303 + taxes 0.009 ≈ 0.47 b/w vs a 0.316 budget. A mechanics
  problem with quantified remedies (bit-granular stride ~0.03–0.05; two-tier budgets
  0.30→~0.15; shared flush 0.094→~0.01–0.03 → projects 10.76–10.85 = under stz), NOT a
  structural wall.
- **Storage-leaning sizes DO beat stz**: W16384_P100 = 10.6977 b/w (−0.185 vs stz) and
  superblock format (a) (4096-sym, 32 rANS lanes, exact two-level index tax 0.008) =
  10.7079 — both at/under floor+0.15. **The order-0 storage question on this set is now
  closed**: floor 10.5583, realized 10.6977, residual is coder mechanics. Further gains,
  storage OR fusible, must come from below-order-0 structure.
- Runtime caveat (carry into any claim): even W≤128 is a different kernel contract than
  0009 (O(1) address, O(W) sequential decode per block); 0009's measured 24% speedup does
  not transfer automatically.
- Verdict: **partial — this operating point falsified, direction alive.** Pre-registered
  v2 bar: combined per-block overhead < 0.316 b/w beats stz; if a competent
  overhead-attack v2 (bit-stride + two-tier budgets + shared flush + column-conditioned
  tables *inside* the block coder — 0014 banned per-weight keying, not this) still can't
  get under 10.88 at W≤128, declare the tile-granular order-0 point falsified for good.

## DIRECTION A FIRES — tile-granular block coding beats realized stz (candidate 0015 v2, 2026-07-02)

The v2 overhead attack (same canonical layer-27 set) got fixed-stride W=128 blocks UNDER the
realized stz baseline. Skeptic verification was maximal: every number re-derived from raw
bits across all 36,864 cells × 256 rows; a REAL container serialized whose byte size equals
the accounted bits exactly; an independently written encoder AND decoder; non-adjacent
blocks (gap 38,975) decoded bit-exact straight from O(1)-computed stride addresses.

- **Best fusible: 10.7004 b/w** (W=128 bit-stride blocks, 4 DP-optimal tier budgets +
  2-bit class flag plane, mantissa-carrying flush) = **beats stz 10.8822 by +0.1818 b/w**
  and passes the entropy-relative bar (floor+0.15 = 10.7083) by 0.0079. Best any-W:
  W256 = 10.6722. Overhead: 0.469 (v1) → **0.1345 b/w**.
- **Lever anatomy**: multi-tier DP budgets were the big lever (+0.238 — tiers subsume BOTH
  percentile padding AND escapes; the escape class empties at P100), mantissa-carrying
  flush refunds the 12-bit flush exactly (+0.089 net), bit-granular stride +0.035.
- **The L4 gate is its own finding**: layer-27 column-conditional structure is essentially
  ABSENT (block-group gain 0.0005 b/w; full-column ceiling 0.0218) — third independent
  confirmation of the layer-identity result. Everything won here is coder mechanics;
  further W≤128 gains need below-order-0 structure that is not column identity.
- **Caveats (both pre-registered in RESULTS.md)**: G2 margin is thin (0.0079 b/w) and the
  winning cell was selected on the set it is scored on; scope is layer 27 only. The
  whole-model projection (~10.7285 b/w ≈ 32.9% vs stz's 31.89%) is selection-optimistic.
  **Required gate before any whole-model claim: frozen-format cross-layer transfer check**
  (also re-tests L4 on early layers, where column structure actually lives).
- Runtime contract caveat unchanged: O(1) block address but O(W)=128 sequential decode per
  block — a register-tile kernel prototype is the remaining runtime-credibility step
  (parked: kernel work is out of current scope).
- **CROSS-LAYER GATE PASSED (2026-07-02)**: the frozen format (W128/T4/P100/L1+L3, per-tensor
  DP budgets as transmitted side info) **transfers** — beats each layer's own realized stz
  on all 6 out-of-selection layers (worst +0.157 at L51, best +0.194 at L3; out-of-selection
  mean +0.1794 ≈ in-selection +0.1817, so selection optimism was only ~0.006 b/w).
  **Honest whole-model number: 10.7346 b/w vs stz 10.8975 (≈32.9%), conservative
  bracket-min over 16 unswept layers** (global-min floor variant 10.7448). Scope
  corrections: the floor+0.15 bar holds MID-MODEL ONLY (fails by 0.0003–0.0107 at layers
  1/3/51 — early layers have higher H(sym)); esc_frac=0 transfers everywhere. L4 re-entry
  on early layers: NO as addressed — the per-column ceiling is real there (0.160/0.125 b/w
  at L1/L3 in sym space) but 64–128-column address-derived groups capture <5% of it; a
  layout-aware form (column-major blocking, ≤16-col groups) is a separate future candidate
  with ~0.02–0.03 b/w whole-model ceiling. Skeptic: not refuted, high confidence (raw-bits
  re-derivation on all 7 layers, freeze verified per row, gate re-measured from raw shard).
- **COMPLETENESS SWEEP DONE (2026-07-02): all 23 expert layers measured — no interpolation
  left. FULLY MEASURED whole-model: 10.7311 b/w = 32.93%** (expert plane frozen-cell,
  non-expert 7% held at stz). G1 pass 23/23 (worst +0.157 @L51, best +0.194 @L3, smooth
  monotone-ish decay with depth), stz parity exact 23/23, SHA-256 round-trips 23/23.
  The bracket-min interpolation (10.7346) was conservative as designed.

## Chooser-levers pre-probe — V3 fractional-m fires everywhere, column family is one pot (2026-07-02)

Three chooser-scale levers priced exactly (adoption-aware, parity vs realized stz exact
768/768, 9+18 SHA-256 round-trips, skeptic re-derived from raw weights — not refuted):

- **V3 fractional-m (grouped-radix index plane, non-power-of-2 K): ADOPT — the strongest
  chooser lever.** +0.0489 b/w model-wide, adopted on 768/768 tensors, and the gain RISES
  with depth (+0.0486 L1 → +0.0540 L27) — the one lever whose mass lives in late layers.
- **V2 per-column int8 BASE (g=1 only): ADOPT for early layers — it is the SAME structure
  as 0014's colkey win, not a second win.** Independent implementations converge (L1:
  +0.0991 vs colkey's +0.0978). Grouping kills it (g16/g64 adopted 0/256 — pure per-column
  shift). **The column family (colkey | V2-g1) is one pot worth ~+0.011–0.014 b/w total**;
  put both in the chooser and let the min-envelope settle supersession at zero risk.
- **V1 per-row escape-k: DROP** (+0.0053 b/w, below the +0.01 bar even before an uncharged
  offset-table cost — the (k−b) conversion rule priced it correctly in advance).
- **Joint projection: +0.0568 b/w ≈ +0.36 pt whole-model** → a .stz chooser v3 takes the
  container from 31.89% to ~32.2% at per-weight random access.

Two-track picture now current: the **.stz track** (per-weight random access, strongest
runtime contract, 31.89% → ~32.2% with these levers) and the **0015 tile track** (W=128
block coding, 10.70 b/w ≈ 32.9% projected, O(W)-sequential-decode contract, cross-layer
gate pending). They share the mantissa/sym anatomy but compete on the index plane; the
container can carry both as per-tensor codec choices.

## Emission peel: sym-side CONVERGED random; the "mantissa is noise" claim CORRECTED (2026-07-02)

Peel-until-random on the 0015-v2 emitted planes (64 tensors, layers 1/13/27/40,
skeptic-verified: every battery re-run with independent code, order-1 coder re-implemented
from spec and round-tripped):

- **Sym-side emission is certified near-random — the recursive peel loop CONVERGES**:
  coded payload ceiling 0.0029 b/w, tier-flag plane 0.0023 (whole plane is only 0.0156).
  Residual above floor is decomposed coder mechanics, not hidden symbol structure.
- **Within-block order-1 context: DEAD as a format change** (+0.0086 realized << 0.05 gate;
  holdout positive in only 4/8 cells). All signal is layer-1 up_proj (+0.0596 there);
  parked into the early-layer bundle (~0.001 b/w whole-model).
- **CORRECTION to the 0012-era mantissa verdict**: the transmitted mantissa plane is NOT
  random. Per-position analysis H(bit | position mod 7) exposes a monotone MSB bias —
  p(1) = [0.416, 0.458, 0.480, 0.491, 0.497, 0.499, 0.500] by bit position — worth a
  **~0.0287 b/w ceiling (MSB alone ~0.020)**, present on 62–64/64 tensors, MI hits at
  native lags 7/14/21/884. The old "6.9875/7, hard random wall, diagnostic terminates"
  claim was a **measurement-geometry artifact**: pooled bit entropy dilutes the positional
  bias ~7×, and byte-aligned compressors (brotli/lzma) cannot see mod-7 phase. Phase is
  fixed-position → known at decode → **fusible-compatible**, and it applies to EVERY
  container that ships mantissas verbatim (stz included). Storage floor drops ~0.03 b/w
  (~10.53); tile-format target if realized: ~10.70 → ~10.67–10.68.
- Tier/budget design headroom (diagnostic lens plane): 0.0335 b/w ceiling, realizable
  slice bounded by the actual pad slack (~0.073) — secondary mechanics probe.

## Super-120B FULL validation — family claim upgraded to measured fact (2026-07-02)

`stream_validate.py` completed all 50 shards of NVIDIA-Nemotron-3-Super-120B-A12B-BF16
(230.2 GB streamed, bounded disk, survived a flaky link via checkpoint/resume):
**ALL 42,642 BF16 tensors round-trip bit-exact lossless; regroup K15 = 28.85%
whole-model (experts 28.79%; byte-split baseline 24.12%)**. BF16 is 100% of seen bytes;
experts 93.5%. The exponent-concentration structure the whole project exploits transfers
from 30B to 120B essentially unchanged (30B plain-regroup ≈ 29.4%). Result:
`0009/tests/artifacts/stream_probe_super_120b_full.json` (supersedes the 1-shard probe).
Remaining generality scope: Ultra-550B full run, cross-modality one-shard probes, HF census.

## Purged tracks — do not re-open (2026-07-01)

Lossy/quantization/QAT/train-time tracks (candidates 0005–0008, 0011, 0013,
`research/traintime`) were explored and deliberately purged on 2026-07-01;
the project scope is strictly lossless per AGENTS.md. Tracked history is
recoverable via git if scope ever deliberately changes. Do not re-open or
re-propose lossy work.

## Open / live

- **0004 — embedding vocab-tail**: Rejected (no untrained tail; immaterial).
- **0009 — fusible exponent codebook**: Confirmed. Lossless ~25–29% fixed-width,
  random-access, exact round-trip (CPU test-001); whole-model bit-exact (test-003); on-device
  exact + fused kernel at **0.756× BF16 time (24% faster) in the bandwidth-bound regime** on
  RTX 4090 (test-002); ties bf16 on H100 (test-004/005). Artifacts: `bench_gpu.py`,
  `gpu_bench_result.json`, `extract_sample.py`, `probe_regroup.py`. Remaining kernel
  optimization (fold escape into the kernel; 12→11.3 b/w for ~29%) is PARKED per the
  2026-07-01 scope note. Pod driven via `runpodctl` (removed after run).
- **0012 — lossless ceiling**: direction complete. Working codec 32.9–33.7% bit-exact;
  frontier ~34–35% storage / ~30% fusible. The only open lossless thread is mantissa
  *statistics*, currently indistinguishable from random (6.99/7 b) by the best tools that
  exist — reopen only with a genuinely new structural hypothesis, not another compressor pass.
- **.stz container (0009 tools)**: Confirmed artifact. Whole model serialized at **31.89%
  (10.8975 b/w BF16), all 13 shards SHA-256-verified**; current best fusible form and the
  baseline every new lever must beat. Next probes queue in
  `research/notes/NEXT_DIRECTIONS.md` (D column-keyed codebooks first, then A block-ANS).
