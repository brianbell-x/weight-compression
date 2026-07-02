# Candidate 0014 — Column-keyed codebooks (Direction D) — RESULTS

**Date:** 2026-07-01 · **Mode:** REAL weights, layer-27 expert set, shard 7
**Constraint:** pure lossless, bit-exact (SHA-256), fusible-only (fixed-width index
plane, address-derived keys).

## Verdict: FALSIFIED — the direction does not fire against the realized stz baseline

All **16** column-keyed variants (per-tensor / shared × g ∈ {1,4,16,64} × index
bits b ∈ {3,4}) are **worse** than the realized stz per-tensor cost. Best variant
`sh_g64_b3` (shared tables, 64-column groups, 3-bit index) still loses
**0.0283 b/w**. The success gate was **≥ +0.09 b/w**; the shortfall is ~0.12 b/w.
The adoption-aware envelope — each of the 256 tensors free to pick any
column-keyed variant or stay on stz — **degenerates exactly to the baseline**
(choice_counts = {baseline: 256}; zero adoption), so this is not a tuning miss:
at stz's realized operating point there is no tensor in the set for which any
tested column keying is the cheaper exact code.

Scope caveat: measured on layer 27 only (the one expert layer wholly inside
shard 7, chosen per the vetting as a representative mid/late layer);
cross-layer transfer unvalidated. Given the loss is mechanistic and monotone
(below), a sign flip on another layer is implausible, but a cheap re-run on an
early MoE layer would upgrade this to a model-wide certificate.

## Headline numbers

| quantity | value |
|---|---|
| targets | 256 tensors (128 experts × {up_proj, down_proj}), 1,277,165,568 params |
| stz baseline (realized, recomputed) | **10.882179 b/w** on this set; 10.8975 whole-model |
| parity gate vs `stz_tensor_stats.jsonl` | max abs diff **0.000000** on all 256 tensors (exact, not just within ±0.01) |
| best column-keyed variant | `sh_g64_b3` = 10.910477 b/w (**−0.0283 b/w**, i.e. worse) |
| projected whole-model (winner) | 10.9238 vs 10.8975 baseline (−0.16 pts of 16) |
| envelope base+pt / base+all (adoption-aware) | 10.882179 = baseline exactly (0/256 adopt) |
| serializer round-trips | 4/4 records (pt+sh × up+down, 8 variants each): bits == plan exactly, SHA-256 bit-exact reconstruction |
| gate (≥ +0.09 b/w) | **FALSIFIED** |
| wall time | ~2.7 min (single 400 s-budget invocation) |

## Full results table (numel-weighted, all side costs charged)

Δ convention: `d vs stz` = baseline − variant; **negative = variant loses**.
`pt` = per-tensor tables; `sh` = tables shared across all 128 experts of the
layer per projection (payload + 128-bit frame charged once, amortized over the
sharers — exact under the uniform adoption these rows assume).

| variant | bpw | d vs stz (b/w) | proj. whole-model b/w | d (pts of 16) | 1st-level table KB |
|---|---|---|---|---|---|
| **stz baseline** | **10.8822** | — | **10.8975** | — | <0.1 |
| pt_g1_b3 | 10.9612 | −0.0791 | 10.9710 | −0.459 | 36.8 |
| pt_g1_b4 | 11.1961 | −0.3139 | 11.1895 | −1.825 | 78.8 |
| pt_g4_b3 | 10.9435 | −0.0613 | 10.9545 | −0.356 | 9.2 |
| pt_g4_b4 | 11.1161 | −0.2339 | 11.1150 | −1.359 | 19.7 |
| pt_g16_b3 | 10.9347 | −0.0525 | 10.9463 | −0.305 | 2.3 |
| pt_g16_b4 | 11.0969 | −0.2147 | 11.0972 | −1.248 | 4.9 |
| pt_g64_b3 | 10.9267 | −0.0445 | 10.9389 | −0.259 | 0.6 |
| pt_g64_b4 | 11.0924 | −0.2103 | 11.0930 | −1.222 | 1.2 |
| sh_g1_b3 | 10.9371 | −0.0549 | 10.9486 | −0.319 | 36.8 |
| sh_g1_b4 | 11.0911 | −0.2090 | 11.0918 | −1.215 | 78.8 |
| sh_g4_b3 | 10.9338 | −0.0516 | 10.9455 | −0.300 | 9.2 |
| sh_g4_b4 | 11.0912 | −0.2090 | 11.0919 | −1.215 | 19.7 |
| sh_g16_b3 | 10.9305 | −0.0483 | 10.9424 | −0.281 | 2.3 |
| sh_g16_b4 | 11.0914 | −0.2092 | 11.0920 | −1.216 | 4.9 |
| **sh_g64_b3** (best) | **10.9105** | **−0.0283** | 10.9238 | −0.164 | 0.6 |
| sh_g64_b4 | 11.0914 | −0.2092 | 11.0921 | −1.216 | 1.2 |
| env(base+pt) | 10.8822 | +0.0000 | 10.8975 | +0.000 | — |
| env(base+all, adoption-aware) | 10.8822 | +0.0000 | 10.8975 | +0.000 | — |

Two clean monotone trends:

- **b=4 is always ruinous** (−0.209 to −0.314 b/w). It does reproduce the
  previously reported escape-rate halving (~2% escapes at g=1), but pays a full
  extra index bit per weight for it — exactly the bit stz's second-level escape
  codebook had already made unnecessary.
- **Finer column granularity monotonically hurts** (g=1 worst, g=64 best, in
  both pt and sh families): per-group table payload plus escape-recoding cost
  grows faster than the conditional-entropy gain it buys.

## Why it failed (mechanism, not tuning)

This is the 2026-07-01 repricing playing out exactly as warned. The direction's
pre-measured evidence — H(exp|col)=2.486 vs ~2.65 order-0, per-column K15
escape halving — was priced against the stale 11.2072 b/w baseline, where an
escape cost a full raw16 fallback. In the realized stz code:

1. **Escapes are already cheap.** The second-level escape codebook recodes them
   in k ∈ {3..6} bits, so halving the escape *rate* converts to a few
   hundredths of a b/w, not the +0.26 originally priced.
2. **The only remaining big prize was b=3 viability** (saving a full index bit
   on tensors stz codes at b=4). Measured: column-keyed K=7 tables still run
   **21–22% escape rates** — the per-column exponent distributions are not
   concentrated enough for 7 symbols — and the recoded escape cost eats the
   saved index bit and more.
3. **The gross ceiling is small.** Column identity offers ~0.16 b/w of
   conditional exponent entropy over order-0 in the storage sense; the
   fixed-width realization overheads measured here (table payload, escape
   recoding, integer index widths) exceed the fraction of that ceiling any
   fixed-width code can capture at stz's operating point.

The comparison was held to the strictest standard available: baseline recomputed
via `stz.plan_regroup` (imported, not reimplemented) with **exact** parity
(abs diff 0.0) against `research/candidates/0009-fusible-exponent-codebook/tests/artifacts/stz/stz_tensor_stats.jsonl`,
and the column-keyed accounting writer-verified (`enc_colkey`/`dec_colkey`
serialize + decode; bits == plan exactly; SHA-256 bit-exact reconstruction).

## Escape forensics on the winner (`sh_g64_b3`) — the optimality certificate

Per NEXT_DIRECTIONS D, a null result here doubles as an optimality certificate
for the per-tensor first-level codebook. The residual escape stream is
near-random:

| tensor (layer 27) | esc rate | fano(row) | fano(col) | binomial ref | adj lift h/v | H(sign) | H(sign\|col) |
|---|---|---|---|---|---|---|---|
| experts.0.down_proj | 0.2220 | 0.97 | 0.99 | 0.778 | 1.000 / 1.000 | 1.000 | 0.9997 |
| experts.64.down_proj | 0.2242 | 1.10 | 1.03 | 0.776 | 1.002 / 0.998 | 1.000 | 0.9997 |
| experts.0.up_proj | 0.2113 | **2.27** | 2.02 | 0.789 | 1.002 / 1.001 | 1.000 | 0.9907 |
| experts.64.up_proj | 0.2058 | **2.30** | 1.49 | 0.794 | 1.002 / 1.001 | 1.000 | 0.9904 |

- No spatial clustering (adjacency lifts ≈ 1.00 both axes) → no run-length or
  2-D mask coding of escapes is available.
- Escape sign carries no column structure (H(sign|col) ≈ H(sign) ≈ 1.0).
- The **only** residual structure anywhere: up_proj **row-wise escape
  overdispersion** (Fano ~2.3 vs ~0.79 binomial) — some rows are escape-heavy.
  Ceiling of exploiting it is small (escapes are already recoded at 3–6 bits;
  a per-row second-level k or row-keyed escape width is worth at most
  ~0.01–0.03 b/w) — a chooser option for the container work, not a candidate.

## Attempt vs direction

**The direction (as priced) is dead, not just this attempt.** Grounds: the
sweep covered the sensible parameter space of the headline mechanism
(first-level column-keyed codebooks: both sharing modes, g from per-column to
64-wide groups, both viable index widths); the free-choice envelope adopted it
on 0 of 256 tensors; the losses are monotone and mechanistic (repricing of
escapes + small gross ceiling), not parametric; and the forensics certify the
residual is near-random. No point in the (g, b, sharing) space is left that
could plausibly swing +0.12 b/w.

Untested variations that are *adjacent* (same keying idea, different
mechanism), with what would have to be true for each to work:

1. **Per-column BASE re-centering** (subtract a shared per-column exponent
   base before the per-tensor codebook; side cost ~C·8 bits/layer, ~0.003 b/w
   amortized). Doesn't widen the index, so it dodges failure mode (2). Fires
   only if column exponent distributions are *shifted copies* of one another
   (translation, not shape variation) so re-centering pulls mass into the
   top-K. The vetting's per-tile BASE falsification (residuals spatially
   stationary) cuts against it but doesn't test the column partition. Ceiling
   ~0.02–0.05 b/w (escape-rate reduction × (k−b) recoding differential) —
   below the 0.09 gate; only worth folding into the stz/container chooser.
2. **Keying the second-level escape codebook by column** instead of the first
   level. Population is thin (2–6% of weights at stz's baseline operating
   point) and forensics show H(sign|col) ≈ H(sign) on escapes; ceiling
   < 0.01 b/w. Do not pursue.
3. **Row-axis keying for up_proj** (the one real forensic signal: row Fano
   ~2.3). Per-row escape-recoder k, or row-keyed pw. Ceiling ~0.01–0.03 b/w;
   chooser option only.
4. **Per-group mixed index width** (b=3 for concentrated groups, b=4
   otherwise, group-stride table keeps random access). Bounded by the same
   economics: the per-tensor free-choice envelope already adopted nothing, and
   mixing must beat *both* uniform bounds by +0.12 b/w. Implausible.
5. **Cross-layer re-run** (one early MoE layer): not a rescue path — a closure
   rider that upgrades "falsified at this operating point" to model-wide.

For *any* column-keyed variant to clear the +0.09 gate, the fixed-width
realization would have to capture more than half of the ~0.16 b/w gross
column-conditional entropy at near-zero overhead — while the realized stz
already sits only ~0.4 b/w above the full-context *storage* floor. The
measured overheads (≥0.19 b/w at b=4; 21–22% escapes at b=3) rule that out.

## Compounding order

- **Column keying should NOT move before (or replace) the per-tensor
  chooser.** The adoption-aware envelope *is* the "column keying inside the
  chooser" experiment — the chooser saw every column-keyed variant priced
  exactly and picked the baseline 256 times out of 256. The per-tensor
  min-envelope chooser + second-level escape recoding already bank the entropy
  column keying targets.
- **Nothing here creates a new compressed form to re-run earlier levers on
  top of** — no column-keyed output was adopted, so the object of study for
  compounding remains stz's realized emission (index plane, escape stream,
  side tables). The forensics above are exactly that recursive
  "peel-until-random" pass, and they came back near-random on the escape mask;
  the one residual (up_proj row overdispersion) is a small chooser lever, not
  a re-ordering.
- Positive by-product: this falsification is itself a certificate that stz's
  per-tensor fixed-width index plane is near-optimal against address-derived
  column conditioning — the fusible-vs-storage gap (~0.4 b/w) must be attacked
  at a different granularity (per-*tile*/block, Direction A) or realized as
  storage (E/F), not by finer weight-level keying.

## Reproduction

From the repo root (`C:/dev/compression`), real shard 7 snapshot at
`models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot`:

```
# smoke (synthetic snapshot, seconds)
uv run python research/candidates/0014-column-keyed-codebooks/tools/probe_column_codebooks.py --synthetic

# real run (resumable; self-limits to ~400 s per invocation and checkpoints;
# re-invoke until complete — the layer-27 set finished in a single invocation)
uv run python research/candidates/0014-column-keyed-codebooks/tools/probe_column_codebooks.py

# summary table + escape forensics + summary JSON
uv run python research/candidates/0014-column-keyed-codebooks/tools/probe_column_codebooks.py --summary
```

To reproduce from scratch, delete `tests/artifacts/colkey_results.jsonl` and
`tests/artifacts/colkey_shared_hists.npz` first (the run otherwise resumes).
The parity gate aborts loudly if the recomputed baseline drifts from
`0009/tests/artifacts/stz/stz_tensor_stats.jsonl`.

## Artifacts

- `tests/artifacts/colkey_summary.json` — table, envelopes, winner, forensics,
  round-trip coverage, verdict (this file is the machine-readable version of
  the tables above).
- `tests/artifacts/colkey_results.jsonl` — per-tensor per-stage records
  (per_tensor / shared / roundtrip).
- `tests/artifacts/colkey_shared_hists.npz` — aggregated per-group histograms
  + membership (atomic checkpoint).
- `tests/artifacts/colkey_*_synthetic.*` — synthetic smoke equivalents.
