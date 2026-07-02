# Candidate: Exact Cross-Checkpoint Delta Coding (Direction C)

## Claim
The same tensor across *training time* (base/pre-RL checkpoint vs the released
checkpoint) is a correlation source no prior candidate has tested — every
falsified delta (0003 cross-expert, emb-vs-lm_head) was *within* one model,
between different functions. If post-training only nudged weights, storing the
release checkpoint as an exact delta against the base checkpoint should cost far
less than the ~10.9 b/w standalone stz floor.

## Why It Might Work
Mechanics (from `research/notes/NEXT_DIRECTIONS.md`, Direction C):
- Sub-half-ulp updates round back bit-identical in BF16, so a match mask over
  uint16 words should have long runs (RLE-cheap) and a possibly large exact-match
  fraction.
- Changed weights rarely change magnitude class, so the XOR high byte / sym field
  should be near-delta-function (~0.2–0.8 b vs ~2.7 standalone): only the ~7
  random mantissa bits get paid on changed words.
- Expected: second checkpoint at ~3–9.5 b/w given the base; the pessimistic
  zero-exact-match case is still ~41–47% via exponent-plane coding of the delta.

## Checkpoint Pair
- Target (local): `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
  (`models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot`, 13 shards).
- Base (sibling): `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`,
  pinned revision `97ab8012882a655dc38df4fee47422aca9caca07` (public, not gated;
  weights uploaded once 2025-12-03 and never modified). Recon verified: index
  weight_map has exactly the same 6243 tensor names, every shape identical.
- Dtype caveat: `backbone.layers.N.mixer.A_log` / `.D` are F32 in the release
  but BF16 in the base (46 tiny [64] tensors, 5888 bytes total) — the probe
  skips and records them; they cannot perturb the conclusion.
- Shard plan (bandwidth-bound, ~10 GB): base shard 1 (embedding + layers 0–3)
  and base shard 7 (expert-heavy layers 24–29). Base shard boundaries are
  shifted by one tensor vs release; 2 stragglers of local shard 7 live in
  adjacent base shards (Range-fetchable later; recorded as skips otherwise).

## Measurement
`tools/probe_ckpt_delta.py` — streams one sibling shard at a time (download,
process, delete), aligns tensors by name, and per BF16 pair measures:
1. % bit-identical uint16 words + run structure of the match mask (concrete
   RLE cost: best of raw mask / Elias-gamma run lengths / Rice-coded gaps).
2. XOR plane split (u16 LE, `sym = u >> 7`, `mant = u & 0x7F`): H0 of the XOR
   high byte and XOR sym/mantissa fields, over all words and over
   non-matching words.
3. An exact delta-coding cost model, ALL side costs charged (mask RLE + count
   fields + selector, non-match XOR syms at H0 plus full histogram table,
   non-match mantissas verbatim at 7 b, per-tensor name/shape header) →
   projected b/w for the release checkpoint given the base. The field-split
   decomposition is mechanically reconstructed and checked bit-exact per tensor.
4. Baselines on the same bytes (mandatory — beat both or it is not a
   contribution): zstd `--patch-from` semantics and xdelta3-class delta. On
   this box neither the `zstd` nor `xdelta3` CLI exists and no xdelta3 python
   module is installed, so patch-from is implemented with the python
   `zstandard` package (raw-content dictionary = base bytes, LDM enabled,
   window sized to cover dict+target, level 19, round-trip verified) both
   per-tensor and whole-shard (when ≤ window limit); xdelta3 availability is
   probed at runtime and recorded — if absent, zstd-patch-from-with-LDM stands
   in as the delta-class baseline and the summary says so.
5. Sanity: standalone stz-class cost of the sibling tensors (candidate 0009
   regroup-K15 accounting, ≈11.2–11.3 b/w vs 10.90 realized .stz) plus a
   streaming standalone zstd (level 9, multithreaded — reference only) of the
   release shard bytes, as the no-base reference.

Direction (post-review fix): TARGET = local release coded GIVEN BASE = the
downloaded sibling, matching the claim exactly; the delta model is
cost-symmetric either way, but patch-from and the stz standalone are now
computed on the release side the ~10.9 b/w floor refers to. A second model
variant prices non-match XOR mantissas at H0 (same table-charge style) next to
the 7 b-verbatim primary, so "delta mantissas are incompressible" is measured,
not assumed. The two boundary-shift stragglers of local shard 7 are
range-fetched individually (`--extra-tensors`) instead of being dropped.

Resumable JSONL (`tests/artifacts/.../results.jsonl`, keyed per tensor/shard) +
`--summarize` mode that rebuilds `summary.json` from the JSONL. A truncated
trailing line (crash mid-write) is auto-repaired on resume/summarize; a work
dir refuses to resume under different parameters (config stamp).

## Promising Result (threshold)
Projected delta cost materially below standalone (~10.9 b/w) — the direction's
own bar is ~3–9.5 b/w — AND below both byte-level baselines (zstd patch-from,
xdelta3-class) on the same shard pairs. If the exact-match fraction is ~0, the
fallback question is whether XOR sym H0 still lands near-delta-function
(~0.2–0.8 b), which alone sustains the ~41–47% pessimistic case.

## Smoke Status (2026-07-02) — passed
`--synthetic` smoke on `models/synthetic/nemotron_tiny/hf_snapshot` (sibling
faked by flipping mantissa LSBs on 3% of BF16 words; a second run additionally
flips the lowest exponent bit on 0.2% to exercise the sym-coding path).
Measured (38 BF16 tensor pairs, 133,696 words; 7 F32 pairs recorded as skips):
- All 38 split-field reconstructions bit-exact; resume skips finished records;
  `--summarize` rebuilds the summary from the JSONL alone.
- Mantissa-only run: 97.02% bit-identical words, H0(XOR sym | non-match) = 0
  (delta function, as constructed), delta model = **0.5558 b/w** vs 11.32 b/w
  stz-class standalone (the synthetic mimics real exponent concentration).
- Sym-flip run: 96.83% match, H0(XOR sym | non-match) = 0.328 — the sym
  table/payload path prices correctly (0.5977 b/w).
- Baselines ran and round-trip verified: per-tensor zstd patch-from 0.551 b/w
  (marginally under the model here — fixed per-tensor side costs dominate on
  ~3.5K-word toy tensors and amortize at real sizes), whole-shard patch-from
  verified, streaming standalone zstd as no-base reference. xdelta3 probed:
  unavailable on this box (no CLI, no module) — recorded in every shard record.
- Post-review re-smoke (2026-07-02, after applying the review fixes): base
  smoke reproduces 0.5558 b/w with the swapped (release-given-base) direction;
  H0-mantissa variant = 0.3625 b/w (prices the constructed LSB-only flips);
  sym-flip variant 0.5977 b/w; truncated-trailing-line repair, config-stamp
  refusal, and the rename/reshape/whole-shard-cap skip branches all exercised.

## Status
Proposed — synthetic smoke passed; real run pending (base shards 1 and 7,
~10 GB download).
