# Candidate 0015 — Block-granular tile codes (NEXT_DIRECTIONS Direction A)

**Status:** PROPOSED — probe authored, adversarially reviewed, coder-measured
rewrite landed (fatal ideal-vs-realizable conflation fixed), synthetic smoke
re-passed, first real tensor verified (2026-07-01). Full real layer-27 run not
yet complete.

**Constraint:** pure lossless, bit-exact reconstruction only. Exact bit
accounting — every side structure (tables, bitmaps, indexes, headers, padding,
alignment) is charged; nothing is waved off.

## Hypothesis

Candidate 0014 CERTIFIED (adversarially verified) that **per-weight fixed-width
keying cannot beat the realized stz baseline** — 10.8975 b/w whole-model,
**10.8822 b/w numel-weighted on the canonical layer-27 target set** (256
tensors: 128 experts × {up,down}_proj, wholly in shard 7). The remaining
fusible-vs-storage gap is ~0.4 b/w (realized 10.88 vs the per-tensor order-0
storage floor H(sym)+7 ≈ 10.5).

That gap is priced as "the cost of random access" — but the constraint was only
ever proven for per-*weight* fixed-width codes. **The matmul reads tiles, not
single weights**, so the addressability tax can be paid per *block*:

> Relax "fixed width per weight" to "fixed byte budget per block of W weights"
> (fixed stride ⇒ O(1) tile address), and entropy-code freely *inside* the
> block. Within-block sequential decode happens transiently in registers during
> the tile fetch — the compressed form is never re-inflated in VRAM.

This is the single untested point on the project's own map: 0012's fusible
probe excluded within-block sequential decode explicitly, and 0014's
certificate covers per-weight codes only.

Field split (stz's exact convention): `u = np.frombuffer(raw, '<u2')`;
`sym = u >> 7` (9-bit sign+exponent, the structured field); `mant = u & 0x7F`
(7-bit mantissa, incompressible, stored verbatim — charged `pad8(7n)` bits
≈ 7.0 b/w). Total b/w = sym-plane bits per weight (including ALL side costs)
+ mantissa.

Blocks are contiguous flat-order (row-major) runs of W weights — the coalesced
read pattern. A 2-D matmul tile is a union of row segments, each of which is a
union of such blocks; block addressability at W ≤ row-segment length gives tile
addressability.

## Formats under test

Per-tensor static distributions throughout (the confirmed-winning granularity).
Both formats code the sym plane with one exactly-named coder against one
per-tensor table quantized to a 12-bit ANS table (M = 4096 states): counts
quantized deterministically (largest-remainder, every present symbol ≥ 1; the
serialized 12-bit field stores q−1, so the degenerate q=4096 constant plane
fits). **The coder:** per-block single-lane bit-renormalizing rANS, state in
[M, 2M), bit-by-bit renorm (emit `x & 1` while `x ≥ 2q`), 12-bit flush storing
`x_final − M`. **All per-block sizes are the MEASURED emitted bits of this
coder** (exact vectorized simulation, bit-identical arithmetic) — not the
quantized-entropy bound `Σ(12 − log2 q)`; the coder's measured excess over that
bound (renorm rounding + flush) is reported per W, and the quantization delta
vs true order-0 entropy is stated separately. Every tensor passes a
deterministic-sample round-trip gate before its record is written: pure-Python
encode → serialized bytes → decode on sampled blocks and lanes, asserting
emitted bits == accounted bits, exact symbol equality, and SHA-256-exact
reconstruction of the original raw BF16 bytes (sym re-merged with the verbatim
mantissa). Any mismatch aborts the run.

### Format (b) — padded fixed-stride blocks (the fusible hero)

Split the sym plane into blocks of W weights, W ∈ {32, 64, 128, 256, 512,
1024, 4096, 16384}. Per tensor and W, choose a byte budget B at percentile
P ∈ {90, 95, 97, 99, 100} of the **measured** per-block emitted bits of the
reference coder (so budgets and escape decisions are derived from real coded
sizes — every kept block provably fits its slot). **The fusible verdict is
capped at W ≤ 128** (Direction A's stated 64–128 row-segment range): random
access inside a kept block costs a sequential rANS decode of up to W symbols,
so larger W is reported as a storage-leaning bracket, never the fusible
headline. Layout and exact charges:

- **Main region:** nb = ceil(n/W) slots of exactly B bytes (fixed stride ⇒
  block j at byte offset j·B, O(1)). Kept blocks are charged the full 8·B bits;
  the gap between measured bits and 8·B is reported as **padding waste**
  (escaped-slot slack is included in the padding figure too).
- **Wholesale escape:** blocks whose measured size exceeds 8·B bits escape
  WHOLESALE to raw 9 b/w. An escaped block occupies its B-byte main slot plus a
  fixed-stride overflow slot of `max(0, ceil(9W/8) − B)` bytes — total
  `max(B, ceil(9W/8))` bytes, i.e. exactly its raw bytes (or the slot floor).
- **Per-block 1-bit escape bitmap:** `pad8(nb)` bits.
- **Escape-rank directory** (O(1) overflow addressing: rank(j) = anchor +
  in-group popcount): one u32 per 512 blocks, `32·ceil(nb/512)` bits.
- **Quantized ANS table:** 512-bit presence bitmap + 12 bits per present
  symbol, byte-padded: `pad8(512 + nnz·12)` bits.
- **Header:** 32 bytes (W, P, B, nb, n, R, C, overflow stride, nnz, …).
- **Mantissa plane:** `pad8(7n)` bits verbatim.

Decode of weight (r, c): block j = flat/W → check bitmap bit j → read B bytes
at j·B (plus overflow slot rank(j) if escaped) → sequential rANS decode of
≤ W symbols in registers → OR with the mantissa at fixed offset 7·flat. No
VRAM re-inflation at any point.

### Format (a) — superblock rANS with offset index (storage-leaning bracket)

4096-symbol superblocks, 32 interleaved lanes of the same reference coder
(lane l takes in-superblock positions l, l+32, …; 12-bit flush per lane),
variable superblock length, byte-aligned. O(1) access at *superblock*
granularity only — this brackets the gap from below and prices what
"almost-storage" costs; it can never carry the fusible verdict. Exact charges
per tensor:

- Per superblock: `pad8(measured lane bits incl. 32·12-bit lane flushes)` —
  byte-aligned so byte offsets address it. Lane flushes live *inside* the
  payload and are reported separately (`lane_flush_bits`), not double-counted
  as tax.
- **Two-level offset index:** level 2 = one u32 per superblock (byte offset
  from group base); level 1 = one u64 absolute anchor per 64 superblocks.
  At 4096 syms that is 32/4096 + 64/(64·4096) ≈ 0.008 b/w tax; lane flushes add
  384/4096 ≈ 0.094 b/w inside the payload. A per-tensor reconciliation assert
  proves payload + padding + tax sums to the charged total.
- Same quantized-table, header, and mantissa charges as format (b).

## Baseline & parity

Baseline is the realized stz cost recomputed per tensor via `stz.plan_regroup`
(imported from candidate 0009, never reimplemented; proven byte-exact by
`enc_tensor`'s internal assert). Parity gate: must match the recorded realized
bpw in `stz_tensor_stats.jsonl` within ±0.01 b/w on every target tensor, and
the numel-weighted reference must reproduce **10.8822 b/w** on the target set;
loud abort otherwise. In `--synthetic` mode the gate is stronger:
`stz.enc_tensor` is run and recomputed bits must equal its realized serialized
bits exactly, and the run aborts if zero tensors exercised the regroup codec.

## Success gates (both reported, numel-weighted over the 256-tensor set)

Both gates are keyed to the **tile-credible fixed-stride grid (W ≤ 128)** —
larger W and format (a) get their own explicitly-labeled storage-leaning
outcome and cannot carry the headline:

- **G1 (beats the crown):** best fixed-stride W ≤ 128 cell < **10.8822 b/w**
  (realized stz on the same tensors, same field split, all side costs charged
  on both sides, measured coder bits on ours).
- **G2 (entropy-relative):** best fixed-stride W ≤ 128 cell ≤ numel-weighted
  per-tensor (H(sym) + 7) + **0.15 b/w** — i.e. block granularity recovers the
  storage floor to within the tax Direction A budgeted.

Verdict: CONFIRMED (fixed-stride, W ≤ 128, measured coder bits) = G1 ∧ G2;
positive (G1 only, fixed-stride) = beats stz but not entropy-tight; positive
(storage-leaning only) = wins need W > 128 or format (a) — not tile-fusible;
FALSIFIED at this operating point = G1 fails everywhere. Any tensor missing
its round-trip gate would downgrade the verdict to PROVISIONAL (in practice
the run aborts on the first mismatch).

## Falsifier

**Heavy-tailed per-block code lengths making padding overhead eat the gain.**
If block ideal-code-length distributions have a fat right tail, then either
(i) B is set at a high percentile and every typical block pays
(p_high − median) b/w of padding, or (ii) B is set low and the tail mass
escapes wholesale at 9 b/w — both can exceed the ~0.4 b/w prize. This is THE
unknown the probe exists to measure: it persists per-block ideal-bit percentile
summaries (p50/p90/p95/p97/p99/max) and full coarse histograms (1/16 b/w bins)
at every W *before* any format verdict, so even a falsified run leaves the tail
map behind. Secondary falsifier: the coder's measured per-block overhead at
small W — the 12-bit flush (12/32 = 0.375 b/w at W=32) plus renorm rounding
plus the 12-bit quantization delta squeezing the viable W window shut from the
other side. First real-tensor measurement (expert 0 down_proj): total measured
excess over quantized entropy is +0.399 b/w at W=32, +0.132 at W=128, +0.044
at W=16384 — independently matching the adversarial reviewer's reference-coder
numbers (+0.035 at W=128, +0.044 at W=4096, renorm-only).

## How to run

```
# smoke (synthetic snapshot, seconds)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_block_codes.py --synthetic

# real run (resumable; each invocation self-limits to ~7 min and checkpoints)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_block_codes.py

# summary table + gates + summary JSON (auto-runs when complete)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_block_codes.py --summary
```

## Artifacts (tests/artifacts/)

- `blockcodes_results[_synthetic].jsonl` — one record per tensor: H(sym),
  floor, stz baseline + parity, quantization delta, measured per-W coded bits
  + coder excess, round-trip gate tallies, per-W ideal-block percentiles +
  coarse histograms, exact format (b) grid (W × P), exact format (a), and the
  accounting stamp (resume refuses to mix rows written under different
  constants). Resumable append (fsync'd, atomic truncated-tail self-repair).
- `blockcodes_summary[_synthetic].json` — numel-weighted table (per W × P and
  format (a): bpw, save vs stz, over floor, escape-block %, padding-waste
  b/w, tax b/w), per-W best cells with G1/fusibility marks, measured coder
  excess per W, ideal-tail diagnostics, quantization delta, gate verdicts
  (keyed to W ≤ 128), whole-model projections (pre-registered W128_P97 and
  post-hoc best, caveats inline).
- `stale-pre-coder-fix/` — pre-review synthetic artifacts (undercharging
  cost model), kept for comparison only; rejected by the accounting stamp.

## Links

- `research/notes/NEXT_DIRECTIONS.md` — Direction A (Tier 1), repricing note
  (the ~0.4 b/w gap), 0014 falsification note.
- `research/candidates/0014-column-keyed-codebooks/` — the per-weight
  fixed-width certificate this candidate relaxes; probe skeleton cribbed.
- `research/candidates/0009-fusible-exponent-codebook/tools/stz.py` — baseline
  cost model reused verbatim; `tests/artifacts/stz/stz_tensor_stats.jsonl` —
  per-tensor parity reference (10.8822 b/w numel-weighted on this set).
- `research/candidates/0012-lossless-ceiling/` — the storage-floor numbers the
  block codes are chasing in addressable form.
