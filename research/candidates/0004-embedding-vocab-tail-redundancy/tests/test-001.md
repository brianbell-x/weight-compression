# Test 001 — Embedding Vocab-Tail Row Redundancy (TRUE weights)

**Date:** 2026-06-29
**Target:** TRUE weights, full 131072 rows (not sampled).
**Tensors:** `backbone.embeddings.weight` (shard 1) and `lm_head.weight` (shard 13), both BF16 `[131072, 2688]`, 704,643,072 bytes each.
**Scripts:** `artifacts/measure_embed_rows.py`, `artifacts/supplement.py` (run with `uv run`).
**Outputs:** `artifacts/results.json`, `artifacts/embeddings_per_row.csv`, `artifacts/lm_head_per_row.csv`.

## Verdict: REJECTED

The hypothesis — that `vocab_size = 131072 = 2^17` implies a large block of **untrained high-ID
tail rows** that are all-zero / constant / exact-duplicate — is **false on the real weights**.
There is no untrained tail. Both tensors are essentially fully trained and row-unique. Reconstruction
is byte-exact, but the row-dedup yields **negative** savings on embeddings and **0.50%** on lm_head,
and **essentially nothing on top of the 0001 plane split**.

## Headline numbers

| Metric | embeddings | lm_head |
|---|---:|---:|
| Total rows | 131072 | 131072 |
| Unique rows (exact bytes) | **131056** | 130362 |
| Duplicate groups (size>1) | 11 | 67 |
| Rows inside dup groups | 27 | 777 |
| Redundant rows removable | **16** | **710** |
| Largest dup group | 4 | 88 |
| Exactly-constant rows | **0** | **0** |
| All-zero rows | **0** | **0** |
| Per-row L2 min / mean / max | 2.7e-6 / 0.855 / 1.26 | 0.309 / 1.050 / 1.56 |
| Byte-exact reconstruction (SHA-256) | **PASS** | **PASS** |
| Dedup bytes saved standalone | **-192,512 (-0.027%)** | **3,538,432 (+0.50%)** |

SHA-256 of reconstructed bytes equals the original tensor bytes for both
(`2e57f847…` embeddings, `eb9d13ae…` lm_head), so the dedup encode/decode is provably lossless —
it just buys (almost) nothing.

## 1-2. Duplicates, constants, zeros

- **Embeddings:** 0 constant rows, 0 zero rows. Only 11 tiny duplicate groups (largest = 4 rows),
  scattered across the whole ID range (e.g. group `{5, 81819, 82977, 114442}`), **not** a contiguous
  tail. Net 16 rows are removable out of 131072 (0.012%).
- **lm_head:** 0 constant rows, 0 zero rows. 710 removable rows, but they are **not** a high-ID tail —
  **740 of 777** dup-involved IDs are **< 1000**, i.e. the low reserved special-token slots
  (`<SPECIAL_0..999>`). The big groups all share L2 ≈ 0.31192 (one near-identical low-norm vector that
  fans out into several exact-dup subgroups differing only in low mantissa bits).

## 3. L2 / distinct-value vs token ID — no untrained tail

CSV `*_per_row.csv` (stride-64 over all 131072 rows). Key aggregates:

- **Embeddings:** L2 mean is flat across the whole range — first-1000 = 0.897, 1000.. = 0.855,
  last-5000 = 0.775. The high-ID tail is fully trained (slightly lower norm, as is normal for rare
  tokens, but unique and non-degenerate). The global L2 argmin is id **25304** (2.7e-6, a single
  near-zero but still-unique trained row), **not** in a tail block.
- **lm_head:** The only anomaly is the **low** end — IDs 0..999 have L2 ≈ 0.315 vs 1.056 for real
  tokens. The reserved special-token *output* rows collapsed toward a shared low-norm value during
  training (they are never a prediction target), which is exactly the 710-row duplication above. The
  high-ID region is fully trained (last-5000 mean L2 = 1.016).

## 4. Tokenizer cross-check

`tokenizer.json` defines a token string for **every** ID in `[0, 131071]` (base vocab = 131072,
0 undefined slots). `config.json`: `vocab_size = 131072`, `tie_word_embeddings = False`,
`hidden_size = 2688`. The 1000 `added_tokens` are `<SPECIAL_0..999>` placed at the **low** IDs 0-999,
not a high tail. So the padded/reserved capacity is the **low special-token block**, and only its
**lm_head** rows (not its embedding rows) show measurable degeneracy. There is no reserved high-ID
tail to exploit.

## 5. Reconstruction + savings, standalone and on top of 0001

- **Encoding:** unique-rows-once block + per-row index (17 bits/row tight = 278,528 bytes).
- **Reconstruction:** byte-exact for both (SHA-256 match, see table).
- **Standalone savings:** embeddings **−192,512 bytes (net loss** — the 17-bit index over 131072 rows
  costs more than the 16 saved rows × 5,376 B). lm_head **+3,538,432 bytes (+0.50%)**.
- **On top of 0001 (de-interleave + entropy on the high plane):** ~nothing. Exact-duplicate / low-norm
  special rows present a nearly-constant high-byte plane (sign+exp+top mantissa bit identical across the
  group), so 0001's entropy coder **already** drives those rows to near-zero bits. Row-dedup and 0001
  target the *same* redundancy here; stacking them is not additive. The incremental value of row-dedup
  over 0001 is negligible (<<0.5% on lm_head, negative on embeddings).

## 6. lm_head vs embeddings cross-check

**0 shared rows** between `backbone.embeddings.weight` and `lm_head.weight` (consistent with
`tie_word_embeddings = False`). No cross-tensor dedup opportunity.

## Payoff regime

Even in its best case (lm_head, 0.50%) this is a **lossless STORAGE win only — Regime A/B**. It is
**not** a per-token-bandwidth or decode win, and it is **negative on the embeddings tensor**. A
resident-VRAM benefit would require holding the table deduped in memory behind an index indirection on
the hot embedding-lookup / lm_head matmul path — not worth it for ≤0.5%, and it would *add* a gather
step to decode. Do not sell this as a runtime win; it is a fraction-of-a-percent storage win that 0001
already captures.

## Conclusion

The specific mechanism in the brief (untrained zero/constant/duplicate **high-ID tail**) does not exist
in these weights. The real structure is the opposite: a small **low-ID** special-token degeneracy that
appears only in `lm_head` (710 rows, 0.50%) and is already covered by 0001. Embeddings are fully unique
and trained; dedup there is a net loss. Reject as a standalone candidate.

## Next Action

None. (Fold the one real observation — lm_head's ~710 near-identical low-norm special-token output rows
— into the 0001 entropy-codec evaluation as already-captured redundancy; no separate row-dedup pass is
warranted.)
