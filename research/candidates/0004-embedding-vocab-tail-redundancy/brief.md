# Candidate: Embedding Vocab-Tail Row Redundancy

## Claim
The embedding table has `vocab_size = 131072` (exactly 2^17), almost certainly
larger than the actually-trained vocabulary, so a block of high-ID rows is likely
untrained — either all-zero, a shared constant, or exact duplicates. Those rows
compress to near-nothing losslessly via row dedup + a reference list, on top of
the per-value plane split.

## Why It Might Work
`131072` is a clean power of two, the classic sign of a padded/reserved vocab.
The embeddings scout sampled only the FIRST 64 rows (common tokens) and found
them all unique with low correlation — but that sample is blind to the tail. Real
tokenizers leave many reserved/unused slots; untrained embedding rows are
typically left at their initialization (often a single constant or zeros) or
never updated, making them exactly identical or exactly zero. Exact-duplicate or
exact-constant rows are perfectly losslessly compressible (store the row once +
an index), unlike the random-mantissa interior of trained rows.

This is a DIFFERENT structure from [[0001-bf16-exponent-plane]] (per-value byte
planes) and from [[0003-cross-expert-base-delta]] (which failed): here the
redundancy is at the ROW level and concentrated in a specific ID range, not
distributed per value.

## Tensor Group
`backbone.embeddings.weight` (BF16 [131072, 2688], 704,643,072 bytes) and the
untied `lm_head.weight` [131072, 2688] (tie_word_embeddings=false).

## Measurement
Over the FULL 131072 rows (not a 64-row sample):
1. Count exactly-duplicate rows (hash each 2688-value BF16 row); report the
   largest duplicate group and total duplicated rows.
2. Count exactly-constant rows (all 2688 values equal) and all-zero rows.
3. Plot per-row L2 norm / distinct-value count vs token ID to locate any
   untrained tail block and its start ID.
4. Cross-check against the tokenizer's real vocabulary size / added-tokens to
   confirm which IDs are reserved.
5. Encode unique-rows-once + index, verify byte-exact reconstruction of the full
   tensor, and report bytes saved (standalone and on top of the 0001 plane split).
6. Repeat for lm_head; check whether embedding and lm_head share identical rows.

## Promising Result
A contiguous tail (or scattered set) of thousands of exactly-duplicate / constant
/ zero rows. Even 5,000 reserved rows * 2688 * 2 B = ~27 MB collapsing to a few
KB is a clean lossless win the byte-plane codec cannot capture (those rows still
cost full mantissa bytes under 0001). If essentially all 131072 rows are unique
and trained, the idea fails and embeddings are just another 0001 target.

## Test Target
True weights directly — this is about the real trained/untrained row pattern,
which the synthetic snapshot (random values, no real vocab structure) cannot
exercise. The embedding tensor is in shard 1, already local.

## Status
Rejected
