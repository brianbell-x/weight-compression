# 0061 GLM-5.2 standalone-codec transfer — RESULTS

**STATUS: GATED WITH SEPARATE EVIDENCE CLASSES (2026-07-13).** The blind gate
recomputed the retained artifact arithmetic and code paths. It did not replay
the raw 1.4 TiB input, serialize or decode the K15 layout at GLM scale, or time
the sparse exact correction path.

- **Experiment:** succeeded — the full 753B GLM-5.2 scan produced **30.168%
  charged K15 bit accounting** (11.173 b/w) and, for a separate byte-split
  representation, **24.967% reduction with all 59,509 BF16 tensors
  round-tripped bit-exact** (12.005 b/w). The K15 representation was not
  independently serialized or decoded at GLM scale. A separate dense 12-bit
  GEMV prototype measured 0.733× BF16 time on an A40; sparse exact correction
  was validated outside the timed kernel.
- **What this means for the project:** sign-and-exponent concentration
  transfers beyond the Nemotron family to a Z.ai checkpoint at 753B scale.
  The 30.168% accounting, 24.967% exact inverse, and dense-path timing are
  independent measurements and must not be combined into a 30% exact-runtime
  claim.
- **Reproduction files:**
  `src/tools/stream_validate.py`
  (16 range-partitioned runs: `--start 18k --shards 18`, k=0..15, last range
  12), `tools/merge_glm.py` (merges the 16 checkpoints → headline JSON),
  `src/tools/extract_glm_sample.py` + `src/tools/bench_kernel_v10.py`
  (kernel benchmark).

## Prior art and distinction

[ZipNN](https://arxiv.org/abs/2411.05239) established exponent-separated
Huffman compression for BF16 storage and transfer, and
[DFloat11](https://arxiv.org/abs/2504.11651) applied variable-length exponent
Huffman coding to GPU-resident inference while restoring BF16 weights before
matrix multiplication. [ZipServ](https://arxiv.org/abs/2603.17435) is the
closest prior runtime design: its fixed-length format reconstructs weights
directly in Tensor Core registers. This repository therefore does not claim the
discovery of BF16 exponent redundancy, roughly 11-bit lossless weights,
fixed-width exact fallbacks, or fused reconstruction. Its distinction is the
specific representation tested here: per-tensor 4-bit codes over 15 joint
sign-and-exponent symbols, raw 7-bit mantissas, and sparse exact escapes. This
0061 result claims cross-family measurement and validation on GLM-5.2, not
first discovery of the underlying compression principle.

## Compression (all 282 shards, 1403.19 GiB, zai-org/GLM-5.2 @ main)

From `tests/artifacts/glm52_standalone_result.json` (merged from the 16
per-range checkpoints in `tests/artifacts/ckpts/`; merge asserts 282 distinct
shards, no overlap, no gap):

| Metric | Value |
|---|---|
| Byte-split BF16 tensors round-tripped bit-exact | **59,509 / 59,509** |
| BF16 weights | 753,329,921,024 (100.0% of bytes; F32 sidecar ≈ 0.0003 GiB) |
| Regroup K=15, charged bit accounting | **30.168%** (11.173 b/w) |
| Byte-split, independently decoded | **24.967%** (12.005 b/w) |
| Accounted K15 size | 1403.19 → **979.87 GiB** |
| Escape rate | 0.0249% of weights (byte-split coder count) |

Family comparison (same codec, same K): Nemotron Nano 30B ≈ 30.03%,
Super 120B = 28.85%, **GLM-5.2 753B = 30.17%**.

Caveats:
- `n_expert_tensors = 0` in the accounting is cosmetic: the expert regex in
  `stream_validate.py` matches Nemotron names (`mixer.experts.*`), not GLM's
  `mlp.experts.*`. The expert mass is inside the BF16 totals; only the
  expert-only breakdown row is unavailable.
- Lossless proof class: `enc_bytesplit_verify` reconstructs the high plane and
  checks `np.array_equal` per tensor (low byte verbatim); the regroup variant
  is a bit-accounting of the same field split (no independent decode), as in
  the gated Super-120B run. Container serialization was not run at this scale
  (stz-class tooling exists but was not part of this test).

## Speed (dense-path prototype, real GLM weights)

From `tests/artifacts/kernel_v10_result.json` (`bench_kernel_v10.py`, A40
46 GB, torch 2.4 + Triton, tensor
`model.layers.10.mlp.experts.0.up_proj.weight` [2048×6144] tiled K=476 ≈ 6.0B
weights to reach the bandwidth-bound regime):

- **BF16 GEMV 19.52 ms vs dense 12-bit prototype 14.31 ms: ratio 0.733,
  or 26.7% less measured time for the dense path.**
- Effective read bandwidth: BF16 ≈ 614 GB/s, prototype ≈ 628 GB/s of A40's
  ~696 GB/s peak. (The script's printed "GBps" field is mislabeled by
  1000× — it divides ms by 1e9; values are GB/ms×1e-3. Corrected here.)
- Escape rate on this tensor is 0.0087%. The sparse host-side correction term
  was validated separately but was not fused into or included in the timing;
  therefore this run does not establish an exact lossless speedup.
- Frame discipline: this is a dense-path GEMV microbenchmark on one GPU model,
  not an exact end-to-end serving result.

## Run economics

8× A40 secure pods ($0.44/hr), 16 range-partitioned streaming workers, whole
1.4 TB processed in ~70 min wall-clock; speed benchmark on 1 pod afterwards.
Total spend ≈ $7.9; all pods deleted (spend/hr = 0, balance $20.13).

## Blind gate (2026-07-12)

Verifier report: `tests/artifacts/BLIND_VERIFICATION_0061.md` — GATE_PASS.
Independently re-derived: shard coverage (282 disjoint, union exact), every
merged headline field (zero differences), the lossless code path, kernel
ratios, the bandwidth-unit correction (613.6 / 628.0 GB/s), and the bit
accounting incl. side costs. Verifier-noted limitations, standing as
disclosed: (1) evidence bundle supports internal consistency + code-path
verification, not a fresh replay from raw shards/hardware (raw 1.4 TB and
the GPU sample were not retained — re-run `stream_validate.py` /
`extract_glm_sample.py` against the public repo to replay); (2) the coder's
resume path trusts a saved checkpoint's `all_lossless` flag — the per-tensor
equality check applies to newly processed tensors (all 282 shards here were
processed fresh in this run, per worker logs); (3) `merge_glm.py` ckpt-path
packaging nit — fixed post-gate (script now also resolves
`tests/artifacts/ckpts/`; re-run reproduces all numbers).

## Discovery questions filed

1. GLM-5.2's escape rate (0.025%) is ~14× lower than Nemotron's (~0.3%) —
   is its exponent distribution tighter, and does that mean K<15 (3-bit
   index) is viable on GLM-class models for ~34%?
2. Z.ai ships FP8 variants of GLM models — if one shares master-weight
   lineage with the BF16, the whole FP8-conditional program (4.2 b/w
   covered) has a second family. (Parked: Brian abandoned the
   FP8-conditional method for now, 2026-07-12.)
3. Does the 2-D exponent context lever (0012, +4 pts storage) transfer to
   GLM experts too — i.e. is the storage ceiling here also ~34%?
