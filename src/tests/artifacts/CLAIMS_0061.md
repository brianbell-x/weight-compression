# Claims under test (bare, no narrative)

Artifacts:
- `ckpts/` — 16 JSON checkpoints, fields: repo, acc{...}, done_shards[]
- `glm52_standalone_result.json` — merged headline numbers
- `kernel_v10_result.json` — GPU benchmark output
- `bench.log` — benchmark stdout
- Merge code: `../../tools/merge_glm.py`
- Coder code: `src/tools/stream_validate.py`
- Kernel code: `src/tools/bench_kernel_v10.py`

Claims:
1. The 16 checkpoints' `done_shards` are pairwise disjoint and their union is
   exactly the 282 shards model-00001..model-00282 of zai-org/GLM-5.2.
2. Summing the 16 `acc` blocks reproduces every number in
   `glm52_standalone_result.json` (reduction 30.168% regroup / 24.967%
   byte-split, 11.173 / 12.0053 b/w, 59,509 BF16 tensors, 753,329,921,024
   weights, total 1403.19 GB, escape rate 0.0249%).
3. `all_lossless` is true in every checkpoint, and in the coder code the
   only path that sets it requires `np.array_equal(high_rec, high)` per BF16
   tensor with the low byte stored verbatim.
4. In `kernel_v10_result.json`: ratio_il_over_bf16 = 0.733 equals us.il/us.bf16;
   WIN=true; rel err 1.31e-07; K=476 with tensor shape [2048, 6144].
5. The printed `GBps` fields (0.61 / 0.63) are a unit mislabel in the script
   (ms treated as s); correctly computed effective bandwidths are ≈614 GB/s
   (bf16) and ≈628 GB/s (il, at 1.5 B/weight) on a ~696 GB/s-peak A40.
6. The bits-accounting in the coder (`enc_bytesplit_verify`, `bits_regroup`)
   charges: 4 index bits/weight (byte-split) or 4 bits + 7 mantissa bits
   (regroup), escapes at 8/9 bits, per-row escape-offset pointers, and the
   codebook — i.e. the 30.168% figure includes those side costs.

Brief: attempt to REFUTE each claim by recomputation from the artifacts and
code. Report discrepancies with numbers. Do not edit any repository files.
Write your report to
`src/tests/artifacts/BLIND_VERIFICATION_0061.md`
and end with an overall verdict line: GATE_PASS or GATE_FAIL.
