# Split12 inference tests

Reconstructed record of the end-to-end serving and kernel work on Split12,
July 13–20, 2026. **The raw artifacts and code (`0062-e2e-serving-test/`,
`retest-results/`) were lost in an unlogged local deletion between 2026-07-20
and 2026-07-23; the GPU pods holding the only other copies were terminated.**
Everything below is reconstructed from the research ledgers — numbers are as
recorded there, but the underlying JSON/logs/kernels no longer exist to
re-inspect.

## Timeline

### 2026-07-13 — GLM-4-9B full-model serving, RTX A6000 (Prime)

All 241 linears converted to exact Split12, zero raw-kept. Matched 269-token
prompt, batch 1, 256 generated tokens:

- Resident weights: **−23.182%** vs BF16; process peak: **−19.985%**
- Decode: 3.664 vs 9.488 tok/s (**2.589× slower**) — unfused exact correction
  is nowhere near throughput-neutral
- Prefill: 7,684.834 vs 775.809 ms

Same day, kernel optimization (Triton CSR fusion into the packed GEMV):
decode 3.664 → 6.741 tok/s, still 24.910% behind the 8.977 tok/s BF16 control;
prefill still 10.525× BF16. Development evidence only (BF16-first startup,
one process per mode).

### 2026-07-14 — Split12 crosses BF16 at 9B dev scope (A6000 + RTX 6000 Ada)

New CUDA kernel family for the batch-1 access pattern (two output rows per
block, inline escape delta, fused QKV with shared exponent base, stateful
operator holding immutable GPU planes):

- A6000: **9.317 vs 7.754 tok/s (+20.147%)**
- RTX 6000 Ada: **21.345 vs 20.641 tok/s (+3.414%)**
- Sampled process peak ~18% below BF16 on both; every 241-matrix inverse
  bit-exact; token hashes matched

### 2026-07-14 — Full GLM-5.2 753B on 8× B300 SXM6 (Prime)

The proving-ladder target running entirely from exact Split12 planes:

- Decode: **14.932 vs 11.752 tok/s (+27.06%)**
- Resident weights **−24.83%**; live Torch allocation −22.47%; process peak
  −11.79%
- All 75 expert banks (38,400 expert matrices) + 688 ordinary linears
  bit-exact; runtime reads packed planes directly, no full-width matrix
  materialized
- **Failed canonical acceptance:** startup was BF16-first, and generated
  tokens diverged from BF16 after a 26/32-token matching prefix (reduction
  order, not codec loss — weights reconstruct exactly). Prefill 52.1% slower.

### 2026-07-15 — Matched production SGLang retest on 8× B300 (r6)

The +27% did not survive a real serving stack. Matched against the optimized
production SGLang configuration (TP8, EAGLE, Triton MoE, FP8 KV, frozen
10-prompt × 3-order workload, 30 sequential 512-token requests):

- **BF16: 207.958 pooled decode tok/s — the earlier ~12 tok/s "baseline" was
  a reference-harness artifact**
- Split12: **73.618 tok/s (64.6% slower)**; TTFT 0.474 vs 0.085 s; TPOT
  13.584 vs 4.809 ms
- Steady device memory: **1,675.266 vs 1,994.102 GiB (−15.99%)**
- Wins kept: server started from the serialized Split12 artifact with no
  full BF16 checkpoint load; 321,432 artifact records verified zero
  mismatches; compatibility fallback down to 4.442 GiB BF16
- Cost: $162.07. Conclusion: the gap is kernel work, not harness or prompts.

### 2026-07-15 → 07-20 — Single-B300 kernel gate

r6 decomposition: 2.8248× decode penalty = **2.1984×** slower per target
verification × **1.2850×** more verification cycles (EAGLE numerics). A
one-B300 gate (TP=8 production shard shapes, batch 1 / EAGLE-verify batch 6 /
prefill batch 128) was frozen before any further 8× spend.

- **v1→v2 batch-tiled scalar kernels (Jul 20, 1× B300, $5.17):** improved
  large shapes (e.g. attn_o 6.69× → 4.65× vs BF16) but nothing crossed 1.0× —
  scalar kernels are issue/latency-bound on per-element integer
  reconstruction; cuBLAS BF16 sits at a ~13–16 µs launch-bound floor.
  Scalar tuning ruled out on Blackwell, confirming the earlier H100 verdict.
- **Tensor-core campaign (Jul 20, thirteen iterations, $76.62):** smem+WMMA
  tiles stalled on the per-weight smem round trip; fragment-register decode
  straight into `mma.sync` operands worked. Final grouped MoE fragment
  kernel: **expert_w2 0.80× BF16** (first sub-1.0× Split12 result on B300),
  expert_w13 1.06×. Dense shapes: BF16 still wins everywhere (best 1.21–1.86×)
  — the ~10–12 µs launch floor leaves small dense shapes no room.

## Where it stands

- Compression and residency wins are proven at full 753B scale; decode-from-
  compressed-planes works end to end.
- Open for a serving win: fuse exact correction into the native Tensor Core
  accumulation schedule (dense shapes), put w13 under 1.0× (2-warp grouped
  block or deeper prefetch), amortize launch floor via CUDA graphs, solve
  the 1.285× EAGLE verification-cycle factor (numerics/reduction-order
  compatibility), and eliminate the remaining 4.442 GiB SGLang compatibility
  fallback.
- Any rerun should follow the frozen r6 protocol; raw protocol definitions
  were in `retest-results/` and are lost — the ledger entries above are the
  surviving specification.
