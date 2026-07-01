# Test 002 — GPU fused kernel: losslessness + per-token bandwidth win (real hardware)

Date: 2026-07-01. Hardware: RunPod RTX 4090 (Ada, ~1 TB/s HBM), torch 2.4.1+cu124,
triton 3.0.0. Script: `tests/artifacts/bench_gpu.py`. Sample: `gpu_sample.npz` (two real
layer-1 experts, up+down, encoded byte-split K=15 = 12 b/w). Raw output:
`tests/artifacts/gpu_bench_result.json`. This closes the one link test-001 left open:
that the narrow form runs *directly* and *faster* on a GPU, not just on paper.

## What was tested
A Triton kernel reads the weights NARROW (4-bit codebook index + 8-bit low byte = 12 b/w),
rebuilds each BF16 value in-register via a precomputed `(nibble,low)->weight` LUT (4096
floats, L1-resident), and does the matvec — the full BF16 is never written to VRAM
(Regime D). Escapes (~0.3%) are corrected by an in-place sparse term (SqueezeLLM
dense+sparse). Compared against: an **identical-structure BF16 Triton kernel** (isolates
the bandwidth lever from kernel quality) and cuBLAS.

## Results

### Correctness + losslessness on-device — PASS
- `lossless_on_gpu: true` for both experts: the codebook+escape form reconstructs the
  **exact** BF16 high-byte plane on the GPU.
- Fused matvec vs BF16 reference: **rel error ~1e-7** (float-rounding only) — the math
  computing directly on the narrow form is numerically exact.

### Per-token bandwidth win — PASS (in the regime that matters)
Two regimes, and the distinction is the whole story:

| regime | fused (12 b/w) | BF16 twin (16 b/w) | cuBLAS | fused/BF16 | bound by |
|---|---|---|---|---|---|
| **single expert** (~10 MB matvec, ~18 µs) | 18.0 µs | 16.1 µs | 15.0 µs | ~1.1 | launch/reduction overhead |
| **bandwidth-bound** (~480–640 MB read) | **507.6 µs @ 0.9 TB/s** | **671.4 µs @ 1.0 TB/s** | 670.7 µs | **0.756** | **HBM bandwidth** |

- In the **bandwidth-bound regime** (tile the weight ×64 so read time dominates launch
  overhead), **fused/BF16 = 0.756 ≈ the ideal 0.75 (=12/16)** — a **24% decode speedup
  matching the 25% fewer bytes almost exactly.** Both BF16 baselines run at ~1 TB/s (the
  4090's memory ceiling), confirming they are memory-bound; fused moves 0.75× the bytes
  and takes 0.75× the time.
- The dense fused dequant is **free**: with the LUT, the fused kernel matches the BF16
  twin per-byte (the 24% gain is entirely the byte reduction, not lost to unpack cost).

### Why the single-expert number shows no win (honest)
At ~18 µs, fixed kernel-launch and the separate sparse-escape launch (~10 µs) dominate the
~4 µs theoretical bandwidth saving, so neither kernel is bandwidth-bound and the win is
invisible. This is a measurement-scale artifact, not a limit: real MoE decode reads on the
order of GBs of experts per token — the bandwidth-bound regime, where the 0.756 ratio holds.

## Verdict
- The user's question is answered on real hardware: **the math CAN use the compressed form
  directly** (lossless + exact on GPU, BF16 only in registers, never re-inflated to VRAM =
  Regime D), **and it is proportionally faster when memory-bound** (0.756× time = 24%
  speedup, matching the 25% byte reduction). The 32% storage idea, re-cast as a fixed-width
  codebook, is now a measured **per-token bandwidth + resident-VRAM win**, still exactly
  lossless.
- **Open engineering (not a limit on the claim):** fold the escape correction into the main
  kernel (one launch) so the small-batch/single-matvec case also benefits; and push the
  byte budget from 12→~11.3 b/w (the 7-bit-mantissa regroup, ~29%) which scales the same
  0.75→~0.71 ratio. These are optimizations; the mechanism and the bandwidth win are proven.

## CORRECTION (test-004, H100): the speedup is kernel- and GPU-dependent
The 0.756× above was measured on a **4090 (~1 TB/s HBM)**. Re-running the identical
benchmark on an **H100 (3.35 TB/s)** overturns the general claim: fused is **~2× SLOWER**
than BF16 there (fused 412 µs @ **1.2 TB/s** vs BF16 twin 207 µs @ 3.1 TB/s at the same
read). Diagnosis: this naive Triton LUT-dequant kernel is **dequant-throughput-limited to
~1.2 TB/s**. On a ~1 TB/s GPU that matches memory bandwidth → it wins; on a 3.35 TB/s GPU
the dequant is the bottleneck → it loses. So "24% faster decode" is **4090-specific and
naive-kernel-limited, NOT general.** The lossless byte reduction (25–30%) is real
everywhere; converting it to speed on high-bandwidth GPUs needs a production dequant kernel
(Marlin/FLUTE-class, shared-mem LUT, vectorized) — which this is not. See `test-004.md`.

## Reproduce
`bench_gpu.py` + `gpu_sample.npz` on any CUDA GPU with Triton:
`pip install --upgrade triton && python bench_gpu.py`. (Ran on a RunPod RTX 4090 driven
via `runpodctl`; ~30 min wall, ~$0.35.)
