# GPU run — closing the fusibility link (candidate 0009)

> **STATUS: DONE (2026-07-01, RTX 4090).** Result in `artifacts/gpu_bench_result.json`,
> analysis in `test-002.md`. Fused kernel is lossless + exact on-device and runs at
> **0.756× BF16 time (24% faster) in the bandwidth-bound regime.** The steps below are
> the reproduction recipe.

Goal: prove on real hardware that reading the experts NARROW (12 bits/weight = 4-bit
codebook index + 8-bit raw low byte) and rebuilding BF16 only in registers is
(1) exactly lossless on-device and (2) FASTER at batch 1 than a BF16 matmul moving
16 bits/weight — a real per-token bandwidth win, not just storage.

## Recommended pod
RTX 4090 (Ada sm_89, ~1 TB/s) — cheapest clean proof, solid Triton + `ncu`.
A100 PCIe (~1.9 TB/s, 80 GB) if you want the canonical datacenter number + batch-sweep headroom.
Avoid Blackwell consumer cards (5090/B200/B300) for now (toolchain immaturity).

## Files to upload
- `bench_gpu.py`            — Triton fused dequant+matvec vs cuBLAS BF16, correctness + latency
- `gpu_sample.npz` (29 MB)  — two REAL layer-1 expert tensors (up+down), pre-encoded
- (optional) `cpu_validate.py` — re-checks the algebra without a GPU

## Run
```bash
pip install --upgrade triton            # torch usually preinstalled on RunPod PyTorch images
python bench_gpu.py                      # prints + writes gpu_bench_result.json
```
Measured HBM bytes/token (the project's real metric):
```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum --target-processes all python bench_gpu.py
```

## What is already proven (CPU, here) — so what's left is ONLY the kernel
- `extract_sample.py` self-checks: reconstruction == original, bit-exact (SHA-256).
- `cpu_validate.py` confirms: (a) lossless high-plane reconstruction, and (b) the
  dense-approx + sparse-escape-correction matvec == exact BF16 matvec (rel err ~1e-7).
  The dense+sparse decomposition is SqueezeLLM-style and is the exact, fused-friendly form.
So `bench_gpu.py`'s correctness path is algebraically validated; only the Triton kernel's
on-device behavior (and the latency numbers) are unverified until you run it.

## Reading the result
- `lossless_on_gpu: true` — the narrow form reconstructs the exact BF16 weights on the GPU.
- `fused_vs_ref_rel` ~1e-6 — fused output matches cuBLAS BF16 (float rounding only).
- `speedup_fused_over_cublas > 1` at batch 1 — the bandwidth win is real. (Expect roughly
  up to ~16/12 ≈ 1.33x if fully bandwidth-bound; less if the v1 strided loads / escape
  correction add overhead — that's the kernel-tuning surface.)

## If the Triton kernel errors or is slow (expected first-pass iteration)
Paste the traceback / numbers back and we iterate. Known v1 simplifications to tune:
- low-byte loads are stride-2 (even/odd) → not fully coalesced; reorder for coalescing.
- escape correction is a separate torch `index_add_`; fold into one launch or precompute.
- try BLOCK ∈ {512,1024,2048}, and the bit-regrouped 7-bit-mantissa layout for the full ~29%.
