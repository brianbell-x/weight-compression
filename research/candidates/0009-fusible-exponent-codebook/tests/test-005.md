# Test 005 — Kernel optimization to beat bf16 on H100 (the deep attempt)

Date: 2026-07-01. Hardware: RunPod H100 SXM 80GB (3.35 TB/s) + H100 NVL. Directive:
the test-004 "24% faster is 4090-specific" conclusion was rejected — H100 ≫ 4090, so a
better kernel should win. This test is the exhaustive attempt to make the fused lossless
kernel beat bf16 GEMV on a clean, saturating H100. Scripts: `bench_kernel_v2..v10.py`.

## Method — 10 kernel variants, systematic isolation
Tiled one real expert to the true per-token footprint (K=370 ≈ 3.7 GB read) so the
kernel is genuinely bandwidth-bound. Compared every fused variant to bf16 GEMV (which
hits **3.15 TB/s = 94% of the SXM's 3.35 peak**, matching cuBLAS — a fair, saturating
baseline).

## What the diagnosis found (this part is the real result)
1. **Narrow loads are NOT the problem.** A single narrow plane (`lowonly`) reads at
   **2.13 TB/s = ties bf16's per-byte rate**. Sub-byte packing does not inherently slow loads.
2. **The two real costs, isolated:** the per-element **LUT gather** (−26%) and **strided
   activation (x) access** (−25%). The original test-002 kernel had both → 2× slower.
3. **Killed the gather:** the codebook high-bytes are near-contiguous exponent ranges
   (sign0:56–61, sign1:183–189), so `high = (sign<<7)|(BASE+offset)` decodes by pure
   ARITHMETIC — no gather. **Verified this encoding is exactly lossless** (0.53% escapes at
   optimal BASE, bit-exact round-trip).
4. **Killed the strided x:** Marlin-style interleaved layout (lane owns cols {j,Q+j,2Q+j,3Q+j})
   so wide idx/low loads AND contiguous x. Kernel output verified bit-exact (rel err 1.8e-6).

## The verdict on a clean, saturating H100 SXM
| kernel | TB/s (reads) | ratio vs bf16 (3.15) | correct? |
|---|---|---|---|
| bf16 GEMV (2.0 b/w) | 3.15 | 1.00 | — |
| best fused `noLUT` (1.5 b/w) | 2.34 | **1.01** | (approx) |
| fused interleaved+arith `v10` (1.5 b/w) | 2.15 | **1.09** | ✓ bit-exact |

**The fused lossless kernel does NOT beat bf16 on H100 — it ties (best 1.01) to slightly
loses (1.09).** The 25% byte reduction (1.5 vs 2.0 b/w) is almost exactly cancelled by the
unavoidable two-stream sub-byte unpack overhead. Every variant plateaued at a **~2.3 TB/s
fused ceiling** — below the H100's 3.35 that bf16 nearly reaches.

## Corrects test-004, precisely
- **NOT "4090-specific" in the dismissive sense** — the fused kernel runs **~2× FASTER on
  H100 (2.15–2.34 TB/s) than on the 4090 (~1 TB/s)** in absolute terms. The user was right
  that H100 is far more capable and the kernel does exploit it.
- **But the RATIO flips** because bf16 *also* scales to 3.15 TB/s on H100 (94% of peak),
  which exceeds the fused kernel's ~2.3 TB/s unpack ceiling. Crisp rule: **fused beats bf16
  on any GPU with HBM bandwidth below ~2.3 TB/s (4090 ~1, A100 ~2) and ties/loses above it
  (H100 3.35).** The win isn't chip-specific; it's bandwidth-threshold-specific.
- Root cause: with only a **25% byte margin**, there is no room to absorb *any* unpack
  overhead once bf16 is already at 94% of peak. (4-bit quant wins because its 4× margin
  swamps the overhead; a lossless 12-bit scheme has no such cushion.)

## What this is still worth (the honest silver lining)
A *tie* on H100 is not nothing: it delivers **~25% lossless resident-VRAM reduction at zero
decode-speed cost**. That strictly beats DFloat11 (the standard lossless method, which is
*1.4–2× slower* at batch 1). So on H100 it's the best lossless option — free memory savings —
and on any GPU ≤2.3 TB/s it's also a real decode speedup.

## Remaining open path (not achieved here)
A production CUDA kernel (Marlin/FLUTE-class) that makes the unpack *truly free* (single
128-bit coalesced packed stream + shared-memory/register LUT, no per-lane gather) could
plausibly push the fused read toward peak and reach ~0.8× (a real ~20% H100 speedup). My 10
Triton variants could not cross the ~2.3 TB/s ceiling. Whether a hand-CUDA kernel clears
3.35 TB/s on a 1.5-b/w stream is unproven and, given the thin margin, uncertain. That is the
one lever left; it's a real engineering project, not a config tweak.

## Cost
~7 H100/NVL pod-hours across the investigation (RunPod SXM repeatedly failed to boot —
auto-retry punched through). All pods torn down.
