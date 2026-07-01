# Test 004 — Rung 2: end-to-end reality check (H100 kernel limit + generate() status)

Date: 2026-07-01. Hardware: RunPod H100 SXM 80GB (3.35 TB/s). Goal: measure the decode
speedup at the real per-token footprint AND a real `generate()` wall-clock side-by-side.
Outcome: an important **correction** to test-002, and an honest **blocked** status on
generate(). Both matter more than a flattering number would.

## Setup that worked
- Real 30B model downloaded on the pod (59 GB, not gated), torch 2.4.1 / triton 3.0.0.
- Per-token active footprint computed from config: 23 MoE layers × (6 routed + 1 shared)
  = **1.84B active MoE params/token = 3.69 GB BF16 read/token** (the dominant decode cost).

## Finding 1 — the speedup does NOT survive on a high-bandwidth GPU (correction)
Same fused-vs-BF16 bandwidth benchmark as test-002, on the H100:

| read size | fused (12 b/w) | BF16 twin (16 b/w) | fused/twin | fused eff. BW |
|---|---|---|---|---|
| 640 MB (K=64) | 412 µs | 207 µs @ 3.1 TB/s | **1.99** | **1.2 TB/s** |
| 3.7 GB (K=370, real/token) | 2350 µs | 1168 µs @ 3.2 TB/s | **2.01** | ~1.2 TB/s |

- The BF16 twin saturates the H100 (~3.1 TB/s ≈ peak) → genuinely bandwidth-bound.
- The fused kernel **caps at ~1.2 TB/s regardless of GPU** — it is limited by the LUT-gather
  dequant, not by memory. On the 4090 (mem ~1 TB/s) 1.2 TB/s dequant keeps up → the byte
  reduction shows as a 24% speedup. On the H100 (3.35 TB/s) the dequant is the bottleneck →
  fused is ~2× slower.
- **Conclusion:** the lossless **byte** reduction (25–30%, test-003) is real on any GPU, but
  converting it to a **speedup** requires a dequant kernel that runs at ≥ the GPU's memory
  bandwidth. This naive Triton kernel does that on a 4090, not on an H100. A production kernel
  (Marlin/FLUTE-class: vectorized packed loads, shared-memory LUT, no per-element gather) is
  required to realize the win on modern datacenter GPUs. That is the honest remaining gap.
- (Also found: an int32 index overflow in the kernel at K≥740 — `r*C` exceeds 2^31; a bug to
  fix for large tensors, not a limit on the claim.)

## Finding 2 — generate() wall-clock: BLOCKED on deploy, but the impl is loop-bound
Could not land the `generate()` number: the model's custom code imports `mamba_ssm`, whose
CUDA extension repeatedly failed to import on the pod (ABI mismatch:
`selective_scan_cuda.so: undefined symbol …c10::cuda…`; pinned-version rebuilds didn't
resolve it in the time box). This is a deployment issue orthogonal to our method.

Independent of that, the reference MoE forward is **provably loop-bound at batch 1**: it
loops over ALL 128 experts every token (and even runs a dummy forward for inactive experts) —
the code itself says *"CALL FOR CONTRIBUTION … expert weights need to be fused to not have
to do a loop here."* So batch-1 decode is dominated by ~128×23 per-expert launches, running
far below the weight-bandwidth ceiling. On this impl, **neither BF16 nor a compressed variant
is bandwidth-bound**, so weight compression cannot speed up `generate()` here. Realizing the
win needs a **fused-MoE serving stack (e.g. vLLM) AND a production dequant kernel** — the
generate() number is only meaningful on such a stack, not the reference loop.

## Honest bottom line for rung 2
- What's solid: **25–30% lossless, whole real model, provably identical outputs** (test-003).
- What's weaker than test-002 implied: the **speed** win is kernel- and GPU-dependent; the
  naive kernel wins on a 4090 but loses on an H100. Not shippable as a general speedup yet.
- What's unproven: an actual `generate()` speedup — needs (a) a production dequant kernel and
  (b) a fused-MoE serving path. The reference model can't demonstrate it (loop-bound), and the
  deploy blocker prevented even the baseline number this run.
- Next if pursued: (1) a Marlin/FLUTE-style kernel to lift dequant throughput above HBM
  bandwidth; (2) measure inside vLLM's fused-MoE path, not the reference loop; (3) an A100/H100
  image with mamba_ssm prebuilt to get the baseline generate() number cleanly.
