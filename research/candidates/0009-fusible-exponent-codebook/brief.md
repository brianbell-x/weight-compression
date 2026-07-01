# Candidate: Fusible Lossless Exponent Codebook (make the 32% a runtime win)

## SCOPE (decided 2026-07-01)
**Deliverable = the lossless weights-compression result:** fixed-width exponent codebook,
**25–30% exact-lossless, whole real 30B model, provably identical outputs** (test-001/003).
This is on-axis for the project and it's DONE/solid.

**Parked (out of current scope):** the runtime/decode-speed exploration (fused GPU kernels,
generate() throughput, serving integration). Fully documented in test-002/004/005 and worth
knowing — verdict: fused decode **beats bf16 on GPUs ≤~2.3 TB/s (4090, A100), ties on H100**;
a real H100 speedup would need a hand-CUDA Marlin-class kernel (thin 25% margin, uncertain).
We are NOT pursuing that now — the project stays focused on weights compression, not
runtime-kernel / serving work. Re-open only if scope deliberately shifts to runtime.

## The question this answers
Candidate 0001 gives an exact-lossless ~32% reduction of the BF16 experts, but it is
**storage-only** (Regime C): its high-byte plane is **variable-length** entropy-coded
(rANS / zstd), so the model must inflate it back to full-width BF16 **in VRAM** before
the matmul runs. The user's question: *if it is the same information, why must we unpack
it in VRAM — can the math read the compressed form directly?*

**Answer (this candidate): yes, with one change.** The reason 0001 must be inflated is
not that "the math can't use compressed weights" — it is that **variable-length codes
have no fixed bit-offset per weight**, and a matmul needs random access (thousands of
GPU lanes each fetch weight (i,j) at a known address, in parallel). Replace the
variable-length code with a **fixed-width codebook index + a sparse escape side-channel**
and every weight regains a known bit-offset → random access → it fuses into the matmul
exactly like INT4/INT8 dequant, the full BF16 reconstructed only transiently in registers
and **never written back to VRAM**. That converts the storage-only win into a
**resident-VRAM + per-token-bandwidth** win (Regime D), and stays **exactly lossless**.

## Claim (measured, this candidate)
Storing each BF16 expert weight as `[fixed-width index into a small per-tensor
sign+exponent codebook] + [raw mantissa]`, with rare sign/exp values handled by an
in-order escape stream, is **exactly lossless** (SHA-256 bit-identical round-trip) and
**fixed-width / random-access by construction**, at **~11.3 bits/weight (~29% reduction)**
for the headline operating point — i.e. it gives up only ~3 points vs 0001's 32% to
become fusible.

## Why it might work / why it does
- **The high (sign+exponent) field is hyper-concentrated** (0001: high-byte entropy
  ~2.9 bits; here, sign+exp8 support is only **~31–56 distinct values** out of 512, and
  the top ~16 cover **>98%** of weights). A few codebook entries capture almost all
  weights; the rest escape.
- **The mantissa is the incompressible floor** (0001: low byte ~7.95 bits of 8). Lossless
  ⇒ the full mantissa must be moved verbatim → ~7–8 bits/weight is a hard floor. So the
  *only* lossless lever is the exponent field, and the *only* question for fusibility is
  whether a fixed-width code of that field stays near its entropy.
- **Fixed-width is the whole point.** A 4–5-bit index has a known stride, so weight (i,j)
  lives at a computable address — the property a matmul needs and a variable-length stream
  destroys. Decode per weight = one tiny table lookup (≤32 entries) + bit-OR with the raw
  mantissa → BF16 in-register. This is strictly simpler than the LUT dequant that shipping
  fused kernels already do (LUT-GEMM, FLUTE), and the escape stream is the same
  dense-and-sparse pattern SqueezeLLM/SpQR fuse.

## The tweak that mattered (don't quit at the first layout)
The naive byte split (codebook the *high byte*, store the *low byte* raw) wastes a bit:
the low byte carries the exponent's LSB mixed with mantissa, so the raw field is 8 bits.
**Regrouping bit-wise** — codebook the full `sign(1)+exponent(8)` field, store only the
`mantissa(7)` raw — saves ~1 bit/weight (the raw field drops 8→7) at the cost of a
slightly larger codebook. Net: byte-split tops out ~27.5%; regroup reaches **~29.4%**.

## Measured operating points (8 up + 8 down layer-1 experts, shard 1)
| scheme | bits/weight | reduction | escape rate | random-access / fusible | lossless |
|---|---|---|---|---|---|
| raw BF16 | 16.0 | 0% | — | n/a | exact |
| **0001 rANS/zstd high-plane** | ~10.8 | **32%** | — | **NO (variable-length)** | exact |
| regroup codebook **K=15** (4-bit idx) | **11.30** | **29.4%** | 3.25% | **YES** | exact ✓ |
| regroup codebook **K=31** (5-bit idx) | 12.01 | 24.9% | **0.06%** | **YES** | exact ✓ |
| byte-split codebook K=7 (3-bit idx) | 11.59 | 27.5% | 7.3% | YES | exact ✓ |

- **K=15 is the headline** (best ratio at a minority escape).
- **K=31 is the conservative choice**: its 0.06% escape sits inside SqueezeLLM's proven
  fusible sparse range (0.05–0.45%), so it is the cleanest to fuse today; K=15's ~3%
  escape is denser than that tested range (more warp divergence / a denser-sparse kernel).
- Exact round-trip verified by SHA-256 on a full real 4,988,928-element tensor for the
  regroup K=15 variant (both up and down). Decode used **only** (codebook, fixed index,
  in-order escape stream, per-row escape offsets, raw mantissa) — no entropy decode.

## Cost-axis honesty (the litmus test)
- **Does the compressed form ever expand to full width in VRAM before use?** No — the
  index + mantissa are read narrow from VRAM; BF16 is rebuilt only in registers via a
  table lookup, exactly as FLUTE's fused kernel does "with no intermediate materialization
  of the dequantized weight tensor." → **Regime D.**
- **Resident VRAM:** −~29% on the expert bulk (experts = 93% of the model).
- **Per-token bandwidth:** −~29% on the active-expert weights read each decode step (the
  dominant decode traffic). Real, by construction.
- **Storage/load:** −~29% (slightly below 0001's 32% — the price of random access).
- **Compute:** a tiny table lookup per weight; hidden under decode's idle compute
  (decode is memory-bandwidth bound).

## What is proven — now including real GPU (test-002, RTX 4090)
**CPU (test-001):** exact losslessness; fixed-width bit budget (~11.3–12 b/w); addressability.
**GPU (test-002):** (1) exact reconstruction on-device (`lossless_on_gpu: true`); (2) a
Triton kernel computes the matvec **directly on the narrow form**, BF16 rebuilt only in
registers, never written to VRAM (Regime D), numerically exact (rel ~1e-7); (3) in the
bandwidth-bound regime the fused kernel runs at **0.756× the time of BF16** (both baselines
pinned at the 4090's ~1 TB/s memory ceiling) — a **24% decode speedup matching the 25%
byte reduction.** The per-token bandwidth win is measured, not just argued.
**Remaining engineering (not a limit on the claim):** fold the escape correction into the
main kernel (one launch) so the tiny single-matvec case also wins; move 12→11.3 b/w
(7-bit-mantissa regroup) for ~29%. Prior art for both: FLUTE/LUT-GEMM, SqueezeLLM/SpQR.

## Ceiling (honest bound)
Lossless ⇒ the random mantissa moves verbatim (~7 b/w floor) + an exponent index (~3–5 b).
So the **lossless-fusible ceiling is ~29–31%**; we are essentially at it. Going below
requires discarding mantissa bits = lossy quantization, which is out of scope
(lossless-only). So this is the best *runtime* win obtainable **without any quality
change at all** — the lossless runtime option.

## Tensor group
Primary: `backbone.layers.*.mixer.experts.*.{up,down}_proj.weight` (BF16, 128 experts/
layer; 93% of params). The mechanism (concentrated exponent, random mantissa) is generic
to BF16, so attention/Mamba/embedding BF16 tensors are likely to extend — **unmeasured**.

## Status
**Storage/VRAM win SOLID (whole real model, lossless). Speed win is kernel/GPU-dependent —
NOT yet general.**
- test-001 (CPU): lossless + bit-budget + addressability on true weights.
- test-003 (real 30B, all shards): **ALL 6,174 BF16 tensors bit-exact lossless**; whole
  model **−24.95% (byte-split) / −30.03% (regroup)** — 58.8→44.1/41.2 GiB. ~100% BF16 so it's
  the whole model. Bit-exactness ⇒ logits provably identical (KL=0 exactly). **This is the
  solid, shippable result: 25–30% smaller, provably unchanged outputs.**
- test-002 (RTX 4090): fused kernel exact on-device + **0.756× BF16 time = 24% faster** in
  the bandwidth-bound regime — BUT see the correction:
- **test-004/005 (H100, deep kernel work): the decode SPEEDUP does not hold on H100 — it
  TIES bf16.** 10 kernel variants on a clean saturating SXM (bf16=3.15 TB/s=94% peak):
  best fused = 2.34 TB/s (ratio 1.01), interleaved+arithmetic-decode = 2.15 (ratio 1.09,
  bit-exact). The fused unpack has a **~2.3 TB/s ceiling**, so it beats bf16 only on GPUs
  **below ~2.3 TB/s** (4090 ~1, A100 ~2) and ties/loses above (H100 3.35). NOT "4090-specific"
  in raw power — the fused kernel runs ~2× faster on H100 than 4090 — but the *ratio* flips
  because bf16 is near-peak on H100 and the 25% byte margin can't absorb the unpack overhead.
  Diagnosis nailed (gather −26%, strided-x −25%; both fixed via arithmetic decode + interleave)
  but the plateau held. Silver lining: a *tie* = 25% lossless VRAM at ZERO speed cost, strictly
  better than DFloat11 (lossless but 1.4–2× slower). Open lever: a Marlin-class CUDA kernel
  (untested, thin margin).

Honest state: **the lossless 25–30% (storage + resident VRAM) is real and whole-model, with
provably identical outputs.** Decode SPEEDUP: real on GPUs ≤~2.3 TB/s; on H100 it's a break-even
tie (free VRAM, no speed tax). See `tests/test-001..005.md`.

## Sources
Inference cost model / Regime C vs D and the litmus test (see AGENTS.md "The Goal"):
- DFloat11 (Regime-C contrast: variable-length, decodes full-width to VRAM) —
  https://arxiv.org/html/2504.11651v3
Fused fixed-width / LUT dequant matmul with no VRAM materialization (the fusibility basis):
- LUT-GEMM — https://arxiv.org/abs/2206.09557
- FLUTE, "Fast Matrix Multiplications for Lookup Table-Quantized LLMs" (fused dequant+matmul,
  single pass, no intermediate materialization) — https://arxiv.org/abs/2407.10960
- Marlin INT4 kernel — https://github.com/IST-DASLab/marlin
Sparse-outlier ("escape") decomposition, fused dense+sparse kernels:
- SqueezeLLM (Dense-and-Sparse, 0.05–0.45% sparse, one fused call) — https://arxiv.org/abs/2306.07629
- SpQR (sparse-quantized representation) — https://arxiv.org/abs/2306.03078
Prior internal results this builds on:
- candidate 0001 (BF16 exponent-plane: 32% lossless, high-byte entropy ~2.9, mantissa
  ~7.95, the variable-length storage-only floor)
```
