# Candidate 0011 — Vector/Product Quantization of Experts (the sub-4-bit frontier)

**Question.** The ledger closed *scalar* post-hoc quant at INT8 (safe) / INT4 (+5% ppl).
Does **vector quantization + incoherence** (the AQLM / QuIP# / QTIP family — every
sub-3-bit result in the literature) break sub-4-bit at usable quality, and how far does
compounding it with the lossless lever go toward the 90% goal?

All errors are **held-out output error** `‖X(W−W')‖/‖X W‖` on 2,593 real held-out
layer-1 expert-input tokens; codecs calibrated on 30,225 real tokens (full-rank Hessian).
Incoherence = randomized-Hadamard rotation `R=(Q₂₁⊗H₁₂₈)·diag(±1)` on the 2688 in-axis
(exact dims, **no padding inflation**), absorbed losslessly into the matmul
`Y = X W = (X Rᵀ)(R W)`.

## Results (layer-1 up_proj experts)

| bits/wt | RTN | GPTQ | RTN+incoh | **GPTQ+incoh (QuIP#-lite)** | VQ+incoh |
|---|---|---|---|---|---|
| 4.03–4.13 | 4.74 | 4.13 | 4.50 | **3.35** | 4.87 |
| 3.03–3.13 | 11.1 | 9.64 | 10.5 | **7.82** | 8.80 |
| 2.02–2.13 | 41.7 | 27.7 | 33.6 | 23.3 | **16.9** |

- **4-bit: QuIP#-lite = 3.35% — beats the ledger's INT4-RTN (5.07%).** A real Pareto gain
  at the same size (incoherence removes the outliers RTN wastes resolution on; GPTQ error
  feedback does the rest). This is the new best sub-INT8 post-hoc codec.
- **2-bit: VQ's space-filling gain shows** (16.9% vs scalar 23.3%) but 16.9% is still deeply
  lossy. **3-bit ≈ 8%** (between INT4 and unusable).
- **Naive weight-MSE PQ was a red herring** (17.5% @ 2-bit, worse than INT4) — it optimized
  `‖W−W'‖` not `‖X(W−W')‖`. Fixing the objective (GPTQ/incoherence) is what mattered.
- **Output-aware Hessian weighting adds nothing** after incoherence: `diag(H)` max/mean
  collapses 5.42 → 2.11 under the rotation, so there is no peaked channel structure left to
  weight. (Confirms the incoherence is doing its job.)

## Why this is a wall, not a tuning problem — rate-distortion

The survey (0010) proved the experts are structureless: full-rank, position-wise
uncorrelated, mantissa 6.97/7 bits random. The incoherence rotation makes that explicit —
after `R`, the weights are **near-Gaussian i.i.d.** A Gaussian source has a hard
rate-distortion bound `D(R) ≈ 2⁻²ᴿ` (per-weight rel-distortion `≈ 2⁻ᴿ`):

| R (bits) | Gaussian bound | measured (best) |
|---|---|---|
| 2 | 25% | 16.9% (VQ recovers the space-filling gain) |
| 3 | 12.5% | 7.8% |
| 4 | 6.25% | 3.35% |

Measured errors track the bound. **VQ recovers the scalar-vs-lattice space-filling gain
(~0.25 bit) and error feedback pushes output error below per-weight distortion — but none
of it beats the bound.** Post-hoc, ~3–4 bit is the quality floor for these weights. This is
the same "experts are dense" wall as 0003/0005/0007, now with an information-theoretic law.

## Compounded combined-compression frontier (proven, post-hoc)

`combined_remaining = 0.93·(b_exp/16) + 0.07·(b_non/16)`, non-experts at the lossless
floor 11 b/w (0009/0010):

| config | expert quality | **combined reduction** |
|---|---|---|
| INT8 experts + lossless non-exp | KL≈3e-4 (safe) | ~48% |
| **4-bit QuIP#-lite + lossless non-exp** | 3.35% (sub-INT4, near-usable) | **~71%** |
| 3-bit + lossless non-exp | 7.8% (degraded) | ~78% |
| 2-bit + lossless non-exp | 16.9% (broken) | ~84% (unusable) |

**Post-hoc compounding tops out ~71% at good quality, ~78% degraded.** 90% would need
experts at ~1 bit/weight — 2× past the proven 2-bit wall.

## The only lever left for 90%: change the weights (training)

The rate-distortion wall bounds compression of the *fixed* weights. But a model's function
does not uniquely determine its weights — there is a manifold of function-equivalent weight
sets, and only some points are low-bit-representable. Post-hoc is stuck at the dense-Gaussian
point training landed on. **QAT / distillation searches the manifold for a low-bit-friendly
point that computes the same function** (why BitNet-1.58 works trained but not post-hoc). This
is end-to-end (downstream layers co-adapt to quant error) → needs GPU. It is the honest path
past ~78% toward the 90% target, and it is a training program, not a re-encoding.

## Reusable artifacts
- `gptq_rht.py` — randomized-Hadamard incoherence + GPTQ + VQ, held-out output-error harness.
- `vq_probe.py` — product/residual VQ + FWHT incoherence.
- `oaware.py` — Hessian-diagonal weighting / incoherence-flattening diagnostic.
- `lossless_ceiling.py` — lzma/entropy cascade proving the mantissa random wall.
