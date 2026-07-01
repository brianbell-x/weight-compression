# Test 001 — Fusible Lossless Exponent Codebook

Date: 2026-06-30. Scripts: `tests/artifacts/probe_byte_split.py`, `probe_regroup.py`.
Run with `uv run`. Raw outputs: `tests/artifacts/byte_split_result.json`,
`regroup_result.json`. Real weights: shard 1, layer-1 routed experts (8 up + 8 down,
each 4,988,928 BF16 elements). No GPU used; all measurements are byte/bit-exact on CPU.

## Goal
Decide whether candidate 0001's lossless ~32% exponent-plane reduction can be made
**fixed-width / random-access (fusible, Regime D)** instead of variable-length
(entropy-coded, storage-only, Regime C) — and at what cost in ratio.

## Method
1. **Distribution of the sign+exponent field.** For each tensor, count distinct values
   (support) and top-K cumulative mass of (a) the high byte `sign+exp7` (byte-split) and
   (b) the regrouped `sign(1)|exp(8)` 9-bit field.
2. **Fixed-width bit budget.** For codebook size K, index width = `ceil(log2(K+1))`
   (K codes + 1 ESCAPE). Total bits/weight =
   `idx_width + raw_field + (escapes×field_bits + per-row escape-offset table + codebook)/n`.
   Byte-split raw field = 8 (low byte); regroup raw field = 7 (mantissa only).
3. **Exact lossless round-trip.** Encode one full real tensor, then decode using **only**
   (codebook, fixed-width index, in-order escape stream, per-row escape offsets, raw
   mantissa) — no entropy decode — and SHA-256 the rebuilt bytes against the original.

## Results

### Distribution (mean over 8 experts)
| field | support (distinct) | top-8 mass | top-16 mass | top-32 mass |
|---|---|---|---|---|
| high byte `sign+exp7` (up) | 31.1 (max 45) | 0.945 | 0.999 | ~1.0 |
| high byte `sign+exp7` (down) | 30.8 (max 42) | 0.977 | 0.9996 | ~1.0 |
| regroup `sign|exp8` (up) | 56.9 (max 78) | 0.83 | 0.978 | 0.9998 |
| regroup `sign|exp8` (down) | 55.6 (max 78) | 0.84 | 0.985 | 0.9997 |

The exponent field is hyper-concentrated: a 16–32 entry codebook covers >98% of weights.
Regroup roughly doubles the support (the exp LSB it pulls in adds detail) but in exchange
shrinks the raw field from 8→7 bits.

### Fixed-width operating points (bits/weight, mean up+down)
| scheme | idx | bits/weight | reduction | escape rate |
|---|---|---|---|---|
| **regroup K=15** | 4-bit | **11.300** | **29.4%** | 3.25% |
| regroup K=31 | 5-bit | 12.010 | 24.9% | 0.06% |
| byte-split K=7 | 3-bit | 11.59 | 27.5% | 7.3% |
| byte-split K=15 | 4-bit | 12.015 | 24.9% | 0.12% |
| reference: 0001 rANS/zstd (variable-length) | — | ~10.8 | ~32% | — |
| reference: raw BF16 / INT8 | — | 16 / 8 | 0% / 50% | — |

- **Regroup K=15 is the headline**: 29.4% lossless, fixed-width, only ~3 points behind the
  variable-length 32% — that gap is the *price of random access*.
- **Regroup K=31** trades ratio for an escape rate (0.06%) inside SqueezeLLM's
  proven-fusible 0.05–0.45% sparse range — the cleanest to fuse with today's kernels.
- The byte-split → regroup change is what moved the frontier from 27.5% to 29.4%.

### Exact lossless round-trip — PASS
- `backbone.layers.1.mixer.experts.0.up_proj.weight` (4,988,928 elems), regroup K=15:
  **SHA-256 rebuilt == original, exact**; 300,101 escapes (~6.0% on this single tensor).
- `...experts.0.down_proj.weight`, regroup K=15: **exact**; 158,172 escapes (~3.2%).
- Decode used only the fixed-width index + codebook + in-order escape stream + per-row
  offsets + raw mantissa. **No entropy decoder anywhere** → confirms the representation is
  truly random-access, not a re-skinned variable-length code.

## Verdict
- The user's question is answered affirmatively and concretely: **the 32% lossless content
  can be carried in a form the matmul reads directly.** The blocker in 0001 was the
  *variable-length* code (no fixed offset), not the information itself. A fixed-width
  exponent codebook + sparse escape restores random access, costs ~3 ratio points
  (32%→~29%), and is **exactly lossless** (verified).
- This is a **Regime-D** method by the project's litmus test: narrow bytes in from VRAM,
  BF16 rebuilt only in registers, never re-materialized in VRAM — so it touches
  **resident VRAM and per-token bandwidth (~−29% on the expert bulk)**, not just storage.
- **Honest limit:** proven here = losslessness + bit budget + addressability. NOT proven
  here = the fused kernel's wall-clock speed at batch 1 (needs a GPU). Fusibility rests on
  shipping prior art (FLUTE/LUT-GEMM fused LUT-dequant matmul with no VRAM materialization;
  SqueezeLLM/SpQR fused sparse escape), which this scheme is strictly simpler than.

## Next action
Build and benchmark a Triton/CUDA fused kernel: fixed-width index → ≤32-entry LUT → bit-OR
raw mantissa → BF16 in-register → matmul, plus a sparse-escape correction pass. Measure
tokens/s and HBM bytes/token vs BF16 cuBLAS at batch 1 on one expert layer. That single
benchmark closes the only remaining unproven link (fusibility-in-practice). Until then the
runtime win is established in representation (bit budget + addressability + losslessness)
but not yet in measured latency.
