# Test 002 — Generalized "dead precision" across the WHOLE model (try-harder)

## Why revisit
Test-001 rejected 0002 on **economics**, not mechanism: the F32 truncation is
exact and lossless, but the F32 family looked KB-scale in shard 1. That rejection
was an *extrapolation from one shard*, and it only tested the narrow F32 case
(low-16-mantissa-bits constant-zero). Try-harder: (a) measure the F32 economics on
the **full model**, not one shard, and (b) test the mechanism at its **general
form** — a *constant bit position across all elements of a tensor* carries zero
information and can be dropped losslessly with no coding (the F32 low-16-zero is
just the special case). Run it over **all 13 shards, BF16 included**, where 93% of
the mass lives.

Scripts: `artifacts/scan_constant_bits.py` (per-tensor AND/OR reduce -> constant
mask, all shards), `artifacts/analyze_constbits.py` (which bits, which categories).
Artifacts: `artifacts/full_model_constbits/constant_bits.json`.

## Result 1 — F32, full model: rejection confirmed (measured, not extrapolated)
Whole-model F32 family = **23,552 bytes** (23 Mamba layers x {A_log[64], D[64]}
clean + 23 MoE `e_score_correction_bias[128]` dirty). Constant-bit free =
**11,648 B (~49.5%)**. So the *entire* addressable F32 win across the 63 GB model
is ~11.6 KB. Immaterial standalone — the shard-1 verdict holds at model scale.

## Result 2 — the mechanism generalizes to 9.72 GB, but it is all EXPONENT
Scanning every BF16 tensor for provably-constant bit positions:

| category | bytes | constant-bit free | note |
|---|---|---|---|
| routed+shared experts | 59.67 GB | **9.238 GB (15.48%)** | the bulk |
| mamba in/out_proj | 1.78 GB | 0.251 GB (14.1%) | |
| embed / lm_head | 1.41 GB | 0.176 GB (12.5%) | |
| other small | 0.30 GB | 0.054 GB (18.3%) | |
| **BF16 total** | **63.16 GB** | **9.72 GB (15.4%)** | |

So the "dead precision" mechanism is **~400,000x larger on BF16 than on the F32
family** it was scoped to (9.72 GB vs 23 KB). But *where* the constant bits sit is
decisive:

- **Sign is essentially never constant** (16 KB of 63 GB) — magnitudes are signed.
- **Mantissa has ZERO constant bits anywhere in the model** (0 of 5934 expert
  tensors, 0 of every other tensor). The 7 mantissa bits are fully live.
- **Every constant bit is a top-of-exponent bit.** Concrete masks:
  `0x7000` (top 3 exponent bits, bits 14-13-12) on 4187 routed-expert tensors;
  `0x6000` (2 bits) on 389; `0x4000` (top 1 bit) on the 1350 big shared-expert
  matrices. Contiguous, exponent-only. Bytes weighted by free-bits/elem: 43.2 GB
  of tensors have exactly 3 constant exp bits, 6.0 GB have 2, 14.0 GB have 1.

## Verdict — real but STRICTLY DOMINATED by 0009
The constant bits are exactly the low-information exponent bits that
[[0009-fusible-exponent-codebook]] already targets. Constant-bit dropping is a
weaker realization of the same structure:
- Its fusible ceiling is drop-the-constant-exponent-prefix: routed experts
  8->5 exp bits = **13 bits/elem (~19%)**; big shared experts 8->7 = **15 bits
  (~6%)**; model-wide **15.4%**.
- 0009's per-group exponent codebook already gets **25-29%** fixed-width,
  random-access, exact, with a confirmed fused GPU kernel.
So dead-precision adds **nothing** over 0009: same field (exponent), less of it.
And the one field 0009 leaves verbatim — the mantissa — is proven here to have
**no dead bits at all**, so dead-precision cannot top it up either.

The idea is not just "too small" (the F32 framing); generalized, it is large but
**subsumed and dominated** by an existing confirmed candidate. That closes the
dead-precision direction properly.

## Salvage / hand-off
- Keep the original `(word & 0xFFFF)==0` F32 clean-detect as a free exact-50%
  sub-rule for the tiny control-tensor family in any general codec (unchanged from
  test-001).
- New, sharper project fact for the ledger: **the BF16 mantissa is the hard
  lossless frontier.** No provably-constant bits exist in any mantissa; all
  realized lossless gains (0001, 0009) come from sign+exponent. Any further
  lossless progress must attack mantissa *statistics* (higher-order structure),
  not dead bits — dead-bit removal is now exhausted and mapped.

## Next action
None for 0002 — direction closed (dominated by 0009, mantissa has no dead bits).
The live lossless frontier is the mantissa's statistics, which belongs to a new
probe, not this candidate.
