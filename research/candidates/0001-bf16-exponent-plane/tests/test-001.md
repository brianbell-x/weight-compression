# Test 001 — BF16 Exponent-Plane Codec

Date: 2026-06-29. Scripts: `tests/artifacts/codec.py`, `test_synthetic.py`, `test_real.py`.
Run with `uv run`. Raw outputs: `tests/artifacts/synthetic_result.json`,
`real_result.json`, `real_entropy.csv`.

## What was tested
1. De-interleave BF16 -> low byte (exp-lsb + 7 mantissa) / high byte (sign + 7 exp),
   sign-fold high -> mag7 + sign, then re-interleave; exact bit-for-bit check by SHA256.
2. Order-0 Shannon entropy of low / high / mag7 / sign planes.
3. Cross-expert KL of the high-byte histogram (up vs down separately) -> shared-table validity.
4. A real static rANS (order-0, 14-bit, shared table) encode/decode of a real expert's
   folded high plane, with full original-tensor-bytes hash round-trip.
5. Headline bytes: method (rANS-floor) vs zstd-max on raw interleaved AND on de-interleaved planes.

## Synthetic (mechanism proof) — PASS
- 38 BF16 tensors in `models/synthetic/nemotron_tiny`: de-interleave + sign-fold +
  bit-pack sign + re-interleave == original, **all 38 exact** (SHA256).
- Static rANS self-test: exact round-trip; realized 1.9196 bits/sym vs 1.9194 entropy
  (0.0002 overhead). Coder is correct and hits the order-0 floor.

## True weights — shard 1, layer-1 experts (16 up + 16 down sampled)
Each up/down tensor = 4,988,928 BF16 elements (~9.98 MB). Layer 1 holds 128 experts.

### Plane entropy (bits/symbol, mean over 16 experts)
| plane | up | down |
|---|---|---|
| low byte (exp-lsb+mantissa) | 7.955 | 7.954 |  -> effectively incompressible
| high byte (sign+exp7) | 2.920 | 2.731 |
| mag7 (folded, sign removed) | 1.920 | 1.731 |
| sign bit | 1.000 | 1.000 |

The high byte is hyper-concentrated (~2.7-2.9 bits of 8); the low byte is ~uniform.
**Sign-fold gives zero order-0 benefit**: H(high) = H(mag7) + H(sign) exactly
(2.920 = 1.920 + 1.000). Sign is perfectly balanced and independent of magnitude, so
peeling it out only relocates one incompressible bit — it does not lower total bits.
(`fold_total_bytes` == `nofold_total_bytes` in `real_result.json`.)

### Shared-table validity (cross-expert KL of high-byte hist, bits)
| kind | mean vs expert0 | max vs expert0 | max vs mean-hist |
|---|---|---|---|
| up | 0.0273 | 0.0381 | 0.0917 |
| down | 0.0245 | 0.0354 | 0.0913 |

KL is tiny (<0.1 bit worst case). One static table per projection is justified;
header cost amortizes to ~nothing across 128 experts. **Validated.**

### Real static-rANS exact round-trip (one real expert, 4,988,928 symbols)
- mag7 round-trip exact: **true**; full original tensor bytes exact (SHA256): **true**.
- Realized 2.0164 bits/sym vs 1.9929 entropy (table shared from 4 experts). Near floor.
- Encode 1.0 s / decode 0.8 s (pure-Python loop).

### Headline (16 up + 16 down, 159.6 MB raw each block)
| | up ratio | down ratio | reduction |
|---|---|---|---|
| **Method** (de-interleave + rANS high, raw low) | **0.6825** | **0.6707** | ~32% |
| zstd-max raw interleaved (lvl 19/22) | 0.7826 / 0.7828 | 0.7684 | ~22% |
| zstd-max on de-interleaved planes | 0.6908 / 0.6907 | 0.6723 | ~32% |

## Verdict
- Core claim CONFIRMED on true weights: the codec is **exactly lossless** and shrinks
  the routed-expert BF16 to **~0.67-0.68x (~32% reduction, ~1.3 GB on the 4.13 GB MoE block)**,
  beating off-the-shelf zstd-max-on-the-file by ~10 percentage points (~0.41 GB net).
- Two brief hypotheses did NOT hold:
  1. **Sign-fold is a dead lever** at order-0 (no bit savings; sign is i.i.d. 1.0 bit).
  2. **The custom static rANS does not beat zstd applied to the same de-interleaved planes**
     (0.6825 vs 0.6908 up; 0.6707 vs 0.6723 down — a tie). The entire win is the
     **de-interleave**, which exposes the low-entropy exponent plane; a generic coder then
     captures essentially all of it. The bespoke entropy stage adds ~0 over `zstd(planes)`.
- The real, defensible result: **de-interleaving BF16 before compression is worth ~10% over
  naive zstd**, is exactly reversible, and the high plane is the only compressible part
  (low plane ~7.95 bits is a hard floor). To beat ~0.67x further you must model the high
  plane with order-1+/context (cross-element exponent correlation) — order-0 is exhausted.

## Next Action
Test whether a context model on the high (exponent) plane beats the order-0 floor: measure
order-1 conditional entropy H(high[i] | high[i-1]) and column-wise (per output-channel)
exponent structure on a few real experts. If conditional entropy drops materially below
2.7 bits, prototype an order-1 rANS / per-column predictor on the high plane only; if it
does not, freeze this candidate at "de-interleave + zstd planes" (~32%, ~1.3 GB) and move on.
