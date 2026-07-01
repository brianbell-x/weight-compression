# RUNTIME.md — Runtime-Real Lossless Verdict (candidate 0012)

**Scope:** whole-model, numel-weighted, of the fusible predictive codec
(`sign(1) + exp_residual_code(index+escape) + mantissa(7)`, with a separable
`row_base[i] + col_base[j] - grand` exponent predictor) versus the non-predictive
0009-style fusible baseline. All numbers are the *fusible* bpw the kernel actually
reads: `index_bits + esc·8 + (R+C)·8/n` for the exponent field, plus 1 sign + 7
mantissa. This is a real runtime read width, not a storage-only entropy figure.

Tool: `research/candidates/0012-lossless-ceiling/tests/artifacts/predictive_wholemodel.py`

## Whole-model number (numel-weighted over all 13 shards)

- Total weights: **31,577,790,464**
- **Predictive fusible: 11.1976 bpw → 30.01% smaller than BF16 (69.99% of 16-bit)**
- Baseline fusible: 11.2072 bpw → 29.95% smaller
- Predictive vs baseline: **−0.0096 bpw / +0.06 pt**
- Predictive vs 0009's stated floor (11.30 bpw / 29.37%): **+0.64 pt**
- **Round-trip: all 13 shards bit-exact (SHA-256 / bit-equality), roundtrip_ok = true for every shard.**

The headline "+0.67 pt" is a **shard-1-only** result. Model-wide the predictor
essentially ties the separable baseline: it wins decisively only on shard 1
(embeddings / early layers, 11.30 → 11.19), while on shards 4–13 the baseline is
already at ~11.19–11.20 and the predictor neither helps nor hurts beyond noise.
The honest whole-model gain of the predictor *over the fusible baseline* is
**+0.06 pt**; its gain *over the 0009 floor* is **+0.64 pt**.

## Per-shard table

| Shard | Weights | Predictive bpw | Predictive % | Baseline bpw | Baseline % | Δpt (pred−base) | Round-trip |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 00001 | 2,495,565,824 | 11.1934 | 30.04 | 11.3000 | 29.37 | +0.67 | ok |
| 00002 | 2,496,253,952 | 11.1880 | 30.07 | 11.2415 | 29.74 | +0.33 | ok |
| 00003 | 2,496,253,952 | 11.1928 | 30.04 | 11.2034 | 29.98 | +0.06 | ok |
| 00004 | 2,497,802,240 | 11.1965 | 30.02 | 11.1957 | 30.03 | −0.01 | ok |
| 00005 | 2,490,228,736 | 11.1983 | 30.01 | 11.1930 | 30.04 | −0.03 | ok |
| 00006 | 2,499,663,872 | 11.1967 | 30.02 | 11.1905 | 30.06 | −0.04 | ok |
| 00007 | 2,496,253,952 | 11.2003 | 30.00 | 11.1940 | 30.04 | −0.04 | ok |
| 00008 | 2,496,253,952 | 11.2007 | 30.00 | 11.1942 | 30.04 | −0.04 | ok |
| 00009 | 2,497,802,240 | 11.2018 | 29.99 | 11.1950 | 30.03 | −0.04 | ok |
| 00010 | 2,496,253,952 | 11.2014 | 29.99 | 11.1948 | 30.03 | −0.04 | ok |
| 00011 | 2,497,802,240 | 11.1979 | 30.01 | 11.1917 | 30.05 | −0.04 | ok |
| 00012 | 2,497,802,240 | 11.1991 | 30.01 | 11.1953 | 30.03 | −0.02 | ok |
| 00013 | 1,619,853,312 | 11.2046 | 29.97 | 11.2040 | 29.98 | −0.01 | ok |
| **Whole model** | **31,577,790,464** | **11.1976** | **30.01** | **11.2072** | **29.95** | **+0.06** | **all ok** |

## Kernel fusibility verdict

The predictor is **fusible and bandwidth-positive**; its overhead does not eat the
win. Reconstruction is register-only — the compressed form is never re-inflated to
full-width BF16 in VRAM before the matmul consumes it.

- **Extra per-weight work is negligible.** The delta over 0009 is `+ row_base[i] +
  col_base[j]` (2 int adds) plus 2 loads that hit shared/registers, not HBM. The
  unpack sits at ~4–6 int-ops/byte, well under the ~10 ops/byte compute-bound
  threshold on A100/H100 — the kernel stays memory-bound with margin; the adds hide
  under memory latency and cost ~0 wall-clock in the bandwidth-bound limit.
- **Side vectors are effectively free.** `row_base`/`col_base` are `(R+C)` int8
  vectors (1.8–10.3 KB each for Nemotron shapes). A `Tr×Tc` tile touches only `Tr`
  row-bases + `Tc` col-bases, staged once in shared memory and reused across the
  whole tile → O(R+C) reads against O(R·C) work → asymptotically zero added
  bandwidth. Same shape of side data as a per-group scale that Marlin/FLUTE already
  carry.
- **Stays bandwidth-bound at the narrower read.** Sign+mantissa (8 bits) are
  byte-identical in both codecs; the entire difference lives in the exponent field.
  The gain surfaces either as a lower escape rate (fewer irregular out-of-band
  fetches) or, where residuals tighten enough, a 1-bit `index_bits` drop that
  literally shrinks the fixed-width main plane.
- **No new addressing hazards.** Fixed-width index plane → identical coalesced,
  stride-regular, random-access loads as 0009. Base lookups are uniform/broadcast
  (tile-contiguous ranges), the residual LUT is ≤32 entries in registers/constant,
  and escape handling is the same discipline as 0009 (and hit *less* often).

**Realizability call:** a Marlin/FLUTE-class fused kernel can carry this codec —
the predictor is just a per-group scale split into a separable row-part + col-part.
The limit is the *size* of the win, not the predictor cost. Whole-model the
predictor only buys +0.06 pt over the fusible baseline (concentrated almost
entirely in shard 1), which lands inside the ~80–95%-of-peak-BW efficiency noise of
a real kernel **unless** the residual tightening also knocks `index_bits` down a
notch. Pursue it specifically where it converts the statistical gain into a hard
fixed-width read reduction (the embedding/early-layer shards); elsewhere it is a
tie, not a regression, and it remains exactly lossless.

## Ceilings, side by side

| Regime | Size vs BF16 | Fusible? | Notes |
|---|---:|:--:|---|
| 0009 baseline (established) | 29.4% | yes | fixed-width, random-access floor |
| Fusible baseline (this codec, no predictor) | 29.95% | yes | whole-model, numel-weighted |
| **Predictive fusible (this codec)** | **30.01%** | **yes** | **whole-model, bit-exact, runtime-real** |
| Storage-lossless ceiling | ~34% | **no** | variable-length entropy coding; breaks fixed-width random access |

**One-line honest summary:** of the ~4.6 pt of storage headroom above 0009's floor
(29.4% → ~34%), only ~0.6 pt (→ 30.0%) is runtime-real, because that is all the
gain a fixed-width, random-access, register-decodable form can hold; the remaining
~4 pt lives only in variable-length codes, which are not fusible and re-inflate to
full width, so they stay a storage-only tool.

## Proposed findings-ledger addendum (not yet applied)

> **Runtime-real lossless ceiling ≈ 30.0% (candidate 0012).** The separable
> exponent predictor (`exp_residual = exp − round(row_base[i] + col_base[j] −
> grand)`, O(R+C) int8 side vectors) was measured whole-model, numel-weighted over
> all 13 Nemotron-30B shards: **11.1976 fusible bpw = 30.01% smaller than BF16**,
> with every shard round-tripping bit-exact (SHA-256). It is genuinely fusible —
> reconstruction is register-only (2 int adds + tile-cached base loads), the kernel
> stays bandwidth-bound (~4–6 ops/byte vs ~10 threshold), and it preserves 0009's
> coalesced fixed-width random access. But the predictor's whole-model edge over the
> non-predictive fusible baseline (11.2072 bpw / 29.95%) is only **+0.06 pt**,
> concentrated almost entirely in shard 1 (embeddings/early layers, +0.67 pt); vs
> 0009's 29.4% floor the gain is +0.64 pt. This pins the runtime-real lossless
> ceiling at ~30%, roughly 4 pt below the ~34% storage ceiling — the gap is
> variable-length entropy coding that is not fixed-width and therefore not fusible.
> Next lever: chase cases where residual tightening drops `index_bits` by a full bit
> (a hard main-plane read reduction), rather than the statistical escape-rate gain.
