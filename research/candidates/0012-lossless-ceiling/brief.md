# Candidate 0012 — Lossless ceiling & the exponent-context lever

**Status:** in progress (adversarial whole-model verification running: workflow
`research/tools/lossless_exhaustion.workflow.js` → this dir's `RESULTS.md`).

**Constraint:** pure lossless, bit-exact reconstruction only. No lossy, no quantization,
no combination with lossy.

## The finding (new this session)

Beyond 0009's order-0 exponent codebook (~30% lossless), the sign+exponent field has
**exploitable 2-D spatial structure** — a genuinely-lossless lever that lifts the whole-model
ceiling to **~34%**.

| lever | measurement | effect |
|---|---|---|
| within-tensor 2-D exponent context (left+up neighbor) | expert_up 2.87→2.64 b, expert_down 2.66→2.50 b | −0.17…0.23 b on exponent |
| cross-tensor shared column profile | per-column exp 99.65% correlated across 32 experts | −0.20 b (structure is within-column, so small) |
| **best lossless bits/weight** | ideal sign 1.0 + context-exp ~2.5 + mantissa ~7.0 | **~10.5 b/w = ~34%** |

Uniform across roles: experts 33.9–34.4%, attention 33.8%, embeddings 33.6%. That ~34%
equals the model's global order-0 value-entropy floor (10.50 b/w) — context modeling reaches
the entropy floor; the floor is 10.5 b because **sign (1.0 b) + mantissa (~7.0 b) are random**.

## The ceiling / the 90% verdict

**Lossless maxes at ~34%.** 90% lossless is information-theoretically impossible: 8 of every
16 bits per weight are random (sign 1.000 b; mantissa order-0 7.96/8, lzma/bz2 ~7.0, byte-delta
no help, zero dead mantissa bits in any of 6174 tensors — 0002/0010). Random bits cannot be
losslessly compressed (pigeonhole). Only the ~2.7-bit exponent is compressible, and it is now
squeezed to its context floor (~2.5 b).

## Caveat (runtime vs storage)

The 2-D-context and cross-tensor exponent codes are **variable-length** (storage-only, not
random-access). 0009's fixed-width exponent codebook (~29%) remains the runtime-real (fusible)
lossless form. The +4-pt improvement here is a **storage** result.

## Working codec + strongest-compressor confirmation (lossless_codec.py)

A real end-to-end codec (encode→bytes→decode, asserts `np.array_equal` on raw u16) achieves
**32.9–33.7% bit-exact** on real tensors (plane split: sign 1b + exp brotli-11 2.60–2.73b +
mant 7b). Round-trip exact on every tensor tested.
- **Strongest-tool mantissa attack** (brotli-q11 / lzma-9-extreme): expert_up mantissa →
  **6.84 b** (a 0.16b sliver of structure in some early-layer experts), expert_down 6.97,
  embeddings **7.00** (pure noise). The mantissa wall holds against the best compressors that
  exist; the residual sliver is tensor-dependent, ≤0.16b, variable-length, immaterial.
- **Specialized exponent models beat general compressors**: 2-D-context/separable get the
  exponent to ~2.5–2.6b vs brotli-11's 2.73b (they exploit 2-D spatial structure a 1-D stream
  coder misses). Peeling the exponent fully needs the specialized predictor, not off-the-shelf.
- Storage max with all structure peeled ≈ 1.0 (sign) + ~2.5 (exp) + ~6.9 (mant) ≈ 10.4b ≈ **~35%**.

## Mantissa hunt — the last detectable pattern, chased to 0.078% (mantissa_hunt.py)

Applied the AGENTS.md diagnostic ("any detectable pattern is compression on the table") to the
mantissa across 11 tensors spanning roles/layers, with brotli-11 (strongest tool):
- **Model-wide mantissa = 6.9875 b (numel-weighted) → 0.078% of the model.** Immaterial.
- The 6.84b sliver is a LOCALIZED anomaly: only layer-1 expert-0 (up 6.84 / down 6.97). Every
  other tensor (expert-1, shared, attn, embeddings, layer-2) is **6.99–7.00 b**.
- H(mant|exp) saves only 0.04–0.18 b (tensor-dependent); magnitude-split doesn't help
  (small-|w| mantissa 6.83–7.00 ≈ large). Not a systematic lever — it is the exp=4 near-zero
  cluster (survey 0010), a few early tensors, ~0.08% of mass.
- **Verdict: the mantissa is noise, confirmed at sub-tensor + magnitude granularity by the
  strongest compressor. The diagnostic terminates — what remains looks random.**

## FINAL lossless frontier (complete)
- Storage: **~34–35%** (sign 1.0 + specialized-exponent ~2.5 + mantissa ~6.9). Variable-length.
- Runtime/fusible: **~30%** (separable predictor, verified whole-model, bit-exact).
- Working codec: 32.9–33.7% bit-exact round-trip (`lossless_codec.py`), at/above SOTA (DFloat11 ~30%).
- 90% lossless is information-theoretically impossible (8 random bits/weight). Every field peeled.

## Artifacts (under tests/artifacts/ in this dir)
- `lossless_battery.py` / `_result.json` — exponent order-0 vs 2-D context vs real compressors, per role.
- `lossless_crosstensor.py` / `_result.json` — cross-expert column-profile correlation + conditional entropy.
- `lossless_ceiling.py` / `_result.json` — plane entropies + lzma on mantissa (the random wall).
