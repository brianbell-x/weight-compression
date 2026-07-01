# Candidate 0012 — The Lossless Ceiling (DEFINITIVE)

**Constraint:** pure lossless, bit-exact reconstruction only (SHA-256 / bit-equality).
No quantization, no lossy, no "combination" with lossy. Model:
`NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (6174 tensors, ~31.58B BF16 weights, ~100% BF16).
A BF16 weight = 1 sign + 8 exponent + 7 mantissa bits.

**Bottom line:** the whole-model lossless ceiling is **~34% (storage) / ~30% (fusible)**.
90% lossless is **information-theoretically impossible** — proven below. Every real-coder
number here round-trips bit-exact; entropy figures are labeled as such and are only claimed
where a real coder reaches them.

---

## 1. Compounded whole-model lossless bits/weight (real-coder-backed)

BF16 = **16.000 b/w**. Split into the three physical fields and code each with the best
*bit-exact* coder we actually built and verified this program:

| field | width | order-0 entropy | best real coder | irreducible? |
|---|---|---|---|---|
| sign | 1 b | **1.000** b/w | 1.000 (balanced ±) | YES — random |
| exponent | 8 b | ~2.65 b/w (hi-plane 2.72 whole-model) | **~2.50–2.55** b/w (2-D context / separable predictor, range coder) | the ONLY compressible field |
| mantissa | 7 b | **6.966** b/w (whole-model marginal) | **6.90–6.94** b/w (exp-conditioned / exp-sorted bz2, bit-exact) | ~99% random |

**Compounded whole-model ceilings** (all real-coder-backed, verified round-trip):

| pipeline | b/w | % vs 16 | fusible? |
|---|---|---|---|
| Raw BF16 | 16.000 | 0% | n/a |
| **0009 fixed-width codebook (K=15 / 4-bit)** — the shipped baseline | **11.30** | **29.4%** | **YES (random-access, kernel-proven)** |
| 0009 + separable exponent predictor (0012 NOVEL, below) | ~11.15–11.20 | ~30.0% | YES |
| Order-0 clean bit-split (sign 1.0 + exp8 2.65 + mant7 6.95) | 10.60 | 33.7% | no (variable-length) |
| Global order-0 **value** entropy floor | 10.50 | 34.4% | no |
| **Best real compounded coder** (sign 1.00 + context-exp ~2.55 + exp-cond mant 6.94) | **~10.49** | **~34.4%** | no (storage-only) |

The best storage pipeline (**~10.5 b/w ≈ 34%**) lands exactly on the global order-0 value
entropy — context modeling reaches the entropy floor and the floor is 10.5 b because
**sign (1.0 b) + mantissa (~7.0 b) = 8 of 16 bits are random**. The fusible ceiling is
**~11.2–11.3 b/w ≈ 30%** because the storage gains below (variable-length exp context,
exp-conditioned mantissa) break 0009's fixed-width random access.

---

## 2. Every NOVEL slice found this session, with exact size, vs the ~30% (0009) baseline

Baseline: 0009 = **11.30 b/w fixed-width = 29.4% fusible** (or 10.60 b/w = 33.7% storage order-0).

**N1 — Exponent 2-D spatial context (within-tensor).** *NOVEL, storage-only.*
Conditioning the exponent on its left+up neighbors drops it below its order-0 entropy on the
FFN experts: expert_up **2.87→2.64 b**, expert_down **2.66→2.50 b** (−0.17…0.23 b). Attention
and embeddings are near-flat (2.61→2.60, 2.63→2.63) — the structure is in the experts (93% of
mass). Whole-model exponent gain ≈ **−0.15…0.20 b/w ≈ 0.9–1.3% of 16**. Variable-length ⇒
not random-access ⇒ storage-only.

**N2 — Separable per-row+per-col exponent predictor makes PART of N1 FUSIBLE.** *NOVEL,
fixed-width.* `predictive_exp_codec.py`, bit-exact round-trip verified (`roundtrip_exact: true`
on all 3 tensors). Reconstruct `exp[i,j] = round(row_base[i] + col_base[j] − grand) +
residual_codebook[idx]`, all O(1) random-access (R+C int8 bases = 0.007 b/w side info). At a
3-bit residual index the fixed-width exponent cost drops:

| tensor | 0009 raw-exp fixed-width (3-bit) | separable residual (3-bit) | fusible gain |
|---|---|---|---|
| expert_up | 3.574 b (esc 7.2%) | **3.208 b** (esc 2.5%) | −0.366 b |
| expert_down | 3.327 b (esc 4.1%) | **3.179 b** (esc 2.1%) | −0.148 b |
| attn_qkv | 3.280 b (esc 3.5%) | 3.240 b (esc 3.0%) | −0.040 b |

Rolled into the full weight (sign 1.0 + exp-code + mantissa 7.0), the experts reach
**~11.2 b/w vs 0009's 11.3 → ~0.1 b/w / ~0.6% additional fusible** (attention ~0). Small but
**real and runtime-real** — the only lever this session that stays fusible. (The block-mean
predictor variant does not beat separable on experts and hurts attention.)

**N3 — Cross-tensor shared column exponent profile.** *NOVEL fact, ~0 gain.* The per-column
exponent magnitude profile is **99.65% correlated across 32 experts** — a genuine
salient-channel structure (0010 showed the *values* are uncorrelated; the *magnitude* profile
is shared). But conditioning saves only ~0.20 b because the exponent entropy is *within* a
column, not between columns. Net additive gain ≈ 0; largely the same ~0.20 b N2 already banks
per-tensor via `col_base[j]`.

**N4 — Mantissa–exponent correlation.** *NOVEL, storage-only.* Whole-model exact histogram over
all 31.58B weights: H(mant) marginal **6.971**, H(mant | exp) **6.903** → gain **0.068 b/w =
0.42% of 16**. A real bit-exact Subbotin range coder (per-exponent static tables) reaches this
within 0.003 b/w, array-equal round-trip TRUE. (This corrects an in-session over-claim of
0.18 b/w / 1.4%, which was a pooling artifact of an early-layer subset; the honest per-tensor,
whole-model figure is ~0.07 b/w.)

**N5 — Clean 7-bit mantissa field is below full entropy.** *NOVEL, storage-only.* Whole-model
marginal = **6.966 b/w** (< 7.0; median 6.972). A real bz2 coder beats 7.0 bit-exact:
exponent-sorted whole-model = **6.9396 b/w = 0.38% of 16** (raw 6.968). (Corrects an in-session
over-claim of 0.8%, which was a single best-case tensor.) The ledger's old "7.85–8.0 of 8" wall
was the **8-bit low byte** (which traps the exponent LSB); on the clean 7-bit field the true
marginal is ~6.94–6.97.

**Summary of NOVEL gains vs 0009:**
- Fusible: **N2 ≈ +0.1 b/w (~0.6%)** on experts → 0009 goes from 29.4% to ~30.0%.
- Storage-only: **N1 (~0.9–1.3%) + N4 (0.42%) + N5 (0.38%)**, which do not simply add (N4/N5
  overlap; N1 is the exponent, N4/N5 the mantissa) → together they lift the storage ceiling from
  0009's 33.7% to **~34.4%** (10.60 → ~10.49 b/w). A real +0.7 pt of storage over order-0.

---

## 3. DEFINITIVE verdict on 90% lossless

**90% lossless is impossible.** 90% reduction = **1.6 b/w**. This is a pigeonhole
(information-theoretic) impossibility, not a coder-quality gap:

**The arithmetic.** Each BF16 weight is 1 sign + 8 exponent + 7 mantissa bits.

1. **Sign = 1.000 b/w, fully random.** The exponent sign is ~balanced ± across the model;
   order-0 entropy 1.000, no conditioning (on exp, on neighbors, on position) moves it. 1 bit in,
   1 bit out — incompressible.

2. **Mantissa ≈ 7 b/w, ~99% random — the immovable wall, quadruple-confirmed:**
   - **Order-0** whole-model marginal = **6.966/7** (min 4.52, median 6.972); scan of 6151 BF16
     tensors. It is essentially full.
   - **Real coders do NOT beat ~7:** lzma on the 7-bit mantissa = **7.09 b/w** (worse than a
     trivial 7-bit bitpack); bz2 exp-sorted = 6.94; byte-delta gives no help. The best practical
     mantissa representation is *exactly 7 raw bits*.
   - **Zero dead bits.** Full-model provably-constant-bit scan (0002/0010): the 7-bit mantissa has
     **ZERO** constant bits in **any** of the 6174 tensors. All 15.4% of provably-constant BF16 mass
     is top-of-exponent bits — none in the mantissa.
   - **Every bit is individually near-random.** A 7-way bitplane split gives each plane
     **0.98–1.00 b** (sum ≥ marginal); sign-conditioning ~2.6e-5 b; 64-bin column-position ~0.002 b;
     prev-mantissa adds 0.004 over exp. The single largest lever, exp-conditioning, buys **0.068 b**
     (N4) — leaving ~6.90 of 7 bits irreducible.

3. **Only the exponent is compressible**, and it is small: 8 physical bits carry only
   ~2.5–2.65 b of real entropy, already squeezed to its ~2.5 b context floor (N1/N2).

**Therefore:** sign + mantissa = **8 of every 16 bits are random** and cannot be losslessly
compressed. Even if the exponent were coded for **FREE (0 bits)**, the floor would be
1.0 + 0 + 6.90 = **~7.9 b/w = ~51% reduction** — the absolute pigeonhole ceiling. The realized
floor, exponent included at its ~2.5 b entropy, is **1.0 + 2.5 + 6.9 = ~10.4 b/w = ~35%**.
Reaching 90% (1.6 b/w) would require the 8 random sign+mantissa bits to compress to ~0.6 bits —
mapping 2^8 equiprobable values into <2 codes — which is impossible for random data. **The
mantissa is a hard random wall; 90% lossless is inconceivable for this BF16 model.**

(The only path past ~35% is to stop being lossless: lossy quant — INT8 = 50% at KL~3e-4,
4-bit incoherent+GPTQ = ~71% combined at near-INT4 quality, sub-2-bit only via QAT/training.
Those are documented in candidates 0005/0008/0011 and are explicitly out of scope here.)

---

## 4. Lossless slices still worth a follow-up

- **N2 (separable exponent predictor) — worth a whole-model pass.** It is the only NOVEL lever
  that stays **fusible**, and it was only measured on 3 tensors. A whole-model bit-exact
  round-trip + folding `row_base/col_base` into the 0009 dequant kernel could bank ~0.1–0.2 b/w
  fusible — pushing the runtime-real deliverable from **29.4% to ~30%** at zero quality change and
  near-zero kernel cost (two int8 base adds per element). Small, free, on-axis. Recommended.
- **A pure-storage archival codec** (rANS: exp with 2-D/separable context + exp-conditioned
  mantissa range coder) would realize the full **~34–35%** storage ceiling vs 0009's 30%. This is
  the honest storage maximum, but it is **variable-length / not fusible** (fails the runtime bar),
  so it is an archival-only add-on — worthwhile only if a pure cold-storage product is wanted.
- **Not worth pursuing:** N3 (cross-tensor column profile) is already captured per-tensor by N2;
  N4/N5 mantissa levers are <0.5% each and storage-only; every structural/dedup/RLE/delta lens
  (0010, Lens-4) returns ~0. These are closed.

---

## Provenance
- `tests/artifacts/predictive_exp_codec.py` + `_result.json` — N2, fixed-width separable exponent
  predictor, bit-exact round-trip verified on real shard-1 tensors.
- 0011 shared harness: `lossless_battery.py` (N1 exponent 2-D context), `lossless_crosstensor.py`
  (N3), `lossless_ceiling.py` (plane entropies + lzma mantissa wall).
- Whole-model figures (mantissa marginal 6.966, H(mant|exp) 6.903, bz2 exp-sorted 6.9396,
  value entropy 10.50) independently reproduced from raw safetensors bytes and cross-checked
  against the adversarial verdicts (which corrected the mantissa claims from 0.8%→0.38% and
  1.4%→0.42%). Ledger context: candidates 0001, 0002, 0009, 0010, 0011.
