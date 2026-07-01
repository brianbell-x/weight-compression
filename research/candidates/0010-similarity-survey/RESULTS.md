# Candidate 0010 — Exhaustive Whole-Model Similarity Survey

**Model:** NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 — 6243 tensors (6174 BF16 + 69 F32),
31,577,934,400 BF16 weights, 63.156 GB. Fingerprints are **exact, not sampled**
(global BF16 histogram over all ~31.6B weights; per-tensor SHA-256 + hi_hist[256]).

**Purpose.** Five mechanical "similarity lenses" swept the entire model to catch any
lossless structure the earlier targeted probes (0001–0009) may have stepped over.
Every candidate finding was then re-derived from raw safetensors bytes by an
adversarial verifier. **Bottom line: the sweep found no new exploitable lossless
structure. Its value is that it converts many "measured on layer 1" negatives into
"confirmed model-wide."**

---

## 1. What the sweep found — three buckets

### Bucket A — Exact duplicates (byte-identical)
- **Zero.** `exact_dup_groups = []` over all 6243 tensors. No two tensors are
  byte-identical by SHA-256 — not two experts, not two layers, not any
  shared/routed pair. Independently re-verified by fingerprinting every tensor
  (nbytes + blake2b of head/tail/strided body, escalate collisions to SHA-256):
  **6243 distinct fingerprints, 0 collisions.**
- The classic "tied embeddings" case is **not** tied: `backbone.embeddings.weight`
  (SHA `2e57f847…`) vs `lm_head.weight` (`eb9d13ae…`), both [131072,2688] /
  704,643,072 bytes — different bytes, true full-vector cosine **+0.031** (≈orthogonal).
- **Whole-tensor / whole-model byte dedup reclaims 0 bytes.**

### Bucket B — Value / structural similarity
- **Near-duplicate expert/projection pairs are artifacts of the block-mean signature.**
  Re-decoding both tensors exactly and taking the true full-vector cosine collapses
  every flagged pair: down_proj L51 e54/e89 sig 0.836 → **|cos| 0.0057**; up_proj
  0.782 → **0.046**; shared_experts up 0.798 → 0.045, down 0.774 → 0.005; mamba
  in_proj 0.314 → 0.003; out_proj 0.418 → 0.0004; gate 0.380 → 0.172. Four random
  cross-expert down_proj pairs: |cos| 0.0001–0.0018. **Experts/projections are
  independent model-wide** (extends ledger 0003/0007 to every projection family).
- **High RAW cosines on 1D norm/bias/D vectors are DC-offset**, not shared fluctuation:
  `e_score_correction_bias[128]` raw 1.000 → centered **0.031**; `dt_bias[64]` 0.862
  → 0.049; `mixer.norm.weight[4096]` 0.757 → 0.006; `mixer.D[64]` 0.710 → −0.253.
- **One genuine cross-layer value cluster exists but is byte-immaterial:** the input
  `norm.weight[2688]` family (52 tensors, RMSNorm gains) has real shared per-channel
  structure — centered cosine max **0.996** (L38/L40), 10 pairs > 0.99, rank-1 explains
  ~50% of centered energy, graded by layer distance. *Adversarial correction:* Lens A
  called this "one isolated pair"; it is actually a smooth cross-layer cluster. But the
  **entire family is 279,552 bytes = 0.00047% of the model** and already inside 0009's
  lossless coverage → exploitable gain **0**.
- All seven 1D norm/bias/D families combined = **777,088 bytes = 0.00123%** of the model.
- 69 F32 tensors (A_log, D, gate bias) = **23,552 bytes total**; not BF16-truncatable
  (differ in sub-BF16 mantissa bits), so ledger-0002's free-F32 trick does not apply.
  Max plausible lossless saving ≈ 1e-5% → 0.

### Bucket C — Byte-layout / distribution similarity
- **The high/low byte-plane split (0001) holds model-wide with no byte-significant
  violator.** Byte-weighted over all 6174 BF16 tensors: high byte (sign+exp)
  **H = 2.7214 b**, low byte (mantissa) **H = 7.9622 b** (max 8). Every bulk role sits
  at the same split (experts, embeddings, lm_head, attn q/k/v/o, mamba in/out_proj all
  hi 2.71–2.76 / lo 7.96–7.97).
- **~99.997% of BF16 mass shares ONE high-byte distribution.** Single-linkage clustering
  (symKL < 0.5 b) yields one cluster of all matmul-weight roles; each role centroid is
  symKL 0.001–0.30 / cosine 0.964–1.0 from the routed-up centroid. Cross-expert
  sign+exp near-identity reproduces 0001 exactly: layer-1 pooled 128 experts centroid
  **KL = 0.02592 b** (ledger said ~0.027); all 2944 routed_up centroid-KL mean 0.0178
  (cosine 0.9969), routed_down 0.0129 (0.9990).
- **The exponent plane is the only compressible part and it is already fully exploited
  by 0009.** Whole-model high-byte H = 2.7395 b (129/256 values, 9 cover 98%); true
  8-bit exponent 2.5937 b; sign bit exactly **1.0000 b** (p1=0.5004).
- **The mantissa is the hard lossless floor and is essentially incompressible:** true
  7-bit mantissa global H = **6.9710 / 7 b = 99.58%** of maximum, near-uniform (needs
  125/128 values for 98% coverage). Low-plane order-0 saving = 0.0378 b/byte = **0.236%
  of the model.**
- **Whole-model order-0 value entropy = 10.4969 b/w (34.39% below 16 b).** A single
  global codebook is competitive with per-tensor but never beats it: global regroup
  (4-bit index over sign+exp8 + 7 raw mantissa + escape 2.294%) = **11.2065 b/w = 29.96%**,
  vs 0009's whole-model per-tensor regroup ~11.195 b/w / 30.03% — global worse by ~0.011
  b/w. Pooling penalty: full value +0.0211 b/w, exponent plane +0.0181 b/w.

---

## 2. NOVEL vs KNOWN

### Genuinely-new EXPLOITABLE structure that survived adversarial verify (gain > 0)
**None that is fusible.** Exactly one item has a nonzero storage number:

- **Early-layer expert mantissa heterogeneity** (Lens C, verdict CONFIRMED). A thin
  subpopulation of layer-3/6 experts has genuinely low mantissa entropy from a large
  near-zero cluster (biased-exp field = 4, value ≈ 9.4e-38): L3 e101 down_proj verifies
  to low-byte H = 5.8620, true-7-bit-mantissa H = **5.2265**, exp field=4 = 38.33% of
  elements (frac_zero = 0.0). Independent recompute of total mantissa order-0 headroom
  beyond 0009's raw-7-bit mantissa = **132.8 MB = 0.21% of the model** (of which the
  early-layer cluster is only ~27 MB / 0.043%; the rest is a uniform ~0.028 b/elem bulk
  deficit). **Harvesting requires variable-length mantissa entropy-coding → breaks
  fixed-width fusibility (Regime C, storage-only). Fusible gain toward the project bar = 0.**

### Model-wide CONFIRMATIONS of prior results (the real deliverable of this sweep)
Each of these upgrades a targeted layer-1/sampled negative into a whole-model fact:
- **0001** (32% exponent-plane ceiling; shared table slightly worse): confirmed and
  quantified model-wide — global-vs-per-tensor pooling costs +0.0211 b/w on value,
  +0.0181 b/w on exponent; per-role sharing max-saves only MI(role;hi) = **0.00288 b/w**
  vs 0009's per-tensor MI(tensor;hi) = **0.0181 b/w** (6× more) → per-role coding is
  strictly worse.
- **0002** (mantissa fully live; F32 control tensors immaterial): confirmed — mantissa
  6.9710/7 b, 69 F32 tensors total 23,552 bytes.
- **0003** (experts share a distribution, not aligned values): confirmed — true
  cross-expert cosine ~0.003–0.05 across every projection family.
- **0007** (experts full-rank, no shared basis): consistent — no near-duplicate survives.
- **0009** (fixed-width regroup ≈ 30% is the fusible floor): confirmed — global codebook
  ties it (never beats it); nothing to amortize by sharing (per-tensor codebooks total
  ~0.18 MB in a 44 GB file).
- Structural census: 52 blocks = 23 Mamba2 + 6 attention + 23 MoE; 128 routed experts +
  1 shared + gate per MoE layer; routed experts = 29.375B params = **58.75 GB = 93.02%**
  of the model — reconciles the ledger's runtime-track sizing exactly.

**Honest summary: this survey is ~99% confirmation. That is its point — the "no
exploitable cross-tensor / near-duplicate / shared-codebook structure" verdict is now
established over all 31.6B weights, not inferred from a layer-1 sample.**

---

## 3. Single strongest lead worth a real follow-up

**There is no fusible lead.** The only nonzero number in the whole sweep is the
**early-layer-expert near-zero mantissa cluster (~0.21% storage, non-fusible)**, and it
is the only thing not already exhausted at order-0.

If any follow-up is run, it should be a *storage-track* probe, clearly outside the
runtime bar:

- **Concrete next experiment:** take the ~200 lowest-mantissa-entropy expert tensors
  (all L3/L6, identified by `lo_entropy` in `report.records.jsonl`), and measure a
  **two-regime split within a single tensor**: segregate the exp-field=4 near-zero
  cluster (≈38% of L3 e101) into its own run and entropy-code *only its* mantissa, while
  the bulk keeps 0009's fixed-width raw-7-bit path. Question to answer: does a
  *per-tensor, block-structured* code keep enough of the elements fixed-width to remain
  fusible while clawing back the ~27 MB? Predicted answer from this sweep: **no** — the
  near-zero elements are spatially scattered (position-wise uncorrelated, 0003), so any
  gather that isolates them is itself a variable-length index. **Expected outcome:
  confirms the 0009 ceiling; do not expect a fusible win.** This is worth at most a
  half-day to formally close the "mantissa has no fusible slack anywhere" question.

---

## 4. Completeness critic — similarity cuts this survey did NOT measure

Ranked backlog (higher = more likely to still hide lossless structure). Everything above
was order-0 / block-mean / whole-vector; the following are structurally distinct cuts:

1. **Higher-order / context entropy of the value stream (order-1+).** The whole sweep is
   order-0. If the mantissa or the (exp,mantissa) stream has *conditional* redundancy —
   e.g. mantissa byte correlated with its row/column neighbor, or exponent runs — order-0
   would miss it entirely. This is the single most likely remaining lossless lever because
   it directly attacks the mantissa floor (the only ~7 b/w that order-0 cannot touch).
   *Probe:* measure H(byte | previous byte) and H(mantissa | exponent) per tensor on a few
   experts vs the order-0 7.96 b; any gap > ~0.1 b is real. Note it is likely non-fusible.
2. **Intra-tensor row/column repeats & structure.** Ledger 0004 tested row-dedup (immaterial),
   but this sweep did not scan for repeated *rows/columns*, constant rows, low-cardinality
   columns, or periodic tiling inside the large [1856,2688]/[131072,2688] matrices. Cheap to
   check via per-row SHA-256 within a tensor.
3. **Cross-tensor delta after alignment (permutation / sign / scale).** Cosines were computed
   on the natural ordering only. Experts could be similar *up to a row permutation or
   per-channel sign/scale flip* (a known MoE symmetry). This survey would score such a pair as
   orthogonal. *Probe:* Hungarian/greedy row-matching between two same-shape experts, then
   residual entropy. Higher risk/effort, but the one cut that could revive "shared basis."
4. **Transpose / rotation similarity.** up_proj[1856,2688] vs down_proj[2688,1856] are
   transposes in shape; never compared as A vs Bᵀ. Low prior, but untested.
5. **Bit-plane / mantissa-column correlation across the tensor.** Treat each of the 7 mantissa
   bits as a plane and test spatial autocorrelation (e.g. XOR-with-neighbor entropy). Would
   catch structured mantissas that look random marginally.
6. **Sub-block value quantization residual (VQ / k-means codebook over full BF16 values).**
   The value codebook here was per-symbol; a *vector* codebook over 2- or 4-weight groups was
   not measured. Order-0 says the joint value space is too spread
   (top-256 = 33.6%), so prior is low, but grouped VQ is a different statistic.
7. **Cross-shard / file-level byte redundancy** (zstd-dictionary across shard boundaries) —
   almost certainly nil given 0 exact dups and random mantissa, listed for completeness.

Items 1–3 are the ones that could still, in principle, beat 0009 losslessly; items 1 and 2
are the cheapest to falsify next.

---

## Proposed findings-ledger entry (do not apply — text only)

> **## 0010 — Exhaustive whole-model similarity survey: no new exploitable structure (Confirmed)**
> A mechanical 5-lens sweep over ALL 6243 tensors / 31.58B weights (exact fingerprints:
> global BF16 histogram, per-tensor SHA-256 + hi_hist[256]), with every candidate finding
> re-derived from raw safetensors bytes by an adversarial verifier. Result: **zero new
> fusible lossless structure; broad model-wide confirmation of the existing negatives.**
> (1) **No exact duplicates anywhere** — 6243 distinct SHA-256; whole-tensor dedup = 0 bytes;
> embeddings vs lm_head are NOT tied (cos +0.031). (2) **No value/structural near-dups** — the
> block-mean "similar experts" collapse to true |cos| 0.003–0.05 across every projection family
> (extends 0003/0007 model-wide); 1D norm/bias high cosines are DC-offset (centered ≤0.05); the
> one real cross-layer cluster (input norm.weight[2688], centered cos up to 0.996) is 0.28 MB =
> 0.0005% and already inside 0009. (3) **Byte-plane split holds model-wide** — hi 2.7214 b / lo
> 7.9622 b, ~99.997% of mass in ONE high-byte distribution; per-role sharing max-saves 0.00288 b/w
> vs 0009's per-tensor 0.0181 b/w (6× worse), so a shared codebook is strictly worse (confirms
> 0001). Whole-model order-0 value entropy 10.4969 b/w (34.4%); global regroup 11.2065 b/w /
> 29.96% ties 0009's per-tensor 30.03% but never beats it. Mantissa is the hard floor: 6.9710/7 b
> = 99.58%, sign exactly 1.0000 b. **Only nonzero number:** early-layer (L3/L6) experts have a
> large exp-field=4 near-zero cluster (L3 e101: 38% of elements, mantissa H 5.23 b) giving ~132.8
> MB = 0.21% total mantissa headroom — but harvesting needs variable-length coding (non-fusible,
> storage-only), so fusible gain = 0. **Net: 0009 remains the floor; the "no cross-tensor /
> near-dup / shared-table structure" verdict is now whole-model, not layer-1-sampled.** Untested
> cuts left as backlog: order-1+ context entropy of the value stream, intra-tensor row/column
> repeats, and cross-tensor delta after permutation/sign/scale alignment.

---

**File:** `C:/dev/compression/research/candidates/0010-similarity-survey/RESULTS.md`
