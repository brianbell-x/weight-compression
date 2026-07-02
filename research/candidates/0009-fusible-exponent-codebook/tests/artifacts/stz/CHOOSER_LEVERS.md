# .stz chooser-scale levers — pre-probe verdicts

## 2026-07-02 — V1 / V2 / V3 pre-probe on real layers 1 / 13 / 27

Scope: direction E "New leads" (see `research/notes/NEXT_DIRECTIONS.md`). Three
per-tensor OPTIONS for the .stz min-envelope chooser, each priced as an
adoption-aware envelope gain vs the **realized** stz baseline
(`stz_tensor_stats.jsonl`, parity gate 768/768 tensors stats-exact). Real MoE
expert tensors, 3 layers x 256 tensors (128 experts x {up,down}_proj,
1,277,165,568 params/layer). All costs exact bit accounting; 9/9 writer-verified
roundtrips (serialized bits == priced bits AND SHA-256 bit-exact, on adopted
configs). Wall: ~6.5 min.

Levers:

- **V1** — per-row(-group) second-level escape width k for up_proj-class
  tensors (row-overdispersed escape mask, Fano ~2.3); side stream of per-group
  k choices; priced through the (k−b) conversion rule. Groups swept {1, 8, 64}.
- **V2** — per-column(-group) BASE re-centering before the top-K codebook; one
  int8 base per column group charged. Groups swept {1, 16, 64}.
- **V3** — per-tensor fractional-m repricing of the index plane: non-power-of-2
  K packed at fractional log2(K) bits via grouped radix coding in fixed-size,
  row-restarting groups (random access survives at group granularity); group
  padding charged exactly. The 2026-07 vetting salvage, now priced against the
  realized baseline for the first time.

### Full table

```
chooser-levers pre-probe -- REAL layers 1/13/27, 256 MoE expert tensors each (128 experts x {up,down}_proj, 1,277,165,568 params/layer), baseline = realized stz (parity-gated exact, 768/768 stats-exact)

layer  baseline   v1_g1     v1_g8     v1_g64    v2_g1     v2_g16   v2_g64   v3        | lever-best env gain b/w (adopted/256)          joint env
       bpw        env-gain  env-gain  env-gain  env-gain  env-gain env-gain env-gain  | V1              V2              V3              gain b/w
1      11.0310    +0.003827 +0.002343 +0.000044 +0.099116 +0.0     +0.0     +0.048558 | +0.003828 (110) +0.099116 (220) +0.048558 (256) +0.109354
13     10.8931    +0.006053 +0.002335 +0.000800 +0.004622 +0.0     +0.0     +0.051080 | +0.006053 (209) +0.004622 ( 35) +0.051080 (256) +0.052770
27     10.8822    +0.005880 +0.001250 +0.000213 +0.000307 +0.0     +0.0     +0.053965 | +0.005880 (202) +0.000307 (  8) +0.053965 (256) +0.053965

decay-weighted model-wide projection (23 MoE layers, expert/total BF16 numel = 0.9302, bar >= +0.01 b/w):
  V1  +0.005253 b/w  below bar
  V2  +0.013578 b/w  PASSES
  V3  +0.048877 b/w  PASSES
  joint (per-tensor min over single-lever options) +0.056839 b/w  PASSES

roundtrips (writer-verified cost models): 9/9 (3 levers x 3 layers) serialized bits == priced bits AND SHA-256 bit-exact, all on ADOPTED configs, e.g. V3 L27 32.up_proj {M:11,k:4}, V2 L1 45.down_proj {g:1}, V1 L27 32.up_proj {group:1,b:3,cap:4}
```

### Verdicts (pre-registered bar: model-wide adoption-aware envelope >= +0.01 b/w)

| lever | model-wide projection | bar | verdict |
|-------|----------------------:|:---:|---------|
| **V3** fractional-m index-plane repricing | **+0.048877 b/w** (~+0.31 pt) | PASS | **ADOPT.** Adopted on 768/768 tensors; typical winner M=11–12, k=3–4. Gain does **not** decay with depth — it *rises* (+0.0486 L1 → +0.0511 L13 → +0.0540 L27), so the early-decay weighting (built for column effects) makes this projection conservative. Strongest lever by far; at L27 the joint envelope equals V3 alone. |
| **V2** per-column BASE re-centering, g=1 only | **+0.013578 b/w** (~+0.085 pt) | PASS | **ADOPT, early layers only, with the colkey caveat below.** Fires only at g=1 (g16/g64 adopted 0/256 at every layer — column bases do not survive any grouping: per-column *shift*, not shape, structure that averages out immediately). Essentially all mass is layer 1 (+0.0991) plus the interpolated ramp over unmeasured layers 2–12; the bar pass is interpolation-sensitive at this margin. |
| **V1** per-row second-level escape k | **+0.005253 b/w** | FAIL | **DROP.** Best variant is group=1 rows (g8/g64 collapse toward zero), and even that fails the bar *before* charging the O(1) per-group code-offset table (~0.01 b/w order at group=1) required for strictly-O(1) row random access. Both facts point the same way. The row-overdispersion signal is real but converts at only (k−b) bits per converted escape — exactly the 0014 pricing rule. |

### Interaction with the queued colkey chooser variant (+0.065 pt, layers 1–10)

**V2 and colkey are the SAME structure — overlapping, not additive; V2-g1
likely supersedes colkey.**

- Layer-1 envelopes match to three decimals across independent implementations:
  V2 +0.0991 vs colkey +0.0978 (0014 cross-layer sweep). Same decay shape
  (to ~0 by L13: V2 +0.0046 vs colkey +0.0029 at L13; both ~0 at L27). This is
  the known early-layer column-*shift* structure; a per-column int8 base
  captures what column-keyed codebooks capture, with a far cheaper side stream
  and no per-column codebook machinery.
- Therefore **do not book +0.065 pt (colkey) and +0.085 pt (V2) separately** —
  the model-wide upside for the column family is ~+0.011–0.014 b/w experts
  (~+0.065–0.085 pt whole-model) TOTAL, whichever mechanism realizes it.
  Cheapest resolution: put V2-g1 into the chooser; colkey then only enters
  where it beats V2 per tensor (expected: rarely/never), so the chooser
  envelope over both settles the supersession question empirically at zero
  risk.
- Interpolation caveat cuts toward colkey's number: colkey's *measured* 7-layer
  decay is convex (L3 +0.0697, L6 +0.0274, L8 +0.0191, L10 +0.0208) — below
  V2's linear L1→L13 interpolation — so V2's +0.0136 is probably nearer
  ~+0.010–0.011 b/w under the measured decay shape. Still at/near the bar, and
  still the same single pot of early-layer column mass.
- **V3 is mechanically orthogonal to colkey/V2** (index-plane *packing* vs
  symbol *re-centering*) and its mass lives where colkey/V2 are zero (layers
  >= 13 carry most of the model), so V3 + column-family gains are approximately
  additive model-wide. Exact composition is unpriced (the joint here is a
  per-tensor min over single-lever options, not lever composition): on the few
  early layers where both fire, re-centering changes the index alphabet and
  hence V3's best (M, k), so the composed early-layer gain needs one repricing
  pass — expected second-order.
- **V1 has no interaction** (escape side stream, different plane) and is
  dropped regardless.

### Recommendation

1. **Fold V3 (fractional-m grouped-radix index plane) into the .stz chooser.**
   +0.049 b/w model-wide, conservative, adopted everywhere, depth-stable.
2. **Fold V2-g1 (per-column int8 base) in as the realization of the queued
   colkey early-layer win** — one column-family option, not two stacked wins
   (~+0.07–0.09 pt whole-model total). Optionally keep colkey as a competing
   chooser variant to confirm supersession per tensor.
3. **Drop V1.** Below bar even before its uncharged side-table cost.
4. Joint projection with per-tensor best single lever: **+0.0568 b/w
   (~+0.36 pt whole-model)** on top of the realized 31.89% .stz — realizable in
   the chooser upgrade; composed V2∘V3 on layers 1–3 is possible upside beyond
   the min-only joint, needs one repricing pass at implementation time.

Artifacts: `../chooser_levers/levers_layer{1,13,27}.jsonl`, `../chooser_levers/summary.json`.
Baselines: `stz_tensor_stats.jsonl` (parity-gated exact). Cross-refs:
`research/candidates/0014-column-keyed-codebooks/RESULTS.md` (colkey cross-layer
certificate), `research/notes/NEXT_DIRECTIONS.md` ("New leads", direction E).
