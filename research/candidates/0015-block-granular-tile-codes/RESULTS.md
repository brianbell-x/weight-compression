# Candidate 0015 — Block-granular tile codes: RESULTS (first real probe)

**Date:** 2026-07-01 · **Scope:** canonical layer-27 target set (shard 7, 128 experts ×
{up,down}_proj = 256 tensors, 1,277,165,568 params) · **Accounting stamp:** `5d7b9e9c4613`
· **Wall time:** ~16 min CPU.

**Verdict: PARTIAL — "positive (storage-leaning only)".** Both pre-registered
tile-fusible gates (G1, G2 at W ≤ 128) **fail**, but the pre-registered falsifier
(heavy per-block tails) **did not trigger**, and the identical format beats the realized
stz crown once blocks are large: best overall **10.6977 b/w at W16384_P100** (−0.185 vs
stz 10.8822, floor +0.139), with superblock format (a) at **10.7079 b/w** also under the
floor+0.15 bar. What kills the tile-granular configs is fixed per-block coder overhead
(flush + renorm + budget slack ≈ 0.44 b/w at W=128), not tail mass — a coder-mechanics
problem with quantified headroom, not a structural wall. Direction A is *not* falsified;
this operating point (independent 12-bit-flush rANS blocks, byte-granular single-tier
budgets, order-0 per-tensor tables) is.

All numbers are numel-weighted over the 256-tensor set, with **all** side costs charged
(quantized ANS tables, escape bitmaps, rank directories, offset indexes, headers, byte
padding) and block sizes are **measured emitted bits** of the exact named coder
(round-trip verified: 10,493 blocks + 512 lanes, emitted bits == accounted bits, SHA-256
exact BF16 reconstruction). Per-tensor stz parity: max |Δbpw| = 0.000000 vs
`stz_tensor_stats.jsonl`; reference reproduced at 10.882182 b/w.

## The four decisive questions

### 1. Does block-granular coding beat realized stz (10.8822 b/w), all taxes charged?

**At tile granularity (fixed-stride, W ≤ 128 — the pre-registered fusible claim): NO.**
Best cell W128_P99 = **11.0349 b/w**, i.e. **−0.153 vs stz** (worse). Projected
whole-model at the pre-registered W128_P97: 11.0403 vs stz's realized 10.8975 — worse.
G1 fails at every W ≤ 128 cell.

**At storage-leaning block sizes: YES.** G1 passes for every best cell from W512 up:
W512_P99 = 10.8256 (−0.057), W1024_P99 = 10.7820 (−0.100), W4096_P99 = 10.7306
(−0.152), **W16384_P100 = 10.6977 (−0.185)**. Superblock format (a) = 10.7079 (−0.174).
The crossover sits between W=256 (10.9017, still worse) and W=512.

### 2. How close to the per-tensor order-0 floor (10.5583 b/w)? How much of the 0.324 gap is recoverable?

The stz→floor gap on this set is 10.8822 − 10.5583 = **0.3239 b/w**.

- **Storage-leaning:** W16384_P100 lands at **floor + 0.1394** — recovers **57%** of the
  gap (0.1845 b/w). Its residual is almost entirely coder mechanics, not model
  structure: coder excess 0.0449 + pad 0.0865 + tax 0.0003 + 12-bit quantization delta
  0.0076 ≈ 0.139. Format (a) lands at floor + 0.1496 and squeaks under the G2 bar
  (floor + 0.15 = 10.7083) by 0.0004. **The order-0 storage side of this set is
  effectively closed.**
- **Tile-granular (W ≤ 128):** recovers **nothing** — best cell is floor + 0.4766,
  which is 0.153 *above* stz. G2 fails by 0.327 b/w.

### 3. Are the per-block tails light enough that padding is cheap? (the falsifier)

**Yes — the falsifier did NOT trigger.** The per-block ideal code-length tails are thin:

| W | p50 (b/w) | p99 (b/w) | p99/p50 | max ever |
|---|---|---|---|---|
| 32 | 3.541 | 4.139 | 1.17 | 5.100 |
| 64 | 3.549 | 3.958 | 1.12 | 4.634 |
| 128 | 3.553 | 3.840 | 1.08 | 4.368 |
| 256 | 3.555 | 3.763 | 1.06 | 4.212 |
| 512 | 3.556 | 3.713 | 1.04 | 4.117 |
| 1024 | 3.556 | 3.681 | 1.04 | 4.055 |
| 4096 | 3.556 | 3.632 | 1.02 | 3.809 |
| 16384 | 3.557 | 3.594 | 1.01 | 3.642 |

Block composition barely varies (per-tensor sym distributions are homogeneous), escapes
are rare (0.5–1.8% of blocks at P97–P99), and moving the percentile P95→P99 shifts the
result by only ~0.001–0.05 b/w. **What kills W ≤ 128 is instead the fixed per-block
overhead:** measured coder excess over quantized entropy (12-bit flush + bit-renorm
rounding) is 0.399 b/w at W=32, 0.221 at W=64, 0.133 at W=128 (of which the flush alone
is 12/W = 0.375/0.188/0.094), plus 0.24–0.30 b/w of budget slack (percentile spread +
byte-ceil) at W=128. Overhead arithmetic at W128_P99:

```
floor (H(sym)+7)          10.5583
+ 12-bit table quant       0.0076
+ coder excess (flush+rnd) 0.1328
+ pad (slack + byte-ceil)  0.3027
+ side tax (bitmap/dir/…)  0.0086
+ escape interplay        ~0.025
= 11.0349
```

Budget to beat stz at W=128: total overhead above floor+quant must be **< 0.316 b/w**;
measured is **0.469**. Budget for G2: **< 0.142**. The same format sails under stz at
W ≥ 512 purely because these fixed costs amortize away.

**Sweet spots:** fusible bracket W128_P99 (11.0349; P97 is statistically identical at
11.0358); storage bracket W16384_P100 (10.6977 — at 16K blocks the tail is so thin that
paying max-percentile beats paying escapes).

### 4. Runtime story — the honest caveat

Even the "fusible" W ≤ 128 format is a **different kernel contract than 0009's .stz**:
0009 gives O(1) *address and O(1) work* per weight (fixed-width index + escape); 0015-b
gives O(1) address per block but **O(W) sequential rANS decode work** to reach a weight
inside it. That is fine *only* for a kernel that consumes whole contiguous row-segments
(the coalesced matmul pattern) and decodes tile-into-registers — a kernel that has not
been written or benchmarked. 0009's measured 24% decode speedup does **not** transfer
automatically. And the configurations that actually win here (W ≥ 512, format (a)
superblocks) are O(1) only at 512–16K-weight granularity — for the running kernel those
are storage/load-time wins in the spirit of candidate 0001, not fusible wins. No
tile-fusible headline exists in this run.

## Full grid (numel-weighted, all taxes charged, measured coder bits)

```
        format       bpw  save_stz over_floor    esc%   pad b/w   tax b/w  fusible
----------------------------------------------------------------------------------
       W32_P90   11.6938   -0.8116    +1.1354   8.537    0.3025    0.0335      yes
       W32_P95   11.5851   -0.7029    +1.0267   1.146    0.5370    0.0335      yes
       W32_P97   11.5851   -0.7029    +1.0267   1.146    0.5370    0.0335      yes
       W32_P99   11.7476   -0.8654    +1.1893   0.271    0.7380    0.0335      yes
      W32_P100   12.5706   -1.6884    +2.0122   0.000    1.5725    0.0335      yes
       W64_P90   11.4092   -0.5270    +0.8509   7.809    0.2219    0.0169      yes
       W64_P95   11.2298   -0.3476    +0.6714   1.803    0.3393    0.0169      yes
       W64_P97   11.2298   -0.3476    +0.6715   1.794    0.3398    0.0169      yes
       W64_P99   11.2818   -0.3996    +0.7235   0.314    0.4631    0.0169      yes
      W64_P100   11.8973   -1.0151    +1.3389   0.000    1.0930    0.0169      yes
      W128_P90   11.1599   -0.2777    +0.6016   5.392    0.1797    0.0086      yes
      W128_P95   11.0733   -0.1911    +0.5149   2.936    0.2182    0.0086      yes
      W128_P97   11.0358   -0.1536    +0.4774   1.753    0.2409    0.0086      yes
      W128_P99   11.0349   -0.1527    +0.4766   0.507    0.3027    0.0086      yes
     W128_P100   11.5171   -0.6349    +0.9588   0.000    0.8098    0.0086      yes
      W256_P90   11.1551   -0.2729    +0.5968   7.302    0.1174    0.0044       no
      W256_P95   10.9878   -0.1056    +0.4295   3.531    0.1469    0.0044       no
      W256_P97   10.9339   -0.0517    +0.3756   2.121    0.1662    0.0044       no
      W256_P99   10.9017   -0.0195    +0.3433   0.691    0.2078    0.0044       no
     W256_P100   11.3058   -0.4236    +0.7475   0.000    0.6470    0.0044       no
      W512_P90   11.1434   -0.2612    +0.5850   8.139    0.0811    0.0024       no
      W512_P95   10.9395   -0.0573    +0.3812   3.843    0.1040    0.0024       no
      W512_P97   10.8747   +0.0075    +0.3164   2.329    0.1188    0.0024       no
      W512_P99   10.8256   +0.0566    +0.2673   0.773    0.1512    0.0024       no
     W512_P100   11.1825   -0.3003    +0.6242   0.000    0.5479    0.0024       no
     W1024_P90   11.1362   -0.2540    +0.5779   8.604    0.0582    0.0013       no
     W1024_P95   10.9220   -0.0398    +0.3637   4.263    0.0747    0.0013       no
     W1024_P97   10.8429   +0.0393    +0.2846   2.538    0.0870    0.0013       no
     W1024_P99   10.7820   +0.1002    +0.2237   0.851    0.1152    0.0013       no
    W1024_P100   11.1098   -0.2276    +0.5515   0.000    0.4873    0.0013       no
     W4096_P90   11.1418   -0.2596    +0.5834   9.315    0.0311    0.0005       no
     W4096_P95   10.9000   -0.0178    +0.3417   4.610    0.0412    0.0005       no
     W4096_P97   10.8104   +0.0718    +0.2520   2.789    0.0488    0.0005       no
     W4096_P99   10.7306   +0.1516    +0.1722   0.934    0.0679    0.0005       no
    W4096_P100   10.8601   +0.0221    +0.3018   0.000    0.2467    0.0005       no
    W16384_P90   11.1453   -0.2631    +0.5870   9.529    0.0222    0.0003       no
    W16384_P95   10.8924   -0.0102    +0.3340   4.739    0.0269    0.0003       no
    W16384_P97   10.7949   +0.0873    +0.2366   2.859    0.0305    0.0003       no
    W16384_P99   10.7012   +0.1810    +0.1429   0.939    0.0400    0.0003       no
   W16384_P100   10.6977   +0.1845    +0.1394   0.000    0.0865    0.0003       no
     A_4096x32   10.7079   +0.1743    +0.1496   0.000    0.0009    0.0083       no
```

References: realized stz on this set = **10.8822 b/w** (whole-model 10.8975); order-0
per-tensor floor H(sym)+7 = **10.5583 b/w**; 12-bit ANS quantization delta = +0.0076 b/w.
`fusible = yes` marks the pre-registered tile-credible bracket (fixed stride, W ≤ 128)
only; see §4 for why even that carries a kernel-contract caveat.

**Gates (pre-registered, keyed to fixed-stride W ≤ 128):** G1 (beat 10.8822) **FAIL**
(best 11.0349, Δ = −0.153); G2 (≤ floor + 0.15 = 10.7083) **FAIL** (+0.477 over floor).
Storage-leaning brackets pass both (labels honest: they cannot carry the fusible
headline). Round-trip, parity, and reconciliation gates all PASS.

## Untested variations (what this run does NOT falsify)

The measured overhead decomposition names exactly what a v2 must fix. To beat stz at
W=128, combined per-block overhead (flush + renorm + budget slack + tax) must drop from
the measured **0.469** to **< 0.316 b/w**; to hit G2, **< 0.142 b/w**.

1. **Bit-granular fixed stride** (drop byte-ceil on B): fixed *bit* stride is still O(1)
   address math. Saves ~0.03–0.05 b/w. Untested.
2. **Two-tier / mixed budgets:** 2 budget classes per tensor + a 1-bit class flag per
   block (~0.008 b/w) roughly halves the percentile slack (0.30 → ~0.15 at W=128).
   Untested. A per-group two-tier raw fallback (instead of per-block wholesale escape)
   is the same idea applied to the escape side.
3. **Cheaper flush:** the 12-bit-per-block flush is 0.094 b/w at W=128 and is only
   required because every block carries an independent rANS state. Sharing one flush
   across a G-block group (group-stride addressing, decode ≤ G·W symbols), tANS with a
   small state, or implicit-state tricks could cut it to ~0.01–0.03 b/w. Untested.
   Rough stack of 1–3: ≈ 11.03 − 0.05 − 0.15 − 0.07 ≈ **10.76–10.85 b/w at W=128** —
   plausibly beats stz, likely still short of G2.
4. **Per-block context / below-order-0 tables:** this run used order-0 per-tensor tables
   only. 0014's *confirmed* measurement (H(exp|col) = 2.486 vs order-0 holdout 2.496,
   column identity ≈ 85–100% of the 2-D context gain, address-derived so zero side info)
   was not exploited. 0014's certificate bans per-weight fixed-width *keying*; it says
   nothing about column-conditioned tables feeding a block entropy coder. This is the
   only named source of *new* entropy headroom to fund the tile tax down to G2.
5. **W beyond 16K:** nearly pointless — W16384_P100 already sits at floor + 0.139 with
   tax 0.0003; the residual is coder excess and pad, not amortizable index cost.

**What would have to be true for the direction to fire fusibly:** items 1–3 must land
their arithmetic (mechanical, high-confidence, gets past stz by ~0.03–0.12 b/w), and
item 4 must deliver ≥ ~0.1 b/w of real conditional-entropy gain at block granularity to
reach floor+0.15. If a v2 with 1–3 implemented still cannot get under 10.88 at W ≤ 128,
the tile-granular *order-0* point is falsified and Direction A survives only in its
context-modeled form.

## Compounding order

**Changes, modestly — Direction A stays primary but re-scoped.** (i) The falsifier
budget shifted: the enemy at tile granularity is per-block fixed overhead, not tail
mass — the next probe is coder mechanics (variations 1–3) + column-conditioned tables
(variation 4), not more tail mapping. (ii) The order-0 *storage* question for this set
is closed: 10.6977 vs floor 10.5583 with everything charged means further storage gains
must come from below-order-0 structure — which is the same lever the fusible path needs,
so the two bars now point at one experiment. (iii) A cheap cross-layer transfer check
(scope here is layer 27 only) should gate any whole-model claim. The storage-leaning
result (format (a) / W16384_P100, −0.18 b/w vs stz ≈ 1.2% of total size) is real but
storage-only; per project priority it is a noted option, not the next object of study.

## Reproduction

```
# smoke (synthetic snapshot, seconds)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_block_codes.py --synthetic

# real run (resumable; each invocation self-limits to ~7 min and checkpoints)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_block_codes.py

# summary table + gates + summary JSON (auto-runs when complete)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_block_codes.py --summary
```

Artifacts: `tests/artifacts/blockcodes_results.jsonl` (256 rows, per-tensor exact
accounting, stamp `5d7b9e9c4613`), `tests/artifacts/blockcodes_summary.json` (this
table, gates, projections). Baseline reference:
`../0009-fusible-exponent-codebook/tests/artifacts/stz/stz_tensor_stats.jsonl`.

---

# v2 — overhead attack (levers L1–L4): RESULTS

**Date:** 2026-07-01 · **Scope:** same canonical layer-27 set (shard 7, 256 tensors,
1,277,165,568 params) · **Accounting stamp:** `3c5e38d4c9a9` · **Wall time:** ~7 min this
session (full grid, single invocation) + ~2 min prior invocation (L4 gate phase, resumed
by design) · **Tool:** `tools/probe_block_codes_v2.py` (reuses v1's skeptic-verified
loader/accounting infrastructure).

**Verdict (pre-registered rule applied): DIRECTION A FIRES at tile granularity.** The
best fusible config — **W128_T4_P100_L11L31L40 = 10.7004 b/w** (fixed *bit*-stride
W=128 blocks, 4 DP-optimal tier budgets + 3-bit class flag, top budget = max block size,
L3 flush-carries-mantissa-payload, L4 off by gate) — beats realized stz 10.8822 by
**+0.1818 b/w** (v2-G1 PASS) and also clears the floor+0.15 bar 10.7083 (v2-G2 PASS,
margin **0.0079** — real but thin, and selected on the same set it is scored on). v1's
verdict line ("if a competent v2 with all four levers cannot get under 10.88, the
tile-granular order-0 point is falsified") resolves the other way: the tile-granular
order-0 point is **confirmed**, not falsified. Same evidentiary standard as v1: all side
costs charged, measured emitted bits, per-cell reconciliation (sym-plane components must
sum exactly to the charged total, mismatch aborts), round-trip 6,144 blocks × all coder
variants with emitted bits == accounted bits and SHA-256-exact BF16, stz parity max
|Δbpw| = 0.000000 (reference reproduced at 10.882182).

**Fusible contract (unchanged caveat from v1 §4):** every 128-weight block is
independently decodable at bit offset `base + block_index * slot_bits(class)`, class from
a fixed-stride 3-bit flag plane + u32 rank anchor per class per 512 blocks (O(1) address,
≤512-flag popcount); decode work inside a block is still O(W)=128 sequential rANS steps —
a register-tile kernel contract, not 0009's O(1)-per-weight contract. Under L3 the first
two weights of a block finalize only after the full block decode (the mantissa payload is
the decoder's terminal state).

## Overhead decomposition at the best fusible config (vs v1's W128_P99)

```
                              v1 W128_P99   v2 W128_T4_P100_L11L31L40
floor (H(sym)+7)                  10.5583      10.5583
+ 12-bit table quant               0.0076       0.0076
+ flush (12b/block)                0.0938*      0.0938
− mantissa credit (L3)                 —       −0.0938
+ coder excess (renorm)            0.0390*      0.0439
+ pad slack (tiers vs pct+byte)    0.3027       0.0727
+ escapes                         ~0.0250       0.0000   (esc_frac = 0)
+ tier flag plane                      —        0.0156
+ rank anchors + class dirs        0.0086†      0.0021
+ tables / align / header          0.0002       0.0004
= total                           11.0349      10.7004
overhead above floor+quant         0.4690       0.1345
  vs beat-stz budget < 0.316         OVER        UNDER
  vs G2 budget < 0.142               OVER        UNDER
```

\* v1 reported flush+renorm jointly as coder excess 0.1328; † v1's side tax line.
The v2 attack cut per-block overhead **0.469 → 0.1345 b/w** — a 71% reduction, all of it
mechanics, none of it new entropy.

## Which levers carried it (marginals at the best cell, each = best-with − best-without)

| Lever | Marginal (b/w) | Note |
|---|---|---|
| **L2 multi-tier budgets (T=4, DP-optimal)** | **0.2379** | The big lever — larger than v1's ~0.15 estimate. With 4 tiers the optimal top percentile is P100 everywhere and the escape class empties entirely (esc_frac = 0): tiers subsume both percentile slack *and* escapes. Best-without: W128_T1_P99_L11L31L40 = 10.9384. |
| **L3 flush carries mantissa payload** | 0.0889 | Seeds the rANS state with the first 12 mantissa bits; the 12-bit flush (0.09375) is exactly refunded by the mantissa-plane credit (0.09375) — near the full 12/W theoretical max. True net cost ~0.005 b/w extra renorm excess from the larger seeded state. |
| **L1 bit-granular fixed stride** | 0.0346 | Kills byte-ceil on slots/flags; inside v1's predicted 0.03–0.05 range. |
| **L4 column-conditioned tables** | — (dropped by gate) | Pre-registered gate: re-measured H(sym\|column-group) directly on the 256 layer-27 tensors *before* any grid row. H(sym) = 3.5583; full per-column ceiling H(sym\|col) = 3.5365 (gain 0.0218 b/w); the block column-group conditioning the L4 coder would actually use: W64 gain 0.000535, W128 0.000334, W256 0.000273 b/w. Best 0.0005 « 0.05 threshold → **dropped as pre-registered**. The motivating H(exp\|col) = 2.486 was an early-layer property (0014's layer-identity finding confirmed); at layer 27 column-conditional sym structure is essentially absent. Immaterial to the verdict: the win margin (0.1818) dwarfs even the 0.0218 full-column ceiling. |

Sum of marginals ≈ 0.36 ≈ the 11.06→10.70 movement from v1's nearest anchor; the win is
entirely overhead mechanics (L1+L2+L3), zero new entropy (L4 dead at this layer).

## Grid (numel-weighted, all taxes charged, measured coder bits; best T,P per lever combo)

```
config (best T,P per combo)         bpw     save     ovhd   flush     xs    pad    esc   flag   rank    tab    mcr    P
----------------------------------------------------------------------------------------------------------------------
 W64_T4_L10L30L40   10.9295  -0.0473  0.3636  0.1875  0.0340  0.1066  0.0000  0.0312  0.0040  0.0002  0.0000  100
 W64_T4_L10L31L40   10.8141  +0.0681  0.2482  0.1875  0.0435  0.1067  0.0000  0.0312  0.0040  0.0002  0.1250  100
 W64_T4_L11L30L40   10.9236  -0.0414  0.3577  0.1875  0.0340  0.1007  0.0000  0.0312  0.0040  0.0002  0.0000  100
 W64_T4_L11L31L40   10.7457  +0.1365  0.1798  0.1875  0.0435  0.1008  0.0000  0.0312  0.0040  0.0002  0.1875  100
W128_T4_L10L30L40   10.7929  +0.0893  0.2270  0.0938  0.0391  0.0762  0.0000  0.0156  0.0021  0.0002  0.0000  100
W128_T4_L10L31L40   10.7351  +0.1471  0.1692  0.0938  0.0439  0.0761  0.0000  0.0156  0.0021  0.0002  0.0625  100
W128_T4_L11L30L40   10.7894  +0.0928  0.2235  0.0938  0.0391  0.0727  0.0000  0.0156  0.0021  0.0002  0.0000  100
W128_T4_L11L31L40   10.7004  +0.1818  0.1345  0.0938  0.0439  0.0727  0.0000  0.0156  0.0021  0.0002  0.0938  100  <- BEST FUSIBLE
W256_T4_L10L30L40   10.7181  +0.1641  0.1522  0.0469  0.0416  0.0545  0.0000  0.0078  0.0011  0.0002  0.0000  100  (non-fusible bracket)
W256_T4_L10L31L40   10.6893  +0.1929  0.1234  0.0469  0.0440  0.0546  0.0000  0.0078  0.0011  0.0002  0.0312  100
W256_T4_L11L30L40   10.7167  +0.1655  0.1508  0.0469  0.0416  0.0531  0.0000  0.0078  0.0011  0.0002  0.0000  100
W256_T4_L11L31L40   10.6722  +0.2100  0.1063  0.0469  0.0440  0.0531  0.0000  0.0078  0.0011  0.0002  0.0469  100  <- best any-W
```

References: stz 10.8822 (parity exact); floor 10.5583; G2 bar 10.7083; quant delta
+0.0076. `save` = vs stz; `ovhd` = above floor+quant; `mcr` = mantissa credit.

**Gates:** v2-G1 (beat 10.8822 at W ≤ 128) **PASS**, Δ = +0.1818. v2-G2 (≤ floor+0.15)
**PASS**, Δ over floor = +0.1421 (margin to bar 0.0079 — thin). Round-trip, parity,
reconciliation: PASS.

## Does the storage-leaning result change?

**It stands, and improves slightly — but the storage/fusible split has nearly
collapsed.** v2's best any-W cell (W256_T4_P100_L11L31L40 = 10.6722) edges v1's
W16384_P100 (10.6977) at 64× finer granularity, and the *fusible* W=128 form (10.7004)
now sits within 0.028 b/w of the best storage form. v1's conclusion "the order-0 storage
side of this set is effectively closed" still holds — the remaining 0.1063–0.1345 above
floor is flush/renorm/tier-slack mechanics plus the 0.0076 quantization delta, not model
structure.

## Cross-layer transfer: now REQUIRED before any whole-model claim

Yes — the v2 win *raises* the need v1 only noted. Three reasons: (i) scope is still layer
27 / shard 7 only, and the projected whole-model number at the best fusible config
(**10.7285 b/w**) is selection-optimistic — tier budgets, best P, and the winning cell
were all selected on the same 256 tensors they are scored on; (ii) the G2 margin (0.0079)
is smaller than plausible cross-layer drift; (iii) 0014 proved layer identity is the
dominant hidden variable, and the L4 gate collapse shows layer-27 statistics are *not*
representative of early layers (where H(exp|col) = 2.486 was measured — L4 might even
re-enter early-layer). A cheap transfer check — freeze the v2 format, run per-tensor
DP-tier selection on 2–3 other layers (early/mid/late), score with selection held out —
gates any whole-model headline.

## Anomalies (recorded)

1. **v1 anchor not exactly reproduced:** v2's nearest-v1 cell W128_T1_P99_L10L30L40 =
   11.0561 vs v1's 11.0349 (+0.021). Explained (and printed by the script): v2 escape
   slots are a separate fixed-stride region (not v1's slot+overflow), and v2 escapes
   rb > bit-granular percentile while v1 escaped rb > 8·ceil(pct/8), so v1 kept blocks in
   the percentile-to-byte-ceil gap. Not a parity failure — stz parity is exact (0.000000).
2. **Tiers subsume escapes:** with T=4 DP tiers the optimal top percentile is P100 in
   every winning cell and the escape class is completely empty. L2's measured marginal
   (0.238) exceeded v1's ~0.15 estimate because it eats the escape machinery too.
3. **Flush is storage-free under L3:** flush 0.09375 and mantissa credit 0.09375 cancel
   exactly at the best config; the real net cost is ~0.005 b/w of extra renorm excess.
4. **G2 pass is thin** (0.0079 b/w) and same-set-selected — see transfer check above.
5. **The L4 gate collapse is itself a finding:** layer-27 experts have essentially no
   column-conditional sym structure (block-group gain 0.0005, full-column ceiling
   0.0218). The only *new-entropy* lever in the v2 plan is gone at this layer; everything
   won here came from overhead mechanics. Further gains at W ≤ 128 on this set must come
   from below-order-0 structure not yet identified.

## Compounding order

**Changes — the fusible tile point is now the frontier, and the next probe order is:**
(i) **cross-layer transfer check first** (cheap, gates the whole-model claim, and doubles
as an early-layer L4 re-test since that is where the column structure lives); (ii) then
apply peel-until-random to the **v2 emitted representation itself** — tier-slot streams,
flag plane, renorm bit-stream — since with L4 dead the residual 0.1345 at W=128 is
mechanics-bound and any real further gain needs below-order-0 structure (block-to-block
correlation, within-block symbol order, mantissa-plane structure) rather than more
overhead shaving; (iii) the register-tile decode kernel remains the runtime-credibility
step for the O(W) contract (unwritten, unbenchmarked — 0009's 24% speedup still does not
transfer automatically). The storage-side W256 result (10.6722) stays a noted option, not
the next object of study.

## Reproduction

```
# smoke (synthetic snapshot, seconds)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_block_codes_v2.py --synthetic

# L4 gate phase only (runs first regardless; recorded either way)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_block_codes_v2.py --gate-only

# real run (resumable; each invocation self-limits to ~7 min and checkpoints)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_block_codes_v2.py

# summary table + gates + summary JSON (auto-runs when complete)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_block_codes_v2.py --summary
```

Artifacts: `tests/artifacts/blockcodes_v2_results.jsonl` (per-tensor exact accounting,
stamp `3c5e38d4c9a9`), `tests/artifacts/blockcodes_v2_gate.jsonl` +
`blockcodes_v2_gate_summary.json` (L4 gate measurements),
`tests/artifacts/blockcodes_v2_summary.json` (all 144 cells, marginals, gates,
projection). Baseline reference unchanged:
`../0009-fusible-exponent-codebook/tests/artifacts/stz/stz_tensor_stats.jsonl`.

### Skeptic corrections to the v2 section (2026-07-02, reporting-level only)

- `overhead_decomposition.align_and_header` is 0.000052 b/w, not 0.000225 (with
  the correct value the components sum exactly to 10.700447).
- The winning config realizes a **2-bit** class flag plane (cls=4, escape class
  empty at P100); "3-bit" applies only to 5-class escape cells at P<100. The
  charged 0.015625 b/w = 2/128 was already the correct 2-bit accounting.
Neither slip affects any total, gate, or the verdict.

---

# Cross-layer transfer of the frozen v2 format: RESULTS

**Date:** 2026-07-02 · **Scope:** expert tensors only (128 experts × {up,down}_proj =
256 tensors, 1,277,165,568 params per layer) on six out-of-selection layers — 1, 3
(early/anomalous), 13, 24 (mid), 40, 51 (late) — plus the in-selection layer 27 for the
aggregate · **Format under test (FROZEN, zero re-selection):** W=128 bit-granular
fixed-stride blocks, T=4 DP-optimal tier budgets (per-tensor, transmitted and charged
exactly as in v2 — legitimately per-tensor side info), P100 top budget, levers L1 + L3
ON, L4 OFF · **Tool:** `tools/probe_block_codes_v2.py --frozen --layer N` · **Wall:**
6 layers in parallel, ~7 min scoring wall (~35 CPU-min).

**Verdict: the frozen format TRANSFERS.** G1 (beat that layer's own realized stz, all
taxes charged) passes on **all six** out-of-selection layers; worst margin **+0.1573**
(layer 51), best **+0.1938** (layer 3). The out-of-selection mean delta (**+0.1794**)
is statistically indistinguishable from layer 27's in-selection **+0.1817** — the
selection optimism v2 flagged was real but small (~0.006 b/w on the whole-model
projection). Same evidentiary standard as v1/v2: measured emitted bits, all side costs
charged, bits == accounted, SHA-256-exact BF16 round-trip on every layer, stz parity
max |Δbpw| = 0.000000 on all 7 layers.

**The one caveat that did NOT transfer: G2 (≤ floor + 0.15).** It holds mid/late-mid
(13, 24, 27, 40) and fails by hair margins at layers 1 (+0.1505 over floor, 0.0005 over
the bar), 3 (+0.1607, 0.0107 over), and 51 (+0.1503, 0.0003 over) — exactly the
fragility the thin layer-27 margin (0.0079) predicted. Early layers have higher H(sym)
(3.69/3.66 vs ~3.55 mid/late) and the fixed mechanics overhead sits slightly further
from their floors. The headline is therefore re-scoped: **"frozen fusible W=128 format
beats realized stz on every layer tested" stands unconditionally; "within 0.15 of the
order-0 floor" holds only mid-model.**

## Per-layer scores (numel-weighted per layer; equal numel per layer)

```
 layer   stz b/w  frozen b/w    delta     floor   G1    G2   L4 gate   L4 var  note
-----------------------------------------------------------------------------------
     1   11.0310     10.8443  +0.1867   10.6938 PASS  FAIL    0.0062       --  frozen transfer
     3   11.0176     10.8238  +0.1938   10.6631 PASS  FAIL    0.0069       --  frozen transfer
    13   10.8931     10.7112  +0.1819   10.5664 PASS  PASS    0.0014       --  frozen transfer
    24   10.8842     10.7008  +0.1834   10.5592 PASS  PASS    0.0006       --  frozen transfer
    27   10.8822     10.7004  +0.1817   10.5583 PASS  PASS    0.0005       --  in-selection
    40   10.8706     10.6971  +0.1735   10.5533 PASS  PASS    0.0005       --  frozen transfer
    51   10.8614     10.7041  +0.1573   10.5538 PASS  FAIL    0.0006       --  frozen transfer
-----------------------------------------------------------------------------------
 all-7   10.9200     10.7403  +0.1798   numel-weighted, incl. in-selection layer 27
xfer-6   10.9263     10.7469  +0.1794   out-of-selection layers only
```

G1 = beats realized stz on that layer's own set; G2 = ≤ floor + 0.15; L4 gate = best
block-column-group conditional gain (b/w) vs the pre-registered 0.05 threshold — FAIL
on every layer, so no L4-ON variant row was run anywhere (as pre-registered). Delta vs
stz declines monotone-ish with depth (+0.194 early → +0.157 late): late layers are the
conservative bound; early layers give the *largest* absolute wins even while missing G2.

## Honest whole-model number

**10.7346 b/w vs stz's realized whole-model 10.8975 (−0.1629 b/w).** Method: all 23
expert layers have identical expert numel (1,277,165,568; experts = 93.02% of model
numel, model = 67.09% of BF16), so the expert-plane saving is the plain mean of
per-layer deltas; the 7 measured layers use their measured delta (27 in-selection, 6
out-of-selection), and each of the 16 unswept layers takes the **min** of its two
bracketing measured layers' deltas (conservative — since delta is monotone-ish in
depth, bracket-min mainly discounts the late block); the non-expert 7% of numel is held
at stz, unchanged. Even-more-conservative floor (every unswept layer at the global
minimum measured delta 0.1573): **10.7448**. Both bracket the honest number. v2's
same-set selection-optimistic projection was 10.7285 — the cross-layer-validated
estimate is ~0.006 b/w worse, i.e. selection optimism was real but small.

**How the ledger should state it:** the v2 selection caveat is **resolved** for the
main claim — "0015 v2 frozen fusible format (W=128 bit-stride, T=4 transmitted tiers,
P100, L1+L3) beats realized stz on every layer tested, out-of-selection mean +0.179
b/w, honest whole-model estimate 10.7346 vs 10.8975 (conservative bracket-min over 16
unswept expert layers)". The G2/floor+0.15 claim must be stated re-scoped ("mid-model
layers only; fails by ≤0.011 at layers 1, 3, 51"), not as a whole-model property.

## Does L4 re-enter on early layers? NO for this mechanism — with a real nuance

The pre-registered gate (block column-group conditional gain ≥ 0.05 b/w) **fails on
every layer including early ones**: best gains 0.0062 (layer 1) and 0.0069 (layer 3),
then ≤0.0014 mid/late. So no L4-ON variant ran anywhere; the frozen score is the only
score. But the *full per-column* conditioning ceiling IS real on early layers —
gain_col = **0.160 b/w at layer 1, 0.125 at layer 3** (0014's early-layer column signal
confirmed in sym space) vs ~0.02–0.03 mid/late. The address-derived block column-group
conditioning the L4 coder can actually use (group = start_col // gcd(W, C), i.e.
64–128-column granularity) captures **<5%** of it. The 0.12–0.16 b/w early-layer column
headroom is a layout problem: capturing it needs column-major blocking or ≤16-column
groups — a separate future candidate with its own address math, not a cell of this
format. Whole-model ceiling if captured perfectly: ~0.02–0.03 b/w (it lives in ~2 of 23
layers).

## Anomalies (recorded)

1. **G2 does not transfer universally** — see above; the thin in-selection margin was a
   genuine early warning, now quantified (fails by 0.0003–0.0107 at layers 1, 3, 51).
2. **esc_frac = 0 on every layer** — the layer-27 finding that T=4 DP tiers fully
   subsume the escape class transfers everywhere.
3. **Benign:** layers 3 and 51 sampled 1,023 round-trip blocks instead of 1,024 (one
   duplicate among the {first, last, argmin, argmax} sample indices). Bits == accounted
   and SHA-256-exact reconstruction PASS on all layers regardless.
4. **Per-tensor DP tier budgets remain legitimate out-of-selection:** they are
   transmitted side info charged exactly as in v2; freezing the *format* while letting
   budgets adapt per tensor is the deployment contract, not selection leakage.

## Compounding order

**Confirms the v2 plan; the transfer gate is now cleared.** (i) The whole-model claim
is unblocked at 10.7346 b/w (conservative). (ii) The single next object of study, per
the peel-until-random loop applied recursively: **the v2 emitted representation
itself** — tier-slot streams, 2-bit flag plane, renorm bit-stream, mantissa plane — for
below-order-0 structure (block-to-block correlation, within-block symbol order), since
the residual 0.13–0.19 above floor is mechanics-bound and L4 is dead as-addressed on
every layer. (iii) The early-layer column headroom (0.12–0.16 b/w at layers 1/3) is a
real, named, but modest new-entropy source needing a layout-aware candidate of its own.
(iv) The register-tile decode kernel remains the runtime-credibility step for the O(W)
contract. (v) Optional cheap completeness: score the 16 unswept expert layers with the
frozen format to replace bracket-min interpolation with a fully measured whole-model
number.

## Reproduction

```
# frozen-format transfer score on one layer (resumable, ~5-7 min each)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_block_codes_v2.py --frozen --layer 1
# (repeat for layers 3, 13, 24, 40, 51; layer 27 reference is the v2 run itself)
```

Artifacts: `tests/artifacts/blockcodes_v2_frozen_summary_layer{1,3,13,24,40,51}.json`,
`blockcodes_v2_frozen_results_layer{N}.jsonl` (per-tensor exact accounting),
`blockcodes_v2_gate_summary_layer{N}.json` (per-layer L4 gate measurements), and the
aggregate `blockcodes_v2_frozen_crosslayer_aggregate.json`. Baseline reference
unchanged: `../0009-fusible-exponent-codebook/tests/artifacts/stz/stz_tensor_stats.jsonl`
(parity exact on all 7 layers).
