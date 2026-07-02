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

---

# Emission peel — randomness certification of the v2 planes + within-block order-1: RESULTS

**Date:** 2026-07-02 · **Scope:** 64 real expert tensors (8 experts × {up,down}_proj ×
layers 1, 13, 27, 40; 319,291,392 params), actual streams emitted with the frozen v2
serializer (W=128 bit-stride, T=4 DP tiers, P100, L1+L3, order-0 12-bit tables) and the
frozen reference recomputed on the same sample · **Accounting stamp:** `b4e62994d250` ·
**Tool:** `tools/probe_emission_peel.py` · **Wall:** ~20 min.

**Verdict: PARTIAL — H2 does NOT fire (both gate conditions fail); H1 certifies the
coded sym-side planes near-random and finds one real, transmitted-plane structure: the
mantissa plane is NOT at 7.00 b/w — a fixed-position, phase-dependent MSB bias gives a
quantified ceiling of ~0.0287 b/w, revising the "mantissa is random noise" floor down
~0.03 b/w.** Same evidentiary standard as v1/v2: measured emitted bits, all side costs
charged, round-trip 256 frozen + 1,024 order-1 blocks serializer bit-exact + SHA-256
exact BF16 PASS.

## H1 — per-plane certificates (pre-registered structure bar: 0.01 b/w ceiling)

| Plane | Transmitted? | Plane size (b/w) | Ceiling (b/w) | Verdict |
|---|---|---|---|---|
| **payload** (coded rANS bits) | yes | 3.7382 | 0.0029 | **Random at these tests**, up to sub-bar weak hits: MI above the circular-shift null at lags 1–4 in 58/64 tensors but magnitudes ~1e-6 bits; layer 1 alone reaches 0.0103, other layers 0.0003–0.0005. |
| **flags** (2-bit tier plane) | yes | 0.0156 | 0.0023 | **Weak structure, below bar:** 64/64 MI + 52/64 autocorr hits (lags 2–4, r ≈ 0.02–0.03), but the whole plane is only 0.0156 b/w — exploitable value bounded there. |
| **lens** (per-block DP code lengths) | **no — diagnostic** | (0.0793 as fixed-width strawman) | 0.0335 | **Structure found** — block-to-block autocorrelation at lags 1–4, entropy-bound-driven, 64/64 tensors. But rb is not transmitted (flags+budgets are), so the ceiling is measured against a hypothetical fixed-width length plane; the *realizable* part is tier/budget design headroom bounded by the actual pad slack, not the full 0.0335. |
| **mant** (mantissa plane, new layout) | yes | 6.9063 | **0.0287** (0.0265–0.0351 by layer) | **Structure found — the real finding.** Driver in 62/64 tensors is per-phase entropy H(bit \| position mod 7): mantissa MSB p(1) ≈ 0.416 decaying monotonically to ~0.50 by bit 6 (mean p(1) per position = 0.416, 0.458, 0.479, 0.490, 0.496, 0.498, 0.4995). Pooled bit entropy (h0 = 0.9985) dilutes this ~7×, which is why it was never seen before; MI hits at native lags 7/14/21/884 in 64/64 tensors corroborate. Fixed-position (phase known at decode) → a fusible exploitation path exists. |

Batteries per plane: order-0/1/2 entropy, bit-pair MI vs a circular-shift null (16
shifts, family α = 0.01), autocorrelation of block-level code lengths and tier flags,
lzma -9e. Payload excludes slot padding (pad is already-quantified structure).

## H2 — within-block order-1 context (sym[i] | bucket(sym[i-1]), reset at block starts)

**Does NOT fire — fails both pre-registered conditions:** realized best delta **+0.0086
b/w « 0.05 gate**, and holdout confirms only **4/8** layer×proj cells positive (the
negative cells sit at ~−0.0005 bits/sym — zero structure, not noise). Realized numbers
use fit-on-self tables with ALL side costs charged (C occupied 12-bit-quantized bucket
tables + order-0 start table + pad8(8+C+(C−1)·9)-bit context header), frozen v2
mechanics otherwise unchanged, DP tier budgets recomputed.

```
layer  tensors  params       floor    frozen   C4       C8       C16      C32      bestC  delta    | H1 ceilings b/w: payload flags  lens   mant
L1     16       79,822,848   10.6990  10.8541  10.8485  10.8431  10.8344  10.8249  32     +0.0292  |                   0.0103  0.0030 0.0349 0.0351
L13    16       79,822,848   10.5596  10.7011  10.6998  10.6987  10.6989  10.6987  32     +0.0024  |                   0.0005  0.0020 0.0334 0.0266
L27    16       79,822,848   10.5590  10.7000  10.6986  10.6985  10.6990  10.7001  8      +0.0014  |                   0.0003  0.0019 0.0329 0.0265
L40    16       79,822,848   10.5549  10.7010  10.6975  10.6966  10.6969  10.6980  8      +0.0044  |                   0.0003  0.0021 0.0327 0.0266
ALL    64       319,291,392  10.5931  10.7390  10.7361  10.7342  10.7323  10.7304  32     +0.0086  |                   0.0029  0.0023 0.0335 0.0287
```

- **All of the order-1 signal is one cell:** layer-1 up_proj (realized +0.0596 b/w
  there; holdout gain +0.0380 bits/sym at C32 vs ~0.000 everywhere else, including
  layer-1 down_proj at −0.0013). Any follow-up is an *early-layer-only mode*, not a
  format change — worth ≈ 0.93 × 0.0292/23 ≈ **0.001 b/w whole-model** if adopted for
  layer 1 alone. Below adoption threshold; parked alongside the early-layer column
  headroom already named in the transfer run.
- **Block-boundary reset cost is negligible:** 0.000026 b/w weighted at best C (the
  1/128 of symbols that lose their context cost essentially nothing).
- If folded anyway, projected whole-model = **10.7266** = 10.7346 − 0.93 × 0.0086
  (selection-optimistic). **Recommendation: do not fold** — the gate failed; the frozen
  format's standing number remains **10.7346**.
- Holdout gains (bits/sym, fit on half the experts, scored on the other half), C32:
  L1_up +0.0380, L13_up +0.0013, L40_up +0.0012, L40_down +0.0000, others ≈ 0 or
  negative.

## What this means

1. **The sym-side emission is essentially done.** Payload certified random at these
   tests (pooled ceiling 0.0029 b/w), flags bounded at 0.0023, within-block order-1
   dead as a format change. The "peel until random" loop on the coded planes has
   converged: the residual 0.13–0.19 above the order-0 floor is coder mechanics
   (flush/renorm/tier slack), already decomposed in v2, not hidden symbol structure.
2. **The frontier moves to the mantissa plane.** The ~0.0287 b/w phase-bias ceiling is
   on a *transmitted* plane, is fixed-position (fusible-compatible), and revises the
   project's "mantissa ≈ 7.95/8 bits, incompressible" assumption: the true floor is
   H(sym) + ~6.97, not H(sym) + 7. The MSB alone carries ~0.020 b/w of it
   (h(0.416) ≈ 0.9797).
3. **Layer 1 is consistently the outlier** (payload ceiling 0.0103, order-1 signal,
   mant ceiling 0.0351) — one more datum that early layers hold the only remaining
   *model* structure; everything mid/late is mechanics.

## Anomalies (recorded)

1. All H2 signal concentrated in layer-1 up_proj (see above) — a property of the model,
   not the probe.
2. The lens ceiling (0.0335) must not be read as realizable: rb is not transmitted; it
   bounds tier/budget *design* headroom, realizable part limited by actual pad slack
   (~0.073 b/w pad component at layer 27, of which only a fraction is reachable).
3. Layer-1 payload ceiling alone (0.0103) marginally exceeds the 0.01 bar though the
   pooled payload verdict stays below it (0.0029); payload MI magnitudes ~1e-6 bits sit
   at the null threshold.
4. A pre-existing layer-27 tensor row (same acct stamp `b4e62994d250`) was seeded into
   the combined results file to resume rather than recompute; a concurrent background
   sweep (blockcodes_v2_frozen layer-38 lock, different tool) was left running — CPU
   cost only, no shared state.

## Compounding order

**Confirms the tile format as the primary track for a container v3 — and names its next
lever.** (i) The frozen W=128 format now carries: cross-layer-validated wins over stz
on every layer tested (honest whole-model 10.7346 vs 10.8975), *and* a randomness
certificate on its coded planes — stz has neither. The .stz container's role for expert
tensors should be succeeded by the tile format in a container v3; stz remains the
baseline reference and the container for non-expert tensors until the tile format is
extended there. (ii) The next object of study is the **mantissa plane's phase-dependent
bias** (~0.0287 b/w ceiling, fixed-position, fusible path) — the first new-entropy
lever found since L4 died, and it applies to *any* container that moves mantissas
verbatim, including stz. (iii) Order-1 context is closed as a format change; the
layer-1-only mode (+0.001 b/w whole-model) is parked with the early-layer column
candidate. (iv) Tier/budget design headroom (bounded by pad slack) is the remaining
mechanics lever, secondary to (ii). (v) The register-tile decode kernel remains the
runtime-credibility step.

## Reproduction

```
# smoke (synthetic snapshot, seconds)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_emission_peel.py --synthetic

# real run (resumable; self-limits per invocation and checkpoints; optionally --layer N)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_emission_peel.py

# summary + gates + summary JSON
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_emission_peel.py --summary
```

Artifacts: `tests/artifacts/emission_peel_results.jsonl` (per-tensor exact accounting,
stamp `b4e62994d250`), `tests/artifacts/emission_peel_summary.json` (per-plane
certificates, H2 per-layer/per-C realized numbers, holdout table, gates).

---

# Mantissa-phase realization inside the frozen format (M1/M2/M3): RESULTS

**Date:** 2026-07-02 · **Scope:** 64 real expert tensors (8 experts × {up,down}_proj ×
layers 1, 13, 27, 40; 319,291,392 params — the emission-peel sample), frozen v2
mechanics unchanged (W=128 bit-stride, T=4 DP tiers, P100, L1+L3, 12-bit tables), frozen
reference recomputed on the same sample · **Accounting stamp:** `ffd4e05b2da3` ·
**Tool:** `tools/probe_mantissa_phase.py` · **Wall:** ~4 min (~108 s compute).

**Verdict: FIRES.** The emission peel's mantissa phase bias is realizable inside the
frozen format with exact accounting: **M1 (extended symbol sym10 = sign+exp8+mantMSB,
1024-entry table, remaining 6 mantissa bits verbatim) = 10.6973 b/w, +0.0417 vs the
frozen 10.7390 recomputed on this sample** — more than 2× the pre-registered 0.02 gate,
ALL side costs charged (doubled table, pad, 88-bit per-tensor phase probs where used),
round-trip proven (256 blocks per mechanism, emitted bits == accounted bits, SHA-256
exact BF16). M1 wins on **every** layer. Field split as frozen: u16 LE, sym = u >> 7,
mant = u & 0x7F; O(1) block addressing and the fixed-stride contract are untouched —
the folded MSB simply joins the block's already-sequential rANS decode.

## Realized (numel-weighted, all side costs charged; delta > 0 = saves vs frozen)

```
    mech       bpw    delta    coded     pad     tab    mant    misc     bound     gap
  frozen   10.7390  +0.0000   3.7382  0.0766  0.0002  6.9062  0.0177   10.5931  0.1459
      M1   10.6973  +0.0417   4.6931  0.0798  0.0004  5.9062  0.0177   10.5363  0.1610
      M2   10.8156  -0.0766  10.7113  0.0864  0.0002  0.0000  0.0177   10.5658  0.2498
      M3   10.7171  +0.0220   5.7113  0.0810  0.0009  4.9062  0.0177   10.5192  0.1979
```

Bounds are entropy-level (no side costs): frozen H(sym)+7 = 10.5931; M1 H(sym10)+6;
M2 the revised independent-phase floor H(sym)+Σh(pᵢ) = 10.5658 (headroom 0.0274,
matching the peel's ~0.0287 ceiling); M3 H(sym11)+5. Per-layer M1 delta: L1 +0.0245,
L13 +0.0434, L27 +0.0458, L40 +0.0533 — positive and above-gate everywhere, growing
monotonically with depth.

## Why M1 wins (and what M2/M3 falsify)

- **M1** pays the coder's per-symbol redundancy only once per weight (coded symbols/w
  stays 1.0; redundancy 0.051 → 0.063 b/sym going A=512 → 1024) while removing a full
  biased bit from the verbatim plane. Crucially it also captures **sym-conditioned MSB
  structure** the independent per-phase model cannot see: H(sym10)+6 = 10.5363 is
  **0.0295 below** the revised independent floor 10.5658 — which is why the realized
  +0.0417 exceeds the peel's independent-phase ceiling (~0.0287). That is not an error;
  it means the mantissa MSB and the sign+exponent symbol are mutually informative
  (≈ 0.036 b/w at entropy level), and folding exploits both terms at once.
- **M2** (7 per-position binary rANS lanes) loses 0.0766 b/w *despite modeling the bias
  directly*: coded symbols/w jumps to 7.9, and the frozen bit-by-bit-renorm coder's
  measured +0.018 b/sym redundancy on near-fair binary symbols swamps the ~0.027 b/w
  bias gain. Pre-registered attribution: this falsifies **per-bit binary lanes in THIS
  coder only** — not the direction. A wider-state/multi-bit-renorm lane is a format
  change, out of scope here.
- **M3** (two folded bits, sym11, 2048 entries) stays positive (+0.0220) but table cost
  plus coder redundancy growth (+0.098 b/sym at A=2048) eat most of the second bit's
  gain. **M1 is the sweet spot**; deeper folding needs cheaper renorm first.

## Bias stability (per-tensor vs global)

Pooled p(1) per phase = 0.4157, 0.4584, 0.4788, 0.4900, 0.4957, 0.4984, 0.4995 —
monotone MSB→LSB, matching the peel, effectively universal across layers 1/13/27/40 and
both projections. Coding with pooled global constants costs +0.000099 b/w vs per-tensor
fitted probs; transmitting 7×12-bit per-tensor probs costs 0.000018 b/w (88 bits/tensor)
— **per-tensor fit pays for itself**, but the margin is tiny and global constants would
lose almost nothing. Either choice is fine for a container spec.

## Honest whole-model number

Projected at M1: **10.6923 b/w** (= 10.7311 − 0.93 × 0.0417). Caveats, stated plainly:
the delta is a 64-tensor sample mean (8 experts/proj on 4 layers), and M1 was selected
on the set it is scored on. The conservative bracket — every unswept layer at the worst
measured per-layer delta (+0.0245, layer 1) — gives **10.7083**; honest range
**10.69–10.71 vs the frozen 10.7346 (vs stz 10.8975)**. A full 6-layer frozen-style
re-check is **not required to accept the mechanism** (the bias was already measured
stable across 4 layers and both projections, global constants cost +0.0001 b/w, and the
*worst* layer still clears the gate), but a cheap frozen+M1 re-score on the existing
transfer layers (at minimum 1, 3, 51 — the G2-fragile ones) should gate any *headline*
whole-model claim, replacing the projection with a measured number. Note M1 also lowers
the floor side (bound −0.0569), so the re-scoped G2 margins should be re-stated against
H(sym10)+6 when that runs.

## Should M1 be offered to the .stz container (per-weight track)? NO — with a precise reason

The bias exists in any container that moves mantissas verbatim, including stz, but the
MSB carries h(0.416) ≈ 0.98 bits — the gain is **fractional-bit** (~0.02–0.04 b/w) and
only an entropy coder can realize it. stz's per-weight track is a fixed-width codebook
index: folding the MSB there forces the index width up by a full bit to buy back at
most one mantissa bit — net ≤ 0, before escape interactions. The vehicle for this win
is the tile-format **container v3** (which the emission peel already named as stz's
successor for expert tensors), not a stz retrofit. stz keeps its role as baseline and
non-expert container until the tile format is extended.

## Anomalies (recorded)

1. **Realized gain exceeds the peel's independent ceiling** (+0.0417 > ~0.0287) —
   explained above: sym↔MSB mutual information (≈ 0.036 b/w entropy-level) is captured
   by folding but invisible to the independent per-phase model. The "revised floor"
   10.5658 is therefore itself not the true floor; sym-conditioned mantissa structure
   is worth another ~0.03 b/w at entropy level (M1 bound 10.5363; M3 bound 10.5192).
2. **M2's loss is a coder property, not a direction falsifier** — pre-registered
   attribution in the summary JSON (`m2_attribution`).
3. **Gain grows with depth** (L1 +0.0245 → L40 +0.0533), opposite the frozen-vs-stz
   delta's depth trend.
4. Frozen recomputed on this 64-tensor sample is 10.7390, slightly above the 10.7311
   whole-model reference (sample composition; all deltas are same-sample, unaffected).
5. One pre-existing layer-27 row (identical stamp `ffd4e05b2da3`, round-trip PASS) was
   seeded into the combined results file; the other 63 tensors were computed fresh.
   Concurrent background job left untouched.

## Compounding order

**Confirms and extends the peel's plan — the mantissa lever is real and cashed; three
named next steps, in order.** (i) **Fold M1 into the frozen format** as the v3-container
spec for expert tensors (sym10, 1024-entry tables, 6-bit verbatim plane) and run the
cheap cross-layer frozen+M1 re-score to convert 10.6923 into a measured whole-model
number. (ii) **The next object of study is the sym↔mantissa dependence itself**: the
run's own bounds prove ~0.03 b/w more sits in conditioning mantissa bits on the symbol
(and M3's bound says the second bit still holds ~0.017) — probe H(mantMSB | exponent
bin) to locate where the mutual information lives, then peel the M1 emission
recursively (is the sym10 payload + 6-bit residual plane now random?). (iii) The
blocking lever for any deeper fold is **coder mechanics**: bit-by-bit renorm redundancy
grows 0.051 → 0.063 → 0.098 b/sym at A=512/1024/2048 and already ate M3; a wider-state
/ byte-renorm variant is the named format-change probe, and it also holds ~0.06 b/w of
M1's remaining 0.161 gap to bound. The register-tile decode kernel remains the
runtime-credibility step.

## Reproduction

```
# smoke (synthetic snapshot, seconds)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_mantissa_phase.py --synthetic

# real run (resumable; self-limits per invocation and checkpoints)
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_mantissa_phase.py

# summary table + gate + summary JSON
uv run python research/candidates/0015-block-granular-tile-codes/tools/probe_mantissa_phase.py --summary
```

Artifacts: `tests/artifacts/mantissa_phase_results.jsonl` (per-tensor exact accounting,
stamp `ffd4e05b2da3`), `tests/artifacts/mantissa_phase_summary.json` (realized cells,
bounds, coder-redundancy attribution, phase stability, gate, projection),
`tests/artifacts/mantissa_phase_run.log`.

---

# Frozen+M1 measured whole-model, MSB MI decomposition, M1 re-peel (T1/T2/T3): RESULTS

**Date:** 2026-07-02 · **Scope:** T1 = ALL 23 expert layers (1, 3, 6, 8, 10, 13, 15, 17,
20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51) × 256 expert tensors each =
5,888 tensors, 29,374,808,064 expert params — every expert tensor in the model; T2 = 64
tensors across layers 1/13/27/51 (319,291,392 params, numel-weighted); T3 = 32 tensors
across the same 4 layers · **Format under test (FROZEN + M1, zero re-selection):** W=128
bit-stride, T=4 DP tiers, P100, L1+L3, sym10 = sign+exp8+mantMSB folded into a
1024-entry 12-bit rANS table, 6 mantissa bits verbatim · **Accounting stamp:**
`bb3a86666829` · **Tool:** `tools/measure_m1_full.py` (composes the skeptic-verified
v1 loader/accounting, v2 coder/DP/tables, emission-peel batteries, and the M1 coder
verbatim) · **Wall:** ~25 min across 22 parallel resumable invocations (layer 13 reused
intact from the prior verified run, same stamp).

**Verdict: the projection is replaced by a measurement, and the measurement is better.
Whole-model frozen+M1 = 10.6866 b/w MEASURED** (10.686640), vs the 10.6923 projection —
0.0057 b/w better. M1 beats frozen on **every one of the 23 expert layers** (min delta
+0.0317 @L1, max +0.0643 @L3, mean +0.0478). Same evidentiary standard as every prior
section: all side costs charged, measured emitted bits, stz parity exact on all 5,888
tensors (max |Δbpw| = 0.000000), round-trip 47,097 blocks with emitted bits == accounted
bits and SHA-256-exact BF16, and an aggregate self-gate — the frozen whole-model
recomputed from the same pipeline reproduces the 10.7311 reference to 3e-6. T2 locates
the MSB structure entirely in **exponent magnitude** (99.99% of the MI; sign carries
exactly zero). T3 does **not** issue a converged certificate: the bulk (layers 13/27/51)
passes both randomness bars, but layer 1 fails locally, and the residual 6-bit mantissa
plane is measurably phase-tilted — which is precisely the already-quantified bit-2
signal, not a new unknown.

## T1 — per-layer scores (numel-weighted; equal expert numel per layer)

```
 layer    stz b/w  frozen b/w    M1 b/w    delta   nonpos  parity  RT
---------------------------------------------------------------------
     1    11.0310     10.8443   10.8126  +0.0317   11/256    0.0  PASS
     3    11.0176     10.8238   10.7595  +0.0643    2/256    0.0  PASS
     6    10.9498     10.7618   10.7119  +0.0499    6/256    0.0  PASS
     8    10.9133     10.7305   10.6869  +0.0436    4/256    0.0  PASS
    10    10.8962     10.7138   10.6702  +0.0435    1/256    0.0  PASS
    13    10.8931     10.7112   10.6670  +0.0442    1/256    0.0  PASS
    15    10.8882     10.7058   10.6637  +0.0420    1/256    0.0  PASS
    17    10.8880     10.7073   10.6627  +0.0446    1/256    0.0  PASS
    20    10.8834     10.7021   10.6575  +0.0446    0/256    0.0  PASS
    22    10.8849     10.7025   10.6587  +0.0438    0/256    0.0  PASS
    24    10.8842     10.7008   10.6553  +0.0455    0/256    0.0  PASS
    27    10.8822     10.7004   10.6540  +0.0465    0/256    0.0  PASS
    29    10.8826     10.7009   10.6538  +0.0471    0/256    0.0  PASS
    31    10.8784     10.6979   10.6485  +0.0494    0/256    0.0  PASS
    34    10.8720     10.6949   10.6438  +0.0511    0/256    0.0  PASS
    36    10.8698     10.6947   10.6430  +0.0517    0/256    0.0  PASS
    38    10.8689     10.6947   10.6428  +0.0520    0/256    0.0  PASS
    40    10.8706     10.6971   10.6455  +0.0516    1/256    0.0  PASS
    43    10.8672     10.6927   10.6410  +0.0517    0/256    0.0  PASS
    45    10.8659     10.6928   10.6413  +0.0515    1/256    0.0  PASS
    47    10.8668     10.6943   10.6427  +0.0516    1/256    0.0  PASS
    49    10.8662     10.6987   10.6488  +0.0499    5/256    0.0  PASS
    51    10.8614     10.7041   10.6569  +0.0472    0/256    0.0  PASS
```

`nonpos` = tensors where M1's table cost exceeded the MSB gain (35/5,888 model-wide,
concentrated at layer 1); layer-level M1 wins everywhere. Every tensor's frozen score
was cross-checked bit-identical against the stored `blockcodes_v2_frozen_*` artifacts
where they exist (256/256 per checked layer).

## The measured whole-model number and the ladder

Convention identical to the 10.7311 computation: wm = 10.897505 (stz whole-model) −
0.930232 (expert share) × mean₂₃(stz_ref_L − m1_bpw_L); all 23 expert layers have equal
numel so the expert plane is the plain mean; the non-expert ~7% of numel is held at stz.

```
frozen recomputed   10.7311   (reference 10.7311: REPRODUCED, aggregate gate PASS)
frozen+M1 MEASURED  10.6866   (+0.0445 vs frozen, +0.2109 vs stz 10.8975)
```

**How the ledger should state the full ladder** (savings vs BF16 = 16 b/w, whole-model,
non-expert at stz):

> **stz 31.89% → tile (frozen W128/T4/P100/L1+L3) 32.93% → tile+M1 (sym10 fold)
> 33.21%.** Whole-model **10.6866 b/w, fully MEASURED** on all 23 expert layers × 256
> tensors (5,888 tensors, 29.4B expert params) — no projection, no interpolation, no
> selection caveat (format and mechanism frozen before scoring; M1 wins on every
> layer). stz parity exact; SHA-256-exact round-trip. Replaces the 10.6923 projection
> (measured is 0.0057 better — the 4-layer sample had over-weighted layer 1, the
> smallest-delta layer).

## T2 — where the MSB structure lives: exponent magnitude, entirely

64 tensors, layers 1/13/27/51, numel-weighted: H(msb) = 0.9794, H(msb|sym) = 0.9432,
H(msb|exp8) = 0.9432 — the 8-bit exponent value captures **99.99%** of MI(msb; sym)
(0.036183 of 0.036187 b/w). Sign carries **exactly nothing** (MI = 0.0000 marginal,
4e-6 given exponent). The shape is smooth and monotone in exponent value: p(msb=1|e) ≈
0.50 for all low/mid exponents and collapses only at the top of the magnitude tail —
e=120: 0.473, e=121: 0.400, e=122: 0.191, e=123: ~0.01. The largest-magnitude weights
sit just above their power-of-two boundary, biasing the mantissa MSB toward 0. Total
MSB headroom vs verbatim = 0.0566 b/w (0.0206 marginal bias + 0.0362 conditioning) —
what M1 monetizes. The structure is model-wide, not layer-local: H(msb|sym) varies only
0.9403 (L1) to 0.9444 (L51).

**Second bit (the sym11 question):** same structure one level deeper, decaying ~3.3×:
H(b2) = 0.9950, H(b2|sym10) = 0.9829 — total headroom **0.0171 b/w** (0.0052 marginal +
0.0120 conditioning), again ~98% exponent-driven (MI(b2;exp) = 0.0118 of MI(b2;sym10) =
0.0120).

**Does this justify a sym11 follow-up? Priced: NOT with the current coder; YES as a
rider on a coder-mechanics probe.** The structure's shape is maximally favorable — a
2048-entry table captures it, no sign or spatial keying needed — but M3 already
*measured* the cost: coder redundancy grows +0.035 b/sym going A=1024→2048 (0.063 →
0.098), which exceeds the entire 0.0171 collectible. Net with the current bit-by-bit
renorm coder ≈ **−0.018 b/w** (M3's realized +0.0220 vs frozen = −0.0197 vs M1 confirms
this empirically — already falsified, do not re-run). With a wider-state / byte-renorm
coder that holds A=2048 redundancy at or below today's A=1024 level, the collectible is
**+0.012–0.017 b/w experts ≈ +0.011–0.016 b/w whole-model → landing ≈ 10.670–10.675**.
The coder change is the gate; sym11 is the payload it unlocks.

## T3 — re-peel of the M1 emission: NO converged certificate; two quantified ceilings

32 tensors, layers 1/13/27/51 (8 each); actual emitted M1 payload plane
(serializer-gated bit-identical to the reference byte packer) + residual 6-bit plane.

- **sym10 coded payload: RANDOM on 3 of 4 layers.** Ceilings 0.0005 (L13), 0.0000
  (L27), 0.0001 (L51) b/w — under the 0.01 bar; lzma cannot compress it; order-0/1/2
  bit entropies ~1.0. **NOT random on layer 1:** weighted 0.0123 b/w, dominated by
  `backbone.layers.1.mixer.experts.0.up_proj.weight` at 0.066 b/w (lzma-visible;
  small-lag MI at lags 1–4, 32, 64, 128); experts.36/91 up_proj at 0.012–0.016; all
  four L1 down_proj payloads fully random (<1e-4). Pooled 32-tensor ceiling 0.0032 b/w.
- **Residual 6-bit mantissa plane: NOT phase-flat, by a design-consistent amount.**
  Pooled p(1|pos mod 6) = [0.4586, 0.4786, 0.4899, 0.4956, 0.4982, 0.4993] — the
  monotone approach to 0.5 continues one bit deeper; native-lag MI at 6/12/18/756 on
  32/32 tensors. This **is** the bit-2 signal T2 quantifies: realized-plane ceiling
  ~0.0065 b/w on converged layers (entropy-level bound 0.0171). Layer 1 heavier: 0.0183
  b/w weighted, max 0.093 on experts.0.up_proj. Pooled ceiling 0.0095 b/w.

**Certificate decision: NOT converged** — per-layer convergence is 3/4, and pooling
would hide the L1 signal. Two new ceilings replace the certificate:

1. **Converged bulk (13/27/51 and by extension mid/late model):** the only residual
   structure is the known bit-2 headroom — 0.0171 b/w entropy-level, ~0.0065 realized-
   plane — i.e. exactly what sym11-with-a-better-coder collects. Nothing else found.
2. **Layer-1 ceiling:** payload 0.0123 + mant6 0.0183 b/w on the L1 sample, concentrated
   in early up_proj tensors (expert 0 up_proj alone: 0.066 + 0.093 ≈ 0.16 b/w on that
   tensor). Layer 1 is 1/23 of expert numel → whole-model unclaimed residue **≤ ~0.002
   b/w** — real but small; parked with the other layer-1/early-layer items (order-1
   up_proj mode, column headroom).

## Anomalies (recorded)

1. **Layer 1 is the outlier everywhere:** smallest M1 delta (+0.0317 vs mean +0.0478),
   11/256 non-positive per-tensor deltas, and the only layer failing the T3 re-peel bar
   — concentrated in `experts.0.up_proj`. Consistent with every prior run's early-layer
   findings; layer-level M1 still wins there.
2. **Measured whole-model beat the skeptic-verified projection by 0.0057** — sampling
   composition only (the 4-layer projection gave L1 a 1/4 weight vs its true 1/23); no
   accounting change; the aggregate gate reproduced the frozen 10.7311 to 3e-6.
3. **M1 delta trends up with depth** (0.044 @L8–22 → 0.052 @L34–47) then dips at
   L49/51 — same shape as the frozen-vs-stz profile: the exponent-tail concentration M1
   exploits strengthens mid-to-late model.
4. T3 pooled passes the 0.01 bar (payload 0.0032, mant6 0.0095) but is reported NOT
   converged to avoid pooling away the L1 signal.
5. No gate failures, no parity drift, no lock contention across 22 parallel resumable
   invocations; layer 13 reused intact (same stamp `bb3a86666829`).

## Compounding order

**The measured number closes the M1 chapter; the coder is now the single blocking
lever.** (i) The ledger entry above supersedes 10.7346 and the 10.6923 projection;
frozen+M1 (sym10) is the v3-container spec for expert tensors. (ii) **The single next
object of study: the rANS coder's renorm/flush mechanics** — a wider-state / byte-renorm
variant. It is the third time this lever has been named (M2's per-bit-lane failure, M3's
erosion, now T2/T3's convergence onto bit-2-behind-coder-redundancy), and it now gates
everything that remains: it directly bounds the ~0.06 b/w of measured renorm redundancy
inside M1's standing ~0.15 gap-to-bound, and it is the stated precondition for sym11's
+0.011–0.016 b/w whole-model. One probe, two priced collectibles, landing zone ≈
10.62–10.67 if both pay. (iii) The layer-1 residue (≤0.002 b/w whole-model) stays parked
with the early-layer column candidate. (iv) The register-tile decode kernel remains the
runtime-credibility step — unchanged, and a byte-renorm coder likely *simplifies* it.

## Reproduction

```
# smoke (synthetic snapshot, seconds; runs T1+T2+T3)
uv run python research/candidates/0015-block-granular-tile-codes/tools/measure_m1_full.py --synthetic

# real, one layer (resumable; all tasks or --tasks t1|t2|t3)
uv run python research/candidates/0015-block-granular-tile-codes/tools/measure_m1_full.py --layer 13

# THE whole-model number (combines the 23 per-layer summaries; runs the aggregate gate)
uv run python research/candidates/0015-block-granular-tile-codes/tools/measure_m1_full.py --aggregate
```

Artifacts: `tests/artifacts/m1full_whole_model.json` (the measured number, per-layer
deltas, convention, gates), `m1full_summary_layer{N}.json` × 23 (per-layer T1 accounting),
`m1full_mi_layer{1,13,27,51}.json` (T2 MI decomposition),
`m1full_repeel_layer{1,13,27,51}.json` + `m1full_repeel_results_layer{N}.jsonl` (T3
per-plane certificates). Stamp `bb3a86666829` throughout. Baseline reference unchanged:
`../0009-fusible-exponent-codebook/tests/artifacts/stz/stz_tensor_stats.jsonl` (parity
exact on all 5,888 tensors).
