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
