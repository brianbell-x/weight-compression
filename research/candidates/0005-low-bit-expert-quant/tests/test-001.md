# Test 001 — Stage-1 matmul-fidelity sweep of low-bit routed-expert codecs

Candidate: 0005-low-bit-expert-quant (the runtime track)
Date: 2026-06-29
Stage: **Stage-1 — matmul-fidelity proxy (NECESSARY, NOT SUFFICIENT)**
Payoff regime: **Regime D — resident-VRAM + decode bandwidth** (weights stay narrow into a fused matmul, never re-inflated; the only track that cuts resident VRAM *and* decode cost at once).

## What was measured

Each codec was applied to real BF16 layer-1 routed-expert matrices read per-tensor
from shard 1 via `safetensors.safe_open` (no full-model load — sidesteps the RAM
blocker). Fidelity is the Stage-1 metric: for fixed seeded input batch `X`
(batch=256, Gaussian rows L2-normalized to unit norm, seed=0), relative output
error `rel_err = ‖XW − XW′‖_F / ‖XW‖_F` and mean per-row `cosine(XW, XW′)`, where
`W′` is the dequantized reconstruction. `X` always matches each tensor's
in_features (up_proj 1856, down_proj 2688). Bits/weight **includes** scale/codebook
overhead. Implied resident VRAM = `4.4 GB + 29.4e9 · b/8 / 1024³` (routed experts =
29.4B params; non-expert floor = 4.4 GB).

Harness (verified, INT8 self-test PASS, rel_err 0.683% / cos 0.999977 / 8.125 b/w):
`tests/artifacts/stage1_probe.py`. Lever scripts + CSVs alongside it
(`lever_baselines.py`, `lever_groupsize.py`, `lever_codebook.py`, `lever_salient.py`).

Sample sizes per config: 16–32 real layer-1 expert matrices (experts spread across
the 128, both up_proj and down_proj). All numbers measured, none fabricated.

## Consolidated table — bits/weight vs output error (sorted by bits/weight)

| Config (lever) | bits/wt | rel_err % | mean_cosine | resident VRAM (GB) |
|---|---:|---:|---:|---:|
| codebook8_perexpert — 3b non-uniform (codebook) | 3.000 | 21.49 | 0.97649 | 14.67 |
| uniform_rtn_3b_g128 (codebook ref) | 3.000 | 28.39 | 0.96198 | 14.67 |
| codebook8_pergroup_g128 — 3b non-uniform (codebook) | 3.125 | 19.12 | 0.98152 | 15.10 |
| INT3 uniform (salient ref) | 3.250 | 25.64 | 0.96878 | 15.52 |
| INT3 + 0.5% salient INT8 (salient) | 3.275 | 25.54 | 0.96902 | 15.61 |
| INT3 + 1.0% salient INT8 (salient) | 3.301 | 25.44 | 0.96924 | 15.70 |
| INT3 + 2.0% salient INT8 (salient) | 3.351 | 25.27 | 0.96963 | 15.87 |
| INT3 + 5.0% salient INT8 (salient) | 3.500 | 24.79 | 0.97071 | 16.38 |
| codebook16_perexpert — 4b non-uniform (codebook) | 4.000 | 11.37 | 0.99348 | 18.09 |
| uniform_rtn_4b_g128 (codebook ref) | 4.000 | 12.27 | 0.99255 | 18.09 |
| **INT4_pg128_RTN — baseline** | **4.125** | **12.08** | **0.99279** | **18.52** |
| **codebook16_pergroup_g128 — 4b non-uniform (best 4b)** | **4.125** | **9.66** | **0.99532** | **18.52** |
| INT4_gs128_axis0 (groupsize) | 4.127 | 11.94 | 0.99294 | 18.53 |
| INT4_gs64_axis0 (groupsize) | 4.250 | 10.93 | 0.99409 | 18.95 |
| INT4 uniform (salient ref) | 4.250 | 11.03 | 0.99398 | 18.95 |
| INT4 + 0.5% salient INT8 (salient) | 4.270 | 10.98 | 0.99402 | 19.01 |
| INT4 + 1.0% salient INT8 (salient) | 4.291 | 10.94 | 0.99407 | 19.09 |
| INT4 + 2.0% salient INT8 (salient) | 4.331 | 10.87 | 0.99415 | 19.22 |
| INT4 + 5.0% salient INT8 (salient) | 4.450 | 10.66 | 0.99436 | 19.63 |
| INT4_gs32_axis0 (groupsize) | 4.500 | 9.81 | 0.99523 | 19.80 |
| INT4_gs16_axis0 (groupsize) | 5.000 | 8.58 | 0.99634 | 21.51 |
| int8_rtn_g128_ref (codebook) | 8.000 | 0.69 | 0.99998 | 31.78 |
| **INT8_pg128_RTN — SAFE FLOOR** | **8.125** | **0.67** | **0.99998** | **32.21** |
| INT8_gs64_axis0_ref (groupsize) | 8.250 | 0.60 | 0.99998 | 32.64 |
| INT8 (salient ref) | 8.250 | 0.61 | 0.99998 | 32.64 |

Reference markers:
- **INT8 safe floor** — ~0.6–0.7% output error, ~32 GB resident (brief's rounded
  ~34 GB; the measured GiB value is 32.2). A safe ~2x VRAM win, validated on true weights.
- **INT4 baseline** — ~12% output error, ~18.5 GB resident. The gap to be closed.

## Promising-result bar — did anything clear it?

The bar (from the brief): **per-layer matmul output error under ~1–2% at ≤4
effective bits/weight** — i.e. INT4-class size (~18–19.5 GB) at INT8-class fidelity.

**No config cleared the bar.** The entire ≤4-bit region sits at 9.7–28% output
error. The best point at or below 4 bits is the 16-level per-group non-uniform
codebook (`codebook16_pergroup_g128`, 4.125 b/w, 18.52 GB) at **9.66%** — about
14x worse error than INT8 and roughly 5–10x outside the 1–2% target band. The error
floor does not approach the band until ~8 bits. The bar is **NOT cleared at Stage-1**.

What the four levers showed (all negative for sub-4-bit):
- **Group size** (128→16 at INT4): each halving buys only ~1 absolute point of
  error but costs progressively more scale overhead; gs16 pays 5.0 b/w (21.5 GB,
  near-INT5 storage) for 8.6% error. Weak lever — the dominant error is the 4-bit
  grid itself, not scale granularity.
- **Non-uniform codebook** (Gaussian/k-means fit, shape-shared): a real but small
  lever — cuts 4-bit error 12.3%→9.66% (per-group scale) and 3-bit 28.4%→19.1%.
  Reproduces the brief's predicted ~9.6%. Scale granularity matters as much as
  codebook shape (per-expert scale only reaches 11.4%). Codebook itself costs ~0
  b/w (amortized over 29.4e9); the cost is the 0.125 b/w per-group fp16 scale.
- **Salient-channel mixed precision** (top-k by weight max-abs → INT8): essentially
  flat, no knee. +0.20 b/w (5% channels to INT8) cuts INT4 error only 11.0%→10.7%.
  Per-group RTN already normalizes each group by its own max-abs, so high-magnitude
  channels are not concentrated error sources under unit-norm random X.

## Honest framing of this stage

This is **Stage-1: a matmul-fidelity proxy on random unit-norm inputs**. It is
**necessary, not sufficient**. It cannot see:
- error compounding across the 52 (23 MoE) layers — a 10% per-layer error is
  catastrophic when stacked;
- discrete router top-6 expert-selection flips;
- **activation-aware behavior** — the random-X batch has *no activation outliers*,
  which is precisely the regime where AWQ / LLM.int8-style outlier-channel and
  GPTQ / Hessian-aware error-feedback methods earn their keep. The salient-channel
  experiment used *weight* max-abs, not *activation* energy, so it rules out
  weight-magnitude saliency but can neither confirm nor refute activation-driven
  saliency. The strongest known levers for this exact 4-bit gap are therefore
  **structurally untestable under Stage-1** and remain open.

Survivors of Stage-1 escalate to **Stage-2 single-forward KL** on real prompts
(compress experts, run one offloaded forward, compare next-token distributions vs
BF16). Stage-2 is **blocked on the RAM issue** (63 GB model, 33.7 GB RAM, no CUDA).

## Verdict

- **INT8 per-group RTN is confirmed on the true weights** as a safe ~2x
  resident-VRAM floor (0.67% error, cos 0.99998, 32.2 GB). This is a real,
  bankable Regime-D win — the candidate's INT8 claim holds.
- **Sub-4-bit stays unusable under every structure-only lever tested.** Group size,
  non-uniform codebook, and weight-magnitude salient mixing each shave only a few
  absolute points; none reach INT4-class size at INT8-class fidelity. The 4-bit
  grid error (~10–12%) is the wall, and these levers do not breach it.
- **But the gap-closing methods that the field actually uses for this problem
  (activation-energy saliency / AWQ, Hessian-aware error-feedback rounding / GPTQ)
  require real activations and were not testable here** — they need cached real
  activations or a Stage-2 forward. So the sub-4-bit question is *not closed*, it is
  *blocked at the proxy boundary*. The data-free levers are exhausted and rejected;
  the activation-aware levers are untested.

This is the ambiguous middle: a confirmed INT8 win plus a sub-4-bit gap that the
cheap structural levers cannot close and the strong activation-aware levers cannot
yet be measured against. That maps to **Needs Deep Analysis**, not Rejected —
rejecting now would throw away INT8's validated win and prematurely kill the
activation-aware path that Stage-1 is simply blind to.

## Next Action

Cache real layer-1 expert input activations (a few hundred rows from a handful of
real prompts, captured during a single disk/CPU-offloaded forward, or a Stage-2
forward once a bigger-RAM/GPU box is available), then re-run the Stage-1 probe with
`X = cached real activations` instead of random unit-norm rows. This single change
is what unblocks the two untested levers — activation-energy salient mixing (AWQ)
and Hessian/error-feedback rounding (GPTQ) — which are the only remaining candidates
with a credible shot at INT4-class size (~18.5 GB) at INT8-class fidelity (≤1–2%).
If they still fail with real activations, downgrade the candidate to Rejected for
sub-4-bit and ship INT8 as the runtime-track floor.
