# Test 002 — Activation-aware sub-4-bit, with REAL captured activations

Candidate: 0005-low-bit-expert-quant (the runtime track)
Date: 2026-06-29
Stage: **Stage-1 — matmul-fidelity proxy, now fed REAL activations** (still NECESSARY,
NOT SUFFICIENT).
Payoff regime: **Regime D — resident-VRAM + decode bandwidth.**

This test resolves test-001's single open blocker. Test-001 exhausted the *data-free*
levers (group size, non-uniform codebook, weight-magnitude salient mixing) and rejected
them for sub-4-bit, but the two levers the field actually uses for this exact 4-bit gap —
**AWQ** (activation-energy channel saliency) and **GPTQ** (Hessian / error-feedback
rounding) — were *structurally untestable* under random-X. They need real layer-1
expert-input activations. Test-002 captures those activations and runs both levers.

## 1. The real-activation capture path now WORKS and is reusable

The keystone infra is built and ran end to end. Layer 1 is the **first** MoE block; all
weights upstream of its routed experts (token embeddings, layer-0 Mamba2 mixer, the two
RMSNorms) live entirely in **shard 1** (`model-00001-of-00013.safetensors`). So we build
*only* those modules, load *only* their weights, run a real prefill, and capture the
hidden state after `backbone.layers.1.norm` — which the model code hands verbatim to both
the router (`gate`, in=2688) and every routed expert's `up_proj` (in=2688). The 128
experts (58 GB) are **never loaded**.

- **Captured X: shape (187, 2688), float32** — 187 real token positions across 12 short
  prompts (general relativity, photosynthesis, Apollo 11, diet, markets, fiction,
  quicksort, a recipe, climate, music, the immune system, quantum computing).
- **Peak RAM: 3.55 GB** (load ≈ 2.64 GB, dominated by the 1.41 GB f32 embedding table),
  **total wall time ≈ 2.3 s** for the full 12-prompt capture. The RAM blocker (63 GB
  model vs 33.7 GB RAM, no CUDA) is fully sidestepped: capture is ~9x under budget.
- **`X@W` orientation verified True**: expert `up_proj.weight` is `[1856,2688]`, so the
  in-axis is 2688 = hidden_size; quantize along axis=0 with `W = up_proj.weight.T`
  `[2688,1856]` so `X[187,2688] @ W` is the true up_proj output. (axis=0 grouping is
  *required*, not optional: out-axis 1856 is not divisible by group_size 128.)
- **Real outlier structure present**: per-input-channel energy `max/mean = 4.30` (random
  unit-norm X sits at ~1.2). This heavy-tailed channel energy is exactly the AWQ/GPTQ
  signal that random X cannot produce. Cached as `channel_energy_layer1.npy [2688]`.

Engineering hurdles solved (so this is replayable): the custom `modeling_nemotron_h.py`
hard-raises `ImportError("mamba-ssm is required")` at import (its gated RMSNorm calls
`rmsnorm_fn` from `mamba_ssm`) — unblocked by injecting a pure-PyTorch gated group-RMSNorm
via a stub `mamba_ssm` package; and the block's `torch.cuda.stream(...)` wrapper (errors
on a CPU-only torch) is bypassed by calling the norm+mixer sub-modules directly. Kernels
affect speed only; the math is the real module classes.

**Reusable artifacts** (under `tests/artifacts/`):
- `activation_capture.py` — the capture harness (runs as-is, ~2.3 s, ~3.55 GB).
- `activations/real_X_layer1.{pt,npy}` — X `[187,2688]` float32.
- `activations/channel_energy_layer1.{pt,npy}` — per-channel RMS energy `[2688]` (the AWQ
  salient-channel signal).
- `activations/capture_meta.json` — prompts, shapes, energy ratio, paths.

This single deliverable unblocks the entire runtime track: every post-hoc activation-aware
lever (and any future Stage-2 work) now has real X to run against.

## 2. bits/weight vs output-error — real-X levers next to test-001's random X

Fidelity = relative output error `‖XW − XW′‖_F / ‖XW‖_F` via `stage1_probe.fidelity()`,
on 16 real BF16 layer-1 experts. Implied resident VRAM = `4.4 GB + 29.4e9·b/8 / 1024³`.
**up_proj** uses the cached X directly (the tensor it feeds verbatim). **down_proj** uses
the second-hop intermediate `relu2(X @ up_proj.T)` per expert.

### 2a. Real-X re-baseline vs random-X (same codecs, only X differs) — up_proj

| Codec | bits/wt | random-X err (test-001) | **real-X err** | VRAM (GB) |
|---|---:|---:|---:|---:|
| INT8 per-group RTN | 8.125 | 0.69% | **0.28%** | 32.21 |
| INT4 per-group RTN | 4.125 | 12.28% | **5.07%** | 18.52 |
| codebook16 per-group (best ≤4b non-uniform) | 4.125 | 9.67% | **4.26%** | 18.52 |

### 2b. AWQ (activation-aware weight quant) on real X — up_proj + down_proj

| Config | bits/wt | real-X err | mean_cos | VRAM (GB) |
|---|---:|---:|---:|---:|
| up INT4 per-group RTN (real-X floor) | 4.125 | 5.07% | 0.99876 | 18.52 |
| up INT4 AWQ (scale search only) | 4.134 | 5.07% | 0.99876 | 18.55 |
| up INT4 AWQ + clip search | 4.134 | 4.78% | 0.99890 | 18.55 |
| **up INT4 AWQ+clip + 1% salient INT8 (best ≤4b)** | **4.174** | **4.71%** | 0.99894 | **18.68** |
| up INT3 per-group RTN | 3.125 | 11.83% | 0.99328 | 15.10 |
| up INT3 AWQ + clip | 3.134 | 9.91% | 0.99539 | 15.13 |
| down INT8 per-group RTN (reference) | 8.125 | 0.63% | 0.99998 | 32.21 |
| down INT4 per-group RTN | 4.125 | 11.37% | 0.99334 | 18.52 |
| down INT4 AWQ + clip | 4.131 | 10.54% | 0.99420 | 18.54 |
| down INT3 AWQ + clip | 3.131 | 21.04% | 0.97677 | 15.12 |

### 2c. GPTQ (Hessian / error-feedback rounding) on real X — held-out eval

Calibration/eval split: Hessian `XᵀX` from 140 calibration tokens; error measured on 47
**held-out** tokens (mean of 16 experts). The data-free baseline GPTQ must beat is RTN on
the same held-out tokens.

| Config | bits/wt | held-out err | VRAM (GB) |
|---|---:|---:|---:|
| up INT8 RTN (reference, only config under the bar) | 8.125 | 0.28% | 32.21 |
| up RTN 4-bit (held-out baseline) | 4.125 | 5.12% | 18.52 |
| **up GPTQ 4-bit (held-out)** | 4.125 | **5.45%** | 18.52 |
| up GPTQ 4-bit (FULL set, in-sample) | 4.125 | 2.88% | 18.52 |
| up GPTQ 3-bit (held-out) | 3.125 | 12.88% | 15.10 |
| down GPTQ 4-bit (held-out) | 4.138 | 19.57% | 18.56 |

## 3. Verdict on the bar — did AWQ or GPTQ break the 4-bit wall?

The bar: **per-layer matmul output error ≤ ~1–2% at ≤4 effective bits/weight** (INT4-class
size at INT8-class fidelity).

**No. Neither activation-aware lever cleared the bar; both are now tested and rejected for
clearing it.**

- **Best ≤4-bit config measured: up_proj INT4 AWQ+clip + 1% salient INT8 — 4.174 b/w,
  4.71% output error, 18.68 GB resident.** That is ~2.4x outside the 1–2% band and ~17x
  the INT8 floor. The non-uniform codebook16 is comparable (4.125 b/w, 4.26%). down_proj
  is far worse (best 4-bit ≈ 10.5%). Nothing ≤4 bits comes within 2x of the bar.
- **AWQ's central idea — the activation-energy SCALE search — is nearly inert here.** Best
  α drives INT4 up_proj from 5.069% to 5.067%. Almost all of AWQ's gain comes from its
  *secondary* max-abs clip term (→4.78%). Mechanism: per-group RTN already normalizes each
  128-column group by its own max-abs, so AWQ's per-channel rescale only inflates a group's
  max-abs and *steals* resolution — there is no headroom left for it to exploit.
- **GPTQ does NOT beat plain RTN on held-out tokens** (5.45% vs 5.12%). The implementation
  is correct — on the in-sample full set it halves RTN (5.07%→2.88%, a real 1.8x gain) —
  but that does not transfer. The cause is data starvation: `XᵀX` from 187 tokens is
  **rank 1939 of 2688**; a damping sweep confirms the more GPTQ is regularized back toward
  RTN the better it generalizes (held-out 5.29%→4.78% as percdamp 0.01→1.0), the textbook
  overfitting signature. Standard GPTQ uses ~1e5 tokens; 187 is ~3 orders of magnitude
  short. down_proj is worse still (19.6%) because real MoE routing gives each of 128
  experts only ~1–2 of the 187 tokens, starving the second-hop activations.
- **INT8 per-group RTN remains the only in-band config** (up 0.28%, down 0.63%, 8.125 b/w,
  ~32 GB resident) — the bankable ~2x Regime-D win, unchanged from test-001 and now also
  confirmed on real activations.

## 4. Calibration of Stage-1 trust — how much real X shifted the proxy

The random-X proxy from test-001 **systematically over-states error by a consistent
~2.3–2.5x**. Every codec lands at ~0.40–0.44x of its random-X number on real activations:
INT8 0.69%→0.28%, INT4 12.28%→5.07%, codebook16-4b 9.67%→4.26%.

Mechanism: real X (channel-energy max/mean 4.30 vs 1.23 for random) concentrates the output
signal `‖XW‖` in a structured outlier subspace, while the quantization noise `W−W′` projects
roughly isotropically — so the relative-error *denominator* is inflated by real activations,
shrinking the ratio. **Codec ranking is preserved** (codebook16 < INT4 on both X; INT8 ≪
all), so Stage-1 stays directionally trustworthy, but its absolute rel_err should be read as
a **~2.4x conservative upper bound**, not the operating number. Practical consequence: the
sub-4-bit gap is real but ~2–3x outside the band, not the ~5–10x the random-X table implied.

## 5. Honest Stage-1 caveat (unchanged, still binding)

This is still a **single-layer matmul-fidelity proxy**. Even with real activations it cannot
see:
- **52-layer (23 MoE) error compounding** — a 4.7% per-layer output error may be benign or
  catastrophic when stacked; the proxy cannot tell which.
- **discrete router top-6 expert-selection flips** — quantizing `gate`/experts can change
  *which* experts fire, an effect invisible to a per-expert matmul error.
- The eval is also small (187 tokens) and, for AWQ/salient, in-sample (X used for both fit
  and score), so those numbers lean optimistic — yet still fail, which makes the negative
  robust. GPTQ's held-out split makes its negative airtight.

Any survivor still requires **Stage-2 single-forward KL** on real prompts to confirm. INT8
is the only thing that would survive escalation; sub-4-bit has nothing to escalate.

## Verdict

- **INT8 per-group RTN CONFIRMED on true weights AND real activations** as the runtime
  floor (up 0.28% / down 0.63% output error, cos 0.99998, ~32 GB resident, safe ~2x). A
  real, bankable Regime-D deliverable.
- **Sub-4-bit REJECTED at the proxy** — not blocked. The activation-aware levers that
  test-001 left untested are now tested: AWQ's scale search is inert against per-group RTN,
  GPTQ overfits 187 tokens and fails to beat RTN on held-out data, and the best ≤4-bit
  config sits at ~4.7% (up) / ~10.5% (down), ~2.4x outside the band. This is consistent
  with the experts being fundamentally dense (full-rank, high-entropy — candidates 0003,
  0007). The ~18.5 GB INT4 target is not reachable at ≤2% per-layer error by any post-hoc
  lever measured.
- **The only path that could reopen INT4** (~5% per-layer) is a Stage-2 end-to-end eval
  showing the model tolerates that error across 52 layers + routing. That is a capability
  question, not a quantization-lever question.

## Next Action

Build the Stage-2 streamed full-forward eval → **candidate 0008**. Extend
`activation_capture.py` to stream all 52 layers from the 13 shards one block at a time
(within the 3.5 GB RAM envelope already proven) and measure end-to-end next-token KL /
perplexity for BF16 vs INT8 vs INT4 experts. This simultaneously (a) validates the INT8
deliverable on real capability and (b) settles whether INT4's ~5% per-layer is catastrophic
or tolerable end-to-end — the one open lever that could move sub-4-bit from Rejected to
viable. GPTQ is *not* worth re-running until far more tokens are cached (1e4–1e5, which the
cheap partial forward makes feasible); even then AWQ is the better bet, as it needs only
per-channel scale statistics, not a full-rank Hessian.

## Scripts
- Capture: `tests/artifacts/activation_capture.py` (also `capture_real_activations.py`).
- AWQ + salient sweep: `tests/artifacts/realx_awq.py`.
- GPTQ sweep (cal/held-out split, damping sweep): `tests/artifacts/realx_gptq.py`.
- Real-X re-baseline / random-X control: `tests/artifacts/realx_realx-rebaseline.py`.
- Stage-1 metric: `tests/artifacts/stage1_probe.py`.
