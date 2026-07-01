# Making a High-Capacity MoE Lighter — Final Synthesis

Target model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (31.6B params, 63.2 GB
BF16; a sparse MoE — 128 routed experts/MoE-layer, 6 active per token; hybrid
Mamba2 + MoE + attention). Constraint throughout: CPU-only box, 33.7 GB RAM, no GPU.

## The one-paragraph answer

You cannot make a trained model's **weights** carry more capability per parameter —
they are already dense and ~optimal. We showed this five independent ways (below).
What *does* make the model lighter is spending **fewer bits per weight**
(quantization) and touching **fewer weights per token** (sparsity / MoE). The
validated, shippable result is **INT8 expert quantization: a 2× runtime/VRAM
reduction (63 GB → ~32 GB) that is capability-preserving end-to-end** (perplexity
+0.1%, top-1 99.2%, KL ~0.001). A 3.4× INT4 reduction is *not* free — plain RTN INT4
costs ~5–14% perplexity at scale.

## Cost axes (the framing that reorganized everything)

Five separate axes — storage, load, resident VRAM, per-token bandwidth, compute.
Lossless compression only moves storage/load/resident-VRAM and never speeds decode
(it can slow it). What makes a model cheaper to *run* must keep weights narrow into a
fused matmul (never re-inflated to full width in VRAM). See
`notes/compression-vs-compute-payoff.md`.

## Track 1 — Post-hoc compression of the finished weights

| Idea | Result |
|---|---|
| Lossless BF16 plane-split + entropy code (0001) | ~32% exact, storage-only, capped (mantissa is high-entropy) |
| F32 control-tensor truncation (0002) | exact 50% but ~KB scale, immaterial |
| Cross-expert base+delta (0003) | dead — experts position-wise uncorrelated (|corr|~0.03) |
| Embedding vocab-tail dedup (0004) | dead — no untrained tail |
| Low-bit expert quant (0005) | INT8 safe; sub-4-bit fails every matmul-proxy lever |
| Precision-ladder / residual paging (0006) | dead — residual incompressible (a 2nd full tensor) |
| Low-rank / shared-basis density (0007) | dead — experts full-rank, independent subspaces |
| Streamed Stage-2 end-to-end eval (0008) | **INT8 confirmed 2×; INT4-RTN degraded at scale** |

**Why everything below INT8 failed:** the experts are statistically dense /
random-like — full-rank, high-entropy bits, no shared structure. There is no slack
to extract. Even AWQ's per-channel scaling collapses under fine per-group RTN.

**The methodological lesson (paid for twice):** matmul-fidelity proxies AND tiny
prompt pilots both mislead. The Stage-1 proxy was too pessimistic about INT4
(cross-layer error cancels); then an 8-prompt Stage-2 pilot was too *optimistic*
(+0.5%). Only a diverse, powered end-to-end eval (41 + 77 prompts) gave the truth.

## Track 2 — Train-time density (can weights be built "tighter"?)

Four CPU experiments (capability-per-parameter on a char-LM), `research/traintime/`:
- exp1: structured families ≈ dense at equal params; the only "win" (shared-dict)
  was a redundancy artifact of an easy task.
- exp2: on a capacity-bound task the sharing win COLLAPSES; superposition is real in
  a toy model **but gated by activation sparsity** (sparse inputs → d dims carry
  10–16× more features).
- exp3: a sparse-superposed FFN trails dense (confounded by a rank-K bottleneck).
- exp4 (fair test, confound removed): the only crossing below dense is a small-budget
  **regularization** artifact that vanishes with budget; not sparsity-driven.

**Conclusion: NEGATIVE.** Dense weights are ~optimal per-parameter. No restructuring,
sharing, or superposition buys capability-per-parameter. The only reliable density
lever is **activation sparsity** — a runtime/compute win, which MoE already provides.

## The unifying finding

Capability *is* the weights; you don't compress it into them. Models get lighter by
(1) fewer bits per weight, (2) fewer weights active per token. Superposition lives in
*representations under sparsity*, not in restructured weight matrices — which is
exactly why sparse-MoE activation, not weight-sharing, is the lever that works.

## Deliverables produced

- **Validated: INT8 expert quantization, 2× runtime/VRAM, capability-preserving.**
- Reusable infra (CPU, no GPU): `tools/capture_activations.py` (real activation
  capture, ~3.5 GB peak) and `tools/streamed_forward.py` + `fused_eval*.py`
  (full-model streamed forward + BF16/INT8/INT4 end-to-end eval, ~11 GB peak,
  checkpointed). These turned a RAM-blocked model into something testable on this box.
- A complete, honest findings ledger and per-experiment results.

## What would move the needle next (needs more than this CPU box)

1. **Activation-aware INT4 at power** (GPTQ with ~1e5 calibration tokens / AWQ),
   proven end-to-end — the only remaining shot at a clean 3.4×. Infeasible here
   (calibration capture via the slow streamed forward would take days); needs a GPU.
2. **Finer-grained sparse MoE** (more, smaller experts, fewer active) — pushes the
   one lever that actually works (activation sparsity). An architecture/training
   direction, not a compression-of-existing-weights one.
