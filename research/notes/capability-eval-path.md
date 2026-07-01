# Capability-Eval Path (how to validate lossy runtime-track candidates)

Lossy candidates can't be judged by byte round-trip — they must be judged by
preserved *capability*. This note fixes how, given our constraints. Established
by the runtime-pivot probe (3 agents, real model on this machine).

## Constraint: full inference does NOT run here yet

- Model is 63.2 GB BF16; this machine has 33.7 GB RAM and no CUDA. A naive
  `from_pretrained` on CPU will thrash/OOM. **The blocker is RAM, not kernels** —
  `modeling_nemotron_h.py` has a pure-PyTorch `torch_forward` slow path that runs
  on CPU when the Triton/causal-conv1d fast path is unavailable, so missing
  `mamba-ssm`/`causal_conv1d` does NOT stop CPU inference; memory does.
- To get a real forward later: accelerate disk/CPU offload (slow, disk-bound,
  minutes/prompt), or a reduced/quantized variant, or a bigger-RAM / GPU box.

## Proxy ranking (cheapest → strongest)

1. **Per-tensor reconstruction error** — instant, no inference. Fast gate only;
   blind to which weight directions the model actually uses.
2. **Per-layer matmul output error on activations** — the workhorse. Runnable NOW
   with random right-shaped inputs; stronger with cached real activations.
   Measures error in the metric the next layer sees; weights directions by
   activation energy. Misses cross-layer error compounding and router top-6 flips.
3. **End-to-end next-token logit / KL divergence on real prompts** — the real
   behavior check (KL on the output distribution is the gold cheap-but-real
   signal). Blocked until the RAM issue is solved; few prompts = tiny sample.
4. **Tiny perplexity** — sanity scalar only; hides per-token catastrophes.

## Recommended harness (two stages)

- **Stage 1 — matmul-fidelity probe (build NOW, no inference, no RAM problem).**
  For sampled routed-expert matrices W: load via `safetensors.safe_open`, apply
  the candidate codec to get W', and on the same input batch X (random unit-norm
  + later cached real activations) compute relative output error ‖XW−XW′‖/‖XW‖,
  cosine, and — for router tensors — top-6 selection rank-correlation. Seconds per
  tensor on existing torch-CPU. **This is the validator the runtime track runs on
  until inference is possible.** Always report sample size (e.g. N=128 experts,
  layer L).
- **Stage 2 — single-forward KL (needs RAM blocker solved).** Compress experts,
  run one offloaded forward on a few real prompts, compare next-token
  distributions (KL) vs the BF16 model.

## Honest gap

Stage-1 output error is a proxy, not capability. It cannot see error compounding
across 52 layers or discrete router top-6 flips. Treat a Stage-1 pass as
*necessary, not sufficient*; escalate survivors to Stage 2 before any capability
claim. This is the same discipline as the lossless track's "no exact
reconstruction, no claim" — restated for lossy work as "no capability evidence,
no capability claim."
