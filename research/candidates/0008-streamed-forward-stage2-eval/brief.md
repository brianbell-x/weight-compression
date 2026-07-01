# Candidate: Streamed Full-Forward for Stage-2 End-to-End Eval

## Claim
A layer-by-layer streamed forward (load one layer's weights from the 13 shards,
compute, free, repeat) can run a full forward of this 63 GB model within 33.7 GB
RAM, enabling real end-to-end capability measurement (next-token KL / perplexity)
that the Stage-1 matmul proxy cannot provide.

## Why It Might Work
`tools/capture_activations.py` already proved a partial forward (embeddings +
layers 0-1) runs on CPU within a few GB by materializing only the needed modules.
Extending it to stream ALL 52 layers — keep only the running hidden state
([tokens, 2688], tiny) plus the current layer's weights resident, loading each
layer's tensors from its shard on demand and freeing them after — keeps peak RAM at
~(largest single layer + embeddings) ≈ a few GB, far under 33.7 GB. It is slow
(disk-bound, minutes/prompt) but feasible, and we only need a handful of prompts.

This is the keystone for making CAPABILITY claims instead of proxy claims, which is
the project's standard for the lossy track. It unblocks two pending questions:
- Validate the INT8 deliverable ([[0005-low-bit-expert-quant]]) on real behavior.
- Settle whether INT4's ~5% per-layer proxy error is actually catastrophic
  end-to-end, or tolerable (which would reopen the ~18.5 GB target).

## Tensor Group
Whole model (streamed): embeddings, all 52 layers (Mamba/MoE/attention), norm_f,
lm_head. Reuses the layer-0/1 logic already working in capture_activations.py;
adds attention layers (5,12,19,26,33,42) and the remaining Mamba/MoE layers.

## Measurement
1. Stream a full BF16 forward on ~8 short real prompts; record next-token logits.
   Verify sanity (coherent top-1 tokens, finite, reasonable perplexity).
2. Re-run with experts quantized in-flight: INT8, then INT4 (per-group RTN).
3. Report per-prompt next-token KL(BF16 ‖ quantized) and perplexity for INT8 and
   INT4 vs BF16. Optionally top-1 agreement and router top-6 overlap.

## Promising Result
- INT8 KL ≈ 0 / perplexity ≈ unchanged → confirms the INT8 deliverable preserves
  capability end-to-end (the bankable win is real, not just proxy-good).
- If INT4 KL is also small / perplexity barely moves despite ~5% per-layer proxy
  error → the proxy was too pessimistic and the ~18.5 GB INT4 point is back on the
  table (a second halving). If INT4 KL is large → sub-4-bit is closed for real and
  INT8 ships as the floor.

## Test Target
True weights, streamed (synthetic has no trained behavior). Build/verify the stream
loop on a BF16 pass first (must reproduce sane next-token output) before trusting
any quantized comparison.

## Status
Passed True Weights