# Test 001 — Streamed Full-Forward Stage-2 Eval

Status: **Passed True Weights**
Date: 2026-06-29
Target: true weights, streamed (all 52 layers), 8 short real prompts.

## What was built (reusable Stage-2 tool)

- `tools/streamed_forward.py` — streams the 63 GB model layer-by-layer: load one
  layer's weights from its shard, run all prompts through it, free it, repeat.
  Handles Mamba2 / MoE-with-routing / attention (layers 5,12,19,26,33,42) + norm_f
  + lm_head. **Peak RAM ~11 GB** (vs 33.7 available); the floor is one fp32 MoE
  layer (~7 GB). Forces `eager` attention (the snapshot's SDPA path crashes at
  layer 5: num_heads*head_dim = 4096 != hidden_size = 2688).
- `tools/fused_eval.py` — runs BF16 / INT8-experts / INT4-experts in ONE streamed
  pass (each layer's weights read from disk once, three hidden states carried
  forward; per-condition expert quant applied to a snapshot of the experts).
  Checkpoints after every layer (`fused_ckpt.pt`) so a kill resumes instead of
  restarting — required because a full cold pass (~13 min, MoE layers load
  25–40 s each off disk) exceeds the ~10-min background limit.

## BF16 sanity — PASS (precondition for trusting quant numbers)

High-margin correct factual completions, finite logits, sane perplexity (geomean
11.30, nowhere near random-chance ~131072):
`The capital of France is`→` Paris`; `Water is made of hydrogen and`→` oxygen`;
`The sun rises in the`→` east`; `Roses are red, violets are`→` blue`. A mis-wired
stream cannot produce these, so the quant comparison below is trustworthy.

## Results (8 prompts, next-token distribution vs the BF16 reference)

| Condition | mean KL(BF16‖·) | perplexity (geomean) | top-1 agreement | router top-6 overlap |
|---|---|---|---|---|
| BF16 (ref)    | —        | 11.30 | —     | —      |
| INT8 experts  | 0.00035  | 11.30 | 100%  | 99.4%  |
| INT4 experts  | 0.076    | 11.36 | 96%   | 92.5%  |

Per-group RTN (group 128), routed experts only (gate / shared expert / norms left
fp32). Implied full-model resident VRAM: INT8 ≈ 32 GB, INT4 ≈ 18.5 GB (vs 63 GB BF16).

## Verdict on the two pending questions

1. **INT8 deliverable confirmed on real behavior.** KL ≈ 3e-4, perplexity
   unchanged to 3 sig figs, 100% top-1 agreement, 99.4% router overlap. The
   ~2× resident-VRAM win (Regime D) is real end-to-end, not just proxy-good. This
   is the bankable runtime result and can ship as the floor.

2. **INT4's per-layer error does NOT compound catastrophically — it largely
   washes out.** Despite ~5% per-layer matmul error (0005 Stage-1), end-to-end
   perplexity moves only 11.30→11.36 (+0.5%), KL is small (0.076), top-1 holds at
   96%, router top-6 overlap 92.5%. The Stage-1 matmul proxy was *too pessimistic*
   about cross-layer compounding. **This reopens the ~18.5 GB INT4 target** that
   Stage-1 alone had rejected.

## Honest limits

- Small sample: 8 short prompts, ~50 predicted positions. KL 0.076 and the 4%
  top-1 flips are real but under-powered for a capability claim at scale — this
  shows INT4 is *not broken*, not that it is *production-equivalent*.
- KL/perplexity on next-token distributions is the cheap-but-real signal; it does
  not test multi-token generation drift, long-context, or task accuracy.
- INT4 here is plain per-group RTN. Activation-aware methods (AWQ/GPTQ, 0005
  test-002) could push the same ~18.5 GB point to lower KL, or enable sub-4-bit.

## Next Action
Power up the Stage-2 INT4 evidence before committing: re-run `fused_eval.py` on a
larger, more diverse prompt set (e.g. 100–200 prompts incl. multi-sentence and a
short generation/΄continuation task) and report KL + top-1 + a real perplexity on
held-out text. If INT4 KL stays small at scale, INT4 experts (~18.5 GB, a second
halving) become the headline runtime deliverable alongside the confirmed INT8 floor.
