# Test 002 — Hardened Stage-2 eval (41 diverse prompts)

Status follow-up to test-001. Date: 2026-06-30.
Purpose: the test-001 INT4 verdict rested on 8 short prompts (~50 positions). This
run powers it up to 41 diverse prompts (factual, reasoning, multi-sentence prose,
code, instructions; ~hundreds of predicted positions) using the same fused,
checkpointed streamed forward (`tools/fused_eval.py`, prompts in
`tools/prompts_eval.txt`, ran across 5 background windows resuming from
`fused_large_ckpt.pt`).

## Results vs BF16 reference (geomean ppl 12.45)

| Condition | mean KL | perplexity (geomean) | top-1 agreement | router top-6 overlap |
|---|---|---|---|---|
| INT8 experts | 0.00082 | 12.42 (−0.3%) | 99.71% | 98.9% |
| INT4 experts | 0.0888  | 13.09 (+5.1%) | 91.76% | 92.8% |

Per-group RTN (group 128), routed experts only. Peak RAM 10.6 GB.

## What changed vs test-001 (8 prompts)

| | 8 prompts | 41 prompts |
|---|---|---|
| INT8 KL / ppl / top-1 | 0.0003 / +0% / 100% | 0.0008 / −0.3% / 99.7% |
| INT4 KL / ppl / top-1 | 0.076 / +0.5% / 96% | 0.089 / **+5.1%** / **91.8%** |

INT8 is unchanged — **rock-solid end-to-end** (KL≈0, perplexity flat, ~99.7% top-1,
98.9% router overlap). The bankable ~2× VRAM floor (63 GB → ~32 GB) is confirmed
on real behavior across diverse text.

INT4 is **worse than the small sample suggested**: perplexity rises +5.1% (not the
lucky +0.5%), top-1 next-token flips on ~8% of positions, with heavy-tailed
per-prompt KL (several prompts 0.2–0.57). The small sample was optimistic.

## Verdict

- **INT8: confirmed, capability-safe, ships as the runtime floor.**
- **INT4 (plain RTN, ~18.5 GB): functional but degraded — NOT a free second
  halving.** The model is not broken (perplexity 12.45→13.09, still coherent), but
  +5.1% perplexity and 8% top-1 flips is a real quality cost, not capability-
  preserving in the strong sense. Plain INT4 RTN is a *fits-vs-doesn't-fit* option
  (take it only when ~32 GB won't fit but ~18.5 GB will, accepting the loss), not a
  clean win. Closing the gap to INT8-class behavior at ~4 bits needs activation-
  aware quant (properly powered GPTQ with ~1e5 tokens, or AWQ on salient channels)
  — the 0005 test-002 thread, which was data-starved at 187 tokens.

## Next Action
If a sub-32 GB deployment is a real target, run a properly-powered GPTQ (cache ~1e5
tokens via the streamed forward, build a full-rank Hessian) and re-measure INT4
end-to-end here; the bar is INT4 size at INT8-class KL (≈1e-3) / perplexity. Else
ship INT8 as the deliverable and treat plain INT4 as the documented degraded option.
