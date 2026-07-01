# Test 003 — INT4 power-up at larger scale + held-out perplexity + drift

Corroborates and extends test-002 (which used 41 prompts). This run: 77 items
(75 diverse prompts + 2 held-out Austen passages), 898 predicted positions, adds
held-out perplexity and a generation-drift list. Tool: tools/fused_eval_large.py →
tools/fused_large_summary.json. Peak RAM 10.9 GB.

## Results vs BF16

| Metric | INT8 | INT4 |
|---|---:|---:|
| corpus perplexity (75 prompts, BF16 7.99) | 7.997 (+0.1%) | 8.69 (**+8.8%**) |
| held-out perplexity (BF16 3.02) | 2.97 (−2%) | 3.44 (**+13.8%**) |
| KL mean / p99 / max | 0.0013 / 0.012 / 0.18 | 0.089 / **1.61** / 5.41 |
| top-1 agreement | 99.2% | **90.9%** |
| router top-6 overlap | 98.8% | 92.6% |
| KL by category (INT4) | — | code 0.133, instr 0.135, dialogue 0.117, reasoning 0.096, facts 0.064 |

Generation drift (teacher-forced argmax, ~40 tok/prompt): INT4 diverges 1–5 positions
per prompt, some semantic — recipe "two cups"→"two eggs"; "a faint"→"a light".

## Verdict — agrees with test-002, strengthens it
- **INT8 (2×, ~32 GB): capability-preserving end-to-end. Ships.** (+0.1% corpus /
  −2% held-out perplexity, 99.2% top-1, KL ~0.001.)
- **INT4 (3.4×, ~18.5 GB) plain per-group RTN: NOT capability-preserving.** Two
  independent powered evals (41 and 77 prompts) agree: +5–9% corpus / +14% held-out
  perplexity, ~91% top-1, heavy-tailed KL, visible generation drift. test-001's +0.5%
  was a small-sample/easy-prompt artifact. INT4-RTN is a fits-vs-doesn't-fit option
  with a real quality cost, not a clean second halving.

## Bottom line for the project
The validated runtime deliverable is **INT8 (2×)**. A clean 3.4× would require
activation-aware INT4 (properly-powered GPTQ with ~1e5 cached tokens / AWQ) proven
end-to-end at this bar (INT4 size, INT8-class KL≈1e-3); that remains untested at power.
