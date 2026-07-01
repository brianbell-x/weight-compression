# Exp4 — Fair final superposition test (full-rank W1, d_model=128) — NEGATIVE

Fixed exp3's confound: W1 full-rank dense up-projection (every hidden unit reads all
input dirs); sharing moved to the OUTPUT side (W2 = D[d_out,M] @ S[M,H], M<H); top-k
sparse activation on H; d_model raised 64→128. Native-compute verified. Dense
re-baselined and still capacity-bound (243k→1.474, 334k→1.393, 504k→1.334).

## Result — the win crosses zero but is not real
Best v2: 90k budget, M=32, H=253 (1.44×), sp=1.0 → 2-seed mean gap **−0.032** vs dense
(exp3 was +0.076, so the confound was indeed the problem). BUT:
- **Does not scale**: gap is −0.032 at 90k, a seed-noise TIE at 180k (signs flip across
  seeds), and VANISHES at 350k (v2 trails by +0.024). A real capacity-density advantage
  grows with budget; this shrinks.
- **Not sparsity-driven**: best config has NO activation sparsity (sp=1.0); extreme
  sparsity (k/H=0.06) regresses in every config; no monotone "more sparse = better."
- **Cause = regularization, not superposition**: extra bias terms + wide-hidden low-rank
  output factorization help only when capacity is scarce (small budget).

## Conclusion — train-time weight density: NEGATIVE
Dense up-projection weights are ~optimal per-parameter at this scale. Packing more
effective features into a superposed pool read out sparsely does NOT buy
capability-per-parameter. Consistent with exp1–exp3: the only reliable density lever is
ACTIVATION sparsity as a runtime/compute saving (MoE), not train-time weight density.
This is the fair test, with the confound removed — the negative is clean.
