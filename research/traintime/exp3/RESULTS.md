# Exp3 — Sparsity-gated superposition FFN vs dense (capacity-bound)

Test of the exp2 synthesis: does a WIDE-hidden FFN with top-k SPARSE activation,
built from a SHARED ATOM POOL, beat dense capability-per-param? Reuses the exp2
capacity-bound task (dense val: 134k→1.683 ... 624k→1.558). Native-compute confirmed.

## Result — does NOT beat dense, but the test was confounded
- Best sparse_superpose: H=542 (1.54x), K=16 atoms, sparsity k/H=0.12 → val 1.759 vs
  dense 1.683 at matched 134k params (**gap +0.076, worse**). Every config trailed dense.
- **Sparsity helps directionally** (mechanism real, partial): k/H 1.0→0.12 improves val
  (1.54x: 1.819→1.759; 4.93x: 1.947→1.903). Optimum ~12%; 3% over-sparse regresses.
- **Headline prediction FAILED — and we know why (design confound):** input-side atom
  sharing makes W1 = C@A *rank-K* (16 input directions). Widening H just adds units
  trapped in a 16-dim subspace, and a low-rank W2 compounds it. So widening H made it
  monotonically WORSE — a bottleneck artifact, not a verdict on superposition.

## Interpretation
Not a clean rejection. The sparsity mechanism showed the predicted trend, but the
implementation imposed a rank-K input bottleneck that confounds the result. The fair
test (exp4): keep W1 FULL-RANK, put the sharing/sparsity on a WIDE hidden→output
dictionary, and use a larger d_model (64 is likely too small for superposition to pay —
superposition needs dimensional room for near-orthogonal features).

## Accumulating meta-read (exp1+exp2+exp3)
No weight-structure/superposition scheme has beaten dense capability-per-param at this
scale. The only lever that reliably helps is ACTIVATION SPARSITY — which is a
runtime/compute lever (fewer active params per token), i.e. exactly what MoE already
does. exp4 gives superposition its fairest shot; if it also fails cleanly, the
train-time density conclusion is: dense is ~optimal per-param at small scale, and the
real win is activation sparsity (MoE), not weight density.
