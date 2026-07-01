# Exp2 — Capacity-bound capability-per-parameter + superposition probe

Fixes exp1's flaw (capacity-saturated task). New task: real Austen prose
(Gutenberg #1342), d_model 64 (so the FFN is the binding constraint), seq 96.
Verified capacity-bound: dense val drops monotonically across the sweep.

## Dense baseline (capacity-bound — val now discriminates)
| params | val_loss |
|---:|---:|
| 134k | 1.683 |
| 224k | 1.618 |
| 394k | 1.584 |
| 624k | 1.558 |
Spread 0.126 nats (exp1 was ~0.01 — non-discriminating).

## Result 1 — shared_dict win COLLAPSES under capacity pressure
At matched params, shared_dict is strictly WORSE than dense everywhere:
- 134k → 1.735 vs dense 1.683 (−0.052)
- 224k → 1.673 vs dense 1.618 (−0.055)
Mechanism: with only 4 FFN sites, sharing has no upside once each site needs
independent capacity. **K=4 is expressively equivalent to dense yet trains ~0.05
nats worse at equal params → the coupled parameterization hurts OPTIMIZATION, not
just capacity.** exp1's low-K win was a redundancy artifact of the easy task.
→ Structural weight-sharing does NOT beat dense per-parameter. (storage-only anyway;
forward materializes the matrix.)

## Result 2 — superposition (occupancy density) is REAL, gated by SPARSITY
Toy model of superposition (tied autoencoder, F features into d dims, ReLU readout):
- Dense inputs (p≈0.3): d dims carry ~d features; packing breaks at F/d ≈ 1–2.
- Sparse inputs (p≈0.01): d dims faithfully carry **~10–16× more features than d**.
- Faithful overcompleteness scales ~1/density.
Occupancy density is real — but ONLY when activations are sparse (rare interference).

## Synthesis (both tracks agree)
The weights are dense and incompressible; exploitable structure lives in the
ACTIVATIONS — their sparsity. Density that works exploits activation sparsity, not
weight redundancy. This is why: post-hoc weight compression capped (dense weights),
AWQ got only slight traction (a little activation structure), structural sharing
fails (exp2), and superposition succeeds exactly in the sparse regime.

Connection to the real model: Nemotron is already a sparse MoE (6/128 active) —
the regime where superposition works. The "fundamentally different weights" win, if
it exists, is **sparsity-gated superposition**: pack more effective capacity into a
shared/superposed parameter pool read out SPARSELY.

## Next (exp3)
Decisive test of the synthesis: an FFN with a WIDE hidden layer (many features) that
is (a) read out with top-k SPARSE activation and (b) built from a shared atom pool
(superposed weights) — does it beat dense capability-per-param BECAUSE sparsity makes
the superposed weights non-interfering? This is the train-time version of "finer
sparse experts sharing a superposed parameter pool," mapping directly to the MoE.
