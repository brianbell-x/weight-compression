# Exp1 — Capability-per-parameter: dense vs structured layers (char-LM, CPU)

Task: char-level LM on a deterministic 36k-char trigram-Markov corpus, vocab 49,
fixed 2-block transformer (d_model 96), only the FFN matrices swapped by family.
Metric: val loss (nats) vs total params. Dense ref: 290,304 params → val 1.1718.
Harness: research/traintime/exp1/{task,model,layers,train}.py (seeded, ~1 min/run).

## Val loss vs total params

| params | dense | lowrank | block_monarch | shared_dict |
|---:|---:|---:|---:|---:|
| ~100-150k | 1.184 | 1.253 | 1.235 | 1.188 (K=1) |
| ~190k | — | — | — | 1.179 (K=2) |
| ~210-240k | 1.176 | 1.177 | 1.187 | 1.182 |
| ~290k | 1.172 | 1.178 | 1.183 | 1.181 |
| ~390-450k | 1.175 | 1.179 | 1.179 | 1.185 |

## Findings

1. **No free lunch at equal params.** Low-rank, Monarch, and shared-dict are all
   marginally WORSE than dense at matched param count. Clever factorization of a
   densely-trained matrix does not beat dense capability-per-param.
2. **The only per-param win is parameter SHARING, not structure.** shared_dict at
   low K (one shared atom-dictionary across the 4 FFN sites) recovers ~99% of dense
   val at ~0.5-0.66x params (K=1 → 1.188 at 0.48x; K=2 → 1.179 at 0.66x). This is
   the weight-tying effect: on this small task the 4 FFN sites are redundant, so
   tying them is nearly free. It is a STORAGE win — fails the native-compute gate
   (the matrix is rebuilt each forward; native_compute_ok=false).
3. **Design limitation — task is capacity-saturated.** Dense val is nearly FLAT
   (~1.17-1.19) across the whole param sweep, so the task does not stress capacity
   and cannot strongly discriminate families. Conclusions are weak until the task
   is capacity-bound.

## Through-line

Density/compression wins come from REDUNDANCY. Post-hoc, the experts were dense
(no slack — full-rank, high-entropy, no shared basis). Train-time, the only
per-param win appeared exactly where the small task left the weights redundant
(shared_dict low-K). The real question: does any per-param win survive when the
model actually NEEDS its capacity?

## Next (exp2)

Redesign the task to be CAPACITY-BOUND (dense val keeps dropping with params over
the swept range), then re-test dense vs shared_dict (the only contender) and add a
SUPERPOSITION probe (toy-model-of-superposition: represent more features than dims
with sparse activation — directly measures occupancy density). Decisive read:
in the capacity-bound regime, does anything beat dense capability-per-param?
