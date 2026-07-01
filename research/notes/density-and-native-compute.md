# Steering Note: Density and Native Compute (the "fit more per slot" direction)

*Status: research direction, exploratory. Captures a line of questions worth real
work. Not yet at the testability standard — this note is the seed.*

## The shift in framing

Most compression asks "how do we make this smaller and then rebuild it?" This
direction asks a different question: **can each stored unit carry more of the
model's function, so the model is simply smaller by construction — and runs in
that tighter form without ever being unpacked?**

The analogy that started this: the difference between one person per room and
five people comfortably sharing one room. Not cramming (destructive), but genuine
shared occupancy — the same space doing more work. Applied to weights: not "zip
the file," but "let each parameter (or dimension, or byte) hold more learned
behavior at once."

## Three axes of "smaller" — keep them separate

| Axis | What shrinks | Example | Native compute? |
| --- | --- | --- | --- |
| **Bit-width density** | bits per number | INT8/INT4 quant (candidate 0005) | yes (fused dequant) |
| **Structural density** | *number* of parameters needed to express the same function | factorized / structured weight matrices | yes — math runs on the factors |
| **Occupancy density (superposition)** | how many distinct functions share the same parameters | features sharing directions; tied/shared bases | yes — never separated |

Candidate 0005 works the first axis. This note is about the **second and third** —
which are orthogonal to bit-width and can compose with it.

## The questions, reworded as research questions

1. **Capability per byte, not bytes.** Can we raise how much model capability each
   stored unit carries, instead of only cutting how many units there are? The
   target metric is *function-preserved per parameter*, not file size.

2. **Shared occupancy without interference.** Trained networks already pack more
   features than they have neurons by letting features share dimensions
   (*superposition*). Can we deliberately design weights so multiple pieces of
   learned behavior coexist in the same parameters without destructive collision —
   five comfortably in one room, not five crammed in?

3. **Native form, no unpack step.** Can the tighter representation be the model's
   *actual* form — the forward pass computes directly on it — so there is no
   decode, no re-inflation to a bigger form before the math? The model doesn't get
   compressed; it just *is* smaller and works that way.

4. **A fundamentally different weight shape.** Could the weights take a different
   structure entirely — fewer, denser parameters arranged differently — that
   expresses the same function? The matrix the math sees need not be the object we
   store, as long as the math runs on the stored object directly.

The unifying question: **is there a representation where a high-capacity model is
small-by-construction and runs in that form, rather than a big model we
compress and decompress?**

## What this could look like for weights (directions, not commitments)

These are *families* to explore, chosen because the math runs on the compact form
directly — nothing is unpacked into a full matrix first:

- **Structured matrices computed natively.** A weight matrix expressed as a
  product/sum of small structured factors (low-rank, block, Kronecker, butterfly,
  circulant-like). The matmul runs as a sequence of small ops on the factors; the
  full matrix is never materialized. Fewer stored numbers, same linear map (or a
  close one). This is structural density.
- **Shared dictionary / atoms across experts.** The 128 experts share one learned
  set of building-block vectors ("atoms"); each expert is a sparse/cheap
  combination of atoms. The earlier finding (experts share a *distribution* but
  not position-wise values) means a naive shared base failed — but a *learned*
  shared dictionary the experts are reconstructed from is a different, untested
  question. Atoms resident once; per-expert combination coefficients are tiny.
- **Cross-layer / cross-expert parameter reuse.** Reusing the same parameters in
  more than one place (more function per stored param), tolerated by the network
  because the residual stream re-contextualizes them.
- **Deliberate superposition.** Encode more directions of behavior than there are
  dimensions, accepting controlled overlap the way trained nets already do, and
  read them out with the model's own nonlinearity — no separation step.

## The discipline (so this doesn't become hand-waving)

Any idea here must pass three gates, in order:

1. **Native-compute gate.** Does the forward pass run *directly* on the compact
   form, with no unpack-to-full-matrix step? If it must be expanded to a full
   matrix in memory before the matmul, it fails this direction (it's just storage
   compression wearing a costume). This is the litmus from
   `compression-vs-compute-payoff.md`.
2. **Density gate.** Measure the real ratio: parameters (or bytes) in the compact
   form vs the original, AND the compute cost of the native op vs the dense matmul
   (structural forms can be *cheaper* to run, not just smaller).
3. **Capability gate.** It is lossy by nature, so validate on preserved capability,
   not byte-exactness — Stage-1 matmul-fidelity proxy first
   (`capability-eval-path.md`), Stage-2 behavior/KL once inference runs.

## Relationship to the rest of the project

- This is the deepest expression of the "lighter representation" track in
  AGENTS.md: change the internal form, preserve broad capability.
- It composes with quantization (0005): structural/occupancy density reduces the
  *count* of numbers; bit-width density reduces the *size* of each. Both at once
  is the real prize.
- It is the antidote to the dead end we keep hitting: the experts' fine bits are
  high-entropy (no cheap residual, no shared grid, no position-wise base). Those
  negatives were all about the *current* representation. This direction asks
  whether a *different* representation has structure the current one hides.

## First probe result — post-hoc structural density is dead (and why)

The first concrete probe (candidate 0007) measured whether the experts are
low-rank or share a basis. Both failed hard: experts are full-rank (2% error needs
rank 0.96; factoring stores 1.63x dense) with independent subspaces (shared basis
never reaches usable fidelity below the full ambient dim). Combined with the
lossless (high-entropy mantissa) and base+delta negatives, the picture is
consistent: **the trained expert matrices are statistically dense / random-like —
there is no structural slack to extract from the finished weights.**

This sharpens the direction rather than killing it. Density gains cannot come from
re-representing an already-maximally-trained dense matrix. They live in two places:

1. **Activation structure (post-hoc, exploitable now-ish).** The weights are
   full-rank, but the activations the model produces are low-dimensional and have
   outlier channels. That is the only structure left in the trained model, and it
   is what AWQ/GPTQ-class methods exploit. This unifies with candidate 0005's
   sub-4-bit blocker: everything now routes through **real activations**. The
   keystone is capturing them via a partial early-layer forward (fits in RAM).
2. **Train-time density (the real home of "fundamentally different weights").** The
   five-in-a-room dream — weights that look different and carry the same capability
   tighter — has to be *built in*: structured layers (Monarch/butterfly), trained-in
   superposition, or MoE with shared trained dictionaries, learned from scratch.
   Post-hoc compression cannot create density that training did not put there.

## ACTIVE DIRECTION (chosen) — train-time density

Decision: post-hoc compression of the finished Nemotron is mapped to its floors
(INT8 runtime, ~32% lossless storage, dense experts). The project now pursues
train-time density — building the tighter representation INTO the model rather than
extracting it after. This is where the "fundamentally different weights / five in a
room" win actually lives.

### What "testable" means now (reframed)

The test target is no longer the Nemotron weight tensors — it is a SMALL, trainable
model on CPU. The metric is **capability-per-parameter**: train layer-type variants
to the same task and compare final loss/accuracy vs parameter count. A structured or
superposed layer "wins" if it matches dense capability at materially fewer params,
with the math run natively on the compact form (no unpack).

This keeps the project's discipline: measurements guide decisions, results are
reproducible, and lossy density is validated on preserved capability — just at a
scale we can actually run.

### Layer families to compare (all native-compute, trained from scratch)

- **Dense** (baseline).
- **Low-rank-from-scratch** (W = U V, trained as factors) — post-hoc low-rank failed
  because trained dense is full-rank; does training UNDER the rank constraint reach
  the same capability at fewer params, or does it cap out?
- **Block / Monarch / butterfly-style structured** (a product of small sparse/block
  factors) — more expressive than low-rank at similar param count.
- **Shared-dictionary MoE** (experts = combinations of shared trained atoms) — the
  train-time version of the post-hoc shared-basis idea that failed (0007).

### First experiment (the atomic question)

On one small, CPU-trainable task, sweep parameter budget and plot capability
(final loss) vs params for each layer family. Decisive read: do any structured
families sit ABOVE the dense capability-per-param curve (same loss, fewer params),
or does dense dominate (meaning the params were genuinely needed)? This single curve
tells us whether train-time density is real before investing further.

## Open — superseded by the active direction above; kept for reference

Pick one family above and make it testable: e.g., can a single MoE layer's 128
experts be reconstructed from a shared learned dictionary of K atoms + per-expert
coefficients, measured by matmul-output fidelity vs atom count K and total stored
numbers — with the matmul run natively on (atoms, coefficients) and never
expanding back to 128 full matrices? That turns the direction into a candidate.
