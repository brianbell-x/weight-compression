# AGENTS.md

## Project

Make high-capacity LLMs lighter to store, load, and run **without changing what
they compute** — exact lossless compression of the weights. Working target is
NVIDIA's Nemotron family:

`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

Treat the direction as a learning loop. Evidence can change the path.

## The Goal

Find more lossless compression. We have proven the weights can be made smaller
*and* faster with zero quality change (candidate 0009: whole-model bit-exact
round-trip, ~25–30% smaller, a measured 24% GPU decode speedup). That number is a
floor, not a ceiling — assume there is more structure to exploit and go find it.

The guiding intuition: **perfect lossless compression is indistinguishable from
random noise.** Any detectable pattern left in the compressed stream is
compression still on the table — if you can predict the next bit better than
chance, you can spend fewer bits encoding it. This turns the vague goal into a
concrete hunt with a built-in diagnostic: search for residual structure in the
weights (bias, correlation, repetition, exploitable byte layout), and measure the
entropy of what you emit. Output that still looks non-random is not done.

This already has a concrete shape here, and it's load-bearing. In BF16 the
mantissa carries ~7.95 bits of real entropy out of 8 — it *is* essentially random
noise, so it's the incompressible floor and must move verbatim. The sign+exponent
field is the opposite: hyper-concentrated (a handful of values cover >98% of
weights), which is where all the structure — and all the compression — lives.
Compressing is peeling the structured fields off until what remains looks like
noise.

And the two bars are not a fork you choose between — we have converted storage
wins into runtime wins. The entropy floor is a *storage* number, and the code that
reaches it (arithmetic / range coding, variable-length) has no fixed bit-offset
per weight, so the math can't address it and it must be inflated in VRAM first.
But that's a layout problem, not a wall: candidate 0009 re-expressed the *same*
structure as a **fixed-width codebook index + sparse escape**, regaining random
access — turning candidate 0001's 32% *storage-only* win into a ~29% *fusible* win
with a measured decode speedup. The fusible ceiling sits just below the entropy
floor (here ~3 points), and the bridge between them — fixed-width, block-wise, or
sparse-escape layouts — is the craft. A storage win is a signal that a runtime win
is nearby, not a dead end.

Every path must clear both bars:

1. **Exactly lossless** — the weights (or bytes) reconstruct bit-for-bit. Prove it
   with an exact round-trip (SHA-256 / bit-equality), not a quality metric.
   Lossless ⇒ logits are provably identical, so no capability eval is needed.
2. **Fusible / runtime-real** — the compressed form must be readable *directly* by
   the math (fixed-width, random-access), rebuilt to full width only transiently
   in on-chip registers, never re-inflated to full width in VRAM. A code that has
   to be fully decoded back to BF16 in memory before the matmul is storage-only —
   still worth noting, but it is not the goal.

The litmus test for any new idea: *does the compressed form ever get expanded back
to full width in VRAM before the math uses it?* If yes, it's a storage tool only.
If no, it's a real runtime win — that is what we are hunting.

## Working Style

- Let measurements guide decisions. Record assumptions, experiments, failures, and
  results.
- Build small probes first; prove the mechanism before scaling. Validate every
  lossless claim with an exact round-trip on real weights.
- Before scouting, read what previous experiments already settled
  (`research/notes/findings-ledger.md` and the `research/candidates/*/` statuses).
  Do not re-propose a falsified idea; build on confirmed ones.
- Read `NOTES.md` for the shared vocabulary and mental model — tensor anatomy
  (name/shape/dtype/bytes/bits, expert/projection/row/column structure) and the
  compression principle that repeated patterns and byte layouts matter more than
  exact duplicates. It's the reference for the concepts this work leans on — keep
  it current: when you learn something that belongs in the shared vocabulary or
  mental model, update `NOTES.md` so it stays the source of truth.
- Favor reproducible scripts, deterministic outputs, and clear checkpoints.
- Don't quit an idea too early. One failed attempt falsifies *that attempt*, not
  the idea. Separate "this specific approach didn't work" from "this direction is
  dead," and before abandoning a promising idea, exhaust the obvious variations —
  different angle, parameters, framing, or a weakened version — and say what would
  have to be true for it to work. Only a real falsifying result closes a path.
  This applies at every level — a whole research direction, or one line of a codec.
- Stay open to nontraditional methods. A useful approach may look original or
  unlike standard model-serving practice, as long as it can be measured and
  reconstructed exactly.

## Agent Execution

- Run tasks/agents in the background by default (e.g. `run_in_background`, spawned
  Agent calls) rather than blocking in the foreground, unless the next step
  genuinely depends on the result before proceeding.

## Python Tooling

- Use `uv` only. Run commands with `uv run`; manage packages with `uv add` /
  `uv remove`. Do not use `pip`, `venv`, or `conda` directly.

## Synthetic Test Set

- A small fake Nemotron-like snapshot exists at
  `models/synthetic/nemotron_tiny/hf_snapshot`. It copies the real model's file
  shape (sharded `.safetensors`, index, config, Nemotron-like tensor names, BF16
  weights, F32/Mamba/attention/MoE tensors) but holds no trained values.
- Use it to shake out a new codec, parser change, or reconstruction path with a
  fast exact round-trip before touching the true weights. It is not useful for
  inference quality checks (and lossless work does not need them).
