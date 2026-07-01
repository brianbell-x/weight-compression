# Hypothesis Scout Loop

Status: Ready

## Purpose

Continuously find candidate research paths for making high-capacity LLMs lighter to store or run while preserving broad model capacity.

## Priority & Prior Findings (read before scouting)

See AGENTS.md "Strategic Priority" and `research/notes/findings-ledger.md`.
Also read the latest reports in `research/claim-reviews/` before creating or updating candidates.

- The primary target is making the model cheaper to **run**, not just to store.
  Apply the litmus test: if the compressed form is re-inflated to full width in
  VRAM before the math uses it, it is a storage-only idea — rank it lower.
- Highest-value lever for this MoE: the **resident expert bulk** (~30B resident,
  ~3B active/token), ideally via a lossy, capability-preserving form that stays
  narrow into compute.
- Do not re-propose falsified ideas (cross-expert base+delta, expert pooling,
  sign-fold). Lossless BF16 expert compression is solved/capped (~32%, storage
  axis only). Build on confirmed results; consult the ledger first.

## Current Shape

One agent loops through this repository's notes, model inspections, tensor manifests, synthetic test set, and prior experiment results to find plausible original research paths.

External sources may be used for inspiration, vocabulary, or contrast, but the scout should not try to clone or reimplement existing methods. The goal is to discover or develop a new way to make high-capacity models lighter.

## Trigger

Run every 30 minutes and inspect this repository for new evidence, notes, manifests, experiment outputs, and unfinished research directions.

```text
workflows/
NOTES.md
AGENTS.md
research/
research/notes/          # strategic steering notes — READ THESE; they constrain which ideas are worth proposing
models/synthetic/
models/nvidia/
tools/
```

### Required Reading Before Proposing

Read `research/notes/compression-vs-compute-payoff.md` before creating candidates. It determines whether an idea yields a real runtime/memory win or only a storage win, and where lossless decode on the inference critical path backfires. Prefer candidates whose payoff regime is explicit; do not propose ideas the note shows are net-zero at inference unless they are explicitly framed as storage/transfer-only wins.

The user may also run it manually with a request such as:

```text
run the hypothesis scout
```

Autonomous scout runs should create candidate folders only when an idea meets the testability standard. Do not create low-signal tickets just to fill the queue.

If a latest claim review contains `Blocking` or `High` unresolved claims relevant to a candidate, resolve them before using that claim as evidence. Either ground the claim with an accepted source/proof or downgrade it to explicit hypothesis wording.

When a candidate fully meets the testability standard, create it with:

```text
Status: Testing
```

This lets the tester loop pick it up without user involvement. If an idea is interesting but incomplete, refine it further instead of creating a `Proposed` ticket.

## Duplicate Policy

Before writing a new candidate folder, scan:

```text
research/candidates/*/brief.md
```

Do not create a new candidate if these are substantially the same as an existing candidate:

- `Claim`
- `Tensor Group`
- `Measurement`

If the new thought improves an existing candidate, append a short note to that candidate instead.

## Handoff

Output candidate briefs for the tester loop.

## Loop-Back From Tester

Read completed tester reports inside:

```text
research/candidates/*/tests/
```

If a report has a concrete `Next Action`, the scout may use it as seed material for a follow-up candidate.

Only create the follow-up candidate if it meets the same testability standard. Otherwise, leave the idea alone until more evidence exists.

## Output Location

Save candidate briefs in:

```text
research/candidates/
```

Use one folder per idea:

```text
research/candidates/0001-short-name/
  brief.md
```

## Candidate Brief Format

```md
# Candidate: Short Name

## Claim
One sentence describing the possible structure.

## Why It Might Work
Short reasoning based on this repo's model notes, tensor anatomy, manifests, or observations.

## Tensor Group
Exact tensors or group to inspect.

## Measurement
What the tester should calculate.

## Promising Result
What result means the idea deserves more work.

## Test Target
Synthetic first, true weights only if synthetic teaches something.

## Status
Testing / Parked / Rejected / Escalated
```

## Testability Standard

An idea is ready for the tester only when it names:

- the tensor group to inspect
- the suspected structure
- the measurement to run
- the result that would make it promising

Example:

```text
MoE experts in the same layer may share a hidden base plus small deltas.
Measure same-shaped expert tensors, derive a base tensor, encode deltas, reconstruct, and compare size plus exactness.
```

## Open Questions

None.
