# Hypothesis Tester Loop

Status: Ready

## Purpose

Test concrete compression or lighter-representation ideas produced by the scout loop.

## Current Shape

One agent takes a candidate brief, turns it into the smallest useful experiment, runs it against the synthetic Nemotron test set, and reports whether it is worth escalating to true weights.

Before testing a candidate, read the latest reports in `research/claim-reviews/`. If the candidate depends on an unresolved `Blocking` or `High` unsupported claim, either ground it as part of the test setup or mark the test report `Needs Deep Analysis` with the grounding gap named explicitly.

## Trigger

Run every 30 minutes and check:

```text
research/candidates/
```

Treat candidate briefs like a ticket queue. Only pick up candidates with:

```text
Status: Testing
```

Ignore `Proposed`, `Parked`, `Rejected`, and `Escalated`.

Each candidate lives in its own folder:

```text
research/candidates/0001-short-name/
  brief.md
```

## Handoff

Output test reports with pass, fail, or needs-more-research status.

## Output Location

Write test outputs inside the candidate folder:

```text
research/candidates/0001-short-name/
  brief.md
  tests/
    test-001.md
    artifacts/
```

Update `brief.md` with the latest status and report path.

## Candidate Statuses

Use these statuses:

```text
Testing
Passed Synthetic
Needs Deep Analysis
Escalated True Weights
Passed True Weights
Rejected
Parked
```

Every test report must end with:

```md
## Next Action
One concrete next step, or "None."
```

## Synthetic Pass Standard

For exact-compression ideas, pass means:

- the compressed representation is smaller than the original payload, or shows a clear path to becoming smaller
- reconstruction exactly matches the original tensors
- the result is explained well enough to try on true weights

For lighter-representation ideas, pass means:

- the representation is smaller or cheaper
- reconstruction or error behavior is measured
- the tradeoff is explicit
- the result reveals a structure worth testing on true weights

The synthetic set does not prove model quality. It proves whether the method works mechanically and whether a structure signal exists.

Some candidates require deep analysis before a fair status is possible. In those cases, do not force a shallow pass/fail. Mark the report as `Needs Deep Analysis` and explain exactly what deeper measurement is needed.

## True-Weight Escalation

The tester may work autonomously on the synthetic test set.

The tester may also decide to test against true Nemotron weights without asking the user when all of this is true:

- the synthetic test passed or produced a strong structure signal
- exact reconstruction logic is verified when exactness is required
- the expected true-weight measurement is named
- the true-weight test is scoped to the smallest useful shard, tensor group, or slice
- the report explains why escalation was justified

No user checkpoint is required. Push decisions right by doing the work and leaving a clear report trail.

## Required Input

The tester accepts only candidate briefs that name:

- the tensor group to inspect
- the suspected structure
- the measurement to run
- the result that would make it promising

The tester should not treat ungrounded candidate rationale as established fact. Unsupported rationale can still be tested, but it must be framed as the hypothesis under test.

## Open Questions

None.
