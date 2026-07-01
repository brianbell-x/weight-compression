# Result Reflection Loop

Status: Draft

## Purpose

Read completed research outputs and find the next useful opportunities they imply.

This loop exists because a result is rarely just pass/fail. A failed test can expose a better tensor group, a measurement artifact can suggest a deeper mechanism, and a partial success can point to a narrower or stronger follow-up. The reflection loop turns those openings into clear next directions for the scout and tester.

## Current Shape

One coordinator periodically reviews recent candidate briefs, test reports, claim-grounding reports, notes, and artifacts. When there is enough new material to justify the overhead, the coordinator may split the read across focused subagents, then own the final synthesis, deduplication, ranking, and any repo edits.

Subagents only inspect assigned evidence slices and return cited findings. They must not create candidates. They should not edit the ledger or modify workflow files directly unless the coordinator explicitly delegates one single-writer task after synthesis.

The loop should not invent unrelated ideas. It should start from repo evidence and ask:

- What did this result reveal?
- What follow-up would test the mechanism more directly?
- What should be abandoned or deprioritized?
- What assumption changed?
- What next candidate would a careful researcher create after reading this?

## Trigger

Run every 30 minutes, after the scout, tester, and claim-grounding loops have had time to produce outputs.

Also run manually when the user asks for reflection on recent results.

Priority inputs:

```text
research/candidates/*/brief.md
research/candidates/*/tests/*.md
research/candidates/*/tests/artifacts/
research/claim-reviews/
research/notes/
AGENTS.md
NOTES.md
workflows/
```

## Subagent Fanout

Use subagents when the input set is broad, recent outputs span several evidence types, or a reflection run would benefit from independent read-only perspectives. Do not fan out when there are only one or two small new reports; one coordinator pass is cheaper and less likely to create duplicate recommendations.

Recommended slices:

- **Candidate/test reviewer**: recent `research/candidates/*/brief.md` and `research/candidates/*/tests/*.md`
- **Artifact reviewer**: recent `research/candidates/*/tests/artifacts/`, especially summaries, CSVs, JSON, logs, and scripts that explain measurements
- **Claim/grounding reviewer**: latest `research/claim-reviews/` reports and unresolved `index.json` entries
- **Notes/workflow reviewer**: `research/notes/`, `NOTES.md`, `AGENTS.md`, and `workflows/`

Each subagent should return only:

- files and line ranges reviewed
- signals found
- duplicate/settled-work checks
- candidate-ready opportunities, if any
- risks, uncertainty, or grounding gaps

The coordinator should merge overlapping subagent findings before ranking them. Subagent agreement is useful evidence, but the coordinator must still check source files directly before creating candidates, updating the ledger, or recommending a workflow change.

Prefer a single-writer shape:

```text
parallel read-only subagents -> coordinator synthesis -> optional coordinator-designated single writer
```

If the subagent runner fails or is unavailable, continue with the normal coordinator pass and record the fallback in the reflection report's input summary.

## Output Location

Write reflection reports to:

```text
research/reflections/
```

Use one report per run:

```text
research/reflections/YYYY-MM-DD-HHMM.md
```

## Report Format

```md
# Result Reflection: YYYY-MM-DD HH:MM

## Summary

- Inputs reviewed:
- Subagent slices, if used:
- Opportunities found:
- Highest-value next move:

## Opportunities

### 1. Short opportunity label

Source evidence:
- path/to/report.md:line

What the result revealed:
Plain-language explanation of the signal.

Opportunity:
The next deeper dive, pivot, narrowing, abandonment, or workflow change implied by the result.

Ranking:
- Value:
- Grounding:
- Testability:
- Novelty:
- Cost:

Why this is not already covered:
Name existing candidates/notes checked, or say what gap remains.

Cost axes:
Storage / Load-transfer / Resident VRAM / Per-token bandwidth / Compute

Grounding:
Evidence already in repo, cited paper/source, or experiment needed.

Suggested handoff:
Scout / Tester / Claim Grounding / User / Workflow

Candidate-ready:
Yes / No

If candidate-ready, include:
- Claim
- Tensor Group
- Measurement
- Promising Result
- Test Target
```

## Handoff

If an opportunity is candidate-ready, the reflection loop may create or update a candidate brief directly, following the scout loop's candidate format and duplicate policy. Only the coordinator may create or update candidate briefs. Subagents can recommend candidate-ready opportunities, but they should not write candidate files.

Only create a candidate directly when the opportunity meets the scout loop's full testability standard:

- tensor group to inspect
- suspected structure
- measurement to run
- result that would make it promising

Before creating a candidate, the coordinator must merge overlapping subagent findings and check for duplicates against existing candidates, findings-ledger conclusions, and recent reflection reports.

Directly created candidates must use:

```text
Status: Testing
```

This lets the tester loop pick them up without a second scout pass.

If an opportunity needs more evidence, write it in the reflection report and leave it for the scout or claim-grounding loop.

If a result implies that a direction should be stopped, deprioritized, or reclassified, write the recommendation clearly with source evidence.

The reflection loop may append confirmed research conclusions to:

```text
research/notes/findings-ledger.md
```

Ledger updates must link to the source reports or artifacts that justify the conclusion.

The reflection loop should not edit `AGENTS.md` by default. Treat `AGENTS.md` as behavioral and strategic steering. If a result implies that `AGENTS.md` should change, write a recommendation in the reflection report for user review.

## Ranking

Rank opportunities by expected research value, using these explicit sub-scores:

```text
Value: could this materially change project direction or unlock a runtime/storage win?
Grounding: how directly is it implied by repo evidence?
Testability: can scout/tester turn it into a concrete next experiment?
Novelty: is it different from already-settled or already-queued ideas?
Cost: how expensive is the next useful check?
```

Sort reports by:

```text
Highest Value + enough Grounding + practical Testability
```

Do not rank by ease alone. Do not rank by novelty alone.

## Open Questions

None.
