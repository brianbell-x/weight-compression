# Workflows

This folder defines the recurring agent loops for the research project.

The goal is to keep research moving without letting unsupported claims, stale conclusions, or exhausted idea queues quietly steer the project.

## Loop Map

### 1. Hypothesis Scout

Spec: `hypothesis-scout-loop.md`

Runs every 30 minutes.

The scout reads repo evidence, prior findings, claim reviews, and unfinished directions. It creates test-ready candidate briefs only when an idea meets the testability standard:

- tensor group
- suspected structure
- measurement
- promising result

Output:

```text
research/candidates/<id>-<short-name>/brief.md
```

### 2. Hypothesis Tester

Spec: `hypothesis-tester-loop.md`

Runs every 30 minutes.

The tester polls candidate briefs marked:

```text
Status: Testing
```

It turns each candidate into the smallest useful experiment, runs it, writes a report, and updates the candidate status.

Output:

```text
research/candidates/<id>-<short-name>/tests/test-001.md
research/candidates/<id>-<short-name>/tests/artifacts/
```

Every tester report ends with `Next Action` so future loops can continue from the result.

### 3. Claim Grounding

Spec: `claim-grounding-loop.md`

Runs every 15 minutes on changed files, plus one full sweep per day.

The grounding loop finds claims and assumptions that are not supported by:

- a repo-local executable proof
- a repo-local measurement or test report
- a cited primary or near-primary external source
- explicit hypothesis/assumption wording

It writes separate reports instead of editing source files in place.

Output:

```text
research/claim-reviews/YYYY-MM-DD-HHMM.md
research/claim-reviews/index.json
research/claim-reviews/.last-scan.json
```

Scout and tester must read the latest claim reviews before acting on candidates or conclusions.

### 4. Result Reflection

Spec: `result-reflection-loop.md`

Runs every 30 minutes after the other loops have had time to produce fresh outputs.

The reflection loop reads completed results and asks what new opportunity the evidence implies:

- deeper dive
- pivot
- narrower follow-up
- abandonment or deprioritization
- workflow or memory update

It can create candidate briefs directly when the scout testability standard is met. It can append confirmed conclusions to `research/notes/findings-ledger.md`. It should not edit `AGENTS.md` by default.

Output:

```text
research/reflections/YYYY-MM-DD-HHMM.md
```

## System Flow

```text
repo evidence + prior findings
  -> claim grounding marks unsupported claims
  -> scout creates test-ready candidates
  -> tester runs experiments and reports results
  -> reflection extracts deeper opportunities
  -> scout/tester pick up candidate-ready follow-ups
  -> findings ledger accumulates confirmed conclusions
```

The loops are allowed to create grounding by running experiments. Agent-written explanations alone do not ground claims.

## Timing

```text
Every 15 min: claim grounding incremental scan
Every 30 min: hypothesis scout
Every 30 min: hypothesis tester
Every 30 min: result reflection
Daily: claim grounding full sweep
```

The exact wall-clock offsets can vary, but reflection should run after other loops have had a chance to produce new material.

## Artifact Layout

```text
research/candidates/
  0001-short-name/
    brief.md
    tests/
      test-001.md
      artifacts/

research/claim-reviews/
  YYYY-MM-DD-HHMM.md
  index.json
  .last-scan.json

research/reflections/
  YYYY-MM-DD-HHMM.md
```

The workflows describe the loops. The candidate folders contain the actual research tickets and results.

## Source Of Truth

Each workflow spec is the source of truth for that loop's behavior. This README is the operating map.
