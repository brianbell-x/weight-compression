# Claim Grounding Loop

Status: Draft

## Purpose

Continuously find claims and assumptions in this repository that are not grounded by either:

- a cited paper or external source
- an executable proof in this repo
- a clearly linked prior experiment result in this repo

The loop protects the scout and tester from building on unsupported statements without turning every note into a manual review burden.

## Current Shape

One agent scans repository outputs on a schedule, identifies unsupported claims, and writes separate review reports. It does not edit source research notes, candidate briefs, or test reports in place.

## Trigger

Run every 15 minutes and inspect files changed since the previous claim-grounding run.

Run one full-repository sweep per day.

Priority locations:

```text
AGENTS.md
NOTES.md
workflows/
research/notes/
research/candidates/
tools/
```

Ignore generated artifacts, caches, downloaded model files, virtual environments, and large binary payloads.

## Grounding Standard

A claim is grounded when the source text gives enough evidence for a scout or tester to trust it as more than speculation.

Acceptable grounding:

- a repo-local executable proof, script, or test report that directly supports the claim
- a repo-local measurement artifact with enough context to reproduce or interpret it
- a cited paper, documentation page, or other external source for general technical background
- wording that clearly labels the statement as a hypothesis, assumption, or open question

Unsupported claims should be recorded when they are likely to affect research direction, candidate priority, implementation choices, or conclusions.

Do not flag harmless phrasing, obvious file facts, or claims already scoped as speculative.

External grounding must come from primary or near-primary sources.

Accept:

- peer-reviewed papers
- arXiv or other preprints when clearly relevant
- official documentation from model, framework, library, hardware, or platform vendors
- official repository README/docs/issues/PRs for the exact tool or method
- Hugging Face model cards for model-specific facts

Do not accept as final grounding:

- blog posts unless they cite primary evidence
- forum comments
- agent-written summaries
- unsourced benchmark claims
- vague "common knowledge" claims

## Agent-Created Grounding

Agents may ground a claim themselves by running an experiment or executable proof in this repository.

Self-created grounding is acceptable when it includes:

- the script, command, or test that was run
- the input data or tensor group used
- the output artifact or report
- the exact claim the result supports
- enough detail for another agent to rerun or audit it

Preferred locations:

```text
research/candidates/<candidate>/tests/
research/notes/
tools/
```

Use candidate test reports when the claim belongs to a candidate. Use `research/notes/` for broader project claims that are not tied to one candidate. Use `tools/` for reusable proof scripts.

When an agent creates grounding for an open claim, update `research/claim-reviews/index.json` and cite the new proof in the next claim review report.

Self-created grounding must be executable or measurement-backed. A new agent-written explanation by itself does not ground a claim.

## Severity

Each unsupported claim must be assigned one severity:

```text
Blocking
High
Medium
Low
```

Use the levels this way:

```text
Blocking: candidate, test, or conclusion should not proceed until the claim is grounded or downgraded
High: likely to mislead research direction, status, prioritization, or implementation choices
Medium: useful to ground, but related work can continue cautiously
Low: wording cleanup, background context, or weak claim with limited downstream effect
```

Within each report, order findings by severity first, then by expected downstream impact.

## Output Location

Write review reports to:

```text
research/claim-reviews/
```

Use one report per run:

```text
research/claim-reviews/YYYY-MM-DD-HHMM.md
```

Track the last successful incremental scan in:

```text
research/claim-reviews/.last-scan.json
```

Track unresolved claims in:

```text
research/claim-reviews/index.json
```

The state file should record:

```json
{
  "last_incremental_scan_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "last_full_sweep_utc": "YYYY-MM-DDTHH:MM:SSZ"
}
```

For incremental scans, inspect text files whose modified time is newer than `last_incremental_scan_utc`.

For daily full sweeps, inspect all eligible text files under the priority locations.

## Deduplication

Assign every unsupported claim a stable ID:

```text
CG-YYYYMMDD-001
```

Deduplicate by:

- normalized claim text
- source file path
- nearby line context

The index should track:

```json
{
  "id": "CG-YYYYMMDD-001",
  "status": "open",
  "severity": "High",
  "source": "research/candidates/0009-example/brief.md:14",
  "claim": "Short normalized claim text",
  "first_seen_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "last_seen_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "resolution": null
}
```

When a claim appears again:

- `Blocking` and `High` claims should appear in a `Still Unresolved` summary until resolved.
- `Medium` and `Low` claims should not be repeated unless the source text changed materially.

When the claim is grounded, downgraded to explicit hypothesis wording, or removed, update the index entry:

```json
{
  "status": "resolved",
  "resolution": "Grounded by research/candidates/0009-example/tests/test-001.md",
  "resolved_utc": "YYYY-MM-DDTHH:MM:SSZ"
}
```

## Report Format

```md
# Claim Grounding Review: YYYY-MM-DD HH:MM

## Summary

- Files scanned:
- Unsupported claims found:
- Still unresolved Blocking/High claims:
- Highest-risk unsupported claim:

## Still Unresolved

- CG-YYYYMMDD-001 - Severity - Source - one-line claim

## Unsupported Claims

### 1. Short claim label

ID: CG-YYYYMMDD-001

Source: path/to/file.md:line

Severity: Blocking / High / Medium / Low

Claim:
> Short excerpt or paraphrase of the claim.

Grounding status:
No repo evidence found. No paper citation found. No executable proof found.

Why it matters:
Explain how this could mislead scout/tester work.

Needed grounding:
- cite a paper or external source, or
- add/run a script proving this in the repo, or
- downgrade wording to a hypothesis.

Suggested owner:
Scout / Tester / User / Unknown
```

If no unsupported claims are found, write a short clean report instead of creating no output.

## Handoff

The scout and tester loops should read the latest claim review reports before acting on candidate briefs or conclusions.

The claim grounding loop does not decide whether a claim is false. It only decides whether the repository currently supports it.

## Open Questions

None.
