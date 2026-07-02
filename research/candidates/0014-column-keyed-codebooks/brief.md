# Candidate 0014 — Column-keyed codebooks (NEXT_DIRECTIONS Direction D)

**Status:** COMPLETE — FALSIFIED (2026-07-01). Full real run on all 256 layer-27
expert tensors: every column-keyed variant loses to the realized stz baseline
(best: sh_g64_b3 at −0.0283 b/w; adoption-aware envelope = baseline, 0/256
adopt). See `RESULTS.md` for the full table, forensics, and closure reasoning.

**Constraint:** pure lossless, bit-exact reconstruction only. Fusible-only design:
fixed-width index plane, address-derived keys, no context dependency.

## Hypothesis

The stz codec (candidate 0009, realized whole-model **10.8975 b/w**, experts
10.8949) keys its top-K sign+exponent codebook per *tensor*. But the sym field has
strong **column** structure that a per-tensor table cannot see:

- H(exp|col) = 2.486 on down_proj — column identity alone reproduces ~85–100% of
  the 2-D context gain (`research/notes/NEXT_DIRECTIONS.md` lines 15–17).
- Per-column K15 codebooks halve escape rates: up 6.02%→3.15%, down 3.17%→1.61%
  (lines 18–19). Axis verified: "column" = **axis 1**, the contiguous in-row axis
  (col = flat_index % C) — recon recomputed both orientations against
  gpu_sample.npz and axis 1 reproduces every quoted number exactly, so the
  both-orientations fallback is unnecessary.
- Cross-expert per-column exponent profiles are 99.65% correlated
  (`research/notes/findings-ledger.md`), so column tables can be **shared across
  all 128 experts of a layer** and their side cost amortized /128.

Hypothesis: keying the first-level codebook by column group (one top-K table per
group of `g` columns, group id = `col // g`, address-derived → fully fusible)
lowers realized cost below the stz per-tensor envelope — mainly by making
**b=3 index planes viable** on tensors stz codes at b=4, plus residual escape
savings.

## Critical repricing (2026-07-01)

The old 11.2072 b/w baseline is stale. stz's second-level escape codebook already
recodes escapes in k∈{3..6} bits, so halving the escape *rate* no longer converts
1:1 into savings (NEXT_DIRECTIONS lines 41–53). Every comparison here is against
the **realized stz per-tensor cost** recomputed with stz's exact cost model
(`stz.plan_regroup`, proven byte-exact by the `enc_tensor` assert), parity-gated
against `research/candidates/0009-fusible-exponent-codebook/tests/artifacts/stz/stz_tensor_stats.jsonl`.

## Design (the probe)

`tools/probe_column_codebooks.py`, target = all 256 layer-27 expert tensors
(128 experts × {up_proj [1856,2688], down_proj [2688,1856]}, wholly in shard 7;
2.554 GB raw BF16). Per tensor, exact realized bits for:

(a) **BASELINE** — `stz.plan_regroup` recomputed here (imported, not
    reimplemented). Parity gate: must match `stz_tensor_stats.jsonl` within
    ±0.01 b/w on every target tensor; loud abort otherwise. In `--synthetic`
    mode the gate is stronger: `stz.enc_tensor` is run and our recomputed bits
    must equal its realized serialized bits exactly.

(b) **Per-tensor column-keyed** — one top-K codebook per group of `g` columns,
    sweep g ∈ {1,4,16,64} × index bits b ∈ {3,4}; second-level escape codebook
    (k ∈ {0,3,4,5,6} envelope, stz's rule) stays per-tensor and global.

(c) **Shared column-keyed** — same sweep, but first-level tables built from the
    aggregated per-group histogram over all 128 experts of the layer (per
    projection type) and charged **once per layer**, amortized over the experts
    that use them (ng·K·16 / 128 bits per tensor).

(d) **Escape forensics on the winning variant** — per-row / per-column escape
    densities (Fano factor vs binomial reference), spatial adjacency lift of the
    escape mask (horizontal + vertical), H(sign|col).

### Exact cost model (mirrors stz byte-for-byte)

Every stream independently padded to a byte boundary; u64 length prefix per
stream (4 streams k=0, 6 streams k>0); u16 per codebook entry — first level
ng·K·16 bits (K = 2^b − 1 per group), second level L·16; per-row escape prefix
R·pw and raw-prefix R·pw2 with stz's `_pw` width rule; fixed header
`<BBHIIQQBB'` = 30 B (b, k, g, R, C, n, n_esc, pw, pw2 — ng is derivable from
C and g). Shared tables: full table bits **plus a 128-bit frame** (u64 length +
packed layer/proj/g/b id) charged once per (layer, proj, g, b), divided equally
across the sharing tensors. Nothing is waved off.

The accounting is **writer-verified, not analytic-only**: `enc_colkey` /
`dec_colkey` serialize and decode every (g, b) variant on a deterministic
sample (all tensors in synthetic mode; expert 0 per projection in real mode,
for both per-tensor and shared tables), asserting serialized bits ==
`colkey_plan` bits exactly and SHA-256 bit-exact reconstruction — the same
standard stz was held to. The summary marks any positive verdict PROVISIONAL
unless this coverage exists for the winning variant kind on every projection.

Envelopes: env(base+pt) is exact. The base+all envelope is **adoption-aware**
(fixed-point per-tensor assignment; each adopted shared table charged fully,
once, across its actual adopters) — the earlier uniformly-amortized env_all
undercharged partially-adopted tables and was removed. The per-variant sh rows
still assume uniform adoption (exact in that case), noted in the summary JSON.

Hardening (2026-07-01 review): single-file atomic aggregate checkpoint
(histograms + membership in one npz, one `os.replace`, count-consistency check
on load), exclusive run lock, truncated-JSONL-tail self-repair with fsync'd
appends, verdict scoped to layer 27 (cross-layer transfer unvalidated until an
early/late layer re-run), per-variant first-level table bytes surfaced and
winner ties broken toward the largest g (smallest table). A g=1 win is
storage-leaning until a kernel-side table-lookup benchmark accompanies it
(~56–80 KB tables exceed per-SM shared memory).

### Fusibility

The index plane stays fixed-width b bits, random-access. Decode of weight
(r, c): `table[c // g][idx]` — group id is address-derived, tables live in
SRAM/L2 like stz's single table. Escapes keep stz's per-row prefix machinery.
No variable-length anything.

## Success gate

- **Confirmed:** some variant (all side costs charged) beats the realized stz
  baseline by ≥ **0.09 b/w** numel-weighted on the layer-27 set — ≈ +0.5
  whole-model pt at experts = 93% of BF16 numel (0.09 × 0.93 / 16 ≈ 0.52 pts).
- **Weak positive:** 0 < Δ < 0.09 b/w — record, fold into the stz chooser
  envelope, but not a headline.
- **Falsified at this operating point:** Δ ≤ 0 for every (g, b, sharing)
  combination. Then the null escape-forensics result doubles as an optimality
  certificate for the per-tensor codebook (NEXT_DIRECTIONS line 118).

## How to run

```
# smoke (synthetic snapshot, seconds)
uv run python research/candidates/0014-column-keyed-codebooks/tools/probe_column_codebooks.py --synthetic

# real run (resumable; each invocation self-limits to ~7 min and checkpoints)
uv run python research/candidates/0014-column-keyed-codebooks/tools/probe_column_codebooks.py

# summary table + escape forensics + summary JSON (auto-runs when complete)
uv run python research/candidates/0014-column-keyed-codebooks/tools/probe_column_codebooks.py --summary
```

## Artifacts (tests/artifacts/)

- `colkey_results[_synthetic].jsonl` — one record per tensor per stage
  (per_tensor / shared / roundtrip), resumable append.
- `colkey_shared_hists[_synthetic].npz` — aggregated per-group histograms for
  the shared tables, with the included-tensor membership stored inside the same
  npz (single-file atomic checkpoint).
- `colkey_summary[_synthetic].json` — numel-weighted table, envelopes
  (base+pt exact, base+all adoption-aware), winner, serializer round-trip
  coverage, forensics, whole-model projection, layer-scoped gate verdict.

## Links

- `research/notes/NEXT_DIRECTIONS.md` — Direction D (lines 107–118), fresh
  measurements (lines 9–24), repricing note (lines 41–53).
- `research/candidates/0009-fusible-exponent-codebook/tools/stz.py` — cost model
  and helpers reused verbatim.
- `research/candidates/0012-lossless-ceiling/` — the storage-side context
  numbers column keying is chasing in fusible form.
