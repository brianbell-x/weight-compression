# Test 001 — F32 Dead Precision (low-16-mantissa-bit truncation)

## Goal
Test whether the model's F32 tensors are exactly BF16-representable (low 16
mantissa bits all zero), so the bottom 2 bytes of every element are dead and can
be dropped, then reconstructed bit-exactly by zero-padding — a guaranteed
lossless 50% on the clean subset.

## Mechanism
For each F32 tensor: read raw little-endian bytes, view as `uint32`, test
`(word & 0x0000FFFF) == 0` for ALL elements. Clean tensor -> keep only the high
2 bytes of each word (`<u2` index 1, which IS the BF16 form). Reconstruct by
zero-padding the low 2 bytes and SHA-256 hash-compare to the original bytes.

Script: `artifacts/scan_f32_dead_precision.py` (run with `uv run`).

## Synthetic — mechanism wired and verified exact
`models/synthetic/nemotron_tiny/hf_snapshot` (both shards): 7 F32 tensors, 0
clean (synthetic holds untrained random values, so low bits are noise as
expected — `worst_nonzero_low16 = 94`). The synthetic set cannot exercise the
clean-truncation path with its own data, so I forced it: zero each F32 tensor's
low 16 bits to synthesize a clean tensor, then ran truncate -> zero-pad ->
hash-compare. Result: **round-trip bit-exact for all 4 forced-clean synthetic
F32 tensors**, kept 2/4 bytes = exactly 50%. The `uint32` reinterpret and
zero-pad reconstruction are proven correct independent of data.

## True shard 1 — real coverage
`model-00001-of-00013.safetensors`. Note: in this model the RMSNorm
`*.norm.weight` tensors are **BF16, not F32** (the brief assumed F32). The true
F32 family in shard 1 is only the Mamba/MoE control tensors — 5 tensors, 1536
bytes total:

| tensor | numel | nonzero_low16 | clean | round-trip |
|---|---|---|---|---|
| layers.0.mixer.A_log | 64 | 0 | yes | exact |
| layers.0.mixer.D | 64 | 0 | yes | exact |
| layers.1.mixer.gate.e_score_correction_bias | 128 | **128** | no | — |
| layers.2.mixer.A_log | 64 | 0 | yes | exact |
| layers.2.mixer.D | 64 | 0 | yes | exact |

Summary: 4/5 clean (80%), worst-case dirty = 128/128 elements nonzero,
total F32 bytes = 1536, **bytes saved by truncation = 512** (all 4 clean
tensors round-trip SHA-256-identical to the originals).

Artifacts: `artifacts/true_shard1/scan_results.json`,
`artifacts/synthetic/scan_results.json`.

## Findings
- The claim holds for the **Mamba `A_log` and `D` tensors**: exactly
  BF16-representable, truncate to 50% with a guaranteed bit-exact round-trip and
  zero coding cost. This confirms the Mamba scout's observation.
- The claim **fails for the MoE `gate.e_score_correction_bias`**: all 128
  elements carry real low-mantissa content (full F32 precision is used). This is
  a learned router bias, not a hand-set constant — it is not free to truncate.
  Truncating it would be lossy; it must be left as F32 or sent to an entropy
  coder instead.
- The brief's premise that RMSNorm weights are F32 does not apply here — they
  are stored BF16 already, so there is no F32 norm family to harvest.

## Verdict
Mechanism is correct and proven bit-exact. But the addressable win is tiny: the
entire F32 family in a 4.99 GB shard is 1536 bytes, and the clean subset saves
512 bytes. Extrapolated across all 13 shards (~62 mamba layers x 2 clean F32
tensors x 256 B, minus the ~half-size last-layer effects) the absolute ceiling
is on the order of a few tens of kilobytes — negligible against a ~62 GB model.
The idea is *correct and lossless* but *economically immaterial* on its own. Its
real value is as a sub-rule inside a broader codec: when a generic F32-handling
path can cheaply detect `(word & 0xFFFF)==0` per tensor, it gets a free exact 50%
on the constant control tensors. It is not worth shipping as a standalone
candidate.

## Next Action
None. Mechanism verified exact; coverage measured on true shard 1 (4/5 clean,
512 B saved of 1536 B F32). The win is real but immaterial standalone — fold the
`(word & 0xFFFF)==0` clean-detect as a free sub-rule into a future general tensor
codec rather than pursuing this candidate further. The dirty
`gate.e_score_correction_bias` is the only F32 tensor needing real coding.
