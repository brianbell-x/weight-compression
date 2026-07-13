# Blind Verification 0061

- **Experiment:** succeeded
- **What this means for the project:** All six claims survived independent recomputation and adversarial inspection of the supplied checkpoints, result files, log, merge code, coder code, and kernel code. No numerical contradiction was found. The evidence bundle does not contain the original GLM-5.2 shards or the GPU sample/environment, so it supports internal consistency and code-path verification rather than a fresh replay from raw weights or hardware.
- **Reproduction file:** `src/tools/merge_glm.py` for the published aggregation formulas; independent recomputation used the 16 JSON files under `src/tests/artifacts/ckpts/` with `uv run python`.

## Scope and method

I treated every supplied claim as false until supported. I independently loaded all checkpoint JSON files, formed sets of shard names, checked all pairwise intersections, constructed the exact expected shard-name set, summed each accumulator field without invoking the merge script, regenerated every numeric result field, and recomputed the benchmark ratios and bandwidths from shape, replication factor, and reported times. I also inspected the complete relevant functions in the coder and kernel sources and compared the benchmark JSON with `bench.log`.

## Claim 1: checkpoint coverage

**Result: verified; refutation failed.**

- Found exactly 16 checkpoint files, all with `repo == "zai-org/GLM-5.2"`.
- Total `done_shards` entries: 282.
- Unique entries: 282.
- Nonempty pairwise intersections among the 16 checkpoint sets: 0.
- Missing from the independently generated set `model-00001-of-00282.safetensors` through `model-00282-of-00282.safetensors`: 0.
- Unexpected names outside that set: 0.

Thus the checkpoint shard sets are pairwise disjoint and their union is exactly the claimed 282-shard sequence.

## Claim 2: merged headline numbers

**Result: verified; refutation failed.**

Independent sums of the 16 `acc` blocks were:

| Field | Recomputed total |
|---|---:|
| `total_raw` | 1,506,659,919,872 bytes |
| `bf16_raw` | 1,506,659,842,048 bytes |
| `other_raw` | 77,824 bytes |
| `bf16_enc_bs` | 1,130,493,155,272.0 bytes |
| `bf16_enc_rg` | 1,052,123,459,634.375 bytes |
| `n_bf16` | 59,509 |
| `n_esc_total` | 187,756,789 |
| `dtype_raw.BF16` | 1,506,659,842,048 bytes |
| `dtype_raw.F32` | 77,824 bytes |

From those totals, using `GB = 1024**3` and `n_weights = bf16_raw / 2`, I obtained:

- 753,329,921,024 BF16 weights.
- 1,403.19 GiB total raw size and 1,403.19 GiB BF16 size, with the file labeling these values `GB`.
- Byte-split: 1,052.85 GiB compressed, 24.967% whole-model reduction, 12.0053 bits/weight.
- Regroup K15: 979.87 GiB compressed, 30.168% whole-model reduction, 11.1730 bits/weight.
- Escape rate: `100 * 187,756,789 / 753,329,921,024 = 0.02492357...%`, rounded to 0.0249%.
- BF16 tensors: 59,509.

Every generated numeric or Boolean field matched `glm52_standalone_result.json`; there were zero field differences. The nonnumeric coverage description also agrees with Claim 1.

Adversarial note from the original gate: `merge_glm.py` searched `tools/glm_ckpts/*.json`, whereas the verification bundle stored its copies under `tests/artifacts/ckpts/`. The public method package now checks both locations; this was a reproduction-packaging issue, not an arithmetic discrepancy.

## Claim 3: lossless flag and enforcement path

**Result: verified from the checkpoints and source; refutation failed.**

- All 16 checkpoint `acc.all_lossless` values are Boolean true.
- For every processed BF16 tensor, `process_shard` calls `enc_bytesplit_verify`, obtains `ok`, then executes `acc["all_lossless"] = acc["all_lossless"] and ok`.
- The sole `np.array_equal` call in the coder is `np.array_equal(high_rec, high)` inside `enc_bytesplit_verify`.
- The low bytes are separated as `low = a[0::2]` and charged/stored verbatim as `n * 8` bits. Equality of reconstructed high bytes plus unchanged low bytes implies exact reconstruction of every 16-bit BF16 payload word.

For newly processed tensors, no contradictory update path exists: the flag is initialized true and thereafter remains true only through conjunction with each tensor's `ok`. There is, however, a resume path in `main` that calls `acc.update(saved["acc"])`; a resumed invocation trusts the saved checkpoint flag rather than replaying earlier tensor comparisons. Thus the literal phrase "only path that sets it" needs the normal qualification "for each newly processed BF16 tensor." This does not contradict the supplied completed checkpoints' flags or the per-tensor verification path, but it means the flag is not independently authenticated against a modified checkpoint.

Evidence boundary: the raw GLM-5.2 shard bytes are not in the bundle, so I could not independently replay the 59,509 per-tensor comparisons or authenticate that the checkpoints were produced by this exact source revision. The claim about the supplied flags and listed code path is nevertheless supported.

## Claim 4: kernel result fields

**Result: verified at reported precision; refutation failed.**

- `kernel_v10_result.json` reports `K = 476`, `ratio_il_over_bf16 = 0.733`, `WIN = true`, and relative error `1.31e-07`.
- `bench.log` identifies the up-projection shape as `[2048, 6144]`, prints `USING_K=476`, and repeats the result JSON values.
- From the rounded reported times, `14306.6 / 19521.6 = 0.7328600115`, which rounds to 0.733 at the file's three-decimal precision.
- `14306.6 < 19521.6`, independently reproducing `WIN = true`.
- The kernel source computes the ratio as `ti/tb`, the win as `ti < tb`, and the relative error as the maximum absolute output difference divided by the maximum absolute BF16-reference output magnitude plus `1e-9`.

The relative-error value is consistent between JSON and log and is produced by the stated code path. It cannot be numerically replayed from this bundle because `/workspace/gpu_sample.npz` and the original A40 execution environment are not supplied.

## Claim 5: bandwidth-unit correction

**Result: verified; refutation failed.**

The benchmark covers

`2048 * 476 * 6144 = 5,989,466,112` weights.

Using the reported microsecond timings:

- BF16: `2 * 5,989,466,112 / 0.0195216 / 1e9 = 613.6245 GB/s`.
- Interleaved: `1.5 * 5,989,466,112 / 0.0143066 / 1e9 = 627.9758 GB/s`.

These round to approximately 614 GB/s and 628 GB/s. The source's `cuda_time` returns milliseconds, while the printed `GBps` calculation divides bytes per millisecond by `1e9`; it is missing a factor of 1,000. The printed 0.61/0.63 values are consequently unit-mislabeled/scaled. Multiplying those already rounded fields by 1,000 gives 610/630 GB/s, consistent with the more precise recomputation from the reported times.

Relative to the claim's approximately 696 GB/s A40 peak reference, the recomputed effective values are about 88.16% and 90.23%. The 696 GB/s hardware specification itself is not encoded in the listed artifacts, so only the arithmetic conditional on that reference was recomputable here.

## Claim 6: codec bit accounting

**Result: verified; refutation failed.**

With `K = 15`, the listed coder charges exactly:

- Byte-split: `n*4 + n_esc*8 + R*max(1, ceil(log2(n_esc+1))) + K*8 + n*8` bits.
- Regroup: `n*4 + n_esc*9 + R*max(1, ceil(log2(n_esc+1))) + K*9 + n*7` bits.

These terms correspond respectively to 4-bit indices, 8- or 9-bit escape payloads, one charged escape-offset pointer per row at a width derived from the tensor's escape count, a 15-entry codebook, and the verbatim 8-bit low byte or 7-bit mantissa. `process_shard` accumulates these complete returned bit counts as `bf16_enc_bs` and `bf16_enc_rg`; the merge calculation uses those accumulated encoded sizes directly. Therefore the 30.168% regroup result includes the enumerated side costs.

Narrow accounting caveat: the formulas charge exactly the side costs named in the claim. They do not visibly charge general container/tensor metadata, and the row-pointer term uses `R` entries rather than `R+1`. Neither point contradicts the narrower claim about what the coder charges.

## Overall verdict

All six claims are internally reproduced from the supplied evidence and survive the attempted refutations. The noted limitations concern provenance replay and packaging, not a discovered numerical or source-code contradiction.

GATE_PASS
