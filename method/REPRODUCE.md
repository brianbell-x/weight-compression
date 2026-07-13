# Reproduce it yourself

The full zai-org/GLM-5.2 scan produced two separate measurements: **K15
charged bit accounting of 30.168%** (11.173 bits/weight; 1403.19 GiB to 979.87
GiB), and an **independently decoded byte-split representation at 24.967%**
(12.005 bits/weight) for which all 59,509 BF16 tensors round-tripped bit-for-bit.
The K15 representation was not independently serialized or decoded at GLM
scale. Run every command from the repo root.

Prior-art scope: [ZipNN](https://arxiv.org/abs/2411.05239) and
[DFloat11](https://arxiv.org/abs/2504.11651) established lossless BF16 exponent
compression, and [ZipServ](https://arxiv.org/abs/2603.17435) already
demonstrated fixed-length coding with direct register reconstruction. This
reproduction validates this repository's different 15-symbol joint
sign-and-exponent representation on GLM-5.2; it does not claim those underlying
ideas as new.

## 0. Setup (once)

```bash
git clone https://github.com/brianbell-x/weight-compression
cd weight-compression
uv sync          # installs numpy, safetensors, huggingface_hub, ... (uv: https://docs.astral.sh/uv/)
```

No GPU needed. Peak disk is ~one shard (~5 GB): the validator streams the model
from Hugging Face one shard at a time and deletes each shard after processing it.

## 1. Fastest check: one shard (~5 GB download, minutes)

```bash
uv run verify.py zai-org/GLM-5.2 --shards 1
```

This downloads shard 1 of 282, encodes every BF16 tensor with the fixed-width
codebook codec, verifies the reconstruction is bit-for-bit identical, prints the
running bit accounting, and deletes the shard. Look for:

```
"ALL_BF16_TENSORS_LOSSLESS": true,
"regroup_K15_11p3bw": { "reduction_pct": ~30, ... }
```

The legacy `ALL_BF16_TENSORS_LOSSLESS` field refers to the byte-split inverse
check. The `regroup_K15_11p3bw` result is separately charged bit accounting;
the validator does not decode that layout.

## 2. The headline number: the whole model (all 282 shards, ~1.4 TB streamed)

Same command without `--shards`. It checkpoints after every shard, so it
resumes if interrupted:

```bash
uv run verify.py zai-org/GLM-5.2
```

Single-worker this is bandwidth-bound (~1.4 TB through your connection). The
published run partitioned the shards across 16 workers on 8 cloud pods and
finished in ~70 minutes; to do the same, give each worker its own range and
scratch dir:

```bash
# worker k of 16 (k = 0..15): 18 shards each, last range covers the remaining 12
uv run verify.py zai-org/GLM-5.2 --start $((k*18)) --shards 18 --work work_range_$k
```

Then merge the 16 per-range checkpoints into the headline JSON (asserts 282
distinct shards, no overlap, no gap):

```bash
# collect the checkpoint_GLM-5.2.json files into method/tools/glm_ckpts/, then:
uv run python method/tools/merge_glm.py
```

Expected output (recorded in `tests/artifacts/glm52_standalone_result.json`):

```
"ALL_BF16_TENSORS_LOSSLESS": true,
"n_bf16_tensors": 59509,
"whole_model_reduction_pct": { "byte_split": 24.967, "regroup_K15": 30.168 },
"bf16_only_bits_per_weight": { "byte_split": 12.005, "regroup_K15": 11.173 },
"compressed_GB": { "byte_split": 1052.86, "regroup_K15": 979.87 }
```

The JSON field name `compressed_GB` is historical; these values use GiB
(`bytes / 1024^3`).

## What "lossless" means here, exactly

Per tensor, the coder splits each BF16 value into a high byte (sign + 7
exponent bits) and a low byte, encodes the high plane against a 15-entry
codebook with escapes, **reconstructs the high plane from only
(codebook + indices + escape stream)**, and checks `np.array_equal` against
the original; the low byte is stored verbatim. That check ran fresh on all
59,509 tensors. The regroup (headline) variant is an exact bit accounting of
the same field split; its decode is proven bit-exact by the public verifier
(`tools/reproduce.py` there decodes both layouts end-to-end — point it at any
GLM shard if you want the regroup decode demonstrated on this model too).

The retained artifacts and code paths were blind-gate checked on 2026-07-12:
`tests/artifacts/BLIND_VERIFICATION_0061.md`. This was not a replay from the raw
1.4 TiB source or a serialized K15 GLM artifact.

## Where everything is

```
method/
  RESULTS.md                                        the gated result and caveats
  tools/stream_validate.py                          the streaming validator verify.py runs
  tools/reproduce.py                                portable end-to-end decode proof for both layouts
  tools/merge_glm.py                                merges the 16 range checkpoints -> headline JSON
  tools/bench_kernel_v10.py                         historical dense-path A40 benchmark
  tests/artifacts/
    glm52_standalone_result.json                    the recorded headline numbers
    ckpts/                                          the 16 per-range checkpoints from the published run
    BLIND_VERIFICATION_0061.md                      independent verification report
verify.py                                           repo-root entry point used above
```

Auth note: if the repo is gated on Hugging Face, accept the license on the
model page, then `uv run hf auth login` (or set `HF_TOKEN`) and retry.
