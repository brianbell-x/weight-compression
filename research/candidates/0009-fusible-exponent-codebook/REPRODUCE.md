# Reproduce it yourself

The claim: **BF16 model weights compress ~25-30% losslessly** (bit-for-bit
identical), in a fixed-width form the matmul can read directly. Everything below is
verifiable from a fresh clone. Run every command from the repo root.

## 0. Setup (once)

```bash
git clone https://github.com/brianbell-x/weight-compression
cd weight-compression
uv sync          # installs numpy, safetensors, torch, ... (uv: https://docs.astral.sh/uv/)
```

Only `numpy` is needed for the lossless proof (steps 1-2). `torch` + `triton` are
used only in the optional GPU step (3).

## 1. Fastest check: run it on ANY BF16 model (~1 min, no big download)

`tools/reproduce.py` (this candidate's folder) encodes every BF16 tensor, decodes it back from ONLY
`(codebook + fixed-width index + in-order escape stream + raw mantissa)`, and checks
the reconstructed bytes are bit-for-bit identical with SHA-256. numpy only, no GPU.

```bash
uv run python research/candidates/0009-fusible-exponent-codebook/tools/reproduce.py --model /path/to/any/model.safetensors
```

`--model` takes a single `.safetensors` file, a directory of shards, or an
hf_snapshot dir. The shape of the output (numbers vary by model):

```
  BF16 tensors checked : 30
  weights              : 500,822,720
  bit-exact round-trip : ALL PASS
  byte-split           : 12.011 b/w   -24.9%   escapes 0.055%
  regroup (headline)   : 11.225 b/w   -29.8%   escapes 2.407%
```

`bit-exact round-trip : ALL PASS` is the whole claim: the compressed form
reconstructs the original weights exactly. Exit code is 0 on all-pass, 1 otherwise.
Flags: `--layout both|bytesplit|regroup`, `--limit N` (stop after N tensors).

## 2. Reproduce the headline number on the real model

The published result is the whole **NVIDIA Nemotron-3-Nano-30B-A3B-BF16** (58 GB,
6,174 BF16 tensors).

Download it (HuggingFace account required; if the repo is gated, run
`uv run hf auth login` first):

```bash
uv run hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local-dir models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot
```

Run the whole-model verifier (streams all 13 shards, one tensor in memory at a time,
~15 min, a few GB RAM):

```bash
uv run python research/candidates/0009-fusible-exponent-codebook/tests/artifacts/whole_model_lossless.py
```

It prints and writes `whole_model_lossless_result.json` next to itself:

```
  "ALL_BF16_TENSORS_LOSSLESS": true,
  "n_bf16_tensors": 6174,
  "byte_split_K15_12bw":  { "whole_model_reduction_pct": 24.95, ... },
  "regroup_K15_11p3bw":   { "whole_model_reduction_pct": 30.03, ... }
```

**Don't want 58 GB?** Pull one shard (~4.65 GB) and point `reproduce.py` at it:

```bash
uv run hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    model-00001-of-00013.safetensors \
    --local-dir models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot

uv run python research/candidates/0009-fusible-exponent-codebook/tools/reproduce.py \
    --model models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot/model-00001-of-00013.safetensors
```

## 3. (optional) GPU: the weights run compressed, read directly

Shows the compressed form read *directly* by a matmul, with BF16 rebuilt only in
registers and never re-inflated to full width in VRAM. Needs a CUDA GPU + Triton.
The sample is regenerated from the real shard, so do step 2's one-shard download first.

```bash
# regenerate the sample (two real layer-1 expert tensors, encoded); self-checks bit-exact
uv run python research/candidates/0009-fusible-exponent-codebook/tests/artifacts/extract_sample.py

# fused dequant+matmul vs BF16 cuBLAS: correctness + latency; writes gpu_bench_result.json
uv run python research/candidates/0009-fusible-exponent-codebook/tests/artifacts/bench_gpu.py
```

`lossless_on_gpu: true` means the narrow form reconstructs the exact BF16 weights
on-device. The latency ratio is bandwidth/GPU-dependent -- see `tests/GPU_RUN.md`
and `tests/test-002.md` / `test-005.md` for the full picture.

## Where everything is

```
research/candidates/0009-fusible-exponent-codebook/tools/reproduce.py                                          the portable verifier above
research/candidates/0009-fusible-exponent-codebook/
  brief.md                                            the method and claims
  writeup/index.html                                  visual explainer (also live on GitHub Pages)
  tests/
    test-001.md .. test-005.md                        the write-up behind each result
    GPU_RUN.md                                         GPU reproduction notes
    artifacts/
      whole_model_lossless.py                         whole-model proof (the 6,174-tensor / 30% number)
      probe_regroup.py, probe_byte_split.py           per-shard bit-budget probes (layer-1 experts)
      extract_sample.py, bench_gpu.py, cpu_validate.py  GPU fused-kernel path
      *_result.json                                   the recorded outputs
research/notes/findings-ledger.md                     what has been tried and settled
```

All scripts expect the model under
`models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot/` and are run from the
repo root.
