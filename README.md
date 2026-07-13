# weight-compression

Exact compression research for BF16 LLM weights. The full GLM-5.2 scan produced
two separate results: a **30.168% K15 charged-format estimate** (11.173
bits/weight), and a **24.967% byte-split format** (12.005 bits/weight) that
round-tripped all 59,509 BF16 tensors bit-for-bit. The K15 layout was not
independently serialized or decoded at GLM scale. A separate dense 12-bit GPU
prototype reconstructs weights in registers, but its sparse exact correction
was not fused or timed; a complete exact serving path remains open.

## Prior work and distinction

[ZipNN](https://arxiv.org/abs/2411.05239),
[DFloat11](https://arxiv.org/abs/2504.11651), and
[ZipServ](https://arxiv.org/abs/2603.17435) are prior lossless BF16 compression
systems. ZipNN and DFloat11 use exponent Huffman coding; DFloat11 restores BF16
weights before matrix multiplication, while ZipServ already established
fixed-width coding with direct register reconstruction. This repository does
not claim those ideas, BF16 exponent redundancy, or the roughly 11-bit range
as new. Its tested codec is a distinct format: per-tensor 4-bit codes over 15
joint sign-and-exponent symbols, raw mantissas, and sparse exact escapes, with
checkpoint-wide validation on newer model families.

- **Read it:** the visual writeup is live at
  https://brianbell-x.github.io/weight-compression/
- **Test it:** point the verifier at any BF16 model you have.

## Quick start

```bash
uv sync
# verify losslessness + reduction on any BF16 model:
uv run verify.py --model /path/to/model.safetensors
```

`bit-exact round-trip : ALL PASS` is the whole point. To validate a model too big
for your disk, stream it shard by shard straight from Hugging Face:

```bash
uv run verify.py <org>/<repo>
```

Methods, experiments, and settled findings live under `research/`.
