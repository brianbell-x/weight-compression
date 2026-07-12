# weight-compression

Lossless compression of BF16 LLM weights. Store the weights **~30% smaller** with
outputs that are **bit-for-bit identical** (not quantization, no quality tradeoff), in
a fixed-width form the matmul can read directly.

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
