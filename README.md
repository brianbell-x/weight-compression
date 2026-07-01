# weight-compression

Lossless compression of BF16 LLM weights. Store the weights **~25-30% smaller** with
outputs that are **bit-for-bit identical** (not quantization, no quality tradeoff), in
a fixed-width form the matmul can read directly. Demonstrated on the whole NVIDIA
Nemotron-3-Nano-30B-A3B-BF16 model: all 6,174 BF16 tensors reconstruct exactly.

- **Read it:** the visual writeup is live at
  https://brianbell-x.github.io/weight-compression/
- **Test it:** see [REPRODUCE.md](REPRODUCE.md).

## Quick start

```bash
uv sync
# verify losslessness + reduction on any BF16 model you have:
uv run python reproduce.py --model /path/to/model.safetensors
```

`bit-exact round-trip : ALL PASS` is the whole point. Full instructions, including
reproducing the headline 30% on the real 30B model, are in
[REPRODUCE.md](REPRODUCE.md).
