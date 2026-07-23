# weight-compression

How small can LLM weights actually get without losing a single bit? This repo
is me chasing that question in public.

BF16 is a lazy encoding. Every weight burns 16 bits, but in a trained model
the sign-and-exponent bits are wildly repetitive — a handful of values cover
nearly everything. The entropy floor sits far below what we store, and every
2x peeled off a model's footprint is headroom to run a 2x bigger model on the
same hardware. Each method that proves out gets its own folder here; this
page is the trail.

The bar never moves: **bit-for-bit lossless, verified by exact round-trip on
real weights.** Estimates get labeled as estimates. Anything less doesn't
count.

## Where the trail is right now

**[Split12](Split12/)** — the one that worked. Byte-split BF16: reconstruct
the high byte from its codebook, keep the low byte raw. On the full
`zai-org/GLM-5.2` 753B scan: **24.967% smaller, all 59,509 tensors
round-tripped bit-for-bit.** A charged K15 accounting prices the same idea at
**30.168%** — an estimate, not yet a codec, and the thing I'm prying at next.

Then the question that actually matters: not smaller on disk — smaller *and
faster in VRAM*. The [inference record](Split12/inference/) is the honest
version of that fight: crossed BF16 at 9B dev scale, hit +27% decode on the
full 753B, then a properly matched production SGLang retest ate my lunch
(73.6 vs 208 tok/s). So: kernel campaign on a B300, thirteen iterations, and
the MoE expert-w2 kernel came out at **0.80× BF16** — first sub-1.0× result.
Dense shapes still beat me. That fight is ongoing.

One scar worth keeping visible: the raw artifacts and kernels from those GPU
runs were lost to a bad local cleanup. What's in `Split12/inference/` is
rebuilt from my ledgers — the numbers stand, the code doesn't. Lesson
recorded, backups exist now.

## The trail

- [`Split12/`](Split12/) — the method: format, verifier, references
- [`Split12/inference/`](Split12/inference/) — serving and kernel record
  (A6000 → RTX 6000 Ada → 8× B300 → SGLang retest → tensor-core campaign)
- [The writeup](https://brianbell-x.github.io/weight-compression/Split12/) —
  the visual version

## Run it yourself

Don't trust any of this — rerun it:

```bash
cd Split12 && uv sync
uv run verify.py <org>/<repo>   # streams the model shard by shard from HF
```

`ALL_BF16_TENSORS_LOSSLESS: true` or it didn't happen.
