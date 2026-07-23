# weight-compression

I'm researching ways to losslessly compress model weights.

Not quantize. Not prune. Shrink the exact bits, get every single one back.

## Why

I want SOTA models running locally on consumer hardware, affordably.

The blocker is memory. Frontier models are hundreds of gigabytes, and BF16 is
a lazy encoding. Every weight burns 16 bits, but in a trained model the
sign-and-exponent bits are wildly repetitive. The real entropy floor is far
below what we store.

Lossless compression is my first foot forward. Every 2x saved on a model's
footprint = headroom for a 2x bigger model on the same hardware.

So I run experiments. A lot of them. Most fail. This repo is the ledger of
what survives.

**The bar:** bit-for-bit lossless, proven by exact round-trip on real
weights. Estimates get labeled as estimates.

## Best so far: Split12

[Split12](Split12/) is the current leader.

The idea: byte-split BF16. Reconstruct the high byte from a codebook. Keep
the low byte raw.

**Compression** on the full `zai-org/GLM-5.2` 753B scan:

- **24.967% smaller.** All 59,509 BF16 tensors round-tripped bit-for-bit.
- **30.168%** priced by a charged K15 accounting. Estimate, not yet a codec.
  That's what I'm prying at next.

**Serving** is the other half of the fight. The point is smaller *and*
faster in VRAM. The [inference record](Split12/inference/):

- Beat BF16 at 9B dev scale. 9.32 vs 7.75 tok/s on A6000 (+20%). 21.3 vs
  20.6 on RTX 6000 Ada (+3.4%).
- Full 753B on 8× B300: 14.93 vs 11.75 tok/s (+27%), with 24.8% lower
  resident weights.
- Then a properly matched production SGLang retest ate my lunch: 73.6 vs 208
  tok/s. The earlier baseline was a harness artifact.
- Thirteen kernel iterations on a B300 later: MoE expert-w2 at **0.80×
  BF16**. First sub-1.0× result.
- Dense shapes still beat me. Ongoing.

Method folder: [`Split12/`](Split12/) - format, verifier, references.
Serving and kernel record: [`Split12/inference/`](Split12/inference/).
Writeup: [the visual version](https://brianbell-x.github.io/weight-compression/Split12/).

Verify it yourself:

```bash
cd Split12 && uv sync
uv run verify.py <org>/<repo>   # streams the model shard by shard from HF
```

`ALL_BF16_TENSORS_LOSSLESS: true` or it didn't happen.

## What's next

Split12 is the best result so far. It won't be the last.

The emitted stream still isn't random. That means there's compression left
on the table. New methods get their own folder here when they prove out.

More to come.
