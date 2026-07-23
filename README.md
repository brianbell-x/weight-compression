# weight-compression

I'm researching ways to losslessly compress model weights — not quantize,
not prune, but shrink the exact bits and get every single one back.

The motivation is simple: BF16 is a lazy encoding. Every weight burns 16
bits, but in a trained model the sign-and-exponent bits are wildly
repetitive. The true entropy floor is far below what we store, and every 2x
peeled off a model's footprint is headroom to run a 2x bigger model on the
same hardware. So I run experiments — a lot of them, most of which fail —
and this repo is the ledger of what survives.

One bar never moves: **bit-for-bit lossless, proven by exact round-trip on
real weights.** Estimates get labeled as estimates.

## Best so far: Split12

[Split12](Split12/) is the current leader. Byte-split BF16 — reconstruct the
high byte from its codebook, keep the low byte raw. On the full
`zai-org/GLM-5.2` 753B scan:

- **24.967% smaller** — all 59,509 BF16 tensors round-tripped bit-for-bit
- **30.168%** priced by a charged K15 accounting (estimate, not yet a codec —
  that's the next thing I'm prying at)

Compression on paper is only half of it, though — the point is smaller *and*
faster in VRAM at serving time. The [inference record](Split12/inference/)
tracks that fight: crossed BF16 at 9B dev scale, +27% decode on the full
753B, then a properly matched production SGLang retest ate my lunch (73.6 vs
208 tok/s). Thirteen kernel iterations on a B300 later, the MoE expert-w2
kernel sits at **0.80× BF16** — first sub-1.0× result. Dense shapes still
beat me. Ongoing.

More to come — the emitted stream still isn't random, which means there's
compression left on the table.

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
