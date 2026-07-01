# Research Notes

## Compression Principle

For compression, "not exactly the same" does not kill the idea. Exact duplicates are only the simplest case. The more important question is whether related tensors have repeated patterns, similar byte distributions, shared structure, predictable deltas, or compressible BF16 byte layouts.

### Where the compressibility lives in a BF16 weight (measured, whole model)

A BF16 scalar is `[sign:1][exponent:8][mantissa:7]` (bit 15 → bit 0). Across the
entire model the lossless signal is not spread evenly:

- **Exponent = compressible.** Weight magnitudes sit in a narrow band, so the top
  exponent bits are low-entropy or literally constant per tensor (measured: 15.4%
  of the BF16 mass is provably-constant bits, and *all* of them are top-of-exponent
  bits). This is the field the working codecs exploit — 0001 (entropy-code the
  exponent plane) and 0009 (per-group exponent codebook, fusible 25–29%).
- **Sign = live.** Almost never constant; carries real information.
- **Mantissa = the hard frontier.** The 7 mantissa bits are fully live: order-0
  entropy ~7.95/8, and **zero provably-constant bits in any tensor**. No lossless
  method has compressed it yet. Any further lossless gain beyond 0009 must come
  from mantissa *statistics* (higher-order structure), not from removing dead bits
  — dead-bit removal is now exhausted and lives entirely in the exponent.

## Tensor Anatomy

A tensor is a named, shaped block of values. For this project, do not think of a tensor as only a math idea. Think of it as both:

- a model object: numbers the model uses during inference
- a file object: bytes stored inside `.safetensors` shards

### Core Parts

| Part | Plain meaning | Compression relevance |
|---|---|---|
| Name | The tensor's label, like `backbone.layers.1.mlp.experts.0.up_proj.weight` | Tells us where the tensor belongs: layer, expert, projection, embedding, norm, router, etc. |
| Shape | The tensor's dimensions, like `[2688, 1856]` | Tells us how many values exist and how they are arranged. Same-shaped tensors are good comparison targets. |
| Rank | The number of dimensions in the shape | A vector has rank 1, a matrix has rank 2, higher tensors have rank 3+. |
| Axis / dimension | One direction inside the shape | Helps us ask whether rows, columns, heads, experts, or channels have repeated structure. |
| Element | One slot inside the tensor | The smallest tensor-level unit. |
| Scalar value | The single number stored in one element | Usually one model parameter for weight tensors. |
| Dtype | The number format, such as BF16 or F32 | Determines how many bytes each scalar uses and what byte patterns are possible. |
| Device | Where the tensor is loaded, such as CPU or GPU | Runtime concern. Not usually part of the saved tensor file. |
| Requires grad | Whether training tracks changes for the tensor | Training concern. For inference/compression, usually informational only. |

### Storage Parts

| Part | Plain meaning | Compression relevance |
|---|---|---|
| Bytes | The actual stored data | This is what file compression directly sees. |
| Bits | The 0/1 pieces inside each byte | Deepest storage level. BF16 uses 16 bits per scalar. |
| Byte order / endianness | The order bytes are written for each value | Needed for exact reconstruction. |
| Offset | Where the tensor's bytes start inside the shard | Needed to locate and extract tensor payloads. |
| Length | How many bytes the tensor occupies | Should equal `number of elements * bytes per element`. |
| Stride | How many steps in memory move along each axis | Important in memory; safetensors stores dense contiguous tensors. |
| Contiguity | Whether values are laid out without gaps | Dense contiguous storage is simpler to compress and reconstruct. |
| Header metadata | JSON-like description before the raw bytes | Contains names, dtypes, shapes, and offsets. |

### Model-Meaning Parts

| Part | Plain meaning | Compression relevance |
|---|---|---|
| Layer | The model block the tensor belongs to | Lets us compare repeated structure across depth. |
| Block type | Mamba, attention, MoE, embedding, norm, or output head | Different block types may need different compression strategies. |
| Expert | One MoE sub-network | Experts are a major focus because Nemotron has many same-shaped expert tensors. |
| Projection | A learned matrix that moves values from one width to another | Up, down, gate, query, key, value, output projections are common comparison groups. |
| Row | A horizontal slice of a matrix | Useful for heatmaps, row statistics, and delta comparisons. |
| Column | A vertical slice of a matrix | Useful for finding repeated channel behavior. |
| Distribution | The pattern of values, such as common ranges or repeated byte frequencies | More important than exact duplicates for advanced compression. |

### Important Mental Model

```text
model
  -> shard files
    -> tensor names
      -> tensor metadata: dtype, shape, offsets
      -> tensor payload bytes
        -> scalar values
          -> bytes
            -> bits
```

For lossless compression, the final test is whether we can rebuild the tensor exactly enough for the chosen target:

- exact same bytes, if preserving the original file representation
- exact same tensor values, if repacking into a new format

## The Two Compression Walls (why the numbers stop where they do)

Compression ratios **compound** (multiply) across stages that remove *different*
structure — this is the real lever and how we reach 71–78% combined (lossless exponent ×
lossy experts). But compounding is bounded by two hard floors, both now measured on the
true weights:

1. **Lossless wall ≈ 34%.** A stage can only be lossless down to the source's entropy.
   The BF16 exponent byte is concentrated (~2.7 bits) but the **mantissa is genuinely
   random** — order-0 entropy 7.96/8 bits, and a real compressor (lzma, byte-delta) gets
   no further. So no lossless stack beats ~11 b/w fixed (~31%) / ~10.5 entropy-coded
   (~34%), and it is essentially all exponent. Truly random bits cannot be shrunk
   losslessly — pigeonhole, no exceptions.

2. **Lossy (post-hoc) wall ≈ 3–4 bits/weight.** After a randomized-Hadamard *incoherence*
   rotation (absorbed losslessly into the matmul), the structureless experts are
   near-**Gaussian i.i.d.**, so their lossy compressibility obeys the rate-distortion law
   `D(R) ≈ 2⁻²ᴿ` (per-weight rel-error ≈ `2⁻ᴿ`). Measured errors track it: 4-bit 3.35%,
   3-bit 7.8%, 2-bit 16.9%. Vector quantization recovers only the small scalar-vs-lattice
   "space-filling gain"; error feedback lowers *output* error below per-weight distortion;
   neither beats the bound. This is the information-theoretic form of "the experts are
   dense."

**Compression axes.** Lossless (bit-exact) → capped ~34%, storage/VRAM only. Lossy
quantization (fewer bits/weight) → post-hoc capped ~3–4 b/w at good quality. Sparsity/MoE
(fewer weights active/token) → runtime only, already exploited. Restructuring/superposing
weights → does not help (dense is ~optimal per-param, train-time exp1–4).

## The weight manifold (the only way past the post-hoc wall)

The rate-distortion wall bounds compression of the **fixed** weights training happened to
produce. But a model's *function* does not uniquely determine its weights — there is a
manifold of function-equivalent weight sets, and only some points on it are
low-bit-representable. Post-hoc quantization is stuck at the dense-Gaussian point;
**quantization-aware training / distillation searches the manifold** for a low-bit-friendly
point that computes the same function, with downstream layers co-adapting to the
quantization error. This is why a model trained low-bit (e.g. BitNet-1.58) works where
post-hoc 2-bit does not. Going below ~78% combined requires *changing* the weights
(training), not re-encoding them.
