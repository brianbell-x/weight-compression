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

## The Lossless Wall (why the numbers stop where they do)

A stage can only be lossless down to the source's entropy, and that floor is now
measured on the true weights. The BF16 exponent byte is concentrated (~2.7 bits) but
the **mantissa is genuinely random** — order-0 entropy 7.96/8 bits, and a real
compressor (lzma, byte-delta) gets no further. So no lossless stack beats ~11 b/w
fixed-width (~31%) / ~10.5 entropy-coded (~34%), and it is essentially all exponent.
Truly random bits cannot be shrunk losslessly — pigeonhole, no exceptions. With the
sign bit also fully live (1.0 b), 8 of the 16 bits per weight are provably random,
which is why gains much beyond ~34–35% storage / ~30% fusible are
information-theoretically impossible losslessly.
