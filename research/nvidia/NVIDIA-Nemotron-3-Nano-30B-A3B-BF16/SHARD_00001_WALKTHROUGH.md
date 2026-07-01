# Shard 00001 Walkthrough

This note walks through the one weight shard we currently have:

```text
models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot/model-00001-of-00013.safetensors
```

The goal is to make the inside of a model shard visible without needing to run the model.

## What We Generated

Run this to rebuild the inspection files:

```powershell
uv run python tools\inspect_shard.py
```

Outputs are here:

```text
research/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/shard_00001/
```

Important files:

| File | Open with | What it shows |
| --- | --- | --- |
| `summary.json` | JSON viewer | One-page summary of shard contents. |
| `manifest.csv` | Table | Every tensor, its shape, dtype, byte count, offsets, group, and hashes. |
| `manifest.jsonl` | JSON/text viewer | Same manifest, one JSON object per tensor. |
| `visual_samples/*.csv` | Heatmap, table, or line plot | Small slices from real tensors. |

## First Mental Model

This is your area of focus.

A safetensors shard is basically:

```text
small JSON header
large raw tensor byte block
```

For this shard:

| Part | Value |
| --- | ---: |
| Total file size | `4,991,205,008` bytes |
| Header size | `55,432` bytes |
| Tensor data starts at byte | `55,440` |
| Tensor count | `434` |
| BF16 tensors | `429` |
| F32 tensors | `5` |
| Exact duplicate tensors | `0` |

The header tells us:

```text
tensor name
tensor dtype
tensor shape
tensor byte offsets
```

The raw data block stores the actual numbers.

## What Is Inside A Tensor?

This is your area of focus.

A tensor is a shaped block of numbers. A 2D tensor is easiest to imagine as a table:

```text
row 0: value, value, value, ...
row 1: value, value, value, ...
row 2: value, value, value, ...
```

Example:

```text
backbone.layers.1.mixer.experts.0.up_proj.weight
dtype: BF16
shape: 1856 x 2688
```

That means:

```text
1856 rows
2688 columns
4,988,928 values
9,977,856 bytes
```

Since it is `BF16`, every value is stored as exactly 2 bytes.

Open this file as a table:

```text
visual_samples/bf16_inside_layer1_expert000_up_first512.csv
```

You will see rows like:

```text
flat_index,float32_value,bf16_hex,stored_low_byte,stored_high_byte
0,-0.04296875,0xbd30,48,189
1,0.0152587890625,0x3c7a,122,60
```

Meaning:

- `float32_value` is the human-readable value after converting BF16 to a normal float for display.
- `bf16_hex` is the exact 16-bit BF16 bit pattern.
- `stored_low_byte` and `stored_high_byte` are the two raw bytes in the file.

Why this matters:

Compression does not have to operate only on the displayed decimal values. It can operate on the exact bytes, split bytes into streams, rearrange values, delta rows, group experts, or do other reversible transforms.

## What This Shard Contains

This is your area of focus.

Shard 1 does not contain a clean slice like "layer 0 through layer 3." It contains complete parts and one cut-off part.

| Part | Tensors | Bytes | Status |
| --- | ---: | ---: | --- |
| Embedding matrix | 1 | `704,643,072` | Complete |
| Layer 0 Mamba | 9 | `77,490,048` | Complete |
| Layer 1 MoE | 261 | `2,594,936,576` | Complete |
| Layer 2 Mamba | 9 | `77,490,048` | Complete |
| Layer 3 MoE | 154 | `1,536,589,824` | Partial: experts 0 through 76 only |

Keep an eye on this:

Shard boundaries are packaging boundaries. They do not always match model-layer boundaries. A compression system should not assume one file equals one clean model component.

## The Biggest Signal

This is your area of focus.

Most bytes in this shard are MoE expert weights.

| Group | Bytes |
| --- | ---: |
| MoE routed expert `down_proj` | `2,045,460,480` |
| MoE routed expert `up_proj` | `2,045,460,480` |
| Embedding matrix | `704,643,072` |
| Mamba `in_proj` | `110,788,608` |
| Mamba `out_proj` | `44,040,192` |

That tells us where to spend attention:

```text
MoE experts first
embedding and LM head next
Mamba as a smaller dense baseline
attention later when we have attention shards
```

## Same Shape Does Not Mean Same Tensor

This is your area of focus.

The shard has many tensors with the same shape:

| Shape | Count |
| --- | ---: |
| `BF16 2688x1856` | `205` |
| `BF16 1856x2688` | `205` |
| `F32 64` | `4` |
| `BF16 2688` | `3` |

But exact duplicate tensor payloads found:

```text
0
```

What this means:

The easy case, "two tensors are literally identical," is not present in shard 1. But the useful compression question is broader: do related tensors have similar byte distributions, shared structure, predictable deltas, or compressible BF16 byte layouts?

## Visual Files To Open First

This is your area of focus.

### 1D line plots

Open these as line plots:

```text
visual_samples/layer0_norm_weight.csv
visual_samples/layer0_mamba_A_log.csv
visual_samples/layer1_router_bias.csv
```

What you are seeing:

- A 1D tensor is just a list of values.
- The x-axis is position in the tensor.
- The y-axis is the value at that position.

What to look for:

- spikes
- repeated flat regions
- smooth trends
- outliers

Do not overread these yet. Small vectors are useful for learning, but they are not where most bytes live.

### 2D heatmaps

Open these as heatmaps:

```text
visual_samples/embedding_rows_0_63_cols_0_63.csv
visual_samples/layer0_mamba_in_proj_0_127x0_127.csv
visual_samples/layer1_expert000_up_proj_0_127x0_127.csv
visual_samples/layer1_expert000_down_proj_0_127x0_127.csv
visual_samples/layer1_expert001_up_proj_0_127x0_127.csv
```

What you are seeing:

- A small rectangular slice from a much larger matrix.
- Color represents value.
- Nearby pixels are nearby tensor entries.

What to look for:

- obvious bands
- repeated patches
- different texture between `up_proj` and `down_proj`
- whether expert 0 and expert 1 look statistically similar

Important caution:

A heatmap is not proof of compression. It is a way to build intuition. Compression claims need numbers and exact reconstruction checks.

### BF16 byte table

Open this as a table:

```text
visual_samples/bf16_inside_layer1_expert000_up_first512.csv
```

What you are seeing:

- first 512 values from one real expert tensor
- the displayed numeric value
- the exact BF16 hex value
- the two stored bytes

This is the closest "inside the tensor" view.

### BF16 byte histogram

Open this as a table or line plot:

```text
visual_samples/byte_hist_layer1_expert000_up.csv
```

What you are seeing:

- byte values from `0` to `255`
- how often each value appears in the low byte
- how often each value appears in the high byte

Why this matters:

If high bytes and low bytes have different distributions, a reversible transform that separates them may compress better than treating BF16 as plain mixed bytes.

### Expert statistics table

Open this as a table:

```text
visual_samples/layer1_expert_stats.csv
```

What you are seeing:

- one row per routed expert in layer 1
- mean, standard deviation, min, and max for each expert's `up_proj` and `down_proj`

First observation:

Layer 1 expert means are very close to zero. `down_proj` tensors have slightly higher standard deviation than `up_proj` tensors in this shard.

Observed ranges:

| Measure | Range |
| --- | --- |
| `up_mean` | about `-0.000043` to `0.000069` |
| `up_std` | about `0.01623` to `0.01883` |
| `down_mean` | about `-0.000017` to `0.000054` |
| `down_std` | about `0.01961` to `0.02105` |

Why this matters:

`up_proj` and `down_proj` probably deserve separate compression buckets. They are same-sized populations, but not necessarily same-distribution populations.

## How To Read The Manifest

This is your area of focus.

Open:

```text
manifest.csv
```

Useful columns:

| Column | Meaning |
| --- | --- |
| `name` | Full tensor name. |
| `dtype` | Storage type, usually `BF16`. |
| `shape` | Tensor dimensions. |
| `numel` | Number of values. |
| `byte_count` | Number of raw bytes. |
| `absolute_begin` / `absolute_end` | Exact byte range inside the shard file. |
| `layer` | Layer number, when applicable. |
| `block_type` | `mamba`, `moe`, or `attention`. |
| `expert` | Expert number, when applicable. |
| `projection` | Role like `up_proj`, `down_proj`, `in_proj`, or `norm`. |
| `sha256` | Exact content hash. |
| `xxh3_128` | Faster content hash for comparisons. |

Why this matters:

This is the scoreboard. A compression experiment should be able to say:

```text
I compressed this exact tensor.
I reconstructed this exact byte range.
The hash still matches.
```

## What We Can Claim Exactly Right Now

This is your area of focus.

From shard 1, we can exactly claim:

- The shard has `434` tensors.
- It has `429` BF16 tensors and `5` F32 tensors.
- Every tensor's name, dtype, shape, byte count, and byte range are known.
- Every tensor payload has a SHA256 hash.
- No two tensor payloads in this shard are exact duplicates.
- The embedding matrix is present and complete.
- Layer 0 Mamba is complete.
- Layer 1 MoE is complete.
- Layer 2 Mamba is complete.
- Layer 3 MoE is partial, with routed experts `0` through `76` present.

What we cannot claim yet:

- Whether shards 2 through 13 have exact duplicates.
- Whether all inferred shapes match real shard contents.
- Whether later attention tensors have special structure.
- Whether a given transform improves compression across the whole model.

## Next Learning Step

This is your area of focus.

Use the visualizer in this order:

1. Open `summary.json` to see the shard as a whole.
2. Open `manifest.csv` and sort by `byte_count`.
3. Open `layer1_expert000_up_proj_0_127x0_127.csv` as a heatmap.
4. Open `layer1_expert000_down_proj_0_127x0_127.csv` as a heatmap.
5. Open `bf16_inside_layer1_expert000_up_first512.csv` as a table.
6. Open `byte_hist_layer1_expert000_up.csv` as a line plot or table.
7. Open `layer1_expert_stats.csv` as a table.

The thing to learn first:

```text
A model weight is not magic.
It is a named, shaped block of typed numbers.
Those numbers are stored as exact bytes.
Compression research starts by understanding those bytes without losing their tensor meaning.
```
