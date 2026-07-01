# Nemotron 3 Nano 30B-A3B BF16: Compression-Focused Breakdown

This file is written for one purpose: help us understand only the parts of this model that matter for lossless compression of model parameters.

The target model is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, pinned locally to Hugging Face revision `cbd3fa9f933d55ef16a84236559f4ee2a0526848`.

## How To Read This

Each topic is labeled by how much attention it deserves right now.

| Label | Meaning |
| --- | --- |
| This is your area of focus | Directly affects how we compress, reconstruct, load, or measure the weights. |
| Related because | Not the main target, but it explains why the weight files are shaped or used a certain way. |
| Keep an eye on this | Could become important once we move from file compression to runtime memory or compute. |
| Purely informational | Useful background, but do not spend much time here yet. |

The goal is not to become an LLM expert in every direction. The goal is to understand enough structure to make exact, reversible compression decisions.

## Current Evidence

This is your area of focus.

We do not have the full model downloaded yet. We have enough to map the architecture and inspect the first shard.

Local files used:

- `config.json`
- `configuration_nemotron_h.py`
- `modeling_nemotron_h.py`
- `model.safetensors.index.json`
- `model-00001-of-00013.safetensors`
- tokenizer files, chat template, generation config, model card, safety/privacy notes, and evaluator config

What we know from these files:

- The full model has `13` safetensors weight shards.
- The safetensors index lists `6,243` tensors.
- The index reports `31,577,937,344` total parameters.
- The index reports `63,155,886,464` total tensor bytes.
- We currently have shard `1 / 13`.
- Shard 1 contains `434` tensors.
- Shard 1 is mostly `BF16`, with a few small `F32` tensors.

Why this matters:

Lossless compression work needs exact names, shapes, dtypes, byte counts, and reconstruction checks. The index gives us the map. The shard gives us real bytes to test.

## The Short Version

This is your area of focus.

Nemotron 3 Nano is not a plain dense transformer. It is a hybrid model:

```text
input text
  -> tokenizer
  -> token IDs
  -> embedding matrix
  -> 52 layers made of Mamba2, MoE, and attention blocks
  -> final norm
  -> LM head
  -> next-token scores
```

The compression-relevant fact is this:

```text
Most stored bytes live in MoE expert weights.
Only a small subset of experts is active for each token.
All experts still have to be stored exactly.
```

That means we should separate three ideas:

| Goal | What it means |
| --- | --- |
| Smaller files | Compress the stored weight bytes and reconstruct them exactly. |
| Lower runtime memory | Keep some weights compressed, paged, streamed, or off-device until needed. |
| Lower compute | Do less math without changing the model output. This is harder if we require no quality loss. |

For now, our first serious target is smaller files with exact reconstruction. Runtime memory comes after we understand the tensor structure.

## Terms You Actually Need

This is your area of focus.

### Parameter

A parameter is one learned number in the model. In this model, most parameters are `BF16`, so each one usually takes 2 bytes.

Why it matters:

Compression operates on these stored values. We need to preserve them exactly unless an experiment is explicitly labeled lossy.

### Tensor

A tensor is a named block of parameters with a shape.

Example:

```text
backbone.embeddings.weight: BF16 [131072, 2688]
```

That means:

- name: `backbone.embeddings.weight`
- dtype: `BF16`
- shape: `131072` rows by `2688` columns
- values: `352,321,536`
- bytes: `704,643,072`

Why it matters:

We should compress tensors by their meaning, not as one anonymous 63 GB byte stream.

### Shape

Shape tells us the dimensions of a tensor.

Example:

```text
[1856, 2688]
```

This is a matrix with 1,856 rows and 2,688 columns.

Why it matters:

The same bytes can have different structure depending on shape. Row deltas, column deltas, transposes, tiling, expert grouping, and byte-plane transforms all depend on shape.

### Dtype

Dtype means the storage format for each number.

Important dtypes here:

| Dtype | Bytes per value | Meaning |
| --- | ---: | --- |
| `BF16` | 2 | bfloat16, used for most large learned weights |
| `F32` | 4 | float32, used for a few small sensitive/state tensors |

Why it matters:

BF16 has only 2 bytes per value. The high byte and low byte may have different statistical behavior. That is a major compression clue.

### Hidden Size

Hidden size is the width of the model's internal token vector.

For this model:

```text
hidden_size = 2688
```

After tokenization, each token becomes a vector with 2,688 BF16 numbers as it moves through the model.

Why it matters:

Hidden size appears everywhere in tensor shapes. It is the main channel count connecting embeddings, experts, attention, Mamba, and the LM head.

### Layer

A layer is one repeated processing block inside the model.

This model has `52` layers. Each layer has:

```text
RMSNorm -> one mixer -> residual add
```

The mixer is one of:

- Mamba2
- MoE
- attention

Why it matters:

Layer number and layer type give us natural compression groups. We should compare Mamba weights with Mamba weights, expert weights with expert weights, and attention weights with attention weights.

### Expert

An expert is a small feed-forward network inside an MoE layer.

Each MoE layer has:

- `128` routed experts
- `1` shared expert
- only `6` routed experts active per token

Why it matters:

Experts are the storage giant in this model. They are probably our main research surface.

## What The Name Means

Related because.

| Name piece | Meaning | Compression relevance |
| --- | --- | --- |
| `30B` | Rough stored parameter class. Index says 31.58B. | Tells us this is a huge storage problem. |
| `A3B` | Around 3B to 3.5B active parameters per token. | Explains why runtime compute is lower than full stored size. |
| `BF16` | Most large weights are bfloat16. | Directly affects byte-level compression strategy. |
| `NemotronH` | Custom Hugging Face architecture name. | Tells us local custom code controls loading and execution. |

Do not overfocus on the marketing name. The real compression facts are tensor names, shapes, dtypes, and usage.

## File Layout

This is your area of focus.

The Hugging Face snapshot is not one giant model file. It is a small set of control files plus large tensor shards.

| File | Why we care |
| --- | --- |
| `config.json` | Declares dimensions, layer count, layer pattern, dtype, and MoE settings. |
| `configuration_nemotron_h.py` | Turns config fields into architecture settings. |
| `modeling_nemotron_h.py` | Shows how tensors are used during forward/generation. |
| `model.safetensors.index.json` | Maps every tensor name to a shard file. This is our first manifest source. |
| `model-00001-of-00013.safetensors` through `model-00013-of-00013.safetensors` | Store the actual tensor bytes. We currently have shard 1. |
| `tokenizer.json` | Converts text to token IDs. Useful for inference, not central to weight compression. |
| `chat_template.jinja` | Prompt formatting. Mostly not relevant to parameter compression. |
| `generation_config.json` | Default sampling/stop settings. Mostly not relevant to parameter compression. |

Keep an eye on this:

Safetensors gives us named tensors and metadata without loading the whole tensor into memory. That is exactly the kind of format we want for building compression probes.

## Tensor Names

This is your area of focus.

Most tensors live under these roots:

```text
backbone.*
lm_head.weight
```

Global tensors:

```text
backbone.embeddings.weight
backbone.norm_f.weight
lm_head.weight
```

Layer tensors:

```text
backbone.layers.0.*
backbone.layers.1.*
...
backbone.layers.51.*
```

Why this matters:

Tensor names already tell us the semantic group:

- embedding
- final output head
- layer number
- block type
- expert number
- projection type
- norm
- router

Compression research should preserve and exploit this grouping.

## Main Architecture Numbers

This is your area of focus.

| Setting | Value | Why it matters |
| --- | ---: | --- |
| Stored dtype | `bfloat16` | Most values are 2 bytes. |
| Hidden size | `2688` | Main internal vector width. Appears in many shapes. |
| Vocabulary size | `131072` | Drives embedding and LM head size. |
| Layers | `52` | Gives repeated structure to compare. |
| Mamba layers | `23` | Dense regular tensors. |
| MoE layers | `23` | Main storage cost. |
| Attention layers | `6` | Smaller storage cost, important for runtime KV cache. |
| Routed experts per MoE layer | `128` | Creates huge repeated expert population. |
| Shared experts per MoE layer | `1` | Always active expert path. |
| Routed experts active per token | `6` | Explains active-vs-stored parameter gap. |
| MoE expert intermediate size | `1856` | Expert matrix shape. |
| Shared expert intermediate size | `3712` | Shared expert matrix shape. |

Purely informational for now:

The tokenizer has vocab size `131072`. Important token IDs include `<unk>`, `<s>`, `</s>`, `<|im_start|>`, `<|im_end|>`, `<think>`, and `</think>`. This matters for running the model, but not much for compressing its weights.

## The 52-Layer Pattern

This is your area of focus.

The config uses this pattern:

```text
MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
```

Meaning:

| Symbol | Layer type | Compression meaning |
| --- | --- | --- |
| `M` | Mamba2 | Dense, regular, medium-sized tensors. |
| `E` | MoE | Huge expert tensors. Main focus. |
| `*` | Attention | Smaller stored tensors, runtime cache concern. |

Counts:

- `23` Mamba2 layers
- `23` MoE layers
- `6` attention layers

Layer indices:

| Type | Layers |
| --- | --- |
| Mamba2 | 0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25, 28, 30, 32, 35, 37, 39, 41, 44, 46, 48, 50 |
| MoE | 1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51 |
| Attention | 5, 12, 19, 26, 33, 42 |

Why this matters:

This pattern tells us which tensors should be compared against each other. It also tells us where storage pressure is coming from.

## Inference Flow

Related because.

Inference means using the model to predict the next token.

High-level flow:

```text
text
  -> tokenizer
  -> token IDs
  -> backbone.embeddings
  -> 52 residual blocks
  -> backbone.norm_f
  -> lm_head
  -> logits over 131072 tokens
```

Each layer does:

```text
input hidden states
  -> RMSNorm
  -> Mamba2, MoE, or attention
  -> add original input back in
```

Why this matters:

If we compress only files, we only need exact reconstruction before loading. If we compress runtime memory, we need to know when each tensor is needed during inference.

Do not overfocus here yet:

The exact math inside every operation is less important than the storage map at this stage.

## Mamba2 Layers

Related because.

Mamba2 is a sequence-processing layer. It is not attention, and it does not store a growing key/value cache the same way attention does.

Verified layer 0 tensors from shard 1:

| Tensor | Dtype | Shape | Role |
| --- | --- | --- | --- |
| `backbone.layers.0.norm.weight` | BF16 | `[2688]` | RMSNorm before mixer |
| `backbone.layers.0.mixer.in_proj.weight` | BF16 | `[10304, 2688]` | Projects hidden states into Mamba streams |
| `backbone.layers.0.mixer.conv1d.weight` | BF16 | `[6144, 1, 4]` | Depthwise causal convolution |
| `backbone.layers.0.mixer.conv1d.bias` | BF16 | `[6144]` | Conv bias |
| `backbone.layers.0.mixer.dt_bias` | BF16 | `[64]` | Time-step bias |
| `backbone.layers.0.mixer.A_log` | F32 | `[64]` | State transition parameter |
| `backbone.layers.0.mixer.D` | F32 | `[64]` | Skip/state parameter |
| `backbone.layers.0.mixer.norm.weight` | BF16 | `[4096]` | Gated Mamba norm |
| `backbone.layers.0.mixer.out_proj.weight` | BF16 | `[2688, 4096]` | Projects output back to hidden size |

Layer 0 takes `77,490,048` bytes including its block norm.

Keep an eye on this:

Mamba tensors are dense and regular. They are good for basic compression probes:

- BF16 byte histograms
- high-byte vs low-byte entropy
- row/column deltas
- block tiling
- exact chunk compression

But they are not the dominant storage cost compared with MoE experts.

## MoE Layers

This is your area of focus.

MoE means Mixture of Experts. Instead of one dense feed-forward block, the layer has many expert blocks. A router chooses which experts to use for each token.

Each MoE layer has:

- `128` routed experts
- `1` shared expert
- `1` router gate
- `1` block norm

Each routed expert has two matrices:

| Tensor pattern | Dtype | Shape |
| --- | --- | --- |
| `mixer.experts.N.up_proj.weight` | BF16 | `[1856, 2688]` |
| `mixer.experts.N.down_proj.weight` | BF16 | `[2688, 1856]` |

The shared expert has two larger matrices:

| Tensor pattern | Dtype | Shape |
| --- | --- | --- |
| `mixer.shared_experts.up_proj.weight` | BF16 | `[3712, 2688]` |
| `mixer.shared_experts.down_proj.weight` | BF16 | `[2688, 3712]` |

The router has:

| Tensor | Dtype | Shape |
| --- | --- | --- |
| `mixer.gate.weight` | BF16 | `[128, 2688]` |
| `mixer.gate.e_score_correction_bias` | F32 | `[128]` |

Verified layer 1 in shard 1:

- `261` tensors total
- `256` routed expert tensors
- `2` shared expert tensors
- `2` router tensors
- `1` norm tensor
- `2,594,936,576` bytes

Why this matters:

This is the largest and most promising area for lossless compression research. MoE gives us many same-shaped matrices that have related jobs:

- expert 0 up projection
- expert 1 up projection
- expert 2 up projection
- ...
- expert 127 up projection

That repeated structure is exactly where we should look for redundancy.

Keep an eye on this:

There are several different questions hiding inside "compress MoE":

| Question | Why it matters |
| --- | --- |
| Do experts in the same layer have similar byte distributions? | Helps decide whether to compress per expert or per layer. |
| Do same-numbered experts across layers look related? | Could reveal cross-layer structure. |
| Are `up_proj` and `down_proj` statistically different? | They may need separate codecs/transforms. |
| Can inactive experts stay compressed during runtime? | This could reduce memory pressure without changing outputs. |
| Can routing metadata help prefetch/decompress experts? | This becomes important for runtime compression. |

Do not overfocus here yet:

Do not try to improve the router or change which experts are selected. That risks changing model behavior. Our first pass should preserve all expert tensors exactly.

## Attention Layers

Keep an eye on this.

Attention layers appear at layers `5, 12, 19, 26, 33, 42`.

Expected tensor shapes:

| Tensor | Shape | Reason |
| --- | --- | --- |
| `q_proj.weight` | `[4096, 2688]` | 32 query heads * 128 head dim |
| `k_proj.weight` | `[256, 2688]` | 2 KV heads * 128 head dim |
| `v_proj.weight` | `[256, 2688]` | 2 KV heads * 128 head dim |
| `o_proj.weight` | `[2688, 4096]` | Projects 32 heads back to hidden size |
| `norm.weight` | `[2688]` | RMSNorm before attention |

This is grouped-query attention:

```text
32 query heads / 2 KV heads = 16 query groups per KV head
```

Why it matters:

Attention is not the largest stored-weight target in this model. But attention has a runtime memory issue: during generation, it stores a key/value cache that grows with sequence length.

For our first weight-file compression pass, attention is lower priority than MoE.

For later runtime-memory work, attention becomes more important.

## Embeddings And LM Head

This is your area of focus.

The input embedding tensor is verified in shard 1:

```text
backbone.embeddings.weight: BF16 [131072, 2688]
```

That is:

```text
131072 * 2688 = 352,321,536 values
352,321,536 * 2 bytes = 704,643,072 bytes
```

The output head is separate:

```text
lm_head.weight: [131072, 2688]
```

The config says:

```text
tie_word_embeddings = false
```

That means the input embedding and output head are not the same tensor.

Why this matters:

These two matrices together are about 1.31 GiB in BF16. They have the same shape and related vocabulary meaning, so they are a natural comparison pair.

Good first questions:

- Are embedding rows and LM-head rows statistically similar?
- Are any rows exactly repeated?
- Do token ranges have different entropy?
- Does token frequency correlate with compressibility?

## Loading The Model

This is your area of focus.

The config points Transformers to custom model code:

```json
"auto_map": {
  "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
  "AutoModel": "modeling_nemotron_h.NemotronHForCausalLM",
  "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"
}
```

Normal loading does roughly this:

```text
read config.json
  -> load custom config class
  -> load custom model class
  -> read model.safetensors.index.json
  -> map tensor names to shard files
  -> load safetensors into matching module parameters
```

Local status:

- `AutoConfig.from_pretrained(..., trust_remote_code=True, local_files_only=True)` works.
- `AutoTokenizer.from_pretrained(..., local_files_only=True)` works.
- Full model construction currently fails because `mamba-ssm` is not installed.
- Full weight loading is not possible yet because shards 2 through 13 are missing.

Why this matters:

Lossless file compression can be tested before the model runs. But runtime-memory compression must eventually integrate with this loading path or another serving runtime.

Keep an eye on this:

The model card recommends NeMo Framework 25.11.01. Windows/CPU is fine for inspection, but probably not the final inference environment.

## What "Stored, Shaped, Typed, Grouped, Loaded, And Used" Means

This is your area of focus.

We should call this milestone reached only when we can answer these questions reproducibly:

| Word | Required answer |
| --- | --- |
| Stored | Which shard file contains each tensor? Later: exact byte offsets too. |
| Shaped | What is every tensor's exact shape? |
| Typed | What dtype is every tensor? Which tensors are not BF16? |
| Grouped | Which layer, block type, expert, projection, or norm does each tensor belong to? |
| Loaded | What code path maps shard bytes into model parameters? |
| Used | When does each tensor affect output, cache, memory, or compute? |
| Verified | Can we compress, decompress, hash, reload, and prove exact equality? |

Current progress:

| Area | Status |
| --- | --- |
| Stored | Mostly reached through `model.safetensors.index.json`. |
| Shaped | Reached for shard 1; inferred for the rest from config and code. |
| Typed | Reached for shard 1; inferred mostly BF16 elsewhere. |
| Grouped | Reached from tensor names and the layer pattern. |
| Loaded | Reached at code-path level; full runtime load blocked by missing deps and shards. |
| Used | Reached at architecture level; exact runtime behavior still needs live inference verification. |

## Compression Strategy From This Pass

This is your area of focus.

The first compression direction should be tensor-aware, not file-blind.

Natural buckets:

- embeddings
- LM head
- MoE routed expert `up_proj`
- MoE routed expert `down_proj`
- MoE shared expert projections
- MoE router weights
- Mamba input projections
- Mamba output projections
- Mamba conv/state tensors
- attention Q/O projections
- attention K/V projections
- RMSNorm weights

Why this matters:

Different tensor families probably compress differently. If we mix them too early, we hide the signal.

Best next artifact:

Build a tensor manifest with:

- tensor name
- shard file
- local file present or missing
- shape
- dtype
- number of values
- byte count
- layer number
- block type
- expert ID when present
- projection type
- exact hash when local bytes are available

Then every compression experiment should point back to this manifest.

## What To Ignore For Now

Purely informational.

These topics are real, but not first-order for our immediate goal:

- prompt templates
- chat formatting
- stop tokens
- benchmark scores
- safety notes
- sampling settings
- exact generation quality
- training data scale, except as broad context
- model branding

They matter when running or evaluating the model. They do not decide how the stored weight bytes are organized.

## Open Questions

Keep an eye on this.

- Do all remaining shards follow the inferred shapes and dtypes exactly?
- How much BF16 entropy lives in high bytes versus low bytes?
- Do experts inside the same MoE layer share measurable structure?
- Do same-numbered experts across MoE layers share measurable structure?
- Are `up_proj` and `down_proj` different enough to need separate transforms?
- Can expert tensors stay compressed until the router needs them?
- Which runtime should be the first exact-load target: Transformers, vLLM, TRT-LLM, SGLang, or NeMo?
- Does the local attention path intentionally avoid explicit rotary embeddings, or is position handled elsewhere?

## Practical Next Step

This is your area of focus.

Create the tensor manifest.

That is the next thing that turns this from reading into research. Once we have a manifest, we can run small reversible compression probes on shard 1 and later scale across all 13 shards.

The first experiments should be:

1. Hash every local tensor exactly.
2. Measure raw byte entropy per tensor.
3. Split BF16 into high-byte and low-byte streams and measure each separately.
4. Compare compression ratios by tensor family.
5. Compare routed experts inside layer 1.
6. Test exact reconstruction after each reversible transform.

The rule stays simple:

```text
No exact reconstruction, no claim.
```
