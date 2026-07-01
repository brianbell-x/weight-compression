# Steering Note: "If we just decompress at inference, aren't we adding steps for nothing?"

*Audience: someone learning how LLMs run, without a CS/ML background. Terms are defined as we go.*
*Status: research-steering note. Synthesizes three findings memos plus a verification pass; corrections from verification are applied inline.*

---

## 1. The question, and the short answer

**The worry, plainly:** We want to shrink the model's weight files. But if at run time we just expand them back to their original form before the model uses them, haven't we done the same math as before, plus an extra unpacking step? Wouldn't that be slower, or at best break even?

**The one-paragraph answer:** It depends entirely on *one* thing: **does the compressed data ever get expanded back to full size in the GPU's main memory before the math reads it?** If yes, you were right to worry — you saved disk and download size, but the running cost is unchanged (and on-the-fly lossless unpacking can even make it slower). If instead the data stays small all the way into the chip's compute units, and is expanded only in tiny fast on-chip memory at the last instant before the multiply, then the extra unpacking step is **nearly free** and decoding actually gets **faster**. The reason it can be free is the central fact of LLM inference: generating text is **limited by how fast weights can be moved out of memory, not by how fast the math runs.** The math units are mostly sitting idle waiting for data. So trading "a little more idle-time math" for "a lot fewer bytes moved" is a winning trade — but only if you never re-inflate the bytes in main memory.

---

## 2. Why generating text is "memory-bandwidth bound" (the core mechanism)

A few terms first, in plain language:

- **Weights / parameters:** the billions of numbers learned during training. They *are* the model. Nemotron stores ~30 billion of them.
- **VRAM / HBM:** the GPU's main memory, where weights live while the model runs. ("HBM" = high-bandwidth memory; it is the GPU's RAM.)
- **Memory bandwidth:** how many bytes per second you can move *out of* VRAM into the chip's compute units. On a high-end H100 GPU this is about 3.35 terabytes per second — fast, but finite.
- **FLOPs:** floating-point operations, i.e. the actual arithmetic (multiplies and adds). "Compute" = doing FLOPs.
- **Prefill vs decode:** *Prefill* is reading your prompt (all its words at once). *Decode* is generating the reply one word-piece ("token") at a time. Decode is what dominates the cost of real chatbot use.

Here is the mechanism. When the model generates **one** token, it must pass a single small vector of numbers through every weight matrix it uses. To do that, it reads each of those weights out of VRAM **and uses each one exactly once** (one multiply-add). This is a "matrix times vector" operation, and it has a brutal property: you move a huge number of bytes but do very little math per byte.

We measure this with **arithmetic intensity** = (FLOPs done) ÷ (bytes moved). For decoding one token in BF16 (a 2-byte number format):

- math per weight = 2 FLOPs (one multiply, one add)
- bytes per weight = 2 bytes
- intensity ≈ **1 FLOP per byte.**

Now compare to where the GPU *stops* being starved for data — the **"ridge point"** = (peak math speed) ÷ (memory bandwidth). For an H100 this is on the order of **150–300 FLOPs per byte** depending on which peak you use. (The exact figure is fuzzy because vendors quote "peak" math speeds that include a sparsity trick not used here; the honest dense number lands in that band. It does not matter for the conclusion.)

So decode runs at ~1 FLOP/byte against a ridge of a few hundred — roughly **100–300x below** the point where math would be the bottleneck. **The compute units are idle the vast majority of the time, waiting for weights to arrive from VRAM.** This is what "memory-bandwidth bound" means.

Two important contrasts:

- **Prefill is the opposite.** Reading a whole prompt at once lets each weight be reused across many tokens, pushing intensity into the ~1000 FLOPs/byte range — compute-bound. So prefill is limited by math; decode is limited by memory movement. Compression for *runtime* speed is really about helping decode.
- **Batching changes the math.** Serving many users at once ("batch size" > 1) lets each loaded weight serve several tokens, raising intensity roughly in proportion to the batch. A *dense* model crosses into compute-bound around batch 32–100. Keep this in mind — it is why some methods that lose at batch 1 win at large batch.

**The payoff sentence:** because we are so far below the ridge, *moving fewer bytes per token makes decode faster almost in proportion, and any extra arithmetic we add (to unpack those bytes) is hidden under the idle time we already had.* That is the whole reason compression-at-inference can be a real win rather than wasted steps.

---

## 3. The decision table: which kind of compression saves what

There are five **separate** cost axes. Conflating them is the main way people fool themselves. Keep them apart:

| Axis | Plain meaning | For Nemotron, scales with… |
|---|---|---|
| **Storage** | bytes on disk | all ~30B params (~60+ GB in BF16) |
| **Load / transfer** | download + read-from-disk + copy into GPU, once per startup | all ~30B params, one time |
| **Resident VRAM** | bytes held in GPU memory while serving | all ~30B params (experts can't be predicted, so all must be present) + KV/Mamba state |
| **Per-token bandwidth** | bytes read from VRAM for each generated token | only the **active** ~3B params per token |
| **FLOPs** | arithmetic per token | only the active ~3B; far below the GPU's ceiling at small batch |

Now the regimes. The decisive question for each: **where does the compressed form live during the forward pass, and when (if ever) is it expanded back to full width in VRAM?**

| Regime | Example | Storage | Load | Resident VRAM | Per-token bandwidth | FLOPs | Where unpack happens | Net runtime effect |
|---|---|:--:|:--:|:--:|:--:|:--:|---|---|
| **A. Disk-only, expand at load** | zstd / ZipNN / ZipLLM on the `.safetensors` | yes | yes | no | no | no | CPU, once at startup | **No runtime change.** Pure storage/shipping win. *This is the design the user's worry correctly describes.* |
| **B. Decompress during load into VRAM** | same, unpacked on the way onto the GPU | yes | yes | no | no | no | once, at load | **No runtime change.** Faster cold start only. |
| **C. Stay compressed in VRAM, unpack losslessly on the fly** | DFloat11 (entropy-codes BF16 exponent bits, bit-exact) | yes ~30% | yes ~30% | **yes ~30%** | **no — slightly worse** | adds overhead | GPU, **on the per-token critical path** | **Trade-off.** Saves resident memory (fit bigger / longer context), but *slower* at batch 1 (~1.4–2x latency). Wins only when the alternative is not fitting at all. |
| **D. Compute directly on the small form (fused dequant)** | INT4/INT8/FP8 weight-only quant, AWQ + Marlin kernel | yes 2–4x | yes | **yes 2–4x** | **yes 2–4x** | free (hidden) | GPU, **fused inside the matmul read** — narrow bytes in, never re-inflated to VRAM | **Real win.** Less VRAM *and* faster decode (measured ~3–4x). Cost is accuracy risk (lossy). |

**Reading the table.** A and B are honest storage/transfer wins with zero effect on the cost of *running* the model — exactly the case where "compress then fully decompress before use" is just added steps. C and D are the only regimes that touch the inference loop, and they touch *different* axes. C buys memory headroom at a small speed cost. D buys both memory and bandwidth and usually speeds decode up.

### The two corrections worth internalizing (from the verification pass)

1. **"In-kernel unpacking is free" is NOT a general truth — it is specifically true of *fused fixed-width dequant* (Regime D).** The tempting story "as long as you unpack in fast on-chip memory, it's free" is wrong as stated. The counterexample is Regime C / DFloat11: it unpacks on the GPU, but it writes the **full-size** BF16 weights back into VRAM before the matmul reads them. So per-token VRAM traffic actually goes *up* (read compressed, write full, read full), which is exactly why it is slower at batch 1. The free-ness comes from **never materializing the wide form in VRAM** — which fixed-width quant kernels achieve and a lossless entropy decoder does not.

2. **DFloat11 / lossless on-the-fly decode does NOT save per-token bandwidth.** Its only genuine runtime benefit is a smaller **resident** footprint (~30%), which lets you hold a bigger model or a longer context. Treat it as a *capacity* win, not a *bandwidth* win. The reason it can only shrink ~30% (vs 4x for quant) is that it is lossless: it can squeeze the redundant exponent bits of BF16 but cannot discard the genuinely random mantissa bits. Modest savings, real decode cost — net negative at batch 1 *unless* the alternative is spilling to CPU/disk.

---

## 4. When "compress then decompress" is a real win vs. just added steps

The precise conditions. It is a **real win** when **all** of the following hold:

1. **Decode is memory-bandwidth bound** (true at small/medium batch — the normal case), so there is idle compute to spend.
2. **The compressed form is consumed without ever being re-inflated to full width in VRAM** — i.e. the narrow bytes are read from VRAM and expanded only transiently in registers/on-chip SRAM, immediately before the multiply (Regime D, fused dequant).
3. **The unpacking is cheap and parallel** — a branch-free shift/multiply, not a serial, data-dependent table walk.

It is **just added steps** (or worse) when:

- The data is expanded back to full size in VRAM before use (**Regimes A and B** — storage/transfer win only, no runtime change), **or**
- The unpacker is a **general lossless entropy decoder** (Huffman/ANS-style) sitting on the per-token path of a model that **already fits in VRAM** (**Regime C used at runtime**) — you pay decode time and get no bandwidth back, because you weren't bandwidth-starved on the read of the *compressed* bytes; the matmul still needs the wide ones. DFloat11's measured ~1.4–2x slowdown at batch 1 is the warning label.

There is one important rescue case for the "loser" regimes: **if the model does not fit and the alternative is streaming weights from CPU/disk over a slow link, then compression that shrinks that transfer is a large win**, because now the transfer *is* the bottleneck. Measured MoE examples (experts streamed over PCIe) show ~5–7x speedups from compressing experts to 2–3 bits. But note this is a *fits-vs-doesn't-fit* win, and the speedups do **not** transfer to a model already resident in VRAM.

---

## 5. Implications for this project

### The Nemotron shape (treat the exact counts as "to be confirmed against `config.json`")

Nemotron-3-Nano-30B-A3B is a **sparse Mixture-of-Experts (MoE)** hybrid: ~30B total parameters, but only ~3B "active" per token because each token is routed to only a few experts. It mixes Mamba-2 layers (which carry a small *fixed-size* recurrent state) with a small number of attention layers. The precise figures circulating in the memos (31.6B total, 128 experts, top-6, 2 shared, 6 attention layers) are plausible but were **not independently verified** here — confirm them from the model's own config before using them as load-bearing numbers.

Two consequences follow from the shape, and they pull in *different* directions — this is the crucial nuance:

- **Resident VRAM scales with ALL ~30B params.** Routing is unpredictable, so every expert must be present in memory even though most stay idle on any given token. The memory-saving opportunity is **full-sized.**
- **Per-token bandwidth scales with only the ACTIVE ~3B.** Only a few experts fire per token, so the bytes *moved* per token are already small (~6–7 GB in BF16). This means the *bandwidth* saving from quantizing weights is **smaller in absolute terms than it would be on a dense 30B model.**

**Corrected steering conclusion:** for this MoE, the **resident-memory win dominates the per-token-bandwidth win.** (Memo 1's framing of MoE as "a gift to a bandwidth-reducing approach" overstates it; the gift is mostly to a *resident-memory*-reducing approach.) The hybrid Mamba design further means KV-cache traffic stays small (only the few attention layers grow with context), so **weight handling, not KV cache, is the thing to optimize** across most context lengths. A back-of-envelope batch-1 ceiling from active-weight traffic alone is ~560 tok/s on an H100 (3.35 TB/s ÷ ~6 GB/token) — useful as an upper bound only; real numbers are lower once KV/Mamba/activation reads and imperfect bandwidth use are counted.

### Mapping to the project's two tracks

The project deliberately separates **(i) exact lossless compression** (must rebuild the original bytes/weights exactly) from **(ii) lighter representation** (may change the internal form, must preserve broad capability). The cost model maps cleanly onto these:

- **Exact-lossless track → fundamentally a storage / load / fit-in-VRAM play (Regimes A, B, and the lossless end of C).** Be honest that it will **not** reduce per-token compute or bandwidth once weights are resident. Its legitimate wins are: smaller checkpoints, cheaper downloads, faster cold starts, and — via on-the-fly lossless decode like DFloat11 — a smaller *resident* footprint that buys longer context or avoids offload. Do not sell it as a decode-speed win; at batch 1 it carries a small decode tax.
- **Lighter-representation track → the only track that can win on all of VRAM, per-token bandwidth, AND decode latency at once (Regime D).** This is where fused fixed-width quant lives. It is lossy, so it must be validated on **capability**, not byte-exactness — which is exactly the project's stated standard for this track.

### The two active candidates

- **0001 — BF16 exponent-plane entropy codec.** This is a **Regime C** method (DFloat11 family): lossless, exploits redundancy in BF16 exponent bits, ~30% smaller. **Its honest payoff is storage/transfer and resident-VRAM, not decode speed.** Flag the **on-critical-path lossless-decode risk** explicitly: if its decoder is ever placed on the per-token path of an already-resident model, expect a **net slowdown** (the verified ~1.4–2x batch-1 penalty), because a serial, variable-length entropy decode cannot fuse into the matmul and re-inflates full-width weights to VRAM. Keep it as a **disk/load-time and capacity** tool. If it is ever used at runtime, justify it only by a fits-vs-doesn't-fit argument (e.g. freeing VRAM for longer context), and measure HBM bytes moved per token and tokens/s to confirm.
- **0002 — F32 dead-precision truncation.** Targeting genuinely unused/low-information precision bits in F32 tensors is a clean **lossless-leaning storage** win on the small F32 tensors. It belongs on the exact-lossless track; expect storage/load benefit, neutral on runtime unless paired with a form that is consumed narrow.

### How to weight future research

1. **Rank ideas by which cost axis they touch, and be explicit about it.** A storage-only win and a bandwidth/resident win are *different products* — never report one as the other.
2. **For runtime/serving impact, prioritize Regime-D-style methods on the experts:** keep weights narrow all the way into a fused matmul, never re-inflating to VRAM. For this MoE the biggest lever is **shrinking resident expert storage** (all 30B), with a secondary, smaller bandwidth benefit on the active 3B.
3. **The litmus test for any new idea:** *Does the compressed form ever get expanded to full width and written back to VRAM before it is used?* If yes → storage/transfer tool only (and the user's worry is literally correct). If it stays narrow and is unpacked transiently in on-chip memory (or consumed directly by a fused kernel) → potential real runtime win.
4. **Keep storage-only work, but right-size the investment.** It is real value (cheaper to store and ship a 60 GB model), it satisfies the exact-lossless proof standard, and it teaches us the weight structure. Just don't expect it to make the model *run* cheaper. The bigger prize — making the model lighter to *run* — lives on the lighter-representation track via fused, never-re-inflated formats and via shrinking the resident expert bulk of the MoE.

---

## 6. Sources

Inference cost model / roofline / decode is bandwidth-bound:
- LLM Inference Unveiled: Survey and Roofline Model Insights — https://arxiv.org/html/2402.16363v4
- Prefill is Compute-Bound, Decode is Memory-Bound — https://towardsdatascience.com/prefill-is-compute-bound-decode-is-memory-bound-why-your-gpu-shouldnt-do-both/
- Memory/compute bottlenecks in inference — https://apxml.com/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/memory-compute-bottlenecks-inference
- AI memory wall / inference latency, ridge points — https://www.spheron.network/blog/ai-memory-wall-inference-latency-guide-2026/
- Prefill/decode for concurrent requests (batch crossover) — https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests
- Inference economics of language models — https://arxiv.org/pdf/2506.04645
- Databricks — LLM inference performance best practices — https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices

GPU specs (ridge points):
- A100 vs H100 — https://www.bestgpusforai.com/gpu-comparison/a100-vs-h100
- NVIDIA Hopper architecture — https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

MoE / Nemotron:
- NVIDIA Nemotron-3-Nano Technical Report — https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf
- Nemotron-3-Nano-30B-A3B-BF16 model card — https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- The MoE-ification of the Open Model Ecosystem (intensity, expert load) — https://www.digitalocean.com/community/tutorials/mixture-of-experts-inference-cost
- Bandwidth-Efficient Adaptive MoE — https://arxiv.org/html/2512.17073v1
- OD-MoE / Mixtral expert-offload tok/s — https://arxiv.org/html/2512.03927
- HOBBIT mixed-precision expert offloading — https://arxiv.org/html/2411.01433v2
- llama.cpp MoE offload guide — https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide

Lossless on-the-fly decode (Regime C):
- DFloat11 — https://arxiv.org/html/2504.11651v3 · https://github.com/LeanModels/DFloat11

Weight-only quant / fused dequant (Regime D):
- Marlin INT4 kernel — https://github.com/IST-DASLab/marlin
- AWQ/GPTQ/Marlin throughput in practice — https://theaiengineer.substack.com/p/quantization-in-practice-gptq-vs
- AWQ 3.73x decode — https://www.spheron.network/blog/awq-quantization-guide-llm-deployment/
- Intel — weight-only quantization for LLM inference — https://www.intel.com/content/www/us/en/developer/articles/technical/weight-only-quantization-in-llm-inference.html
- Atom: low-bit quantization for efficient serving — https://arxiv.org/pdf/2310.19102
- Fused INT8 weight-only quant in Pallas — https://huggingface.co/blog/rishiraj/fused-int8-weight-only-quantization-in-pallas
- bitsandbytes 4-bit / QLoRA (memory-only, not faster) — https://huggingface.co/blog/4bit-transformers-bitsandbytes · https://mccormickml.com/2024/09/14/qlora-and-4bit-quantization/
- vLLM quantization impact — https://docs.gpustack.ai/2.0/performance-lab/references/the-impact-of-quantization-on-vllm-inference-performance/

Disk-only lossless model compression (Regimes A/B):
- ZipNN — https://github.com/zipnn/zipnn
- ZipLLM — https://arxiv.org/html/2505.06252v3
- HF transformers ZipNN RFC — https://github.com/huggingface/transformers/issues/34737
