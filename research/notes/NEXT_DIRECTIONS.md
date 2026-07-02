# Next-Directions Scouting — 2026-07-01

Product of a 44-agent adversarial scouting pass (5 repo readers → 10 ideation
lenses → dedupe → 2 adversarial vetters per direction; several vetters ran live
probes on real weights — see `research/candidates/0009-fusible-exponent-codebook/tools/contrarian_probe.py`). Each direction below
survived a kill attempt against the findings ledger and an impact/effort judge.
Verdicts shown as [soundness/value].

## Fresh measurements produced during vetting (real weights, holdout-validated)

- **True joint (left,up) exponent context beats the hashed 8-bit context by only
  ~0.006 b/w** (up 2.6365 vs 2.6426, down 2.4957 vs 2.4960 holdout). The 0012
  "2-D context" numbers went through `ctx=(left ^ up*131)&0xFF` — but the hash
  turns out to be a near-lossless context quantizer. Un-hashing is a dead lever.
- **Column identity alone reproduces ~85–100% of the 2-D context gain**:
  H(exp|col)=2.486 on down_proj beats the true-joint holdout (2.496). Column
  index is address-derived → zero context dependency, fully fusible.
- **Per-column K15 codebooks halve escape rates**: up 6.02%→3.15%, down
  3.17%→1.61%. H(sign|col)=0.978 (sign has a small column lean too).
- **Fractional-radix packing + per-tile BASE re-windowing: falsified during
  vetting** (probe run on shard-2 experts). Per-tile re-windowing is a strict net
  loss (−0.006 to −0.031 b/w; residuals are spatially stationary — tile-local
  windows match the global one). Salvage: per-tensor fractional-m as one more
  option in a codec chooser (~+0.03–0.05 b/w on concentrated tensors only).

## Integrity items (do these regardless of direction)

1. ~~Untracked evidence~~ **DONE 2026-07-01**: `stream_validate.py`, `contrarian_probe.py`,
   both `stream_probe_*.json`, and the .stz evidence are committed. Full Super-120B
   streamed validation kicked off in the background (50 shards, bounded disk).
2. ~~Estimate, not artifact~~ **DONE 2026-07-01**: `stz.py` serializes the whole model —
   63.16→43.02 GB = **31.89%**, all 13 shards SHA-256-verified from the .stz alone
   (`tests/artifacts/stz/stz_manifest.json`, `stz_verify.json`). Direction E's core is
   shipped; remaining E scope (loader, DFloat11/ZipNN same-bytes bench, cold-start
   measurement) is still open.
3. ~~Predictor pathology~~ **DONE 2026-07-01**: the .stz per-tensor min-envelope chooser
   (raw16 fallback) eliminates all >16 bpw cases.
4. ~~Stale 0012 wording~~ **DONE 2026-07-01**: RUNTIME.md carries the wording
   correction (in-memory identity check, .stz is the serialized artifact) and
   0012/brief.md is marked complete with the realized numbers.

## Repricing note (2026-07-01, post-.stz)

The realized .stz codec measures **10.8975 b/w BF16 numel-weighted (experts 10.8949)** —
0.30 b/w better than the 11.1976 accounting baseline all directions below were priced
against. Consequences:
- **The fusible-vs-storage gap is now ~0.4 b/w, not ~0.7** (10.90 realized vs ~10.5
  entropy-coded floor). Direction A's expected band (10.65–11.0) partially overlaps the
  realized number; its remaining upside is ≤~0.25–0.4 b/w, not the 4-pt story.
- **Direction D must benchmark against the realized stz per-tensor costs**
  (`stz_tensor_stats.jsonl`), not 11.2072: the second-level escape codebook already
  recodes escapes cheaply, so "escape rate halves" no longer converts 1:1 into
  the +0.26 b/w priced below. Column keying now has to win on *index-plane* bits
  (3-bit indices on low-escape tensors) plus residual escape savings.

## Tier 1 — new structure, strong on both verdicts

### A. Block-granular entropy coding with O(1) *tile* access  [strong/strong]
The ~0.7 b/w fusible-vs-storage gap is priced as "the cost of random access" —
but the matmul reads **tiles, not single weights**, so pay the addressability tax
per block, not per weight. Two formats: (a) 4096-symbol superblocks, 32
interleaved rANS lanes, two-level offset index (~0.13 b/w tax), per-tensor static
tables (the confirmed-winning granularity); (b) 64–128-weight blocks tANS-coded
and **padded to a fixed byte budget** (fixed stride = random access), wholesale
block-escape bitmap for overflowers. The 11.2 b/w "fusible ceiling" was only ever
measured for per-*weight* fixed-width codes; block granularity is the single
untested point on the project's own map (0012's probe excluded within-block
sequential decode explicitly). Expected: ~10.65–11.0 b/w = **31.2–33.4% fusible**,
recovering most of the 4-pt gap. First probe (~1 day, CPU): per-block ideal code
length histograms (pure numpy) to size the padding overhead FIRST — that is the
only genuine unknown; then exact ANS accounting, block ∈ {32..16K} × padding
percentile ∈ {90..99}; success gate is entropy-relative (realized ≤ per-tensor
order-0 H(sign|exp) + 0.15 b/w). Falsifier: heavy-tailed per-block code lengths
making padding overhead eat the gain.

### B. Lossless compression of low-precision-native checkpoints  [strong/strong]
The frontier ships FP8/FP4, not BF16: DeepSeek-V3/R1 are native FP8 (~688 GB, no
BF16 master), gpt-oss is MXFP4, NVIDIA publishes Nemotron in FP8/NVFP4. BF16's
anatomy replays in miniature: E4M3 = [1|4|3] with sign+exp the structured bits;
E8M0/block-scale planes are pure hyper-concentrated exponent fields; INT4
GPTQ/AWQ code streams are discretized bells (own prior measurement: 2.975/4 bits
→ ~0.8 b/w headroom on every published INT4 artifact). The 0009 recipe (top-K /
band codebook + fixed-width index + sparse escape) ports directly. Expected: FP8
~15–25%, INT4 artifacts ~10–24%, MXFP4/GGUF 5–15%. Novelty caveat from vetting:
arXiv 2508.19263 (ZipNN follow-up) already published FP8 Huffman-on-exponent
numbers — our defensible claim is **first whole-model fixed-width/fusible-form
lossless numbers** for these formats, plus INT4/NVFP4 which remain unclaimed.
First probe (~2 h): extend `stream_validate.py`'s dtype filter to F8_E4M3; pull
DeepSeek-V3's index.json first and pick an **expert-bearing** shard (not shard 1;
experts start at layer 3); field entropies + top-K coverage + escape at 3-bit.

### C. Exact cross-checkpoint delta coding  [strong/strong]
Every falsified delta was within-model between *different functions*
(cross-expert cos ~0.03, emb-vs-lm_head ~0.03). The same tensor across
*training time* is a categorically different, never-tested correlation source.
Mechanics: sub-half-ulp updates round back bit-identical in BF16 (RLE match
flags ≈ free); changed weights rarely change magnitude class, so the XOR/delta
high byte is near-delta-function (~0.2–0.8 b vs 2.7 standalone) and only the 7
random mantissa bits get paid. Verified live targets: Nano Base-BF16 vs release,
Omni-Reasoning / Labs-Elastic siblings, OLMo-2 training revisions, LoRA-merged
pairs. Expected: second checkpoint at ~3–9.5 b/w given the base (50–80% smaller
than standalone); pessimistic zero-exact-match case still ~41–47%. **Cheapest
probe on the list (<1 h, bandwidth-bound)**: stream shard pairs, align by name,
report %-bit-identical uint16 words + XOR high-byte H0. Vetting adjustments:
include an expert-heavy shard pair (not just shard 1) and mandatory baselines
(`zstd --patch-from`, xdelta3) — beat both or it isn't a contribution.

### D. Column-keyed codebooks + escape forensics — **FALSIFIED 2026-07-01 (candidate 0014)**
Probed on shard-7 layer 27 (256 tensors, exact accounting, parity vs stz exact,
skeptic-verified): all 16 pt/shared × g × b variants lose to realized stz; the
adoption-aware envelope picked the baseline 256/256. Mechanism: post
second-level-recoder, escape reductions convert at only (k−b) bits per converted
escape. The null on escape spatial structure delivered the promised optimality
certificate — the escape mask is near-random. Salvage leads moved to "New leads"
below. Do not re-propose per-weight column keying; the untested remnant is
cross-layer transfer of the *certificate* (cheap closure rider), not a rescue.

## Tier 2 — consolidate the crown (product + realization)

### E. Container v2 + "ship it": safetensors-z / .stz  [viable/strong ×2 merged]
Fixes integrity item 2 and harvests guaranteed leftovers in one move: real
serialized container (header + per-tensor codebook/band + 4-bit index plane +
raw mantissa plane + escape stream), CLI compress/verify, transparent
decompress-on-load loader (distribution scope — NOT the parked kernel path).
Joint per-tensor chooser over {raw16, regroup-K, predictor, BASE band,
fractional-m} × index_bits × escape_width (6-bit escape recoding ≈ +0.3 pt
model-wide; chooser kills the 18-bpw pathology). Same-bytes benchmark vs
DFloat11/ZipNN with **decode GB/s as the headline** (our fixed-width plane is
near-memcpy; ZipNN's Huffman is not), both profiles fielded (fixed-width
mmap-able; max-ratio entropy-coded) so the known 3–4 pt gap isn't misread.
Cold-start (download+decode-to-RAM) is the live performance axis now that
kernels are parked — never measured in this repo, half a day to measure.
Expected: +0.15–0.45 pt whole-model AND the flagship number upgrades from
estimate to artifact.

### F. Archival codec at the honest storage floor (~34.6–35.1%)  [viable/strong]
0012 RESULTS.md §4 recommended but never built the full archival rANS codec
(position-mixed contexts: column-group × left-exp → expert exponents ~2.46–2.55
holdout; + H(mant|exp) 0.068 b/w). Un-hashing itself is dead (see fresh
measurements) — the levers are position mixing and realization. Deliverable is
half compression, half **certification**: converts "~34% ceiling" from an
order-0 artifact into a context-model-certified floor, and undercuts the
field's order-0 "Shannon-limit" claims. ~1 day probe on local tensors, then one
mid-model shard.

### G. Generality program  [viable/viable]
Full Super-120B validation (then Ultra-550B as a multi-day background job),
cross-modality one-shard probes (FLUX/diffusion, VLM, audio — needs only an
--index-path flag), top-100 HF census (~200 s + ~4.7 GB per model, delete as you
go) emitting per-model dtype/entropy/ratio stats. The census doubles as a
structure-hunting instrument for every other direction and is a citable artifact
on its own. Fit the exponent-concentration scaling law across 30B/120B/550B.

## Tier 3 — cheap certificates and lottery tickets

- **H. Permutation-gauge hunt** [viable/viable]: hidden-unit permutation is free
  (π ≈ 0.0035 b/w). Run the FREE upper bound first: per tensor,
  H0(exp) − mean per-row H0(exp) (bounds any row ordering); kill if <0.05 b/w.
  Cross-expert Hungarian matching on ±folded row cosines with a **rotation
  null** (prior ~20–30%; if it fires, 2.66–2.87 → ~2.2–2.4 b/w = +1.5–3 pt
  storage — the largest untapped number named anywhere; if not, a definitive
  gauge-blind closure of 0003/0007/0010).
- **I. Adversarial randomness certification** [viable/viable]: every compressor
  battery on disk is LZ-class; no PAQ/zpaq/CM-class tool has ever run. Order:
  packed-popcount 16×64 bit-pair MI census with circular-shift null (15 min,
  gates everything), then bespoke logistic context-mixing on mantissa bit-planes
  (online-adaptive code length = constructive bound). Expected gain ~0; the
  deliverable is terminal closure certificates the ledger currently can't state.
- **J. Hardware-decodable load tier** [viable/viable]: two-tier format — VRAM
  stays 0009 regroup; disk/wire wraps each plane in nvCOMP/Blackwell-DE-decodable
  chunks. Measured foothold: H(sign+exp in index) = 3.81/3.57 of 4 bits → ~0.6
  b/w rides the wire free. Honest framing per vetting: ~1–1.6 GB less wire + the
  fixed-function-decode story vs shipping 0009 as-is; chunk-frame at 64 KB from
  the start or the probe is non-decisive.
- **K. Undertrained-mass atlas** [viable/weak]: Nano headroom is bounded-known
  (0010: 132.8 MB = 0.21% total). Lead with **Ultra** (are ≥506-expert models'
  rarely-routed experts less trained → more compressible?) and re-derive the
  escape-density claim from stream_probe_ultra_550b.json first (30 min gate).

## New leads (2026-07-01, from 0014's escape forensics — chooser-scale, fold into E)

- **Per-row second-level escape k for up_proj**: up_proj escape mask is row-overdispersed
  (Fano ~2.3 vs ~0.79 binomial null) — a per-row k chooser could harvest it. Ceiling
  ~0.01–0.03 b/w; price through the (k−b) conversion rule.
- **Per-column BASE re-centering as a chooser option only**: fires only if column exponent
  distributions are shifted copies (not shape-varying); ceiling ~0.02–0.05 b/w.
- **Extend direction I's certification to stz's emitted planes** (index plane, side
  tables): 0014's forensics already did the escape mask; finishing this states terminal
  closure on the whole emission.
- ~~Cross-layer closure rider~~ **DONE 2026-07-01** (7-layer sweep, `--layer` flag):
  certificate holds for layers ≥13; early layers keep a small adoption-aware win
  (envelope +0.098 L1 → 0 by L13; ≈ +0.011 b/w experts ≈ +0.065 pt model-wide).
  **Concrete E-scope item: add `colkey` to the .stz per-tensor chooser** — measured
  free upside, mostly layers 1–10. Vetting's column numbers were early-layer-only
  (layer identity was the hidden variable).
- **Pricing rule for all future escape-based pitches**: value = (k−b) bits × converted
  escapes, NOT 16 or 9 bits per escape avoided. Reprice before probing.

## Falsified during this scouting pass — do not re-propose

- **Per-tile BASE re-windowing / fractional-radix packing as a headline lever**
  (probe run on real shard-2 experts): per-tile windows match the global window
  (residuals spatially stationary) → strict net loss. Fractional-m survives only
  as a chooser option (~+0.03–0.05 b/w on concentrated tensors).
- **Un-hashing the 2-D context**: true joint ≈ hashed (Δ ~0.006 b/w). The hash
  was never the bottleneck.

## Compounding note (2026-07-01, user directive)

Prefer directions that study the *current best compressed form* over ones that
restart from raw weights (see AGENTS.md "Compound rather than pivot"). Mapped
onto the tiers: **D** (escape forensics on 0009's codebook output), **A** (codes
the residual fusible-vs-storage gap of the current form), **E/F** (realize and
certify the current form), and **I** (randomness-certify the emitted streams)
are compounding moves. **B/C/G/K** start from new substrates (other dtypes,
other checkpoints, other models) — still valid, but they are breadth, not
compounding; sequence them behind at least one depth pass on the 0009/0012
output streams.

## Recommended sequence

1. **Tonight (zero risk)**: commit `stream_validate.py` + probe JSONs +
   `contrarian_probe.py`; kick off full Super-120B streamed validation in the
   background. (G core, integrity item 1)
2. **This week, hour-scale, information-per-hour order** (updated 2026-07-01 after
   D's falsification): A's per-block code-length histogram (~2 h, CPU-local, gates
   the rANS design; now the primary path to the ~0.4 b/w gap) → C's shard-pair
   delta probe (<1 h, when bandwidth frees up — Super-120B validation is hogging
   the flaky link) → B's FP8 expert-shard probe (~2 h, same bandwidth constraint).
3. **Next**: whichever of A–C fires, plus E's remaining scope (loader,
   DFloat11/ZipNN same-bytes bench, cold-start measurement, and the chooser-scale
   "New leads" above).
