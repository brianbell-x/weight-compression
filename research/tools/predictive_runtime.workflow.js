export const meta = {
  name: 'predictive-exp-runtime',
  description: 'Verify the fusible separable-predictive exponent lossless codec across all 13 shards (exact round-trip, whole-model runtime bits/weight vs 0009) and assess GPU-kernel fusibility, then write the runtime-real lossless verdict',
  phases: [
    { title: 'Shards', detail: 'run the fusible predictive codec per shard, numel-weighted bpw' },
    { title: 'Kernel', detail: 'GPU-kernel fusibility / bandwidth analysis of the reconstruction' },
    { title: 'Synthesize', detail: 'whole-model runtime-real lossless % + writeup' },
  ],
}

const SNAP = args?.snap || 'C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot'
const ART = 'C:/dev/compression/research/candidates/0012-lossless-ceiling/tests/artifacts'
const SCRIPT = `${ART}/predictive_wholemodel.py`

const common = `
PURE LOSSLESS, bit-exact, RUNTIME-focused. The codec (fusible, fixed-width, random-access):
per weight = sign(1) + exp_residual_code(index+escape) + mantissa(7), where
exp_residual = exp - round(row_base[i]+col_base[j]-grand); row/col bases are O(1) side vectors.
Reconstruction is all register ops -> fusible (never re-inflated to full BF16 in VRAM).
Established: whole-shard-1 = predictive 11.19 b/w (30.04%) vs 0009 baseline 11.30 b/w (29.37%);
storage-lossless ceiling ~34% (variable-length, NOT fusible). The tool is ${SCRIPT}
(prints an aggregate JSON per shard: baseline/predictive fusible bpw + pct_vs16). Use uv run python.
`

const SHARD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['shard', 'weights', 'predictive_bpw', 'baseline_bpw', 'predictive_pct', 'baseline_pct', 'roundtrip_ok'],
  properties: {
    shard: { type: 'string' }, weights: { type: 'number' },
    predictive_bpw: { type: 'number' }, baseline_bpw: { type: 'number' },
    predictive_pct: { type: 'number' }, baseline_pct: { type: 'number' },
    roundtrip_ok: { type: 'boolean' },
  },
}

phase('Shards')
const shards = Array.from({ length: 13 }, (_, i) =>
  `${SNAP}/model-${String(i + 1).padStart(5, '0')}-of-00013.safetensors`)

const results = (await parallel(shards.map((sh, i) => () =>
  agent(`${common}
Run the codec on ONE shard and report its numbers. Execute:
  uv run python "${SCRIPT}" "${sh}"
It prints a JSON aggregate (baseline_fusible_bpw, predictive_fusible_bpw, baseline_pct_vs16,
predictive_pct_vs16, weights, tensors). The script asserts the exponent round-trip is exact
(pred+resid==exp) per tensor — set roundtrip_ok true iff it ran without assertion error. Return
this shard's numbers.`,
    { label: `shard:${i + 1}`, phase: 'Shards', schema: SHARD_SCHEMA }))))
  .filter(Boolean)

phase('Kernel')
const kernel = await agent(`${common}
Assess GPU-KERNEL FUSIBILITY of the predictive codec vs 0009's plain exponent-codebook kernel.
The dequant per weight: load index (fixed bits), LUT[index] -> residual; add row_base[i] +
col_base[j] (two small per-row/col vectors, cacheable in shared memory / registers); shift to
exponent position; OR sign bit and raw mantissa -> BF16 in register; feed the matmul. Analyze:
(1) extra work vs 0009 (2 adds + 2 small cached loads per weight) — negligible vs memory traffic?
(2) do the row/col base vectors fit in shared memory / L1 for a tile (sizes ~1856-10304 int8)?
(3) does it stay bandwidth-bound (the whole point) at the ~0.67pt-narrower read? (4) any addressing
hazard to random access? Give a concrete verdict: is the +0.67pt bandwidth win realizable by a
Marlin/FLUTE-class fused kernel, or does the predictor overhead eat it? Reason from arithmetic
intensity; you need not run a GPU.`,
  { label: 'kernel', phase: 'Kernel', effort: 'high' })

phase('Synthesize')
const synth = await agent(`${common}
Synthesize the RUNTIME-REAL lossless verdict. Shard results:
${JSON.stringify(results, null, 2)}
Kernel analysis:
${kernel}
Compute the numel-weighted WHOLE-MODEL predictive vs baseline fusible bits/weight and %, confirm
all shards round-tripped exact, and state the honest runtime-real lossless ceiling (~30%?) vs the
storage ceiling (~34%) and 0009's 29.4%. Write C:/dev/compression/research/candidates/0012-lossless-ceiling/RUNTIME.md
with: the whole-model number, the per-shard table, the kernel verdict, and a one-line honest
summary of how much of the +4pt storage gain became runtime-real and why the rest cannot. Also
give a proposed one-paragraph findings-ledger addendum (do not edit the ledger). Return the path
+ a 5-bullet summary.`,
  { label: 'synthesize', phase: 'Synthesize', effort: 'high' })

return { shards: results.length, kernel: !!kernel, synth }
