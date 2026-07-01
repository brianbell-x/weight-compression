export const meta = {
  name: 'lossless-exhaustion',
  description: 'Exhaustively attack lossless BF16 compression of the 30B from every angle (exponent context, mantissa last-stand, whole-model coder ceiling, structural dedup), verify each claim with a real coder, and report the definitive bit-exact ceiling',
  phases: [
    { title: 'Attack', detail: 'independent lossless levers, each measured with a real compressor + entropy' },
    { title: 'Verify', detail: 'adversarially re-derive any claimed gain, bit-exact round-trip' },
    { title: 'Synthesize', detail: 'compounded lossless ceiling + honest 90% verdict' },
  ],
}

const MODEL = args?.model || 'C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot'
const LEDGER = 'C:/dev/compression/research/notes/findings-ledger.md'

const common = `
PURE LOSSLESS ONLY — every weight must reconstruct BIT-EXACT (SHA-256 / bit-equality). No
quantization, no lossy, no "combination" with lossy. Model: NVIDIA-Nemotron-3-Nano-30B-A3B
(BF16, 6174 tensors, ~31.6B weights) at ${MODEL}. A BF16 weight = 1 sign + 8 exp + 7 mantissa.
Read raw bytes at header offsets (idiom in C:/dev/compression/tools/survey.py: read_header,
data_offsets; u16 = frombuffer(dtype=uint16); exp8=(u16>>7)&0xFF, sign=u16>>15, mant=u16&0x7F).
Use uv run python, numpy + stdlib compressors (zlib, bz2, lzma). Report bits/weight and % vs 16.

ESTABLISHED THIS SESSION (do not re-derive; BEAT or CONFIRM):
- Whole-model order-0 value entropy 10.50 b/w; hi-plane(sign+exp) 2.72 b, mantissa 7.96/8 b.
- Mantissa is a hard random wall: lzma 7.85-8.0 of 8, byte-delta no help, ZERO dead mantissa
  bits in any tensor (ledger 0002 test-002).
- Cross-tensor: per-column exponent profile is 99.65% correlated across experts but conditioning
  saves only ~0.20 b (entropy is within-column). 0009 = per-tensor exponent codebook ~2.7 b -> ~30%.
Read ${LEDGER} (candidates 0001,0002,0009,0010,0011) so you build on, not repeat, prior results.
Label each finding NOVEL or KNOWN; a gain only counts if a REAL coder achieves it bit-exact.
`

const FIND = {
  type: 'object', additionalProperties: false,
  required: ['lens', 'summary', 'best_bpw', 'findings'],
  properties: {
    lens: { type: 'string' },
    summary: { type: 'string', description: '4-6 sentences with exact numbers' },
    best_bpw: { type: 'number', description: 'best lossless bits/weight this lens achieves for its target field (or whole weight), real-coder-backed' },
    findings: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['claim', 'evidence', 'real_gain_pct_vs_16', 'status'],
        properties: {
          claim: { type: 'string' },
          evidence: { type: 'string' },
          real_gain_pct_vs_16: { type: 'number', description: 'additional whole-MODEL lossless % a real coder gets from this, beyond 0009 (0 if none)' },
          status: { type: 'string', enum: ['NOVEL', 'KNOWN'] },
        },
      },
    },
  },
}

phase('Attack')
const LENSES = [
  { key: 'exponent-context', prompt: `${common}
LENS 1 — MINIMUM LOSSLESS EXPONENT. Push the sign+exponent field as low as a REAL coder can,
bit-exact. Combine every structure: within-tensor 2-D context (condition exp on left+up
neighbors), cross-tensor shared column profile (+0.2 b known), cross-plane (exp<->mantissa MI
~0.13-0.32 b). Simulate an actual context/arithmetic coder (or use lzma/bz2 on the 2-D exp
plane and delta/predictor-residual views) across a SAMPLE of tensors spanning roles (experts
up/down, attn in/out_proj, embeddings, lm_head) and layers (early/mid/late). Report the true
minimum exp bits/weight and the whole-model % it implies. Is there real structure below 0009's
~2.7 b order-0, or not?` },

  { key: 'mantissa-laststand', prompt: `${common}
LENS 2 — MANTISSA LAST STAND. The mantissa (7 b) is the wall; try to break it anyway. Throw the
STRONGEST attacks: lzma preset 9|extreme and bz2 on the mantissa plane; condition mantissa on
exponent value / on sign / on column position / on neighbor mantissa (measure conditional
entropy for each); bit-plane split (compress each of the 7 mantissa bitplanes separately);
per-tensor scan for ANY tensor whose mantissa is compressible (chase the survey's L3/L6
exp-field=4 132MB cluster). If ANYTHING beats ~7.0 b bit-exact on real data, quantify it; else
prove the wall with the strongest evidence. This is the crux of whether 90% is even conceivable.` },

  { key: 'wholemodel-coder', prompt: `${common}
LENS 3 — WHOLE-MODEL CODER CEILING. What does the best practical GENERAL lossless pipeline get
on real shards, end-to-end? Compare on several full tensors (and a whole ~4.6GB shard if
feasible by streaming): raw BF16 vs zstd/lzma-on-raw vs PLANE-SPLIT then compress each plane vs
0009's bit-regroup exponent-codebook+raw-mantissa. Confirm/beat 0009's ~30% (byte-split 12 b/w
/ regroup 11.3 b/w). Report the best real whole-model bits/weight and % achievable losslessly.` },

  { key: 'structural-dedup', prompt: `${common}
LENS 4 — STRUCTURAL / EXACT REDUNDANCY. Survey 0010 found 0 byte-identical tensors and value
cosines ~0. Attack the residual angles it flagged: (a) intra-tensor exact row/column repeats
(hash all rows within big tensors — any duplicates?); (b) cross-tensor delta after sign/scale
alignment (does |W| or W/scale align between any pair?); (c) run-length / sparsity of exact
zeros or repeated values model-wide. Quantify any exact-dedup lossless slice (likely ~0, but
measure). Bit-exact only.` },
]

const attacks = (await parallel(LENSES.map(l => () =>
  agent(l.prompt, { label: `attack:${l.key}`, phase: 'Attack', schema: FIND, effort: 'high' })
    .then(r => r ? { ...r, key: l.key } : null)))).filter(Boolean)

phase('Verify')
const toV = []
for (const a of attacks)
  for (const f of (a.findings || []))
    if (f.real_gain_pct_vs_16 > 0.5) toV.push({ lens: a.key, ...f })

const VER = {
  type: 'object', additionalProperties: false,
  required: ['claim', 'verdict', 'real_gain_pct_vs_16'],
  properties: {
    claim: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'OVERSTATED'] },
    real_gain_pct_vs_16: { type: 'number' },
    note: { type: 'string', description: 'what you re-derived with a real bit-exact coder' },
  },
}
const verdicts = toV.length ? (await parallel(toV.map(v => () =>
  agent(`${common}
ADVERSARIALLY VERIFY, bit-exact, from raw bytes yourself. Implement the actual coder and confirm
the exact round-trip AND the claimed size. If the "gain" needs variable-length coding note it's
storage-only. Claim: ${v.claim}\nEvidence: ${v.evidence}\nClaimed whole-model gain: ${v.real_gain_pct_vs_16}%`,
    { label: `verify:${v.lens}`, phase: 'Verify', schema: VER, effort: 'high' })))).filter(Boolean) : []

phase('Synthesize')
const synth = await agent(`${common}
SYNTHESIS. Given all lossless attacks and adversarial verdicts:
ATTACKS:\n${JSON.stringify(attacks, null, 2)}
VERDICTS:\n${JSON.stringify(verdicts, null, 2)}
Write C:/dev/compression/research/candidates/0012-lossless-ceiling/RESULTS.md with: (1) the
compounded whole-model lossless bits/weight and % achievable (sign + best-exponent + mantissa),
real-coder-backed; (2) every NOVEL slice found and its exact size, vs the ~30% 0009 baseline;
(3) the DEFINITIVE verdict on 90% lossless with the information-theoretic argument and the full
evidence that the mantissa (~7 b/weight) is random and immovable; (4) any lossless slice still
worth a follow-up. Be exact and honest — if the ceiling is ~33%, say so with proof; if a lever
pushes it higher, quantify it. Return the path + a 6-bullet executive summary as text.`,
  { label: 'synthesize', phase: 'Synthesize', effort: 'high' })

return { attacks: attacks.length, verified: verdicts.length, synth }
