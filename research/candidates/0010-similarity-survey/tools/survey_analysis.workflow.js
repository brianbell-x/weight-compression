export const meta = {
  name: 'similarity-survey-analysis',
  description: 'Exhaustively analyze the whole-model similarity fingerprints across independent lenses, adversarially verify any exploitable structure from raw bytes, and synthesize novel-vs-known findings',
  phases: [
    { title: 'Lenses', detail: 'independent similarity lenses over the fingerprint data' },
    { title: 'Verify', detail: 're-derive any exploitable/surprising claim from raw model bytes' },
    { title: 'Synthesize', detail: 'novel-vs-known, completeness critic, write report' },
  ],
}

const DATA = args?.data || 'C:/Users/bbell/AppData/Local/Temp/claude/C--dev-compression/771b35d1-71b7-45b9-9704-7ab4517510e6/scratchpad/survey_real'
const MODEL = args?.model || 'C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot'
const LEDGER = 'C:/dev/compression/research/notes/findings-ledger.md'

const common = `
You are one lens of an exhaustive similarity survey of NVIDIA-Nemotron-3-Nano-30B (BF16, 6174 tensors).
Goal of the whole project: find MORE lossless structure to exploit (see AGENTS.md). This survey mechanically
catalogs everything "similar" in the weights so we don't miss structure the earlier targeted probes stepped over.

The fingerprint data (already computed, exact — not sampled) lives in ${DATA}:
  - report.json          : dup groups, value_census, similarity_by_shape, global_byte_entropy
  - report.records.jsonl : one row per tensor {name,dtype,shape,numel,sha256,distinct,top1_frac,top16_frac,
                           value_entropy,hi_entropy,lo_entropy,hi_hist[256],mean,std,absmax,frac_zero,frac_nonfinite}
  - report.global.npz    : global_u16[65536] (exact BF16 value histogram over ALL ~29B weights), global_byte[256]
  - *.fp.json            : per-shard; records also carry sig[256] (block-mean value signature) if you need it
Raw model shards (for verifying claims against actual bytes): ${MODEL} (safetensors; header idiom in
C:/dev/compression/research/candidates/0010-similarity-survey/tools/survey.py -> read_header, offset math, (u16<<16).view(f32) decodes BF16 exactly).
Use uv run python. Pure numpy is enough.

BEFORE concluding, read ${LEDGER} and label each finding NOVEL (not previously established) or KNOWN
(re-confirms/quantifies a prior result). Do not re-propose anything the ledger marks falsified as if new.
Report exact numbers. Distinguish "confirmed the negative model-wide" (valuable) from "found new exploitable structure".
`

const FINDINGS = {
  type: 'object', additionalProperties: false,
  required: ['lens', 'summary', 'findings'],
  properties: {
    lens: { type: 'string' },
    summary: { type: 'string', description: '3-6 sentences: what this lens establishes, with exact numbers' },
    findings: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['claim', 'evidence', 'status', 'exploitable'],
        properties: {
          claim: { type: 'string' },
          evidence: { type: 'string', description: 'exact numbers / tensor names backing it' },
          status: { type: 'string', enum: ['NOVEL', 'KNOWN'] },
          exploitable: { type: 'string', enum: ['yes', 'maybe', 'no'], description: 'could this reduce bytes losslessly beyond candidate 0009?' },
        },
      },
    },
  },
}

phase('Lenses')
const LENSES = [
  { key: 'A-duplicates', prompt: `${common}
LENS A — EXACT & NEAR DUPLICATES. (1) From report.json exact_dup_groups: which tensors are byte-identical
(sha256)? Characterize them (role, dtype, size) and compute total bytes reclaimable by dedup+reference.
(2) NEAR-dup: scan similarity_by_shape for any group with max_abs_cos notably above the ~0.03-0.05 baseline
the ledger reports for experts. For EVERY shape group whose top pair exceeds |cos| 0.30, you MUST adversarially
verify: re-read those two tensors' raw bytes from ${MODEL}, decode to f32, compute the EXACT full-vector cosine
(not the block-mean sig). Report whether real value-similarity exists (would be NOVEL) or it was a signature
artifact. Pay special attention to the CROSS-LAYER same-role cut (layer i vs layer j, same projection).` },

  { key: 'B-value-codebook', prompt: `${common}
LENS B — GLOBAL VALUE CODEBOOK / ENTROPY FLOOR. Load report.global.npz global_u16 (exact histogram of all BF16
values). Compute: exact order-0 entropy (bits/weight) of the whole-model value distribution; distinct values used
of 65536; coverage curve (k values for 0.98/0.999/0.9999); implied lossless bits/weight and % vs 16b for a single
GLOBAL codebook+escape. Do the SAME for the exponent-only plane (high byte, derive its global 256-hist by summing
per-tensor hi_hist weighted by numel from records.jsonl, or recompute). Compare a GLOBAL codebook to candidate
0009's PER-TENSOR codebook (~11.3 b/w, 25-30%): is global competitive, better, or worse, and by how much? The
ledger says a shared table was slightly worse at layer scale (0001) — test that claim at WHOLE-MODEL scale with
exact numbers. Also report mantissa (low byte) global entropy — that bounds how random the residual truly is.` },

  { key: 'C-plane-structure', prompt: `${common}
LENS C — BYTE-PLANE STRUCTURE BY ROLE. From records.jsonl, parse each name into a role
(embedding / lm_head / norm / mamba(A_log,D,dt_bias,conv1d,in_proj,out_proj,...) / attention(q,k,v,o) /
moe-router / moe-expert-up / -gate / -down). Tabulate per role: count, total bytes, mean hi_entropy, mean
lo_entropy, mean value_entropy, mean distinct, frac_zero. Which roles have UNUSUALLY LOW mantissa (lo) entropy
(< ~7.5 bits) — i.e. compressible BEYOND the exponent lever? That would be NOVEL structure. Which roles carry the
resident bulk? Confirm/deny model-wide the "high byte compressible (~2.9b), low byte ~random (~7.95b)" split from
0001, and find any family that violates it.` },

  { key: 'D-distribution', prompt: `${common}
LENS D — DISTRIBUTION SIMILARITY. Using hi_hist[256] per tensor (records.jsonl), quantify how similar the
sign+exponent distributions are ACROSS tensors: within the big expert families compute pairwise symmetric-KL /
cosine (sample if a group is huge) and report the spread; do the same across roles. Confirm model-wide the
"cross-expert distribution near-identical (KL~0.027)" result (0001) and, crucially, decide whether that near
-identity is EXPLOITABLE (a shared static code) or not (the ledger says shared table was ~0.4% worse — nothing to
amortize). Find any family whose distribution is an OUTLIER (its own code would help). Cluster roles by distribution.` },

  { key: 'E-structural-census', prompt: `${common}
LENS E — STRUCTURAL CENSUS & CROSS-LAYER MAP. Parse all names into (role, layer_idx, expert_idx, projection).
Produce the full inventory: how many layers, which are mamba/attention/MoE, experts per MoE layer, tensor count &
bytes per role, and the resident-bulk breakdown (should reconcile with the ledger's "experts = 93%, 58.75GB").
Then the CROSS-LAYER question: for each role, are per-tensor stats (mean/std/absmax/value_entropy/distinct) near
-constant across depth, or do they drift? Near-constant stats + the value/near-dup results from other lenses would
bound whether any cross-layer sharing is even conceivable. Cross-check 0003 (position-wise uncorrelated) and 0007
(full-rank, no shared basis) — you are mapping, not re-litigating, those negatives.` },
]

const lensResults = await parallel(LENSES.map(l => () =>
  agent(l.prompt, { label: `lens:${l.key}`, phase: 'Lenses', schema: FINDINGS })
    .then(r => r ? { ...r, key: l.key } : null)))
const lenses = lensResults.filter(Boolean)

// Collect every finding flagged exploitable yes/maybe OR NOVEL -> independent adversarial verification.
phase('Verify')
const toVerify = []
for (const lr of lenses)
  for (const f of (lr.findings || []))
    if (f.exploitable === 'yes' || f.exploitable === 'maybe' || f.status === 'NOVEL')
      toVerify.push({ lens: lr.key, ...f })

const VERDICT = {
  type: 'object', additionalProperties: false,
  required: ['claim', 'verdict', 'reason'],
  properties: {
    claim: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'OVERSTATED'] },
    reason: { type: 'string', description: 'what you recomputed from raw bytes / data and what it showed (exact numbers)' },
    real_lossless_gain_pct: { type: 'number', description: 'best-case additional lossless % beyond 0009 if exploited, else 0' },
  },
}

const verdicts = toVerify.length ? (await parallel(toVerify.map(v => () =>
  agent(`${common}
ADVERSARIALLY VERIFY this claim from an independent angle. Default to skepticism; re-derive from the raw data /
model bytes yourself, do not trust the fingerprint summary. If it asserts exploitable lossless structure, quantify
the ACTUAL additional lossless gain beyond candidate 0009's 25-30% (0 if none). If it merely re-confirms a known
negative, mark CONFIRMED with gain 0.

LENS: ${v.lens}
CLAIM: ${v.claim}
EVIDENCE OFFERED: ${v.evidence}
CLAIMED STATUS: ${v.status} / exploitable=${v.exploitable}`,
    { label: `verify:${v.lens}`, phase: 'Verify', schema: VERDICT }))))
  .filter(Boolean) : []

phase('Synthesize')
const synthesis = await agent(`${common}
You are the SYNTHESIS + COMPLETENESS CRITIC for the whole similarity survey. Inputs:

LENS RESULTS:
${JSON.stringify(lenses, null, 2)}

ADVERSARIAL VERDICTS:
${JSON.stringify(verdicts, null, 2)}

Write a tight markdown report to C:/dev/compression/research/candidates/0010-similarity-survey/RESULTS.md
(create the dir). It must contain:
  1. What the exhaustive whole-model similarity sweep FOUND, in three buckets: exact-duplicate, value/structural,
     byte-layout/distribution. Exact numbers.
  2. NOVEL vs KNOWN: clearly separate any genuinely-new exploitable structure (survived adversarial verify, gain>0)
     from model-wide CONFIRMATIONS of prior results. Be honest if it's mostly confirmation — that itself upgrades
     "tested on layer 1" to "swept whole model".
  3. The single strongest lead worth a real follow-up probe, if any, with the concrete next experiment.
  4. COMPLETENESS CRITIC: what similarity cut this survey did NOT measure (e.g. intra-tensor row/column repeats,
     transpose/rotation similarity, cross-tensor delta after alignment, higher-order/context entropy of the value
     stream) that could still hide lossless structure — as a ranked backlog.
Also append a one-paragraph proposed entry for research/notes/findings-ledger.md (do not edit the ledger yourself;
just include the proposed text). Return the full RESULTS.md path plus a 5-bullet executive summary as your text.`,
  { label: 'synthesize', phase: 'Synthesize', effort: 'high' })

return { lenses: lenses.length, verified: verdicts.length, synthesis }
