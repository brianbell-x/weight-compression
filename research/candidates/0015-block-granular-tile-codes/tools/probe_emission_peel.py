"""probe_emission_peel.py -- candidate 0015: peel the v2 EMITTED representation.

The frozen v2 tile format (W=128 bit-stride blocks, T=4 DP tier budgets,
mantissa-carrying flush [L3], order-0 per-tensor 12-bit-quantized rANS tables,
P100 top budget so no escapes) fired cross-layer at an honest whole-model
10.7346 b/w vs floor ~10.56. Per AGENTS.md the emitted representation is now
the object of study: "peel until random" applies recursively to what we emit.
This probe has two halves:

H1  RANDOMNESS CERTIFICATION of the v2 emitted planes. For a deterministic
    sample of tensors (8 experts x {up,down} = 16 per layer; layers 1/13/27/40
    real, whatever exists synthetic) the probe EMITS the actual streams by
    extending v2's exact vectorized coder simulation to record every renorm
    bit (verified bit-identical against the pure-Python reference serializer
    on sampled blocks, plus full decode + SHA-exact BF16 reconstruction):
      (a) coded rANS payload bits  -- concatenated per-block serialized
          payloads (12-bit flush MSB-first + renorm bits in decode order),
          WITHOUT slot padding (pad slack is already quantified structure in
          the v2 accounting, deterministic zeros);
      (b) the tier-flag plane      -- ceil(log2(classes)) bits/block (2 bits
          at T=4), packed MSB-first, plus the flag symbol sequence;
      (c) per-block DP budget/code-length sequence -- the measured emitted
          bits rb[i] per block (diagnostic plane: rb itself is not
          transmitted -- flags+budgets are -- but block-to-block structure in
          rb is exploitable tier/budget design headroom);
      (d) the mantissa plane in the L3 layout -- per block the 7W-12 bits
          that remain after the first 12 bits move into the flush field
          (weight-major, 7 bits MSB-first per weight, matching the d_seed
          convention), concatenated.
    Batteries per plane: order-0/1/2 entropy (bit-level for bit planes,
    symbol-level for sequences), bit-pair mutual information at fixed lags
    vs a circular-shift null (R=16 deterministic shifts; threshold =
    max(null max, Bonferroni chi2(1) analytic bar)), autocorrelation of
    block-level code lengths and tier flags (lags 1..64 + the row-stride
    lag; 4/sqrt(nb) significance), and one strong general compressor
    (lzma preset 9|EXTREME) per plane. Each plane also gets its NATIVE
    strides: the mant plane (7 bits/weight weight-major) adds MI lags
    {7, 14, 21, 7*Ccols, 7W-12 (the block stride) and its row stride} plus
    a per-phase entropy candidate H(bit | position mod 7) charged
    7 x 2 x 12 model bits -- pure phase bias has exactly zero pairwise MI
    and is diluted ~7x in pooled bit entropy, so only this test can see it;
    the flags plane adds lags {flagb, flagb*row_lag}. Circular-shift null
    draws are constrained to shifts coprime to the plane's natural periods
    (a period-aligned shift lands on structure-aligned pairs and would
    absorb the very signal under test -- it can only HIDE structure).
    Deliverable per plane: "structure found (quantified b/w ceiling)" --
    ceiling = (emitted bits - min(entropy bound, lzma bits)) / n -- or
    "random at these tests" (pre-registered smallness bar
    STRUCT_EPS_BPW = 0.01 b/w; sub-bar MI/autocorr hits are reported as
    weak structure, not a quantified ceiling).

H2  WITHIN-BLOCK ORDER-1 CONTEXT -- the one lever the tile contract gives
    for free. Decode inside a block is already sequential, so conditioning
    sym[i] on ctx(sym[i-1]) (same row; reset at block starts AND row starts)
    costs no random-access penalty -- only context-table side costs and
    coder mechanics. Context quantization: identity-top-C buckets, the
    top C-1 most frequent symbols get identity buckets, everything else
    folds into bucket C-1; C in {4,8,16,32}. Block/row-start symbols use
    the order-0 table (transmitted anyway as the start table). Measured:
      (1) holdout conditional entropy H(sym_i | ctx(sym_{i-1})): bucket map
          + conditional tables fit on half the sampled experts of a layer
          (per projection), evaluated on the other half (add-0.5 smoothing)
          -- shows the structure is not overfit noise;
      (2) EXACT realized accounting under the frozen v2 mechanics: per-
          position tables fed to the same exact coder simulation (L3 seeds
          unchanged), DP T=4 tier budgets recomputed, flag/rank/class-dir/
          header/align/mantissa accounting unchanged; charged side costs =
          C bucket tables (only occupied buckets) + the order-0 start table
          (each pad8(512+nnz*12) as in v2) + a pad8(8+C+(C-1)*9)-bit context
          header: 8-bit mode/C field, C-bit table-occupancy bitmap (the
          decoder cannot parse the table section without it), (C-1) 9-bit
          bucket-map symbol ids. Fit-on-self with table cost charged IS the
          deployment number (tables are fit on the tensor and transmitted);
      (3) realized b/w vs the frozen cell recomputed on the SAME sample,
          per layer and whole-sample, plus projected whole-model
          (10.7346 - 0.93 * weighted delta).
    Block-boundary reset cost at W=128 (1/128 of symbols lose their
    context) is quantified as the smoothed-empirical entropy difference at
    block-start positions that do have a same-row predecessor.
    Round-trip proof: the order-1 coder (pure-Python encode -> serialized
    bytes -> decode with the context reset rules) is verified on sampled
    blocks of EVERY tensor and EVERY C: emitted bits == accounted bits,
    symbols exact, L3 payload recovered from the decoder's final state,
    reconstructed BF16 bytes byte-exact (SHA-256 over sampled spans).

Pre-registered H2 gate: FIRES if the best-C realized fit-on-self number
(all side costs charged) beats the frozen cell by >= 0.05 b/w on the
sampled set AND the holdout gain is positive for that C in every
(layer, proj) cell (not overfit). H1 delivers certificates either way.

Field split (stz convention): u16 LE, sym = u >> 7 (9 bits),
mant = u & 0x7F (7 bits). Coder, quantizer, DP, accounting constants are
IMPORTED from probe_block_codes.py / probe_block_codes_v2.py, not
reimplemented.

Usage:
  uv run python probe_emission_peel.py --synthetic     # smoke (fake snapshot)
  uv run python probe_emission_peel.py                 # real run (resumable)
  uv run python probe_emission_peel.py --layer 13      # one layer only
  uv run python probe_emission_peel.py --summary       # tables + JSON
"""
from __future__ import annotations
import argparse, hashlib, json, lzma, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import probe_block_codes as v1      # noqa: E402
import probe_block_codes_v2 as v2   # noqa: E402

M, M_LOG2, FLUSH_BITS = v1.M, v1.M_LOG2, v1.FLUSH_BITS
pad8, ceil_div, BITLEN, die = v1.pad8, v1.ceil_div, v1.BITLEN, v1.die
ART = v1.ART
BORROW_BITS = v2.BORROW_BITS
RANK_GROUP, RANK_BITS = v2.RANK_GROUP, v2.RANK_BITS
HEAD_BITS, CLASS_DIR_BITS = v2.HEAD_BITS, v2.CLASS_DIR_BITS

# ---- frozen v2 cell (the object of study; no re-selection here)
W = 128                      # frozen block size (bit-stride, fusible)
T_MAX = 4                    # frozen DP tier count
# P100 => top budget = max measured size => no escape blocks (asserted)

# ---- sampling
LAYERS_REAL = (1, 13, 27, 40)
EXPERTS_PER_PROJ = 8         # x2 projections = 16 tensors per layer

# ---- H1 batteries (pre-registered)
STRUCT_EPS_BPW = 0.01        # quantified-ceiling smallness bar (b/w)
MI_LAGS = (1, 2, 3, 4, 8, 12, 16, 24, 32, 64, 128)
MI_NULL_R = 16               # circular-shift null draws (empirical cross-check)
MI_FAMILY_ALPHA = 0.01       # family-wise false-hit rate, Bonferroni over lags
MI_CAP_BITS = 1 << 23        # analysis prefix cap for MI (full plane for entropy)
MI_SEED = 20260702
AC_MAX_LAG = 64
AC_SIG_K = 4.0               # |r| > K/sqrt(n) flags block-to-block structure
LZMA_FILTER = 9 | lzma.PRESET_EXTREME
MODEL_BITS_PER_CELL = 12     # entropy bounds charge 12 bits per (ctx, symbol)
PHASE_P = 7                  # mant plane natural period: 7 bits/weight, weight-major
PHASE_MODEL_BITS = PHASE_P * 2 * MODEL_BITS_PER_CELL  # per-phase bound model cost

# ---- H2 (pre-registered)
CTXS = (4, 8, 16, 32)        # identity-top-C bucket counts
H2_GATE_BPW = 0.05           # fires if best-C realized beats frozen by >= this
CTXMAP_SYM_BITS = 9          # transmitted bucket map: (C-1) 9-bit symbol ids
SMOOTH_ALPHA = 0.5           # add-alpha smoothing for holdout / reset-cost CE
FROZEN_WHOLE_MODEL_BPW = 10.7346
FLOOR_WHOLE_MODEL_BPW = 10.56
EXPERT_FRAC = 0.93

ACCT = {"schema": 2, "probe": "emission_peel", "W": W, "T_MAX": T_MAX,
        "P": 100, "L1": 1, "L3": 1, "L4": 0,
        "M_LOG2": M_LOG2, "FLUSH_BITS": FLUSH_BITS, "BORROW_BITS": BORROW_BITS,
        "RANK_GROUP": RANK_GROUP, "RANK_BITS": RANK_BITS,
        "HEAD_BITS": HEAD_BITS, "CLASS_DIR_BITS": CLASS_DIR_BITS,
        "LAYERS_REAL": LAYERS_REAL, "EXPERTS_PER_PROJ": EXPERTS_PER_PROJ,
        "STRUCT_EPS_BPW": STRUCT_EPS_BPW, "MI_LAGS": MI_LAGS,
        "MI_NULL_R": MI_NULL_R, "MI_FAMILY_ALPHA": MI_FAMILY_ALPHA,
        "MI_CAP_BITS": MI_CAP_BITS,
        "MODEL_BITS_PER_CELL": MODEL_BITS_PER_CELL,
        "PHASE_P": PHASE_P, "PHASE_MODEL_BITS": PHASE_MODEL_BITS,
        "MI_NATIVE_LAGS": ("mant:{7,14,21,7*C,7W-12,(7W-12)*rowlag} "
                           "flags:{flagb,flagb*rowlag}"),
        "MI_NULL_COPRIME": 1,
        "CTX_HDR_BITS": "pad8(8+C+(C-1)*9)",
        "MI_SEED": MI_SEED, "AC_MAX_LAG": AC_MAX_LAG, "AC_SIG_K": AC_SIG_K,
        "CTXS": CTXS, "H2_GATE_BPW": H2_GATE_BPW,
        "CTXMAP_SYM_BITS": CTXMAP_SYM_BITS, "SMOOTH_ALPHA": SMOOTH_ALPHA,
        "FROZEN_WHOLE_MODEL_BPW": FROZEN_WHOLE_MODEL_BPW,
        "EXPERT_FRAC": EXPERT_FRAC,
        "CODER": v2.CODER_SPEC}
ACCT_STAMP = hashlib.sha256(json.dumps(ACCT, sort_keys=True).encode()).hexdigest()[:12]


def check_stamp(rows: list[dict], jsonl: Path):
    bad = [r for r in rows if r.get("acct") != ACCT_STAMP]
    if bad:
        die(f"{len(bad)}/{len(rows)} rows in {jsonl} carry accounting stamp "
            f"{bad[0].get('acct')!r} != current {ACCT_STAMP!r} -- move that "
            f"file aside and re-run")


# ---------------------------------------------------------- emitting coder ---
def rans_sim_emit(qm: np.ndarray, cm: np.ndarray, x0: np.ndarray):
    """v2.rans_sim_blocks extended to RECORD the emitted bits: returns
    (bits_per_block, K, V, x_final) where K[b,j] is the renorm bit count at
    symbol j and V[b,j] the emitted value (low K bits of the pre-shift state,
    first-emitted bit = bit 0). Bit-identical arithmetic to the reference
    encoder; x_final - M is the 12-bit flush."""
    nbk, Wb = qm.shape
    x = x0.astype(np.int64).copy()
    K = np.empty((nbk, Wb), np.int64)
    V = np.empty((nbk, Wb), np.int64)
    bits = np.full(nbk, FLUSH_BITS, np.int64)
    for j in range(Wb - 1, -1, -1):
        qq = qm[:, j]
        thr1 = (qq << 1) - 1
        k = BITLEN[x] - BITLEN[thr1]
        np.maximum(k, 0, out=k)
        k += (x >> k) > thr1
        K[:, j] = k
        V[:, j] = x & ((np.int64(1) << k) - 1)
        bits += k
        x = M + cm[:, j] + ((x >> k) - qq)
    return bits, K, V, x


def emit_payload_plane(K: np.ndarray, V: np.ndarray, xf: np.ndarray,
                       chunk: int = 8192) -> np.ndarray:
    """Serialized payload bit plane (uint8 0/1): per block, 12-bit flush
    MSB-first then, for j = 0..W-1, the K[b,j] bits of V[b,j] MSB-first --
    exactly v1.pack_block's layout (reverse of LSB-first emission order)."""
    nbk, Wb = K.shape
    out = []
    for lo in range(0, nbk, chunk):
        hi = min(lo + chunk, nbk)
        vals = np.concatenate([(xf[lo:hi] - M)[:, None], V[lo:hi]], axis=1).ravel()
        lens = np.concatenate([np.full((hi - lo, 1), FLUSH_BITS, np.int64),
                               K[lo:hi]], axis=1).ravel()
        total = int(lens.sum())
        if total == 0:
            continue
        item = np.repeat(np.arange(lens.size, dtype=np.int64), lens)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        t = np.arange(total, dtype=np.int64) - np.repeat(starts, lens)
        shift = lens[item] - 1 - t
        out.append(((vals[item] >> shift) & 1).astype(np.uint8))
    return np.concatenate(out) if out else np.zeros(0, np.uint8)


def mant_plane_bits(mant: np.ndarray, nb: int) -> np.ndarray:
    """L3-layout mantissa plane: 7 bits/weight MSB-first, weight-major, minus
    the first 12 bits of each block (they ride in the flush)."""
    b = ((mant[:, None] >> np.arange(6, -1, -1)) & 1).astype(np.uint8)
    return np.ascontiguousarray(b.reshape(nb, 7 * W)[:, BORROW_BITS:]).ravel()


def flags_plane_bits(flags: np.ndarray, flagb: int) -> np.ndarray:
    if flagb == 0:
        return np.zeros(0, np.uint8)
    return ((flags[:, None] >> np.arange(flagb - 1, -1, -1)) & 1
            ).astype(np.uint8).ravel()


# ------------------------------------------------------------ H1 batteries ---
def h0_bits(counts: np.ndarray) -> float:
    n = counts.sum()
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def bit_orders(b: np.ndarray) -> dict:
    """Order-0/1/2 bit-level entropy (bits per bit); alphabet 2."""
    n = b.size
    if n < 8:
        return {"A": 2, "h0": None, "h1": None, "h2": None}
    x = b.astype(np.int64)
    h0 = h0_bits(np.bincount(x, minlength=2))
    j1 = np.bincount(x[:-1] * 2 + x[1:], minlength=4).reshape(2, 2)
    h1 = v2.cond_entropy_bits(j1, int(j1.sum()))
    ctx = x[:-2] * 2 + x[1:-1]
    j2 = np.bincount(ctx * 2 + x[2:], minlength=8).reshape(4, 2)
    h2 = v2.cond_entropy_bits(j2, int(j2.sum()))
    return {"A": 2, "h0": round(h0, 6), "h1": round(h1, 6), "h2": round(h2, 6)}


def seq_orders(x: np.ndarray) -> dict:
    """Order-0/1/2 symbol-level entropy (bits per symbol) over a dense-mapped
    integer sequence. Order-1/2 are fit-on-self (upper bound on structure;
    small-sample overfit is flagged via samples-per-context)."""
    n = x.size
    u, xi = np.unique(x, return_inverse=True)
    A = int(u.size)
    out = {"A": A, "h0": round(h0_bits(np.bincount(xi, minlength=A)), 6),
           "h1": None, "h2": None, "n": int(n)}
    if n >= 4 and A * A <= (1 << 24):
        j = np.bincount(xi[:-1] * A + xi[1:], minlength=A * A).reshape(A, A)
        out["h1"] = round(v2.cond_entropy_bits(j, int(j.sum())), 6)
        out["h1_samples_per_ctx"] = round(n / (A * A), 2)
    if n >= 8 and A ** 3 <= (1 << 24):
        ctx = xi[:-2] * A + xi[1:-1]
        j = np.bincount(ctx * A + xi[2:], minlength=A ** 3).reshape(A * A, A)
        out["h2"] = round(v2.cond_entropy_bits(j, int(j.sum())), 6)
    return out


def bit_mi_battery(b: np.ndarray, extra_lags=(), periods=()) -> dict:
    """MI(bit_i; bit_{i+lag}) at MI_LAGS + plane-native extra lags vs a
    circular-shift null (deterministic RNG). Null shifts are constrained to
    lags coprime to the plane's natural periods: a period-aligned shift lands
    on structure-aligned pairs and would absorb the very signal under test
    (directionally it can only HIDE structure, the wrong direction for a
    randomness certificate). Threshold = max(null max, analytic bar)."""
    x = b[:min(b.size, MI_CAP_BITS)].astype(np.int64)
    n = x.size
    if n < 256:
        return {"skipped": f"plane too small ({n} bits)"}

    def mi(u, v):
        c = np.bincount(u * 2 + v, minlength=4).astype(np.float64).reshape(2, 2)
        N = c.sum()
        p = c / N
        pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = p * np.log2(p / (pa * pb))
        return float(np.nansum(t))

    lags = sorted({int(l) for l in (*MI_LAGS, *extra_lags) if l and l < n // 4})
    real = {l: mi(x[:-l], x[l:]) for l in lags}
    pers = sorted({int(p) for p in periods if p and p > 1})
    rng = np.random.default_rng(MI_SEED)
    nulls, rejected = [], 0
    while len(nulls) < MI_NULL_R:
        r = int(rng.integers(n // 7, n - n // 7))
        if any(r % p == 0 for p in pers):
            rejected += 1
            if rejected > 64 * MI_NULL_R:
                die("null rejection runaway -- periods misconfigured")
            continue
        y = np.roll(x, r)
        nulls.append(mi(x[:-1], y[:-1]))
    null_max = max(nulls)
    # analytic threshold: under independence 2N ln2 * MI ~ chi2(1); Bonferroni
    # over the tested lags at family alpha (a hit must clear the empirical
    # circular-shift null max AND the analytic bar)
    from statistics import NormalDist
    p = MI_FAMILY_ALPHA / max(len(real), 1)
    chi2_thr = NormalDist().inv_cdf(1 - p / 2) ** 2
    mi_thr = max(null_max, chi2_thr / (2 * n * math.log(2)))
    hits = {l: m for l, m in real.items() if m > mi_thr}
    return {"n_bits": int(n), "null_max": round(null_max, 9),
            "null_mean": round(float(np.mean(nulls)), 9),
            "null_periods": pers, "null_rejected": rejected,
            "mi_thresh": round(mi_thr, 9),
            "mi": {str(l): round(m, 9) for l, m in real.items()},
            "hits": {str(l): round(m, 9) for l, m in hits.items()},
            "n_hits": len(hits)}


def autocorr_battery(x: np.ndarray, extra_lags=()) -> dict:
    n = x.size
    if n < 16:
        return {"skipped": f"sequence too small ({n})"}
    f = x.astype(np.float64)
    f -= f.mean()
    denom = float((f * f).sum())
    if denom == 0:
        return {"note": "constant sequence", "sig": []}
    lags = list(range(1, min(AC_MAX_LAG, n // 4) + 1))
    for l in extra_lags:
        if l and l not in lags and l <= n // 4:
            lags.append(int(l))
    r = {int(l): float((f[:-l] * f[l:]).sum() / denom) for l in lags}
    thr = AC_SIG_K / math.sqrt(n)
    sig = sorted(((l, v) for l, v in r.items() if abs(v) > thr),
                 key=lambda kv: -abs(kv[1]))
    return {"n": int(n), "thresh": round(thr, 6),
            "max_abs_r": round(max(abs(v) for v in r.values()), 6),
            "max_abs_lag": int(max(r, key=lambda l: abs(r[l]))),
            "sig": [[int(l), round(v, 6)] for l, v in sig[:8]],
            "n_sig": len(sig)}


def lzma_bits_of(by: bytes) -> int:
    return 8 * len(lzma.compress(by, preset=LZMA_FILTER))


def plane_cert(bits_total: int, n: int, orders: dict, lzma_b: int,
               mi: dict | None, ac: dict | None, unit_count: int,
               transmitted: bool = True, extra_ent: list | None = None) -> dict:
    """Per-plane certificate. Entropy bound at order k = H_k x units +
    A^k x A x 12 bits of charged model cost (fit-on-self entropies without a
    transmitted model are not a code); extra_ent adds plane-native candidates
    ({name, h, model_bits, ...}: bound = h x units + model_bits); ceiling =
    (emitted - min(entropy bounds, lzma)) / n weights, clamped at 0. lzma
    carries its own model."""
    A = int(orders.get("A", 2))
    ent_cands = []
    for k, key in enumerate(("h0", "h1", "h2")):
        h = orders.get(key)
        if h is not None:
            ent_cands.append(h * unit_count + (A ** k) * A * MODEL_BITS_PER_CELL)
    ee_out = []
    for e in (extra_ent or []):
        bb = e["h"] * unit_count + e["model_bits"]
        ent_cands.append(bb)
        ee_out.append({**e, "bound_bits": int(bb)})
    ent_bound = min(ent_cands) if ent_cands else None
    cands = [b for b in (ent_bound, lzma_b) if b is not None]
    best = min(cands) if cands else bits_total
    ceiling = max(0.0, (bits_total - best) / n)
    mi_hit = bool(mi and mi.get("n_hits"))
    ac_hit = bool(ac and ac.get("n_sig"))
    if ceiling >= STRUCT_EPS_BPW:
        verdict = f"structure found (ceiling ~{ceiling:.4f} b/w)"
    elif mi_hit or ac_hit:
        verdict = (f"weak structure (MI/autocorr above null; ceiling "
                   f"{ceiling:.4f} < {STRUCT_EPS_BPW} b/w)")
    else:
        verdict = "random at these tests"
    return {"bits": int(bits_total), "bpw": round(bits_total / n, 6),
            "orders": orders,
            "extra_ent": ee_out or None,
            "entropy_bound_bits": None if ent_bound is None else int(ent_bound),
            "lzma_bits": int(lzma_b),
            "ceiling_bpw": round(ceiling, 6),
            "mi_hit": mi_hit, "ac_hit": ac_hit,
            "mi": mi,
            "transmitted_plane": bool(transmitted),
            "verdict": verdict}


# --------------------------------------------------------- realized cells ---
def realized_cell(rb: np.ndarray, nb: int, n: int, tab_bits: int) -> dict:
    """Frozen v2 mechanics at P100 (no escapes): DP T=4 tier budgets over the
    measured block bits, bit-granular slots, flag/rank/class-dir/header/align
    charged exactly as v2's cell formula, L3 mantissa plane."""
    kept, budgets, slots, counts = v2.tier_dp(rb, True, T_MAX)[T_MAX]
    assert int(rb.max()) <= budgets[-1]          # P100: nothing escapes
    classes = len(budgets)
    flagb = int(math.ceil(math.log2(classes))) if classes > 1 else 0
    flag = nb * flagb
    rank = classes * RANK_BITS * ceil_div(nb, RANK_GROUP) if classes > 1 else 0
    cdir = classes * CLASS_DIR_BITS if classes > 1 else 0
    sym_raw = kept + flag + rank + cdir + tab_bits + HEAD_BITS
    sym_total = pad8(sym_raw)
    mant_bits = pad8(7 * n - BORROW_BITS * nb)
    flags = np.searchsorted(np.asarray(budgets), rb)
    return {"bpw": round((sym_total + mant_bits) / n, 6),
            "sym_bits": int(sym_total), "mant_bits": int(mant_bits),
            "coded_bits": int(rb.sum()), "kept_slot_bits": int(kept),
            "pad_bits": int(kept - rb.sum()),
            "flag_bits": int(flag), "rank_bits": int(rank),
            "cdir_bits": int(cdir), "tab_bits": int(tab_bits),
            "align_bits": int(sym_total - sym_raw),
            "budgets": [int(x) for x in budgets],
            "class_counts": [int(x) for x in counts],
            "classes": int(classes), "flagb": int(flagb),
            "_flags": flags}


# --------------------------------------------------------- order-1 context ---
def build_ctx(sym: np.ndarray, hist: np.ndarray, n: int, Ccols: int, Cq: int):
    """Identity-top-C bucket context. Returns (ctx per position with value Cq
    at start positions, bucket map array (512,), start mask, reset mask)."""
    order = np.argsort(-hist, kind="stable")
    top = order[:Cq - 1]
    top = top[hist[top] > 0]
    bmap = np.full(512, Cq - 1, np.int64)
    bmap[top] = np.arange(top.size)
    idx = np.arange(n, dtype=np.int64)
    ctx = np.empty(n, np.int64)
    ctx[0] = Cq
    ctx[1:] = bmap[sym[:-1]]
    start = (idx % W == 0) | (idx % Ccols == 0)
    ctx[start] = Cq
    reset = (idx % W == 0) & (idx % Ccols != 0)   # block starts w/ same-row pred
    return ctx, bmap, start, reset, int(top.size)


def o1_enc_block(seq, tids, QL, CL, x0):
    x = x0
    bits = []
    ap = bits.append
    for s, tid in zip(reversed(seq), reversed(tids)):
        qq = QL[tid][s]
        t = qq << 1
        while x >= t:
            ap(x & 1)
            x >>= 1
        x = M + CL[tid][s] + (x - qq)
    return x - M, bits


def o1_dec_block(data, nbits, g0, Ccols, bmap, QL, CL, S2S, start_tid):
    """Forward decode with the context reset rules: position p = g0+i uses the
    start table when i == 0 (block start) or p % Ccols == 0 (row start), else
    the bucket table of the previously decoded symbol."""
    if nbits < FLUSH_BITS:
        return None
    f = 0
    for i in range(FLUSH_BITS):
        f = (f << 1) | ((data[i >> 3] >> (7 - (i & 7))) & 1)
    x = M + f
    pos = FLUSH_BITS
    out = []
    for i in range(W):
        p = g0 + i
        tid = start_tid if (i == 0 or p % Ccols == 0) else int(bmap[out[-1]])
        slot = x - M
        s = S2S[tid][slot]
        x = QL[tid][s] + slot - CL[tid][s]
        while x < M:
            if pos >= nbits:
                return None
            x = (x << 1) | ((data[pos >> 3] >> (7 - (pos & 7))) & 1)
            pos += 1
        out.append(s)
    if pos != nbits or not (M <= x < 2 * M):
        return None
    return out, x


def sparse_counts(arr2: np.ndarray) -> dict:
    """(k,2) index rows -> sparse {'(a,b)': cnt} via flat bincount."""
    flat = arr2[:, 0] * 512 + arr2[:, 1]
    c = np.bincount(flat, minlength=0)
    nzi = np.flatnonzero(c)
    return {"i": (nzi // 512).tolist(), "j": (nzi % 512).tolist(),
            "c": c[nzi].tolist()}


def sparse_hist(vals: np.ndarray) -> dict:
    c = np.bincount(vals, minlength=512)
    nzi = np.flatnonzero(c)
    return {"i": nzi.tolist(), "c": c[nzi].tolist()}


# --------------------------------------------------------------- per-tensor ---
def analyze_tensor(raw: bytes, t: dict) -> dict:
    u = np.frombuffer(raw, "<u2")
    n = u.size
    R, Ccols = t["shape"]
    assert n == R * Ccols, (t["name"], n, R, Ccols)
    if n % W:
        die(f"n={n} not divisible by W={W} on {t['name']}")
    nb = n // W
    sym = (u >> 7).astype(np.int64)
    mant = (u & 0x7F).astype(np.int64)
    hist = np.bincount(sym, minlength=512).astype(np.int64)
    H = h0_bits(hist)

    starts_pos = np.arange(nb, dtype=np.int64) * W
    d_seed = (mant[starts_pos] << 5) | (mant[starts_pos + 1] >> 2)
    x0 = (M + d_seed).astype(np.int64)

    # ---- frozen cell (order-0 per-tensor table, L3 seeds)
    q0, cum0, clq0, tab0_bits, nnz0 = v2.build_table(hist, n)
    qm = q0[sym].reshape(nb, W)
    cm = cum0[sym].reshape(nb, W)
    rb, K, V, xf = rans_sim_emit(qm, cm, x0)
    chk = v2.rans_sim_blocks(qm, cm, x0)
    if not np.array_equal(rb, chk):
        die(f"emitting sim disagrees with v2 sim on {t['name']}")
    frozen = realized_cell(rb, nb, n, tab0_bits)
    flags = frozen.pop("_flags")

    # ---- H1: emit the planes
    payload = emit_payload_plane(K, V, xf)
    if payload.size != int(rb.sum()):
        die(f"payload plane size {payload.size} != accounted {int(rb.sum())}")
    mplane = mant_plane_bits(mant, nb)
    fplane = flags_plane_bits(flags, frozen["flagb"])

    # serializer + round-trip gate: pure-Python reference on sampled blocks
    off = np.concatenate([[0], np.cumsum(rb)])
    ql, cl = q0.tolist(), cum0.tolist()
    pres = np.flatnonzero(q0)
    s2s = np.repeat(pres, q0[pres]).tolist()
    sha_o, sha_r = hashlib.sha256(), hashlib.sha256()
    rt_blocks = 0
    for i in sorted({0, nb - 1, int(np.argmin(rb)), int(np.argmax(rb))}):
        s0 = int(starts_pos[i])
        seq = sym[s0:s0 + W].tolist()
        x0i = int(x0[i])
        ctxs = f"{t['name']} frozen block {i}"
        fl, bits = v2.rans_enc_block(seq, ql, cl, x0i)
        data, nbits = v1.pack_block(fl, bits)
        if nbits != int(rb[i]):
            die(f"SERIALIZER ({ctxs}): emitted {nbits} != accounted {int(rb[i])}")
        ref = np.unpackbits(np.frombuffer(data, np.uint8))[:nbits]
        if not np.array_equal(ref, payload[off[i]:off[i] + nbits]):
            die(f"SERIALIZER ({ctxs}): vectorized plane bits != reference bytes")
        dec = v2.rans_dec_block(data, nbits, W, ql, cl, s2s)
        if dec is None or dec[0] != seq or dec[1] != x0i:
            die(f"ROUND-TRIP ({ctxs}): decode failed")
        mrec = mant[s0:s0 + W].copy()
        d_rec = dec[1] - M
        if d_rec != int(d_seed[i]):
            die(f"ROUND-TRIP ({ctxs}): payload {d_rec} != {int(d_seed[i])}")
        mrec[0] = d_rec >> 5
        mrec[1] = ((d_rec & 31) << 2) | (mrec[1] & 3)
        rec = ((np.array(dec[0], np.int64) << 7) | mrec).astype("<u2")
        orig = raw[2 * s0:2 * (s0 + W)]
        if rec.tobytes() != orig:
            die(f"ROUND-TRIP ({ctxs}): reconstructed bytes != original")
        sha_o.update(orig)
        sha_r.update(rec.tobytes())
        rt_blocks += 1

    # per-plane batteries; each plane gets its NATIVE strides
    row_lag = Ccols // W if (Ccols >= W and Ccols % W == 0) else None
    mant_stride = 7 * W - BORROW_BITS      # 884: block stride in the mant plane
    mant_lags = [PHASE_P, 2 * PHASE_P, 3 * PHASE_P, 7 * Ccols, mant_stride]
    if row_lag:
        mant_lags.append(mant_stride * row_lag)   # exact same-column row stride
    mant_periods = (PHASE_P, mant_stride)
    flagb0 = frozen["flagb"]
    flag_lags = [flagb0] + ([flagb0 * row_lag] if row_lag else [])
    flag_periods = (flagb0,)

    # per-phase entropy H(bit | position mod 7): the mant plane's one
    # physically-native structure mode. idx = 7*w + k within a block, so
    # (position + BORROW_BITS) mod 7 == k, the MSB-first bit-in-weight index.
    # Pure phase bias has zero pairwise MI and is diluted ~7x in pooled
    # bit-level entropy -- only this candidate can see it.
    phase_pat = np.arange(BORROW_BITS, 7 * W) % PHASE_P
    m2 = mplane.reshape(nb, 7 * W - BORROW_BITS)
    ph_h, ph_p1 = 0.0, []
    for ph in range(PHASE_P):
        colsel = phase_pat == ph
        tot = int(nb) * int(colsel.sum())
        ones = int(m2[:, colsel].sum())
        ph_p1.append(round(ones / tot, 6))
        ph_h += (tot / mplane.size) * h0_bits(
            np.array([tot - ones, ones], np.int64))
    mant_extra = [{"name": "phase7", "h": ph_h,
                   "model_bits": PHASE_MODEL_BITS, "p1": ph_p1}]

    h1 = {
        "payload": plane_cert(payload.size, n, bit_orders(payload),
                              lzma_bits_of(np.packbits(payload).tobytes()),
                              bit_mi_battery(payload), None, payload.size),
        "flags": (plane_cert(fplane.size, n, bit_orders(fplane),
                             lzma_bits_of(np.packbits(fplane).tobytes()),
                             bit_mi_battery(fplane, flag_lags, flag_periods),
                             autocorr_battery(flags, (row_lag,)), fplane.size)
                  if fplane.size else
                  {"bits": 0, "bpw": 0.0, "verdict": "empty (single class)"}),
        "lens": plane_cert(nb * int(rb.max()).bit_length(), n, seq_orders(rb),
                           lzma_bits_of(rb.astype("<u2").tobytes()),
                           None, autocorr_battery(rb, (row_lag,)), nb,
                           transmitted=False),
        "mant": plane_cert(mplane.size, n, bit_orders(mplane),
                           lzma_bits_of(np.packbits(mplane).tobytes()),
                           bit_mi_battery(mplane, mant_lags, mant_periods),
                           None, mplane.size, extra_ent=mant_extra),
    }
    h1["flags"]["seq_orders"] = seq_orders(flags)
    h1["flags"]["ac"] = autocorr_battery(flags, (row_lag,))
    h1["lens"]["ac"] = autocorr_battery(rb, (row_lag,))
    # block-to-block predictability of the code lengths specifically:
    # order-1 gain over order-0 with the order-1 model cost charged
    so = h1["lens"]["orders"]
    if so.get("h1") is not None:
        gap = ((so["h0"] - so["h1"]) * nb
               - so["A"] * so["A"] * MODEL_BITS_PER_CELL) / n
        h1["lens"]["o1_gap_bpw"] = round(max(0.0, gap), 6)

    # ---- H2: within-block order-1 context, exact realized accounting
    marg_sm = (hist + SMOOTH_ALPHA) / (n + SMOOTH_ALPHA * 512)
    h2 = {}
    for Cq in CTXS:
        ctx, bmap, start_mask, reset_mask, n_top = build_ctx(
            sym, hist, n, Ccols, Cq)
        # per-bucket tables (occupied only) + start table = order-0 table
        Qt = np.zeros((Cq + 1, 512), np.int64)
        Ct = np.zeros((Cq + 1, 512), np.int64)
        CLQ = np.zeros((Cq + 1, 512))
        tabs_bits = 0
        used = []
        joint = np.zeros((Cq, 512), np.int64)
        for b in range(Cq):
            sel = sym[ctx == b]
            if sel.size == 0:
                continue
            hb = np.bincount(sel, minlength=512).astype(np.int64)
            joint[b] = hb
            qb, cb, clb, tbb, _ = v2.build_table(hb, int(hb.sum()))
            Qt[b], Ct[b], CLQ[b] = qb, cb, clb
            tabs_bits += tbb
            used.append(b)
        Qt[Cq], Ct[Cq], CLQ[Cq] = q0, cum0, clq0
        # context header: 8-bit mode/C field + Cq-bit table-occupancy bitmap
        # (the decoder cannot parse the table section without knowing which
        # buckets carry tables) + (Cq-1) 9-bit bucket-map symbol ids
        ctxmap_bits = pad8(8 + Cq + (Cq - 1) * CTXMAP_SYM_BITS)
        tab_total = tabs_bits + tab0_bits + ctxmap_bits

        qpos = Qt[ctx, sym].reshape(nb, W)
        cpos = Ct[ctx, sym].reshape(nb, W)
        rb1 = v2.rans_sim_blocks(qpos, cpos, x0)
        cell = realized_cell(rb1, nb, n, tab_total)
        cell.pop("_flags")

        # conditional entropies: empirical (non-start positions) + quantized
        nz = joint > 0
        rowsum = joint.sum(1, keepdims=True)
        n_cond = int(rowsum.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            pj = joint / np.where(rowsum > 0, rowsum, 1)
            hrow = np.where(nz, -pj * np.log2(np.where(nz, pj, 1.0)), 0.0).sum(1)
        h_cond_emp = float((rowsum[:, 0] / n_cond * hrow).sum())
        q_bits = float(CLQ[ctx, sym].sum())          # quantized code length sum

        # block-boundary reset cost (smoothed-empirical, entropy-level)
        rpos = np.flatnonzero(reset_mask)
        rc = 0.0
        if rpos.size:
            prev = sym[rpos - 1]
            cur = sym[rpos]
            bprev = bmap[prev]
            cnt_b = rowsum[:, 0]
            p1 = ((joint[bprev, cur] + SMOOTH_ALPHA)
                  / (cnt_b[bprev] + SMOOTH_ALPHA * 512))
            rc = float((np.log2(p1) - np.log2(marg_sm[cur])).sum() / n)

        # order-1 round-trip on sampled blocks (bit-exact, reset rules live)
        QL = [Qt[i].tolist() for i in range(Cq + 1)]
        CLl = [Ct[i].tolist() for i in range(Cq + 1)]
        S2S = []
        for i in range(Cq + 1):
            pr = np.flatnonzero(Qt[i])
            S2S.append(np.repeat(pr, Qt[i][pr]).tolist() if pr.size else [])
        rt1 = 0
        for i in sorted({0, nb - 1, int(np.argmin(rb1)), int(np.argmax(rb1))}):
            s0 = int(starts_pos[i])
            seq = sym[s0:s0 + W].tolist()
            tids = ctx[s0:s0 + W].tolist()
            x0i = int(x0[i])
            ctxs = f"{t['name']} o1 C{Cq} block {i}"
            fl, bits = o1_enc_block(seq, tids, QL, CLl, x0i)
            data, nbits = v1.pack_block(fl, bits)
            if nbits != int(rb1[i]):
                die(f"O1 ROUND-TRIP ({ctxs}): emitted {nbits} != "
                    f"accounted {int(rb1[i])}")
            dec = o1_dec_block(data, nbits, s0, Ccols, bmap, QL, CLl, S2S, Cq)
            if dec is None or dec[0] != seq:
                die(f"O1 ROUND-TRIP ({ctxs}): decode failed / symbols differ")
            if dec[1] != x0i:
                die(f"O1 ROUND-TRIP ({ctxs}): final state != seed")
            mrec = mant[s0:s0 + W].copy()
            d_rec = dec[1] - M
            if d_rec != int(d_seed[i]):
                die(f"O1 ROUND-TRIP ({ctxs}): payload mismatch")
            mrec[0] = d_rec >> 5
            mrec[1] = ((d_rec & 31) << 2) | (mrec[1] & 3)
            rec = ((np.array(dec[0], np.int64) << 7) | mrec).astype("<u2")
            orig = raw[2 * s0:2 * (s0 + W)]
            if rec.tobytes() != orig:
                die(f"O1 ROUND-TRIP ({ctxs}): reconstructed bytes != original")
            sha_o.update(orig)
            sha_r.update(rec.tobytes())
            rt1 += 1

        h2[f"C{Cq}"] = {
            "cell": cell,
            "delta_vs_frozen_bpw": round(frozen["bpw"] - cell["bpw"], 6),
            "tables_used": len(used), "n_top_identity": n_top,
            "tab_bits_total": int(tab_total), "ctxmap_bits": int(ctxmap_bits),
            "n_cond_positions": n_cond,
            "H_cond_emp": round(h_cond_emp, 6),
            "cond_qentropy_bpw": round(q_bits / n, 6),
            "reset_cost_bpw": round(rc, 6),
            "n_reset_positions": int(rpos.size),
            "roundtrip_blocks": rt1,
        }

    if sha_o.digest() != sha_r.digest():
        die(f"ROUND-TRIP ({t['name']}): SHA-256 mismatch over sampled spans")

    # ---- sparse stats for the layer-level holdout (C-agnostic)
    nonstart = np.flatnonzero(
        ~((np.arange(n) % W == 0) | (np.arange(n) % Ccols == 0)))
    o1_pairs = np.stack([sym[nonstart - 1], sym[nonstart]], axis=1)
    start_syms = sym[(np.arange(n) % W == 0) | (np.arange(n) % Ccols == 0)]

    return {
        "name": t["name"], "layer": t["layer"], "expert": t["expert"],
        "proj": t["proj"], "ho": t["ho"], "n": int(n), "R": int(R),
        "C": int(Ccols), "nb": int(nb), "acct": ACCT_STAMP,
        "H_sym": round(H, 6), "floor_bpw": round(H + 7.0, 6),
        "frozen": frozen,
        "h1": h1,
        "h2": h2,
        "roundtrip": {"frozen_blocks": rt_blocks,
                      "o1_blocks": sum(h2[k]["roundtrip_blocks"] for k in h2),
                      "sha256_ok": True},
        "stats": {"hist": sparse_hist(sym),
                  "o1_joint": sparse_counts(o1_pairs),
                  "start_hist": sparse_hist(start_syms)},
    }


# ------------------------------------------------------------------ holdout ---
def dense_from_sparse_hist(s: dict) -> np.ndarray:
    h = np.zeros(512, np.int64)
    h[np.array(s["i"], np.int64)] = np.array(s["c"], np.int64)
    return h


def dense_from_sparse_joint(s: dict) -> np.ndarray:
    j = np.zeros((512, 512), np.int64)
    j[np.array(s["i"], np.int64), np.array(s["j"], np.int64)] = \
        np.array(s["c"], np.int64)
    return j


def holdout_eval(recs: list[dict]) -> dict:
    """Per (layer, proj, C): bucket map + conditional model fit on the pooled
    train-half experts, cross-entropy evaluated on the test half (add-alpha
    smoothing on the 512 alphabet), vs the order-0 train marginal."""
    out = {}
    cells = sorted({(r["layer"], r["proj"]) for r in recs})
    for (L, proj) in cells:
        grp = [r for r in recs if r["layer"] == L and r["proj"] == proj]
        tr = [r for r in grp if r["ho"] == "train"]
        te = [r for r in grp if r["ho"] == "test"]
        if not tr or not te:
            out[f"L{L}_{proj}"] = {"skipped": "missing train or test half"}
            continue
        Jtr = sum(dense_from_sparse_joint(r["stats"]["o1_joint"]) for r in tr)
        Jte = sum(dense_from_sparse_joint(r["stats"]["o1_joint"]) for r in te)
        marg_tr = sum(dense_from_sparse_hist(r["stats"]["hist"]) for r in tr)
        mtr = Jtr.sum(1).astype(np.float64)           # train non-start marginal
        p0 = (Jtr.sum(0) + SMOOTH_ALPHA) / (Jtr.sum() + SMOOTH_ALPHA * 512)
        te_marg = Jte.sum(0)
        N_te = int(te_marg.sum())
        ce0 = float(-(te_marg * np.log2(p0)).sum() / N_te)
        cell = {"n_test_pairs": N_te, "H0_holdout": round(ce0, 6), "per_C": {}}
        for Cq in CTXS:
            order = np.argsort(-marg_tr, kind="stable")
            top = order[:Cq - 1]
            top = top[marg_tr[top] > 0]
            bmap = np.full(512, Cq - 1, np.int64)
            bmap[top] = np.arange(top.size)
            fold_tr = np.zeros((Cq, 512), np.int64)
            fold_te = np.zeros((Cq, 512), np.int64)
            np.add.at(fold_tr, bmap, Jtr)
            np.add.at(fold_te, bmap, Jte)
            rows = fold_tr.sum(1, keepdims=True).astype(np.float64)
            p1 = (fold_tr + SMOOTH_ALPHA) / (rows + SMOOTH_ALPHA * 512)
            ce1 = float(-(fold_te * np.log2(p1)).sum() / N_te)
            cell["per_C"][f"C{Cq}"] = {
                "H_cond_holdout": round(ce1, 6),
                "gain_bits_per_sym": round(ce0 - ce1, 6)}
        out[f"L{L}_{proj}"] = cell
    return out


# ------------------------------------------------------------------ summary ---
def summarize(tg: list[dict], jsonl: Path, summaryp: Path, synthetic: bool):
    rows = v1.load_rows(jsonl)
    check_stamp(rows, jsonl)
    rec = {}
    for r in rows:
        rec.setdefault(r["name"], r)
    names = [t["name"] for t in tg]
    miss = [nm for nm in names if nm not in rec]
    if miss:
        die(f"summary requires all tensors done; {len(miss)} missing "
            f"(first: {miss[0]}) -- re-invoke without --summary to resume")
    recs = [rec[nm] for nm in names]
    n_tot = sum(r["n"] for r in recs)
    layers = sorted({r["layer"] for r in recs})
    wsum = lambda f: sum(f(r) for r in recs)

    floor_w = wsum(lambda r: r["floor_bpw"] * r["n"]) / n_tot
    frozen_w = wsum(lambda r: r["frozen"]["bpw"] * r["n"]) / n_tot
    rt_ok = all(r["roundtrip"]["sha256_ok"] for r in recs)
    rt_frozen = wsum(lambda r: r["roundtrip"]["frozen_blocks"])
    rt_o1 = wsum(lambda r: r["roundtrip"]["o1_blocks"])

    # ---- H1 aggregate: per plane, weighted ceilings + hit counts
    def agg_plane(key):
        rs = [r for r in recs if r["h1"][key].get("bits", 0) > 0]
        if not rs:
            return {"verdict": "empty on all tensors"}
        bits = sum(r["h1"][key]["bits"] for r in rs)
        bound = sum(min(x for x in (r["h1"][key].get("entropy_bound_bits"),
                                    r["h1"][key].get("lzma_bits"))
                        if x is not None) for r in rs)
        nn = sum(r["n"] for r in rs)
        ceil_w = max(0.0, (bits - bound) / nn)
        mi_hits = sum(1 for r in rs if r["h1"][key].get("mi_hit"))
        ac_hits = sum(1 for r in rs if r["h1"][key].get("ac_hit"))
        struct = sum(1 for r in rs
                     if r["h1"][key]["verdict"].startswith("structure"))
        if ceil_w >= STRUCT_EPS_BPW:
            verdict = f"structure found (ceiling ~{ceil_w:.4f} b/w)"
        elif mi_hits or ac_hits:
            verdict = (f"weak structure ({mi_hits} MI / {ac_hits} autocorr "
                       f"tensor hits; ceiling {ceil_w:.4f} < {STRUCT_EPS_BPW})")
        else:
            verdict = "random at these tests"
        return {"tensors": len(rs), "bits": int(bits),
                "bpw_w": round(bits / nn, 6),
                "ceiling_bpw_w": round(ceil_w, 6),
                "tensors_mi_hit": mi_hits, "tensors_ac_hit": ac_hits,
                "tensors_structure": struct, "verdict": verdict}

    h1_agg = {k: agg_plane(k) for k in ("payload", "flags", "lens", "mant")}

    # ---- H2 aggregate: realized per layer / whole sample
    def w_over(rs, f):
        nn = sum(r["n"] for r in rs)
        return sum(f(r) * r["n"] for r in rs) / nn

    h2_layers = {}
    for L in layers:
        rs = [r for r in recs if r["layer"] == L]
        row = {"tensors": len(rs), "params": sum(r["n"] for r in rs),
               "floor_bpw": round(w_over(rs, lambda r: r["floor_bpw"]), 6),
               "frozen_bpw": round(w_over(rs, lambda r: r["frozen"]["bpw"]), 6)}
        for Cq in CTXS:
            k = f"C{Cq}"
            row[k] = {
                "bpw": round(w_over(rs, lambda r: r["h2"][k]["cell"]["bpw"]), 6),
                "delta_bpw": round(
                    w_over(rs, lambda r: r["h2"][k]["delta_vs_frozen_bpw"]), 6),
                "H_cond_emp": round(
                    w_over(rs, lambda r: r["h2"][k]["H_cond_emp"]), 6),
                "reset_cost_bpw": round(
                    w_over(rs, lambda r: r["h2"][k]["reset_cost_bpw"]), 6),
                "tab_bpw": round(w_over(
                    rs, lambda r: r["h2"][k]["tab_bits_total"] / r["n"]), 6)}
        best = max(CTXS, key=lambda c: row[f"C{c}"]["delta_bpw"])
        row["best_C"] = int(best)
        row["best_delta_bpw"] = row[f"C{best}"]["delta_bpw"]
        h2_layers[f"L{L}"] = row

    whole = {"frozen_bpw": round(frozen_w, 6)}
    for Cq in CTXS:
        k = f"C{Cq}"
        whole[k] = {
            "bpw": round(wsum(lambda r: r["h2"][k]["cell"]["bpw"] * r["n"])
                         / n_tot, 6),
            "delta_bpw": round(
                wsum(lambda r: r["h2"][k]["delta_vs_frozen_bpw"] * r["n"])
                / n_tot, 6)}
    best_C = max(CTXS, key=lambda c: whole[f"C{c}"]["delta_bpw"])
    best_delta = whole[f"C{best_C}"]["delta_bpw"]

    ho = holdout_eval(recs)
    ho_ok_cells, ho_cells = 0, 0
    for cell in ho.values():
        if "per_C" not in cell:
            continue
        ho_cells += 1
        if cell["per_C"][f"C{best_C}"]["gain_bits_per_sym"] > 0:
            ho_ok_cells += 1
    ho_ok = ho_cells > 0 and ho_ok_cells == ho_cells

    fires = bool(best_delta >= H2_GATE_BPW and ho_ok)
    proj_wm = FROZEN_WHOLE_MODEL_BPW - EXPERT_FRAC * best_delta
    scope = ("synthetic smoke -- carries no evidential weight" if synthetic
             else f"sampled experts on layers {layers}; frozen reference "
                  f"recomputed on the same sample")

    verdict_h2 = (f"H2 {'FIRES' if fires else 'does NOT fire'}: best C={best_C} "
                  f"realized delta {best_delta:+.4f} b/w vs gate "
                  f">= {H2_GATE_BPW} (holdout non-overfit: "
                  f"{ho_ok_cells}/{ho_cells} cells positive)")
    if not rt_ok:
        verdict_h2 = "PROVISIONAL: " + verdict_h2

    mode = "SYNTHETIC (smoke only)" if synthetic else "REAL (sampled)"
    print(f"\n=== candidate 0015 -- emission peel: H1 certificates + H2 "
          f"order-1 context [{mode}] ===")
    print(f"sample: {len(recs)} tensors, {n_tot:,} params, layers {layers}; "
          f"acct stamp {ACCT_STAMP}")
    print(f"frozen cell recomputed on sample: {frozen_w:.4f} b/w | floor "
          f"H(sym)+7 = {floor_w:.4f} | frozen whole-model ref "
          f"{FROZEN_WHOLE_MODEL_BPW}")
    print(f"round-trip: {rt_frozen} frozen + {rt_o1} order-1 blocks, "
          f"serializer bit-exact + SHA-256 exact: {'PASS' if rt_ok else 'FAIL'}")

    print("\nH1 -- per-plane randomness certificates (weighted over sample):")
    hdr = (f"{'plane':>9}{'bpw':>9}{'ceiling':>9}{'MI-hit':>8}{'AC-hit':>8}"
           f"  verdict")
    print(hdr)
    print("-" * 78)
    for k in ("payload", "flags", "lens", "mant"):
        a = h1_agg[k]
        if "bpw_w" not in a:
            print(f"{k:>9}  {a['verdict']}")
            continue
        print(f"{k:>9}{a['bpw_w']:>9.4f}{a['ceiling_bpw_w']:>9.4f}"
              f"{a['tensors_mi_hit']:>8}{a['tensors_ac_hit']:>8}  {a['verdict']}")
    print("  (lens is a diagnostic plane: rb is not transmitted -- flags+"
          "budgets are; its structure = tier-design headroom)")

    print("\nH2 -- realized order-1 vs frozen (b/w, weighted; fit-on-self "
          "tables, ALL side costs charged):")
    hdr = (f"{'layer':>7}{'frozen':>9}"
           + "".join(f"{'C' + str(c):>9}" for c in CTXS)
           + f"{'bestC':>7}{'delta':>9}")
    print(hdr)
    print("-" * len(hdr))
    for L in layers:
        row = h2_layers[f"L{L}"]
        print(f"{L:>7}{row['frozen_bpw']:>9.4f}"
              + "".join(f"{row[f'C{c}']['bpw']:>9.4f}" for c in CTXS)
              + f"{row['best_C']:>7}{row['best_delta_bpw']:>+9.4f}")
    print(f"{'ALL':>7}{whole['frozen_bpw']:>9.4f}"
          + "".join(f"{whole[f'C{c}']['bpw']:>9.4f}" for c in CTXS)
          + f"{best_C:>7}{best_delta:>+9.4f}")

    print("\nH2 holdout (fit on half the experts, eval on the other half; "
          "bits/sym gain of ctx over order-0):")
    for key, cell in sorted(ho.items()):
        if "per_C" not in cell:
            print(f"  {key}: {cell['skipped']}")
            continue
        gains = "  ".join(f"C{c}={cell['per_C'][f'C{c}']['gain_bits_per_sym']:+.4f}"
                          for c in CTXS)
        print(f"  {key}: H0={cell['H0_holdout']:.4f}  {gains}")

    rc_w = wsum(lambda r: r["h2"][f"C{best_C}"]["reset_cost_bpw"] * r["n"]) / n_tot
    print(f"\nblock-boundary reset cost at W={W} (best C={best_C}, smoothed "
          f"entropy level): {rc_w:.4f} b/w "
          f"(~1/{W} of symbols start a block and lose their context)")
    print(f"projected whole-model at best C: {proj_wm:.4f} b/w "
          f"(= {FROZEN_WHOLE_MODEL_BPW} - {EXPERT_FRAC} x {best_delta:+.4f}; "
          f"sample-selected C -- selection-optimistic)")
    print(f"H2 gate (>= {H2_GATE_BPW} b/w realized, holdout-positive): "
          f"{verdict_h2}")
    print(f"scope: {scope}")

    summary = {
        "mode": "synthetic" if synthetic else "real", "scope": scope,
        "acct_stamp": ACCT_STAMP, "acct": ACCT,
        "targets": len(recs), "total_params": int(n_tot),
        "layers": [int(x) for x in layers],
        "floor_bpw_weighted": round(floor_w, 6),
        "frozen_bpw_weighted": round(frozen_w, 6),
        "frozen_whole_model_ref_bpw": FROZEN_WHOLE_MODEL_BPW,
        "roundtrip": {"frozen_blocks": int(rt_frozen), "o1_blocks": int(rt_o1),
                      "all_ok": bool(rt_ok)},
        "h1": {"planes": h1_agg,
               "prereg": {"struct_eps_bpw": STRUCT_EPS_BPW,
                          "mi_lags": list(MI_LAGS), "mi_null_r": MI_NULL_R,
                          "ac_sig_k": AC_SIG_K},
               "note": ("payload/mant/flags are emitted bit planes (payload "
                        "excludes slot padding -- pad is already-quantified "
                        "structure); lens is diagnostic (not transmitted)")},
        "h2": {"per_layer": h2_layers, "whole_sample": whole,
               "best_C": int(best_C), "best_delta_bpw": best_delta,
               "gate_bpw": H2_GATE_BPW, "fires": fires,
               "holdout": ho,
               "holdout_positive_cells": f"{ho_ok_cells}/{ho_cells}",
               "reset_cost_bpw_bestC_weighted": round(rc_w, 6),
               "projected_whole_model_bpw": round(proj_wm, 6),
               "verdict": verdict_h2},
        "accounting_note": (
            "realized cells use the frozen v2 mechanics exactly (W=128 "
            "bit-stride, DP T=4 tier budgets, P100 no-escape, L3 flush "
            "payload, per-block class flags, u32 rank anchors per 512 "
            "blocks, 96-bit class descriptors, 32-byte header, pad8 record "
            "align, pad8(7n-12nb) mantissa plane); order-1 cells charge C "
            "occupied bucket tables + the order-0 start table (each "
            "pad8(512+nnz*12)) + a pad8(8+C+(C-1)*9)-bit context header "
            "(8-bit mode/C field, C-bit table-occupancy bitmap, (C-1) 9-bit "
            "bucket-map ids); block sizes are measured emitted bits of the "
            "named coder, round-trip verified per tensor per C on sampled "
            "blocks with SHA-exact BF16 reconstruction over sampled spans"),
    }
    summaryp.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written: {summaryp}")
    return summary


# --------------------------------------------------------------------- main ---
def sample_targets(snap: Path, synthetic: bool, layer: int | None) -> list[dict]:
    allt = v1.enum_targets(snap, True)          # all layers; filter ourselves
    have = sorted({t["layer"] for t in allt})
    want = have if synthetic else list(LAYERS_REAL)
    if layer is not None:
        want = [layer]
    missing = [L for L in want if L not in have]
    if missing:
        die(f"requested layers {missing} not in snapshot (available: {have})")
    out = []
    for L in want:
        for proj in ("up", "down"):
            cand = sorted((t for t in allt
                           if t["layer"] == L and t["proj"] == proj),
                          key=lambda t: t["expert"])
            if not cand:
                die(f"no {proj}_proj experts on layer {L}")
            k = min(EXPERTS_PER_PROJ, len(cand))
            sel = sorted({int(i) for i in
                          np.linspace(0, len(cand) - 1, k).round()})
            for pos, i in enumerate(sel):
                t = dict(cand[i])
                t["ho"] = "train" if pos % 2 == 0 else "test"
                out.append(t)
    return out


def run(a, snap: Path, jsonl: Path, summaryp: Path):
    tg = sample_targets(snap, a.synthetic, a.layer)
    if a.summary:
        return summarize(tg, jsonl, summaryp, a.synthetic)

    prior = v1.load_rows(jsonl)
    check_stamp(prior, jsonl)
    done = {r["name"] for r in prior}
    t0, processed = time.time(), 0
    for i, t in enumerate(tg):
        if t["name"] in done:
            continue
        if a.limit and processed >= a.limit:
            break
        if time.time() - t0 > a.budget_s:
            print(f"\n[budget] {a.budget_s:.0f}s reached after {processed} "
                  f"tensors -- progress saved, re-invoke to resume.", flush=True)
            sys.exit(0)
        raw = v1.read_raw(snap, t)
        rec = analyze_tensor(raw, t)
        with jsonl.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        done.add(t["name"])
        processed += 1
        print(f"[{i + 1}/{len(tg)}] {t['name']} done "
              f"({time.time() - t0:.0f}s)", flush=True)

    if a.limit and processed >= a.limit and len(done) < len(tg):
        print(f"\n[limit] {a.limit} tensors this invocation -- re-invoke to resume.")
        sys.exit(0)

    print(f"\nall {len(done)}/{len(tg)} tensors done ({time.time() - t0:.0f}s)")
    summarize(tg, jsonl, summaryp, a.synthetic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run against the synthetic tiny snapshot (smoke)")
    ap.add_argument("--summary", action="store_true",
                    help="summary tables + JSON (requires all tensors done)")
    ap.add_argument("--layer", type=int, default=None,
                    help="restrict to one layer (default: layers "
                         f"{LAYERS_REAL} real / all present synthetic)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max tensors this invocation (0 = no cap)")
    ap.add_argument("--budget-s", type=float, default=420.0,
                    help="soft wall-clock budget; exits cleanly when exceeded")
    a = ap.parse_args()

    snap = v1.SYN_SNAP if a.synthetic else v1.REAL_SNAP
    tag = (("_synthetic" if a.synthetic else "")
           + (f"_layer{a.layer}" if a.layer is not None else ""))
    ART.mkdir(parents=True, exist_ok=True)
    jsonl = ART / f"emission_peel_results{tag}.jsonl"
    summaryp = ART / f"emission_peel_summary{tag}.json"

    lockp = ART / f"emission_peel{tag}.lock"
    try:
        fd = os.open(lockp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = lockp.read_text().strip() or "?"
        except OSError:
            holder = "?"
        die(f"lock file {lockp} exists (pid {holder}); if no run is live, "
            f"delete it and retry")
    with os.fdopen(fd, "w") as lf:
        lf.write(str(os.getpid()))
    try:
        run(a, snap, jsonl, summaryp)
    finally:
        try:
            lockp.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
