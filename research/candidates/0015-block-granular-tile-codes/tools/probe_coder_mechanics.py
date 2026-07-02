"""probe_coder_mechanics.py -- candidate 0015: a lower-redundancy coder for the
frozen tile+M1 format contract (R1/R2/R3), exact accounting, round-trip proven.

The measured whole-model tile+M1 number (10.68664 b/w, measure_m1_full.py)
converged on ONE blocking lever: the frozen coder is a bit-by-bit-renorm rANS
(M=4096, state in [M,2M), renorm emits x&1 while x>=2q) whose measured
per-symbol redundancy over quantized entropy GROWS with alphabet size --
~0.051 b/sym at A=512, 0.063 at A=1024 (M1/sym10), 0.098 at A=2048. That
redundancy killed M2 (binary lanes), ate most of M3's second folded bit, and
per the T2 MI decomposition (m1full_mi_layer*.json) is the only thing standing
between the format and the sym11 collectible (bit-2 headroom 0.0171 b/w,
~98% exponent-driven). This probe prices a WIDER-STATE / BYTE-RENORM coder for
the SAME format contract -- fixed-stride blocks, O(1) block address, W=128
sequential in-block decode, DP T=4 tier budgets at P100, mantissa-carrying
flush -- and asks whether the cheaper coder (a) beats the M1 reference and
(b) makes folding the second mantissa bit net-positive.

THE VARIANTS (pre-registered; every cost charged exactly):
  M1ref  the standing tile+M1 cell recomputed on the sample with the FROZEN
         bit-renorm coder (probe_mantissa_phase.py's M1 machinery imported
         verbatim, asserted cell-identical): sym10 = u >> 6 (sign+exp8+mantMSB,
         A=1024, 12-bit table), 6-bit verbatim low plane, 12-bit flush seeded
         with the block's first 12 low-plane bits (credit 12). Reference
         10.6973 b/w on the 64-tensor sample convention (RESULTS.md).
  R1     sym10 through the WIDE coder: 32-bit state in [L, 256L), L = 2^24,
         renorm emits one BYTE (x & 0xFF) while x >= q*(L/M)*256; flush stores
         the 32-bit final state; the encoder's INITIAL state is seeded
         x0 = L + d with d = the first 31 bits of the block's 6-bit low plane
         (x0 < L + 2^31 < 256L, valid), so the decoder's final state returns a
         31-bit mantissa payload -- the wider state carries MORE credit than
         L3's 12 (net flush cost = 32 - 31 = 1 bit/block, charged exactly;
         the low plane shrinks to 6W - 31 bits/block). Tables 12-bit (M=4096).
  R2     R1 + sym11 = u >> 5 (A=2048, 12-bit table, 5-bit low plane, credit
         31): does bit-2 become net-positive once the coder is cheap?
  R3     R1 with 14-bit tables (M=16384; finer quantization, table charged
         pad8(A + nnz*14)): is the 12-bit quantization delta now a visible
         fraction of the residual?
  R4     R2 with 14-bit tables (sym11, A=2048, M=16384): the sym11 question
         at the finer table precision, so the SYM11 verdict is rendered under
         the BEST coder rather than pinned to 12-bit tables (A=2048 into 4096
         slots is the coarsest quantization of any variant and would
         systematically overstate sym11's quant penalty).

THE WIDE CODER (named exactly; sizes are measured emitted bits, never bounds):
per-block single-lane byte-renorm rANS; quantized table sums to M = 2^m_log2
(m_log2 = 12 or 14, deterministic largest-remainder, every present symbol
>= 1); state x in [L, 256L), L = 2^24; encode consumes symbols in reverse from
x0 = L + d (d < 2^31): renorm emits (x & 0xFF) while x >= q*(L>>m_log2)*256,
then x = (x//q)*M + cum[s] + (x%q); flush stores x_final (32 bits). Decode:
x = 32-bit flush; per symbol slot = x & (M-1), s = slot2sym[slot],
x = q[s]*(x>>m_log2) + slot - cum[s], then read bytes while x < L; after W
symbols x == x0 and d = x - L is the payload. Same fixed-stride / O(1)-address
contract as the frozen format: the DP tier budgets are over the measured
per-block bits, nothing else moves. Contract note: the first ceil(31/k_low)
weights of a block (6 for sym10, 7 for sym11) finalize only after the full
block decode (was 2 under L3) -- same register-tile kernel contract.

Accounting is the frozen v2 cell EXACTLY (v2.tier_dp T=4, P100 asserted
no-escape, per-block class flags, u32 rank anchors per 512 blocks, 96-bit
class descriptors, 32-byte header, pad8 record align, pad8(A + nnz*m_log2)
table, pad8(k_low*n - credit*nb) verbatim low plane). Per tensor and variant
the realized bpw is decomposed EXACTLY (asserted to 1e-8) into
  bound (H(symA) + k_low)  + quant (table quantization delta)
  + flush_net ((flush - credit)*nb/n) + renorm (coded - qentropy - flush*nb)
  + pad (DP tier slack) + tab + misc (flags/rank/cdir/header/aligns).
Every variant is round-trip proven on sampled blocks of EVERY tensor:
pure-Python encode -> serialized bytes -> independent-path decode, asserting
emitted bits == accounted bits, symbols exact, final state == seed, payload
== the block's leading low-plane bits, and byte-exact BF16 reconstruction
with the flush-borne fields DESTROYED before the rebuild (zero_borrowed) so
the credit provably flows from the decoded payload -- SHA-256 over all
sampled spans. Generalization parity gates per tensor: the generalized
quantizer == mp.quantize_hist_a at (A=1024, M=4096); the generalized seed
== mp.d_seed_k at credit=12; the generalized cell == mp.realized_cell_k at
credit=12 on the M1ref block sizes.

Pre-registered gates:
  R-GATE   fires if the best R-variant beats the M1 reference (recomputed on
           the same sample) by >= 0.02 b/w, all costs charged, round-trip
           proven on sampled blocks of every tensor (accounting for the
           remaining blocks via the bit-identical emitted-bit simulation);
           only a full-sample non-synthetic run can fire it.
  SYM11    separate verdict, recorded either way: bit-2 is net-positive under
           the BEST coder iff bpw(best of {R2,R4}) < bpw(best of {R1,R3});
           the matched-table-precision pairs (R1 vs R2 at 12-bit, R3 vs R4 at
           14-bit) are reported alongside, against the entropy-level headroom
           bound(sym10) - bound(sym11) (~0.017 b/w on real).

Field split (stz convention): u16 LE, sym = u >> 7, mant = u & 0x7F;
sym10 = u >> 6, sym11 = u >> 5. Sample: 8 experts x {up,down}_proj on layers
1/13/27/40 (>=16 tensors/layer, the mantissa-probe convention, via
ep.sample_targets). M1 whole-model reference 10.686640 b/w (measured);
projection = 10.686640 - 0.93 * best delta (selection-optimistic).

Usage:
  uv run python probe_coder_mechanics.py --synthetic     # smoke (fake snapshot)
  uv run python probe_coder_mechanics.py                 # real run (resumable)
  uv run python probe_coder_mechanics.py --layer 13      # one layer only
  uv run python probe_coder_mechanics.py --summary       # tables + JSON
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import probe_block_codes as v1        # noqa: E402  -- verified infrastructure
import probe_block_codes_v2 as v2     # noqa: E402  -- coder, DP, accounting
import probe_emission_peel as ep      # noqa: E402  -- sampling convention
import probe_mantissa_phase as mp     # noqa: E402  -- M1 machinery (verbatim)

M, M_LOG2, FLUSH_BITS = v1.M, v1.M_LOG2, v1.FLUSH_BITS
pad8, ceil_div, die = v1.pad8, v1.ceil_div, v1.die
ART = v1.ART
RANK_GROUP, RANK_BITS = v2.RANK_GROUP, v2.RANK_BITS
HEAD_BITS, CLASS_DIR_BITS = v2.HEAD_BITS, v2.CLASS_DIR_BITS

# ---- frozen format cell (unchanged; no re-selection here)
W = mp.W                       # 128, bit-stride, fusible
T_MAX = mp.T_MAX               # 4 DP tiers, P100 (no escapes, asserted)

# ---- the wide coder (R-family)
WIDE_LOG2_L = 24
WIDE_L = 1 << WIDE_LOG2_L      # state in [L, 256L) = [2^24, 2^32)
WIDE_B_LOG2 = 8                # renorm chunk: one byte
FLUSH_W_WIDE = 32              # flush stores the full 32-bit final state
CREDIT_WIDE = 31               # seed payload d in [0, 2^31): L + d < 256L

# ---- gates / references (pre-registered)
GATE_BPW = 0.02                # best R-variant must beat M1ref by >= this
M1_SAMPLE_REF_BPW = 10.6973    # M1 on the 64-tensor sample (RESULTS.md)
M1_SAMPLE_REF_TOL = 0.0005     # recomputed reference must reproduce it
M1_WHOLE_MODEL_BPW = 10.686640 # measured whole-model tile+M1 (T1)
EXPERT_FRAC = 0.93
SYM11_HEADROOM_REF = 0.0171    # T2 entropy-level bit-2 headroom (context)

VARIANTS = ("M1ref", "R1", "R2", "R3", "R4")
R_VARIANTS = ("R1", "R2", "R3", "R4")
SYM10_VARIANTS = ("R1", "R3")          # sym10 R-variants (12- / 14-bit table)
SYM11_VARIANTS = ("R2", "R4")          # sym11 R-variants (12- / 14-bit table)
SYM11_PAIRS = (("R1", "R2"), ("R3", "R4"))  # matched table precision
SPEC = {
    "M1ref": {"A": 1024, "shift": 6, "k_low": 6, "coder": "bit",
              "m_log2": 12, "flush": 12, "credit": 12},
    "R1":    {"A": 1024, "shift": 6, "k_low": 6, "coder": "byte",
              "m_log2": 12, "flush": FLUSH_W_WIDE, "credit": CREDIT_WIDE},
    "R2":    {"A": 2048, "shift": 5, "k_low": 5, "coder": "byte",
              "m_log2": 12, "flush": FLUSH_W_WIDE, "credit": CREDIT_WIDE},
    "R3":    {"A": 1024, "shift": 6, "k_low": 6, "coder": "byte",
              "m_log2": 14, "flush": FLUSH_W_WIDE, "credit": CREDIT_WIDE},
    "R4":    {"A": 2048, "shift": 5, "k_low": 5, "coder": "byte",
              "m_log2": 14, "flush": FLUSH_W_WIDE, "credit": CREDIT_WIDE},
}

WIDE_CODER_SPEC = (
    "per-block single-lane byte-renorm rANS; quantized table sums to "
    "M=2^m_log2 (12 or 14); state in [L,256L), L=2^24; renorm emits x&0xFF "
    "while x >= q*(L>>m_log2)*256; encode x=(x//q)*M+cum+(x%q); flush = "
    "32-bit final state; x0 seeded L + first-31-low-plane-bits (decoder "
    "final state returns them); sizes are measured emitted bits, "
    "round-trip verified on samples")

ACCT = {"schema": 1, "probe": "coder_mechanics", "W": W, "T_MAX": T_MAX,
        "P": 100, "L1": 1, "L3": 1, "L4": 0,
        "M_LOG2": M_LOG2, "FLUSH_BITS": FLUSH_BITS,
        "WIDE_LOG2_L": WIDE_LOG2_L, "WIDE_B_LOG2": WIDE_B_LOG2,
        "FLUSH_W_WIDE": FLUSH_W_WIDE, "CREDIT_WIDE": CREDIT_WIDE,
        "RANK_GROUP": RANK_GROUP, "RANK_BITS": RANK_BITS,
        "HEAD_BITS": HEAD_BITS, "CLASS_DIR_BITS": CLASS_DIR_BITS,
        "GATE_BPW": GATE_BPW,
        "VARIANTS": {k: dict(v) for k, v in SPEC.items()},
        "TABLE_BITS_RULE": "pad8(A + nnz*m_log2) per table",
        "MANT_PLANE_RULE": "pad8(k_low*n - credit*nb)",
        "LAYERS_REAL": list(ep.LAYERS_REAL),
        "EXPERTS_PER_PROJ": ep.EXPERTS_PER_PROJ,
        "M1_SAMPLE_REF_BPW": M1_SAMPLE_REF_BPW,
        "M1_WHOLE_MODEL_BPW": M1_WHOLE_MODEL_BPW,
        "EXPERT_FRAC": EXPERT_FRAC,
        "SYM11_HEADROOM_REF": SYM11_HEADROOM_REF,
        "CODER_BIT": v2.CODER_SPEC, "CODER_WIDE": WIDE_CODER_SPEC}
ACCT_STAMP = hashlib.sha256(json.dumps(ACCT, sort_keys=True).encode()).hexdigest()[:12]


def check_stamp(rows: list[dict], jsonl: Path):
    bad = [r for r in rows if r.get("acct") != ACCT_STAMP]
    if bad:
        die(f"{len(bad)}/{len(rows)} rows in {jsonl} carry accounting stamp "
            f"{bad[0].get('acct')!r} != current {ACCT_STAMP!r} -- move that "
            f"file aside and re-run")


# ------------------------------------------- generalized quantizer / tables ---
def quantize_hist_gen(hist: np.ndarray, n: int, A: int, Mq: int) -> np.ndarray:
    """mp.quantize_hist_a generalized to a table total Mq (mp hardcodes 4096).
    Identical algorithm: deterministic largest-remainder, every present
    symbol >= 1. Asserted equal to mp.quantize_hist_a at Mq=4096 per tensor."""
    nz = np.flatnonzero(hist)
    assert 1 <= nz.size <= Mq, "present symbols must fit the table"
    tgt = hist[nz].astype(np.float64) * (Mq / n)
    q = np.maximum(np.floor(tgt), 1.0).astype(np.int64)
    guard = 0
    while True:
        d = Mq - int(q.sum())
        if d == 0:
            break
        guard += 1
        if guard > 2 * Mq:
            die("ANS quantization did not converge (progress guard)")
        if d > 0:
            rem = tgt - q
            order = np.lexsort((nz, -rem))
            q[order[:min(d, nz.size)]] += 1
        else:
            surplus = -d
            elig = np.flatnonzero(q > 1)
            if elig.size == 0:
                die("ANS quantization cannot reach the table total")
            order = elig[np.lexsort((nz[elig], tgt[elig] - q[elig]))]
            cap = q[order] - 1
            ccap = np.cumsum(cap)
            k = int(np.searchsorted(ccap, surplus))
            if k >= order.size:
                q[order] = 1
            else:
                if k:
                    q[order[:k]] -= cap[:k]
                q[order[k]] -= surplus - (int(ccap[k - 1]) if k else 0)
    out = np.zeros(A, np.int64)
    out[nz] = q
    return out


def build_table_gen(hist: np.ndarray, n: int, A: int, m_log2: int):
    """Quantized m_log2-bit table over an A-symbol alphabet. Table cost =
    pad8(A + nnz*m_log2): A-bit presence bitmap + m_log2 bits per present
    symbol (the mp rule, field width following the table precision)."""
    Mq = 1 << m_log2
    q = quantize_hist_gen(hist, n, A, Mq)
    nnz = int((hist > 0).sum())
    present = np.flatnonzero(q)
    cum = np.zeros(A, np.int64)
    cum[present] = np.cumsum(q[present]) - q[present]
    cl_q = np.zeros(A)
    cl_q[q > 0] = m_log2 - np.log2(q[q > 0])
    return q, cum, cl_q, pad8(A + nnz * m_log2), nnz


# ------------------------------------------------- generalized seed helpers ---
def d_seed_gen(mlow: np.ndarray, starts: np.ndarray, k: int,
               borrow: int) -> np.ndarray:
    """First `borrow` bits of the k-bit MSB-first weight-major low plane of
    each block (mp.d_seed_k generalized from borrow=12; asserted equal
    there). borrow=31, k=6 walks 6 weights (6,6,6,6,6,1 bits)."""
    acc = np.zeros(starts.size, np.int64)
    got, w = 0, 0
    while got < borrow:
        take = min(k, borrow - got)
        acc = (acc << take) | (mlow[starts + w] >> (k - take))
        got += take
        w += 1
    return acc


def apply_seed_gen(mrec: np.ndarray, d: int, k: int, borrow: int):
    """Rebuild the borrowed leading fields of a block's k-bit low plane from
    the recovered payload (inverse of d_seed_gen, one block)."""
    got, w = 0, 0
    while got < borrow:
        take = min(k, borrow - got)
        chunk = (d >> (borrow - got - take)) & ((1 << take) - 1)
        keep = (1 << (k - take)) - 1
        mrec[w] = (chunk << (k - take)) | (int(mrec[w]) & keep)
        got += take
        w += 1


def zero_borrowed_gen(mrec: np.ndarray, k: int, borrow: int):
    """Destroy the flush-borne fields of a block's k-bit low plane (exactly
    the bits apply_seed_gen writes) before the round-trip rebuild, so the
    credit provably flows from the DECODED payload."""
    got, w = 0, 0
    while got < borrow:
        take = min(k, borrow - got)
        mrec[w] = int(mrec[w]) & ((1 << (k - take)) - 1)
        got += take
        w += 1


# --------------------------------------------------------- the wide coder ---
def rans_sim_wide(qm: np.ndarray, cm: np.ndarray, x0: np.ndarray,
                  m_log2: int) -> np.ndarray:
    """Exact vectorized emitted-bit simulation of the byte-renorm coder
    (bit-identical arithmetic to wide_enc_block). qm/cm: (nb, W) int64
    per-symbol quantized / exclusive-cumulative counts; x0 in [L, 256L).
    Returns FLUSH_W_WIDE + 8 * renorm-byte-events per block."""
    nbk, Wb = qm.shape
    Mq = np.int64(1 << m_log2)
    kf = np.int64(WIDE_L >> m_log2)
    x = x0.astype(np.int64).copy()
    bits = np.full(nbk, FLUSH_W_WIDE, np.int64)
    for j in range(Wb - 1, -1, -1):
        qq = qm[:, j]
        thr = (qq * kf) << WIDE_B_LOG2
        while True:
            mask = x >= thr
            if not mask.any():
                break
            x[mask] >>= WIDE_B_LOG2
            bits[mask] += 8
        x = (x // qq) * Mq + cm[:, j] + (x % qq)
    return bits


def wide_enc_block(syms: list, ql: list, cl: list, x0: int, m_log2: int):
    """Reference encoder (pure Python). Returns (x_final, bytes in emission
    order). x0 in [L, 256L); under the seed contract x0 = L + d, d < 2^31."""
    assert WIDE_L <= x0 < (WIDE_L << 8)
    Mq = 1 << m_log2
    kf = WIDE_L >> m_log2
    x = x0
    out = []
    ap = out.append
    for s in reversed(syms):
        qq = ql[s]
        thr = (qq * kf) << WIDE_B_LOG2
        while x >= thr:
            ap(x & 0xFF)
            x >>= 8
        x = (x // qq) * Mq + cl[s] + (x % qq)
    return x, out


def wide_pack_block(xf: int, byts: list) -> tuple[bytes, int]:
    """Serialize: 32-bit final state big-endian, then renorm bytes in REVERSE
    emission order (decoder pops LIFO)."""
    by = int(xf).to_bytes(4, "big") + bytes(reversed(byts))
    return by, 8 * len(by)


def wide_dec_block(data: bytes, nbits: int, L_syms: int, ql: list, cl: list,
                   s2s: list, m_log2: int):
    """Reference decoder (independent path). Returns (symbols, final_state)
    -- final_state == encoder's x0, payload = final_state - WIDE_L -- or
    None on any inconsistency (starvation, leftovers, state out of range)."""
    if nbits < FLUSH_W_WIDE or nbits % 8:
        return None
    nby = nbits // 8
    x = int.from_bytes(data[:4], "big")
    if not (WIDE_L <= x < (WIDE_L << 8)):
        return None
    Mq = 1 << m_log2
    mask = Mq - 1
    pos = 4
    out = []
    ap = out.append
    for _ in range(L_syms):
        slot = x & mask
        s = s2s[slot]
        x = ql[s] * (x >> m_log2) + slot - cl[s]
        while x < WIDE_L:
            if pos >= nby:
                return None
            x = (x << 8) | data[pos]
            pos += 1
        ap(s)
    if pos != nby or not (WIDE_L <= x < (WIDE_L << 8)):
        return None
    return out, x


# ------------------------------------------------------ realized cell (gen) ---
def realized_cell_gen(rb: np.ndarray, nb: int, n: int, tab_bits: int,
                      k_low: int, credit: int) -> dict:
    """mp.realized_cell_k generalized to an arbitrary flush-borne credit
    (mant plane = pad8(k_low*n - credit*nb)). credit=12 is asserted equal to
    mp.realized_cell_k per tensor (on the M1ref block sizes)."""
    assert k_low * W >= credit, "block low plane must cover the seed"
    kept, budgets, slots, counts = v2.tier_dp(rb, True, T_MAX)[T_MAX]
    assert int(rb.max()) <= budgets[-1]          # P100: nothing escapes
    classes = len(budgets)
    flagb = int(math.ceil(math.log2(classes))) if classes > 1 else 0
    flag = nb * flagb
    rank = classes * RANK_BITS * ceil_div(nb, RANK_GROUP) if classes > 1 else 0
    cdir = classes * CLASS_DIR_BITS if classes > 1 else 0
    sym_raw = kept + flag + rank + cdir + tab_bits + HEAD_BITS
    sym_total = pad8(sym_raw)
    mant_bits = pad8(k_low * n - credit * nb)
    cell = {"bpw": round((sym_total + mant_bits) / n, 6),
            "sym_bits": int(sym_total), "mant_bits": int(mant_bits),
            "coded_bits": int(rb.sum()), "kept_slot_bits": int(kept),
            "pad_bits": int(kept - rb.sum()),
            "flag_bits": int(flag), "rank_bits": int(rank),
            "cdir_bits": int(cdir), "tab_bits": int(tab_bits),
            "align_bits": int(sym_total - sym_raw),
            "budgets": [int(x) for x in budgets],
            "class_counts": [int(x) for x in counts],
            "classes": int(classes), "flagb": int(flagb)}
    comp = (cell["kept_slot_bits"] + cell["flag_bits"] + cell["rank_bits"]
            + cell["cdir_bits"] + cell["tab_bits"] + HEAD_BITS
            + cell["align_bits"])
    if cell["sym_bits"] != comp:
        die("CELL RECONCILIATION: sym-plane components do not sum")
    return cell


DECOMP_KEYS = ("bound_bpw", "quant_bpw", "flush_net_bpw", "renorm_bpw",
               "pad_bpw", "tab_bpw", "misc_bpw")


def decompose(cell: dict, qE_bits: float, H: float, spec: dict,
              nb: int, n: int) -> dict:
    """Exact redundancy decomposition of the realized cell; the seven named
    components are asserted to sum to the charged bpw (1e-8). renorm =
    measured coded bits minus quantized entropy minus the gross flush; the
    seed payload is credited at face value against the flush (flush_net)."""
    flushb, credit, k_low = spec["flush"], spec["credit"], spec["k_low"]
    coded = cell["coded_bits"]
    mant_align = cell["mant_bits"] - (k_low * n - credit * nb)
    assert mant_align >= 0
    misc = (cell["flag_bits"] + cell["rank_bits"] + cell["cdir_bits"]
            + HEAD_BITS + cell["align_bits"] + mant_align)
    d = {"bound_bpw": H + k_low,
         "quant_bpw": qE_bits / n - H,
         "flush_net_bpw": (flushb - credit) * nb / n,
         "renorm_bpw": (coded - qE_bits - flushb * nb) / n,
         "pad_bpw": cell["pad_bits"] / n,
         "tab_bpw": cell["tab_bits"] / n,
         "misc_bpw": misc / n}
    total = sum(d[k] for k in DECOMP_KEYS)
    exact = (cell["sym_bits"] + cell["mant_bits"]) / n
    if abs(total - exact) > 1e-8:
        die(f"DECOMPOSITION RECONCILIATION: {total!r} != {exact!r}")
    d["flush_gross_bpw"] = flushb * nb / n
    d["credit_bpw"] = credit * nb / n
    return {k: round(v, 8) for k, v in d.items()}


# --------------------------------------------------------------- round-trip ---
def rt_wide(raw: bytes, name: str, var: str, symk: np.ndarray,
            mlow: np.ndarray, spec: dict, q: np.ndarray, cum: np.ndarray,
            x0v: np.ndarray, dsv: np.ndarray, rb: np.ndarray,
            starts: np.ndarray, sha_o, sha_r) -> int:
    """Round-trip gate for the wide-coder variants: encode -> serialize ->
    independent-path decode on sampled blocks; verify emitted bits ==
    accounted, symbols exact, final state == seed, payload == the block's
    leading low-plane bits, and byte-exact BF16 reconstruction with the
    flush-borne fields destroyed before the rebuild."""
    ql, cl = q.tolist(), cum.tolist()
    pres = np.flatnonzero(q)
    s2s = np.repeat(pres, q[pres]).tolist()
    k_low, shift = spec["k_low"], spec["shift"]
    m_log2, credit = spec["m_log2"], spec["credit"]
    nb = starts.size
    done = 0
    for i in mp.sample_ids(rb, nb):
        s0 = int(starts[i])
        seq = symk[s0:s0 + W].tolist()
        x0i = int(x0v[i])
        ctx = f"{name} {var} block {i}"
        xf, byts = wide_enc_block(seq, ql, cl, x0i, m_log2)
        data, nbits = wide_pack_block(xf, byts)
        if nbits != int(rb[i]):
            die(f"ROUND-TRIP ({ctx}): emitted {nbits} != accounted {int(rb[i])}")
        dec = wide_dec_block(data, nbits, W, ql, cl, s2s, m_log2)
        if dec is None or dec[0] != seq:
            die(f"ROUND-TRIP ({ctx}): decode failed / symbols differ")
        if dec[1] != x0i:
            die(f"ROUND-TRIP ({ctx}): final state {dec[1]} != seed {x0i}")
        d_rec = dec[1] - WIDE_L
        if not (0 <= d_rec < (1 << credit)) or d_rec != int(dsv[i]):
            die(f"ROUND-TRIP ({ctx}): payload {d_rec} != {int(dsv[i])}")
        mrec = mlow[s0:s0 + W].copy()
        zero_borrowed_gen(mrec, k_low, credit)  # credit must flow from payload
        apply_seed_gen(mrec, d_rec, k_low, credit)
        rec = ((np.array(dec[0], np.int64) << shift) | mrec).astype("<u2")
        orig = raw[2 * s0:2 * (s0 + W)]
        if rec.tobytes() != orig:
            die(f"ROUND-TRIP ({ctx}): reconstructed bytes != original")
        sha_o.update(orig)
        sha_r.update(rec.tobytes())
        done += 1
    return done


# --------------------------------------------------------------- per-tensor ---
def analyze_tensor(raw: bytes, t: dict) -> dict:
    u = np.frombuffer(raw, "<u2")
    n = u.size
    R, Ccols = t["shape"]
    assert n == R * Ccols, (t["name"], n, R, Ccols)
    if n % W:
        die(f"n={n} not divisible by W={W} on {t['name']}")
    nb = n // W
    starts = np.arange(nb, dtype=np.int64) * W

    sym10 = (u >> 6).astype(np.int64)
    m6 = (u & 0x3F).astype(np.int64)
    sym11 = (u >> 5).astype(np.int64)
    m5 = (u & 0x1F).astype(np.int64)
    hist10 = np.bincount(sym10, minlength=1024).astype(np.int64)
    hist11 = np.bincount(sym11, minlength=2048).astype(np.int64)
    H10, H11 = mp.h0_bits(hist10), mp.h0_bits(hist11)

    # generalization parity gates vs the verified mp machinery
    if not np.array_equal(quantize_hist_gen(hist10, n, 1024, M),
                          mp.quantize_hist_a(hist10, n, 1024)):
        die(f"QUANTIZER PARITY on {t['name']}: gen != mp at (A=1024, M=4096)")
    if not np.array_equal(d_seed_gen(m6, starts, 6, 12),
                          mp.d_seed_k(m6, starts, 6)):
        die(f"SEED PARITY on {t['name']}: d_seed_gen(12) != mp.d_seed_k")

    sha_o, sha_r = hashlib.sha256(), hashlib.sha256()
    cells, decomp, rt, nnzs = {}, {}, {}, {}

    # ---- M1ref: the standing tile+M1 cell, frozen bit-renorm coder (mp path)
    q10, cum10, tab10, nnz10 = mp.build_table_a(hist10, n, 1024)
    cl10 = np.zeros(1024)
    cl10[q10 > 0] = M_LOG2 - np.log2(q10[q10 > 0])
    qE10 = float((hist10 * cl10).sum())
    ds12 = mp.d_seed_k(m6, starts, 6)
    x0b = (M + ds12).astype(np.int64)
    rb0 = v2.rans_sim_blocks(q10[sym10].reshape(nb, W),
                             cum10[sym10].reshape(nb, W), x0b)
    cells["M1ref"] = mp.realized_cell_k(rb0, nb, n, tab10, 6)
    if realized_cell_gen(rb0, nb, n, tab10, 6, 12) != cells["M1ref"]:
        die(f"CELL PARITY on {t['name']}: realized_cell_gen(credit=12) != "
            f"mp.realized_cell_k")
    nnzs["M1ref"] = nnz10
    decomp["M1ref"] = decompose(cells["M1ref"], qE10, H10, SPEC["M1ref"], nb, n)
    rt["M1ref"] = mp.rt_extended(raw, t["name"], "M1", sym10, m6,
                                 mp.MECH_SPEC["M1"], q10, cum10, x0b, ds12,
                                 rb0, starts, sha_o, sha_r)

    # ---- R1: sym10, wide byte-renorm coder, same 12-bit table
    ds31_6 = d_seed_gen(m6, starts, 6, CREDIT_WIDE)
    x0w6 = (WIDE_L + ds31_6).astype(np.int64)
    rb1 = rans_sim_wide(q10[sym10].reshape(nb, W),
                        cum10[sym10].reshape(nb, W), x0w6, 12)
    cells["R1"] = realized_cell_gen(rb1, nb, n, tab10, 6, CREDIT_WIDE)
    nnzs["R1"] = nnz10
    decomp["R1"] = decompose(cells["R1"], qE10, H10, SPEC["R1"], nb, n)
    rt["R1"] = rt_wide(raw, t["name"], "R1", sym10, m6, SPEC["R1"], q10,
                       cum10, x0w6, ds31_6, rb1, starts, sha_o, sha_r)

    # ---- R2: sym11 (A=2048, 12-bit table), wide coder
    q11, cum11, cl11, tab11, nnz11 = build_table_gen(hist11, n, 2048, 12)
    qE11 = float((hist11 * cl11).sum())
    ds31_5 = d_seed_gen(m5, starts, 5, CREDIT_WIDE)
    x0w5 = (WIDE_L + ds31_5).astype(np.int64)
    rb2 = rans_sim_wide(q11[sym11].reshape(nb, W),
                        cum11[sym11].reshape(nb, W), x0w5, 12)
    cells["R2"] = realized_cell_gen(rb2, nb, n, tab11, 5, CREDIT_WIDE)
    nnzs["R2"] = nnz11
    decomp["R2"] = decompose(cells["R2"], qE11, H11, SPEC["R2"], nb, n)
    rt["R2"] = rt_wide(raw, t["name"], "R2", sym11, m5, SPEC["R2"], q11,
                       cum11, x0w5, ds31_5, rb2, starts, sha_o, sha_r)

    # ---- R3: sym10, wide coder, 14-bit table (finer quantization, charged)
    q14, cum14, cl14, tab14, nnz14 = build_table_gen(hist10, n, 1024, 14)
    qE14 = float((hist10 * cl14).sum())
    rb3 = rans_sim_wide(q14[sym10].reshape(nb, W),
                        cum14[sym10].reshape(nb, W), x0w6, 14)
    cells["R3"] = realized_cell_gen(rb3, nb, n, tab14, 6, CREDIT_WIDE)
    nnzs["R3"] = nnz14
    decomp["R3"] = decompose(cells["R3"], qE14, H10, SPEC["R3"], nb, n)
    rt["R3"] = rt_wide(raw, t["name"], "R3", sym10, m6, SPEC["R3"], q14,
                       cum14, x0w6, ds31_6, rb3, starts, sha_o, sha_r)

    # ---- R4: sym11 (A=2048), wide coder, 14-bit table (sym11 at the finer
    #      precision so the SYM11 verdict can be rendered under the best coder)
    q11f, cum11f, cl11f, tab11f, nnz11f = build_table_gen(hist11, n, 2048, 14)
    qE11f = float((hist11 * cl11f).sum())
    rb4 = rans_sim_wide(q11f[sym11].reshape(nb, W),
                        cum11f[sym11].reshape(nb, W), x0w5, 14)
    cells["R4"] = realized_cell_gen(rb4, nb, n, tab11f, 5, CREDIT_WIDE)
    nnzs["R4"] = nnz11f
    decomp["R4"] = decompose(cells["R4"], qE11f, H11, SPEC["R4"], nb, n)
    rt["R4"] = rt_wide(raw, t["name"], "R4", sym11, m5, SPEC["R4"], q11f,
                       cum11f, x0w5, ds31_5, rb4, starts, sha_o, sha_r)

    if sha_o.digest() != sha_r.digest():
        die(f"ROUND-TRIP ({t['name']}): SHA-256 mismatch over sampled spans")
    rt["sha256_ok"] = True

    deltas = {v: round(cells["M1ref"]["bpw"] - cells[v]["bpw"], 6)
              for v in R_VARIANTS}

    return {
        "name": t["name"], "layer": t["layer"], "expert": t["expert"],
        "proj": t["proj"], "n": int(n), "R": int(R), "C": int(Ccols),
        "nb": int(nb), "acct": ACCT_STAMP,
        "H_sym10": round(H10, 6), "H_sym11": round(H11, 6),
        "nnz": {k: int(v) for k, v in nnzs.items()},
        "cells": cells,
        "decomp": decomp,
        "deltas": deltas,
        "roundtrip": rt,
    }


# ------------------------------------------------------------------ summary ---
def summarize(tg: list[dict], jsonl: Path, summaryp: Path, synthetic: bool,
              full_sample: bool):
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

    def var_bits(r, v):
        c = r["cells"][v]
        return c["sym_bits"] + c["mant_bits"]

    bpw = {v: wsum(lambda r: var_bits(r, v)) / n_tot for v in VARIANTS}
    delta = {v: bpw["M1ref"] - bpw[v] for v in R_VARIANTS}
    dw = {v: {k: wsum(lambda r: r["decomp"][v][k] * r["n"]) / n_tot
              for k in DECOMP_KEYS} for v in VARIANTS}

    rt_ok = all(r["roundtrip"]["sha256_ok"] for r in recs)
    rt_blocks = {v: wsum(lambda r: r["roundtrip"][v]) for v in VARIANTS}

    # the recomputed reference must reproduce the standing sample number
    ref_note = None
    if not synthetic and full_sample:
        d_ref = abs(bpw["M1ref"] - M1_SAMPLE_REF_BPW)
        if d_ref > M1_SAMPLE_REF_TOL:
            die(f"M1 reference recomputed {bpw['M1ref']:.4f} != standing "
                f"{M1_SAMPLE_REF_BPW} (|d|={d_ref:.4f} > {M1_SAMPLE_REF_TOL})")
        ref_note = (f"recomputed M1 reference {bpw['M1ref']:.4f} reproduces "
                    f"the standing {M1_SAMPLE_REF_BPW} (|d|={d_ref:.6f})")

    best_r = min(R_VARIANTS, key=lambda v: bpw[v])
    best_delta = delta[best_r]
    delta_meets_gate = bool(best_delta >= GATE_BPW and rt_ok)
    # the gate can only FIRE on a full-sample, non-synthetic run (the M1
    # reference reproduction cross-check above is skipped otherwise)
    gate_valid = bool(full_sample and not synthetic)
    r_fires = bool(delta_meets_gate and gate_valid)
    # sym11 verdict (separate, recorded either way): best sym11 coder vs best
    # sym10 coder, with the matched-table-precision pairs alongside
    best_s10 = min(SYM10_VARIANTS, key=lambda v: bpw[v])
    best_s11 = min(SYM11_VARIANTS, key=lambda v: bpw[v])
    bit2_net = bpw[best_s10] - bpw[best_s11]  # >0: folding bit 2 pays
    bit2_pairs = {f"{SPEC[a]['m_log2']}bit_{b}_minus_{a}":
                  round(bpw[b] - bpw[a], 6) for a, b in SYM11_PAIRS}
    bit2_headroom = dw[best_s10]["bound_bpw"] - dw[best_s11]["bound_bpw"]
    proj_wm = M1_WHOLE_MODEL_BPW - EXPERT_FRAC * best_delta

    scope = ("synthetic smoke -- mechanics + round-trip proof only. "
             "M1ref-vs-R1/R3 deltas are transferable coder-mechanics signals "
             "(same tables, same symbols); R2/R4's A=2048 table amortization "
             "over 3,072-weight tensors is a synthetic artifact (vanishes on "
             "multi-million-weight real tensors), so the sym11 verdict here "
             "is NOT evidential"
             if synthetic else
             f"sampled experts on layers {layers}; M1 reference recomputed "
             f"on the same sample with the frozen bit-renorm coder"
             + ("" if full_sample else
                " [PARTIAL SAMPLE -- layer-restricted; M1 reference "
                "reproduction check skipped; gate cannot fire]"))
    verdict = (f"R-GATE {'FIRES' if r_fires else 'does NOT fire'}: best "
               f"variant {best_r} realizes {best_delta:+.4f} b/w vs the M1 "
               f"reference (gate >= {GATE_BPW}, all side costs charged; "
               f"round-trip {'proven' if rt_ok else 'FAILED'} on sampled "
               f"blocks of every tensor, all other blocks accounted via the "
               f"bit-identical emitted-bit simulation)"
               + (" [smoke only -- no evidential weight]" if synthetic else "")
               + ("" if full_sample else
                  " [partial sample -- delta indicative only, gate held]"))
    sym11_verdict = (
        f"SYM11 under the best coder ({best_s11} vs {best_s10}, both at "
        f"{SPEC[best_s11]['m_log2']}/{SPEC[best_s10]['m_log2']}-bit tables): "
        f"bit-2 is {'NET-POSITIVE' if bit2_net > 0 else 'net-negative'} "
        f"({best_s11} - {best_s10} = {-bit2_net:+.4f} b/w realized; "
        f"matched-precision pairs "
        f"{bit2_pairs}; entropy headroom bound(sym10)-bound(sym11) = "
        f"{bit2_headroom:+.4f}, T2 reference ~{SYM11_HEADROOM_REF})"
        + (" [smoke only -- table amortization artifact]" if synthetic else ""))

    mode = "SYNTHETIC (smoke only)" if synthetic else "REAL (sampled)"
    print(f"\n=== candidate 0015 -- coder mechanics: wider-state byte-renorm "
          f"rANS for the tile+M1 format [{mode}] ===")
    print(f"sample: {len(recs)} tensors, {n_tot:,} params, layers {layers}; "
          f"acct stamp {ACCT_STAMP}")
    print(f"wide coder: {WIDE_CODER_SPEC}")
    if ref_note:
        print(ref_note)
    print(f"round-trip: " + ", ".join(f"{v}={rt_blocks[v]}" for v in VARIANTS)
          + f" blocks; bits==accounted + SHA-256 exact: "
            f"{'PASS' if rt_ok else 'FAIL'}")

    print("\nrealized (b/w, numel-weighted; ALL side costs charged; delta > 0 "
          "= saves vs M1ref):")
    hdr = (f"{'variant':>8}{'bpw':>10}{'delta':>9}{'bound':>10}{'quant':>9}"
           f"{'flushN':>9}{'renorm':>9}{'pad':>8}{'tab':>8}{'misc':>8}"
           f"{'gap':>8}")
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANTS:
        c = dw[v]
        d = 0.0 if v == "M1ref" else delta[v]
        print(f"{v:>8}{bpw[v]:>10.4f}{d:>+9.4f}{c['bound_bpw']:>10.4f}"
              f"{c['quant_bpw']:>9.4f}{c['flush_net_bpw']:>9.4f}"
              f"{c['renorm_bpw']:>9.4f}{c['pad_bpw']:>8.4f}"
              f"{c['tab_bpw']:>8.4f}{c['misc_bpw']:>8.4f}"
              f"{bpw[v] - c['bound_bpw']:>8.4f}")
    print("  (bound = H(symA)+k_low, entropy level; quant = table "
          "quantization delta; flushN = (flush-credit)*nb/n; renorm = coded "
          "- qentropy - gross flush; components sum exactly to bpw)")

    print("\ncoder redundancy per coded symbol (renorm b/sym; 1 sym/weight "
          "in every variant):")
    for v in VARIANTS:
        cd = "bit-renorm" if SPEC[v]["coder"] == "bit" else "byte-renorm"
        print(f"{v:>8}  {dw[v]['renorm_bpw']:+.6f} b/sym  ({cd}, "
              f"A={SPEC[v]['A']}, {SPEC[v]['m_log2']}-bit table; history: "
              f"bit-renorm 0.051@512 / 0.063@1024 / 0.098@2048)")
    print("  (note: for the wide variants this component also carries the "
          "seed/flush state-boundary effect -- the 31-bit seed x0=L+d has "
          "E[log2 x0] < 32 while the flush is charged a full 32 bits -- so "
          "it is not a pure renorm-quantization number; for M1ref the 12-bit "
          "seed and 12-bit flush cancel almost exactly. Totals, deltas and "
          "the gate are unaffected: the decomposition sums exactly.)")

    print("\nper-layer (b/w, weighted):")
    hdr = (f"{'layer':>6}{'tensors':>9}"
           + "".join(f"{v:>10}" for v in VARIANTS) + f"{'best':>6}{'delta':>9}")
    print(hdr)
    print("-" * len(hdr))
    per_layer = {}
    for L in layers:
        rs = [r for r in recs if r["layer"] == L]
        nn = sum(r["n"] for r in rs)
        row = {"tensors": len(rs), "params": int(nn)}
        for v in VARIANTS:
            row[v] = round(sum(var_bits(r, v) for r in rs) / nn, 6)
        ds = {v: row["M1ref"] - row[v] for v in R_VARIANTS}
        row["best"] = max(ds, key=lambda v: ds[v])
        row["best_delta"] = round(ds[row["best"]], 6)
        per_layer[f"L{L}"] = row
        print(f"{L:>6}{row['tensors']:>9}"
              + "".join(f"{row[v]:>10.4f}" for v in VARIANTS)
              + f"{row['best']:>6}{row['best_delta']:>+9.4f}")
    print(f"{'ALL':>6}{len(recs):>9}"
          + "".join(f"{bpw[v]:>10.4f}" for v in VARIANTS)
          + f"{best_r:>6}{best_delta:>+9.4f}")

    print(f"\n{verdict}")
    print(sym11_verdict)
    print(f"projected whole-model at best variant: {proj_wm:.4f} b/w "
          f"(= {M1_WHOLE_MODEL_BPW} - {EXPERT_FRAC} x {best_delta:+.4f}; "
          f"sample-selected -- selection-optimistic)")
    print(f"scope: {scope}")

    summary = {
        "mode": "synthetic" if synthetic else "real", "scope": scope,
        "acct_stamp": ACCT_STAMP, "acct": ACCT,
        "targets": len(recs), "total_params": int(n_tot),
        "layers": [int(x) for x in layers],
        "m1_sample_ref_bpw": M1_SAMPLE_REF_BPW,
        "m1_ref_recomputed_bpw": round(bpw["M1ref"], 6),
        "m1_ref_reproduced": ref_note,
        "realized_bpw": {v: round(bpw[v], 6) for v in VARIANTS},
        "delta_vs_m1ref_bpw": {v: round(delta[v], 6) for v in R_VARIANTS},
        "decomposition_bpw": {v: {k: round(dw[v][k], 6) for k in DECOMP_KEYS}
                              for v in VARIANTS},
        "renorm_b_per_sym": {v: round(dw[v]["renorm_bpw"], 6)
                             for v in VARIANTS},
        "bit_renorm_redundancy_history_b_per_sym":
            {"A512": 0.051, "A1024": 0.063, "A2048": 0.098},
        "per_layer": per_layer,
        "roundtrip": {"blocks": {v: int(rt_blocks[v]) for v in VARIANTS},
                      "all_ok": bool(rt_ok)},
        "gate": {"gate_bpw": GATE_BPW, "best_variant": best_r,
                 "best_delta_bpw": round(best_delta, 6),
                 "delta_meets_gate": delta_meets_gate,
                 "full_sample": bool(full_sample),
                 "synthetic": bool(synthetic),
                 "valid": gate_valid,
                 "fires": r_fires, "evidential": gate_valid},
        "sym11": {"best_sym10": best_s10, "best_sym11": best_s11,
                  "best_sym11_minus_best_sym10_bpw": round(-bit2_net, 6),
                  "bit2_net_bpw": round(bit2_net, 6),
                  "net_positive": bool(bit2_net > 0),
                  "matched_precision_pairs_bpw": bit2_pairs,
                  "r2_minus_r1_bpw": round(bpw["R2"] - bpw["R1"], 6),
                  "r4_minus_r3_bpw": round(bpw["R4"] - bpw["R3"], 6),
                  "entropy_headroom_bpw": round(bit2_headroom, 6),
                  "t2_headroom_ref_bpw": SYM11_HEADROOM_REF,
                  "verdict": sym11_verdict},
        "projected_whole_model_bpw": round(proj_wm, 6),
        "verdict": verdict,
        "accounting_note": (
            "all cells use the frozen v2 mechanics exactly (W=128 bit-stride, "
            "DP T=4 tier budgets over measured emitted bits, P100 no-escape, "
            "per-block class flags, u32 rank anchors per 512 blocks, 96-bit "
            "class descriptors, 32-byte header, pad8 record align); table "
            "charged pad8(A + nnz*m_log2); low plane pad8(k_low*n - "
            "credit*nb); wide variants charge the 32-bit flush in every "
            "block's measured bits and credit exactly 31 seed-borne mantissa "
            "bits against the plane; M1ref is mp's M1 machinery verbatim "
            "(cell asserted identical); every variant round-trip proven per "
            "tensor on sampled blocks with the flush-borne plane fields "
            "destroyed before the rebuild and SHA-256-exact BF16 "
            "reconstruction over the sampled spans"),
    }
    summaryp.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written: {summaryp}")
    return summary


# --------------------------------------------------------------------- main ---
def run(a, snap: Path, jsonl: Path, summaryp: Path):
    tg = ep.sample_targets(snap, a.synthetic, a.layer)
    full_sample = a.layer is None
    if a.summary:
        return summarize(tg, jsonl, summaryp, a.synthetic, full_sample)

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
        print(f"\n[limit] {a.limit} tensors this invocation -- re-invoke to "
              f"resume.")
        sys.exit(0)

    print(f"\nall {len(done)}/{len(tg)} tensors done ({time.time() - t0:.0f}s)")
    summarize(tg, jsonl, summaryp, a.synthetic, full_sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run against the synthetic tiny snapshot (smoke)")
    ap.add_argument("--summary", action="store_true",
                    help="summary tables + JSON (requires all tensors done)")
    ap.add_argument("--layer", type=int, default=None,
                    help="restrict to one layer (default: layers "
                         f"{ep.LAYERS_REAL} real / all present synthetic)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max tensors this invocation (0 = no cap)")
    ap.add_argument("--budget-s", type=float, default=420.0,
                    help="soft wall-clock budget; exits cleanly when exceeded")
    a = ap.parse_args()

    snap = v1.SYN_SNAP if a.synthetic else v1.REAL_SNAP
    tag = (("_synthetic" if a.synthetic else "")
           + (f"_layer{a.layer}" if a.layer is not None else ""))
    ART.mkdir(parents=True, exist_ok=True)
    jsonl = ART / f"coder_mechanics_results{tag}.jsonl"
    summaryp = ART / f"coder_mechanics_summary{tag}.json"

    lockp = ART / f"coder_mechanics{tag}.lock"
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
