"""probe_mantissa_phase.py -- candidate 0015: cash the mantissa-phase bias
INSIDE the frozen v2 tile format, exact accounting, round-trip proven.

The emission peel (probe_emission_peel.py, RESULTS.md 2026-07-02) certified
that the transmitted mantissa plane of the frozen 0015-v2 cell is NOT random:
H(bit | position mod 7) shows a fixed-position per-phase bias (MSB p(1)~0.416
rising monotonically to ~0.500 by bit 6), a quantified ceiling of ~0.0287 b/w
(~0.020 from the MSB alone, h(0.416)=0.9797), present on 62-64/64 sampled
tensors across layers 1/13/27/40. The phase is fixed-position (mantissa bits
sit at known offsets of the fixed bit-stride layout), so the bias is
exploitable without giving up O(1) block addressing. This probe prices THREE
in-format mechanisms that realize it, all preserving the frozen contract
(fixed bit-stride slots via DP tier budgets; every block independently
decodable from an O(1)-computable address; mantissa bits may join the block's
sequential rANS decode -- within-block sequential decode is already the L3
contract):

  M1  extended symbol (sym10): fold the mantissa MSB into the coded symbol.
      sym10 = u >> 6 (sign + exp8 + mant MSB; 1024-entry table, charged
      pad8(1024 + nnz*12) -- the ~2x table cost); the remaining 6 mantissa
      bits stay verbatim at fixed width (6W - 12 bits per block; the first
      12 ride the flush exactly as in L3).
  M2  per-phase mantissa coding: ALL 7 mantissa bit-planes join the block's
      rANS stream as binary symbols with 7 static per-tensor probabilities,
      each transmitted as a 12-bit quantized count (q1 = clip(round(p_i *
      4096), 1, 4095); side cost pad8(7*12) = 88 bits/tensor, charged).
      Decode order per block: the W syms (order-0 table, unchanged), then
      the 7W-12 mantissa bits in plane order (the first 12 still ride the
      flush = the L3 payload, unchanged). No verbatim mantissa plane
      remains: the whole block is one sequential decode (allowed -- the L3
      cell already finalizes weights 0-1 only after the full-block decode).
  M3  two extended bits (sym11 = u >> 5; 2048-entry table, k=5 verbatim
      plane): the bracket that shows where table cost overtakes the (tiny)
      phase-1 gain.

Also reported: the REVISED storage floor per tensor, H(sym) + sum_i h(p_i),
vs the old H(sym) + 7 -- plus the entropy-level bound of each mechanism
(M1: H(sym10) + 6, M3: H(sym11) + 5; these also capture sym-conditioned MSB
structure that the independent-phase model cannot).

Accounting is the frozen v2 cell EXACTLY (probe_emission_peel.realized_cell,
asserted bit-identical at k=7 per tensor): W=128 bit-stride blocks, DP T=4
tier budgets over the MEASURED emitted bits of the named coder, P100 (no
escapes), per-block class flags + u32 rank anchors per 512 blocks + 96-bit
class descriptors + 32-byte header + pad8 record align + the mechanism's
table(s) + the mechanism's verbatim plane pad8(k*n - 12*nb), k = 7/6/5
(0 for M2: no plane). Block sizes are exact vectorized coder simulation
(bit-identical arithmetic; the generalized quantizer is asserted equal to
v1's on the 512-bin histogram of every tensor). Every mechanism is
round-trip proven on sampled block ranges of EVERY tensor: pure-Python
encode -> serialized bytes -> decode, asserting emitted bits == accounted
bits, symbols exact, L3 payload recovered from the decoder's final state,
and SHA-256-exact BF16 reconstruction (sym + full mantissa path) over the
sampled spans -- with the flush-borne fields of the low plane destroyed
before the rebuild, so the borrowed bits provably flow from the DECODED
payload, not from the original data.

Pre-registered gate: FIRES if the best mechanism realizes >= 0.02 b/w
improvement over the frozen cell recomputed on the same sample, ALL side
costs charged, round-trip proven. Side question (recorded either way): is
the bias per-tensor stable -- per-tensor fitted probs (transmitted, 88
bits/tensor) vs pooled global constants, compared at entropy level from the
stored per-tensor phase counts.

Field split (stz convention): u16 LE, sym = u >> 7 (9 bits), mant = u & 0x7F
(7 bits); phase i = MSB-first bit index in the mantissa (phase 0 = MSB).
Sampling identical to the emission peel: 8 experts x {up,down} per layer,
layers 1/13/27/40 real (all layers synthetic). Frozen whole-model reference
10.7311 b/w (fully measured); projection = 10.7311 - 0.93 * best delta.

Usage:
  uv run python probe_mantissa_phase.py --synthetic     # smoke (fake snapshot)
  uv run python probe_mantissa_phase.py                 # real run (resumable)
  uv run python probe_mantissa_phase.py --layer 13      # one layer only
  uv run python probe_mantissa_phase.py --summary       # tables + JSON
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import probe_block_codes as v1        # noqa: E402  -- verified infrastructure
import probe_block_codes_v2 as v2     # noqa: E402  -- coder, DP, tables
import probe_emission_peel as ep      # noqa: E402  -- sampling, frozen cell

M, M_LOG2, FLUSH_BITS = v1.M, v1.M_LOG2, v1.FLUSH_BITS
pad8, ceil_div, BITLEN, die = v1.pad8, v1.ceil_div, v1.BITLEN, v1.die
ART = v1.ART
BORROW_BITS = v2.BORROW_BITS
RANK_GROUP, RANK_BITS = v2.RANK_GROUP, v2.RANK_BITS
HEAD_BITS, CLASS_DIR_BITS = v2.HEAD_BITS, v2.CLASS_DIR_BITS

# ---- frozen v2 cell (the fixed reference; no re-selection here)
W = 128                       # frozen block size (bit-stride, fusible)
T_MAX = 4                     # frozen DP tier count
# P100 => top budget = max measured size => no escape blocks (asserted)

# ---- mechanisms (pre-registered)
GATE_BPW = 0.02               # best mech must beat frozen by >= this (b/w)
PHASE_TAB_BITS = pad8(7 * 12) # M2: 7 transmitted 12-bit phase probs = 88 bits
MECHS = ("frozen", "M1", "M2", "M3")
MECH_SPEC = {
    "frozen": {"A": 512,  "shift": 7, "k_low": 7},   # baseline (v2 cell)
    "M1":     {"A": 1024, "shift": 6, "k_low": 6},   # sym10 = sign+exp8+MSB
    "M2":     {"A": 512,  "shift": 7, "k_low": 0},   # 7 binary phase lanes
    "M3":     {"A": 2048, "shift": 5, "k_low": 5},   # sym11 = +2 mant bits
}
FROZEN_WHOLE_MODEL_BPW = 10.7311   # fully measured whole-model (frozen fmt)
EXPERT_FRAC = 0.93
PEEL_CEILING_BPW = 0.0287          # emission-peel mant-plane ceiling (context)

ACCT = {"schema": 1, "probe": "mantissa_phase", "W": W, "T_MAX": T_MAX,
        "P": 100, "L1": 1, "L3": 1, "L4": 0,
        "M_LOG2": M_LOG2, "FLUSH_BITS": FLUSH_BITS, "BORROW_BITS": BORROW_BITS,
        "RANK_GROUP": RANK_GROUP, "RANK_BITS": RANK_BITS,
        "HEAD_BITS": HEAD_BITS, "CLASS_DIR_BITS": CLASS_DIR_BITS,
        "GATE_BPW": GATE_BPW, "PHASE_TAB_BITS": PHASE_TAB_BITS,
        "MECHS": {k: dict(v) for k, v in MECH_SPEC.items()},
        "TABLE_BITS_RULE": "pad8(A + nnz*12) per table; M2 adds pad8(7*12)",
        "MANT_PLANE_RULE": "pad8(k_low*n - 12*nb) if k_low>0 else 0",
        "LAYERS_REAL": list(ep.LAYERS_REAL),
        "EXPERTS_PER_PROJ": ep.EXPERTS_PER_PROJ,
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


def hbin(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def h0_bits(counts: np.ndarray) -> float:
    n = counts.sum()
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


# ----------------------------------------------- generalized quantizer/table ---
def quantize_hist_a(hist: np.ndarray, n: int, A: int) -> np.ndarray:
    """v1.quantize_hist generalized to alphabet size A (v1 hardcodes 512).
    Identical algorithm: deterministic largest-remainder to M=4096, every
    present symbol >= 1. Asserted equal to v1's output at A=512 on every
    tensor's sym histogram (the parity gate in analyze_tensor)."""
    nz = np.flatnonzero(hist)
    assert 1 <= nz.size <= M, "present symbols must fit the table"
    tgt = hist[nz].astype(np.float64) * (M / n)
    q = np.maximum(np.floor(tgt), 1.0).astype(np.int64)
    guard = 0
    while True:
        d = M - int(q.sum())
        if d == 0:
            break
        guard += 1
        if guard > 2 * M:
            die("ANS quantization did not converge (progress guard)")
        if d > 0:
            rem = tgt - q
            order = np.lexsort((nz, -rem))
            q[order[:min(d, nz.size)]] += 1
        else:
            surplus = -d
            elig = np.flatnonzero(q > 1)
            if elig.size == 0:
                die("ANS quantization cannot reach the 4096 total")
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


def build_table_a(hist: np.ndarray, n: int, A: int):
    """Quantized 12-bit table over an A-symbol alphabet. Table cost =
    pad8(A + nnz*12): A-bit presence bitmap + 12 bits per present symbol,
    the same rule v2.build_table charges at A=512."""
    q = quantize_hist_a(hist, n, A)
    nnz = int((hist > 0).sum())
    present = np.flatnonzero(q)
    cum = np.zeros(A, np.int64)
    cum[present] = np.cumsum(q[present]) - q[present]
    return q, cum, pad8(A + nnz * 12), nnz


# --------------------------------------------------------- L3 seed helpers ---
def d_seed_k(mlow: np.ndarray, starts: np.ndarray, k: int) -> np.ndarray:
    """First BORROW_BITS(=12) bits of the k-bit MSB-first weight-major low
    plane of each block (the L3 flush payload). k=7 reproduces the frozen
    d = (mant[s]<<5)|(mant[s+1]>>2) exactly."""
    acc = np.zeros(starts.size, np.int64)
    got, w = 0, 0
    while got < BORROW_BITS:
        take = min(k, BORROW_BITS - got)
        acc = (acc << take) | (mlow[starts + w] >> (k - take))
        got += take
        w += 1
    return acc


def apply_seed_k(mrec: np.ndarray, d: int, k: int):
    """Rebuild the borrowed leading fields of a block's k-bit low plane from
    the recovered 12-bit payload (inverse of d_seed_k, one block)."""
    got, w = 0, 0
    while got < BORROW_BITS:
        take = min(k, BORROW_BITS - got)
        chunk = (d >> (BORROW_BITS - got - take)) & ((1 << take) - 1)
        keep = (1 << (k - take)) - 1
        mrec[w] = (chunk << (k - take)) | (int(mrec[w]) & keep)
        got += take
        w += 1


def zero_borrowed(mrec: np.ndarray, k: int):
    """Destroy the flush-borne fields of a block's k-bit low plane (exactly
    the bits apply_seed_k writes), keeping only the verbatim-plane bits.
    Run before apply_seed_k in the round-trip so the byte-exact check
    provably reconstructs the borrowed bits from the DECODED payload --
    seeding mrec from the original plane alone would let an apply_seed_k
    no-op pass silently (skeptic defect, 2026-07-02)."""
    got, w = 0, 0
    while got < BORROW_BITS:
        take = min(k, BORROW_BITS - got)
        mrec[w] = int(mrec[w]) & ((1 << (k - take)) - 1)
        got += take
        w += 1


# ----------------------------------------------------------- M2 coder paths ---
def mant_bits_matrix(mant: np.ndarray, nb: int) -> np.ndarray:
    """(nb, 7W-12) uint8: the block's mantissa bits in L3 plane order
    (7 bits/weight MSB-first, weight-major, minus the 12 borrowed bits).
    Coded position j has phase (j + BORROW_BITS) % 7."""
    b = ((mant[:, None] >> np.arange(6, -1, -1)) & 1).astype(np.uint8)
    return np.ascontiguousarray(b.reshape(nb, 7 * W)[:, BORROW_BITS:])


def rans_sim_m2(qm: np.ndarray, cm: np.ndarray, mb2: np.ndarray,
                q1ph: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Exact vectorized emitted-bit simulation of the M2 block coder: decode
    order = W syms then 7W-12 binary mantissa bits, so the encoder consumes
    the mantissa bits in reverse first, then the syms in reverse.
    Bit-identical arithmetic to v2.rans_sim_blocks / the reference encoder."""
    nbk = qm.shape[0]
    nmb = mb2.shape[1]
    x = x0.astype(np.int64).copy()
    bits = np.full(nbk, FLUSH_BITS, np.int64)
    q0ph = M - q1ph
    for j in range(nmb - 1, -1, -1):
        ph = (j + BORROW_BITS) % 7
        b = mb2[:, j].astype(np.int64)
        qq = np.where(b == 1, q1ph[ph], q0ph[ph])
        cc = np.where(b == 1, q0ph[ph], 0)
        thr1 = (qq << 1) - 1
        k = BITLEN[x] - BITLEN[thr1]
        np.maximum(k, 0, out=k)
        k += (x >> k) > thr1
        bits += k
        x = M + cc + ((x >> k) - qq)
    for j in range(W - 1, -1, -1):
        qq = qm[:, j]
        cc = cm[:, j]
        thr1 = (qq << 1) - 1
        k = BITLEN[x] - BITLEN[thr1]
        np.maximum(k, 0, out=k)
        k += (x >> k) > thr1
        bits += k
        x = M + cc + ((x >> k) - qq)
    return bits


def m2_enc_block(seq: list, mb: list, ql: list, cl: list,
                 q1ph: np.ndarray, x0: int):
    """Reference M2 encoder (pure Python): consumes mantissa bits in reverse,
    then syms in reverse, from the L3-seeded initial state."""
    assert M <= x0 < 2 * M
    x = x0
    bits = []
    ap = bits.append
    for j in range(len(mb) - 1, -1, -1):
        ph = (j + BORROW_BITS) % 7
        q1 = int(q1ph[ph])
        q0b = M - q1
        if mb[j]:
            qq, cc = q1, q0b
        else:
            qq, cc = q0b, 0
        t = qq << 1
        while x >= t:
            ap(x & 1)
            x >>= 1
        x = M + cc + (x - qq)
    for s in reversed(seq):
        qq = ql[s]
        t = qq << 1
        while x >= t:
            ap(x & 1)
            x >>= 1
        x = M + cl[s] + (x - qq)
    return x - M, bits


def m2_dec_block(data: bytes, nbits: int, ql: list, cl: list, s2s: list,
                 q1ph: np.ndarray, nmb: int):
    """Reference M2 decoder: W syms via the order-0 table, then nmb binary
    mantissa bits via the per-phase tables. Returns (syms, mant_bits,
    final_state) -- final_state - M is the L3 payload -- or None."""
    if nbits < FLUSH_BITS:
        return None
    f = 0
    for i in range(FLUSH_BITS):
        f = (f << 1) | ((data[i >> 3] >> (7 - (i & 7))) & 1)
    x = M + f
    pos = FLUSH_BITS
    syms = []
    for _ in range(W):
        slot = x - M
        s = s2s[slot]
        x = ql[s] + slot - cl[s]
        while x < M:
            if pos >= nbits:
                return None
            x = (x << 1) | ((data[pos >> 3] >> (7 - (pos & 7))) & 1)
            pos += 1
        syms.append(s)
    mb = []
    for j in range(nmb):
        ph = (j + BORROW_BITS) % 7
        q1 = int(q1ph[ph])
        q0b = M - q1
        slot = x - M
        b = 1 if slot >= q0b else 0
        if b:
            qq, cc = q1, q0b
        else:
            qq, cc = q0b, 0
        x = qq + slot - cc
        while x < M:
            if pos >= nbits:
                return None
            x = (x << 1) | ((data[pos >> 3] >> (7 - (pos & 7))) & 1)
            pos += 1
        mb.append(b)
    if pos != nbits or not (M <= x < 2 * M):
        return None
    return syms, mb, x


# --------------------------------------------------------- realized cell (k) ---
def realized_cell_k(rb: np.ndarray, nb: int, n: int, tab_bits: int,
                    k_low: int) -> dict:
    """ep.realized_cell generalized to a k-bit verbatim low plane
    (mant plane = pad8(k*n - 12*nb) bits; k=0 -> no plane, everything but
    the 12 flush-borne bits is inside the coded block). k=7 is asserted
    bit-identical to ep.realized_cell per tensor."""
    kept, budgets, slots, counts = v2.tier_dp(rb, True, T_MAX)[T_MAX]
    assert int(rb.max()) <= budgets[-1]          # P100: nothing escapes
    classes = len(budgets)
    flagb = int(math.ceil(math.log2(classes))) if classes > 1 else 0
    flag = nb * flagb
    rank = classes * RANK_BITS * ceil_div(nb, RANK_GROUP) if classes > 1 else 0
    cdir = classes * CLASS_DIR_BITS if classes > 1 else 0
    sym_raw = kept + flag + rank + cdir + tab_bits + HEAD_BITS
    sym_total = pad8(sym_raw)
    mant_bits = pad8(k_low * n - BORROW_BITS * nb) if k_low > 0 else 0
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


# --------------------------------------------------------------- per-tensor ---
def sample_ids(rb: np.ndarray, nb: int) -> list[int]:
    return sorted({0, nb - 1, int(np.argmin(rb)), int(np.argmax(rb))})


def rt_extended(raw: bytes, name: str, mech: str, symk: np.ndarray,
                mlow: np.ndarray, spec: dict, q: np.ndarray, cum: np.ndarray,
                x0v: np.ndarray, dsv: np.ndarray, rb: np.ndarray,
                starts: np.ndarray, sha_o, sha_r) -> int:
    """Round-trip gate for frozen/M1/M3 (extended-symbol family): encode ->
    serialize -> decode sampled blocks; verify bits, symbols, payload, and
    byte-exact BF16 reconstruction via the k-bit verbatim plane + payload."""
    ql, cl = q.tolist(), cum.tolist()
    pres = np.flatnonzero(q)
    s2s = np.repeat(pres, q[pres]).tolist()
    k_low, shift = spec["k_low"], spec["shift"]
    nb = starts.size
    done = 0
    for i in sample_ids(rb, nb):
        s0 = int(starts[i])
        seq = symk[s0:s0 + W].tolist()
        x0i = int(x0v[i])
        ctx = f"{name} {mech} block {i}"
        fl, bits = v2.rans_enc_block(seq, ql, cl, x0i)
        data, nbits = v1.pack_block(fl, bits)
        if nbits != int(rb[i]):
            die(f"ROUND-TRIP ({ctx}): emitted {nbits} != accounted {int(rb[i])}")
        dec = v2.rans_dec_block(data, nbits, W, ql, cl, s2s)
        if dec is None or dec[0] != seq:
            die(f"ROUND-TRIP ({ctx}): decode failed / symbols differ")
        if dec[1] != x0i:
            die(f"ROUND-TRIP ({ctx}): final state {dec[1]} != seed {x0i}")
        d_rec = dec[1] - M
        if d_rec != int(dsv[i]):
            die(f"ROUND-TRIP ({ctx}): payload {d_rec} != {int(dsv[i])}")
        mrec = mlow[s0:s0 + W].copy()
        zero_borrowed(mrec, k_low)      # borrowed bits must flow from payload
        apply_seed_k(mrec, d_rec, k_low)
        rec = ((np.array(dec[0], np.int64) << shift) | mrec).astype("<u2")
        orig = raw[2 * s0:2 * (s0 + W)]
        if rec.tobytes() != orig:
            die(f"ROUND-TRIP ({ctx}): reconstructed bytes != original")
        sha_o.update(orig)
        sha_r.update(rec.tobytes())
        done += 1
    return done


def rt_m2(raw: bytes, name: str, sym: np.ndarray, mb2: np.ndarray,
          q0: np.ndarray, cum0: np.ndarray, q1ph: np.ndarray,
          x0v: np.ndarray, dsv: np.ndarray, rb: np.ndarray,
          starts: np.ndarray, sha_o, sha_r) -> int:
    """Round-trip gate for M2: syms + coded mantissa bits + flush payload
    must reconstruct the block's BF16 bytes exactly."""
    ql, cl = q0.tolist(), cum0.tolist()
    pres = np.flatnonzero(q0)
    s2s = np.repeat(pres, q0[pres]).tolist()
    nmb = mb2.shape[1]
    nb = starts.size
    pw = (1 << np.arange(6, -1, -1)).astype(np.int64)
    done = 0
    for i in sample_ids(rb, nb):
        s0 = int(starts[i])
        seq = sym[s0:s0 + W].tolist()
        mb = mb2[i].tolist()
        x0i = int(x0v[i])
        ctx = f"{name} M2 block {i}"
        fl, bits = m2_enc_block(seq, mb, ql, cl, q1ph, x0i)
        data, nbits = v1.pack_block(fl, bits)
        if nbits != int(rb[i]):
            die(f"ROUND-TRIP ({ctx}): emitted {nbits} != accounted {int(rb[i])}")
        dec = m2_dec_block(data, nbits, ql, cl, s2s, q1ph, nmb)
        if dec is None or dec[0] != seq or dec[1] != mb:
            die(f"ROUND-TRIP ({ctx}): decode failed / symbols or mant bits differ")
        if dec[2] != x0i:
            die(f"ROUND-TRIP ({ctx}): final state {dec[2]} != seed {x0i}")
        d_rec = dec[2] - M
        if d_rec != int(dsv[i]):
            die(f"ROUND-TRIP ({ctx}): payload {d_rec} != {int(dsv[i])}")
        full = np.empty(7 * W, np.int64)
        for b in range(BORROW_BITS):
            full[b] = (d_rec >> (BORROW_BITS - 1 - b)) & 1
        full[BORROW_BITS:] = dec[1]
        mant_rec = (full.reshape(W, 7) * pw).sum(1)
        rec = ((np.array(dec[0], np.int64) << 7) | mant_rec).astype("<u2")
        orig = raw[2 * s0:2 * (s0 + W)]
        if rec.tobytes() != orig:
            die(f"ROUND-TRIP ({ctx}): reconstructed bytes != original")
        sha_o.update(orig)
        sha_r.update(rec.tobytes())
        done += 1
    return done


def analyze_tensor(raw: bytes, t: dict) -> dict:
    u = np.frombuffer(raw, "<u2")
    n = u.size
    R, Ccols = t["shape"]
    assert n == R * Ccols, (t["name"], n, R, Ccols)
    if n % W:
        die(f"n={n} not divisible by W={W} on {t['name']}")
    nb = n // W
    starts = np.arange(nb, dtype=np.int64) * W

    sym9 = (u >> 7).astype(np.int64)
    mant = (u & 0x7F).astype(np.int64)
    sym10 = (u >> 6).astype(np.int64)
    m6 = (u & 0x3F).astype(np.int64)
    sym11 = (u >> 5).astype(np.int64)
    m5 = (u & 0x1F).astype(np.int64)

    hist9 = np.bincount(sym9, minlength=512).astype(np.int64)
    hist10 = np.bincount(sym10, minlength=1024).astype(np.int64)
    hist11 = np.bincount(sym11, minlength=2048).astype(np.int64)
    H9, H10, H11 = h0_bits(hist9), h0_bits(hist10), h0_bits(hist11)

    # per-phase bias (fit on the tensor; the transmitted M2 model)
    ones = [int(((mant >> (6 - i)) & 1).sum()) for i in range(7)]
    p1 = [o / n for o in ones]
    q1ph = np.clip(np.round(np.array(p1) * M), 1, M - 1).astype(np.int64)

    floors = {
        "floor7": round(H9 + 7.0, 6),                       # old floor
        "floor_rev": round(H9 + sum(hbin(p) for p in p1), 6),  # revised floor
        "bound_m1": round(H10 + 6.0, 6),
        "bound_m3": round(H11 + 5.0, 6),
    }

    # generalized-quantizer parity gate vs v1 (A=512, the frozen table)
    if not np.array_equal(quantize_hist_a(hist9, n, 512),
                          v1.quantize_hist(hist9, n)):
        die(f"QUANTIZER PARITY on {t['name']}: generalized != v1 at A=512")

    sha_o, sha_r = hashlib.sha256(), hashlib.sha256()
    rt = {}
    cells = {}

    # ---- frozen (k=7) -- the baseline, asserted equal to the peel's cell
    q0, cum0, _clq0, tab0_bits, nnz9 = v2.build_table(hist9, n)
    ds7 = d_seed_k(mant, starts, 7)
    x0_7 = (M + ds7).astype(np.int64)
    rb7 = v2.rans_sim_blocks(q0[sym9].reshape(nb, W),
                             cum0[sym9].reshape(nb, W), x0_7)
    cells["frozen"] = realized_cell_k(rb7, nb, n, tab0_bits, 7)
    epc = ep.realized_cell(rb7, nb, n, tab0_bits)
    epc.pop("_flags")
    if epc != cells["frozen"]:
        die(f"FROZEN PARITY on {t['name']}: realized_cell_k(7) != "
            f"ep.realized_cell")
    rt["frozen"] = rt_extended(raw, t["name"], "frozen", sym9, mant,
                               MECH_SPEC["frozen"], q0, cum0, x0_7, ds7, rb7,
                               starts, sha_o, sha_r)

    # ---- M1 (sym10, k=6)
    q10, cum10, tab10_bits, nnz10 = build_table_a(hist10, n, 1024)
    ds6 = d_seed_k(m6, starts, 6)
    x0_6 = (M + ds6).astype(np.int64)
    rb10 = v2.rans_sim_blocks(q10[sym10].reshape(nb, W),
                              cum10[sym10].reshape(nb, W), x0_6)
    cells["M1"] = realized_cell_k(rb10, nb, n, tab10_bits, 6)
    rt["M1"] = rt_extended(raw, t["name"], "M1", sym10, m6, MECH_SPEC["M1"],
                           q10, cum10, x0_6, ds6, rb10, starts, sha_o, sha_r)

    # ---- M3 (sym11, k=5)
    q11, cum11, tab11_bits, nnz11 = build_table_a(hist11, n, 2048)
    ds5 = d_seed_k(m5, starts, 5)
    x0_5 = (M + ds5).astype(np.int64)
    rb11 = v2.rans_sim_blocks(q11[sym11].reshape(nb, W),
                              cum11[sym11].reshape(nb, W), x0_5)
    cells["M3"] = realized_cell_k(rb11, nb, n, tab11_bits, 5)
    rt["M3"] = rt_extended(raw, t["name"], "M3", sym11, m5, MECH_SPEC["M3"],
                           q11, cum11, x0_5, ds5, rb11, starts, sha_o, sha_r)

    # ---- M2 (order-0 sym table + 7 binary phase lanes, k=0)
    mb2 = mant_bits_matrix(mant, nb)
    rb2 = rans_sim_m2(q0[sym9].reshape(nb, W), cum0[sym9].reshape(nb, W),
                      mb2, q1ph, x0_7)
    cells["M2"] = realized_cell_k(rb2, nb, n, tab0_bits + PHASE_TAB_BITS, 0)
    rt["M2"] = rt_m2(raw, t["name"], sym9, mb2, q0, cum0, q1ph, x0_7, ds7,
                     rb2, starts, sha_o, sha_r)

    if sha_o.digest() != sha_r.digest():
        die(f"ROUND-TRIP ({t['name']}): SHA-256 mismatch over sampled spans")
    rt["sha256_ok"] = True

    deltas = {m: round(cells["frozen"]["bpw"] - cells[m]["bpw"], 6)
              for m in MECHS if m != "frozen"}

    return {
        "name": t["name"], "layer": t["layer"], "expert": t["expert"],
        "proj": t["proj"], "n": int(n), "R": int(R), "C": int(Ccols),
        "nb": int(nb), "acct": ACCT_STAMP,
        "H_sym": round(H9, 6), "H_sym10": round(H10, 6),
        "H_sym11": round(H11, 6),
        "nnz": {"sym9": nnz9, "sym10": nnz10, "sym11": nnz11},
        "phase": {"p1": [round(p, 6) for p in p1], "ones": ones,
                  "q1": q1ph.tolist()},
        "floors": floors,
        "cells": cells,
        "deltas": deltas,
        "roundtrip": rt,
    }


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

    def mech_bits(r, m):
        c = r["cells"][m]
        return c["sym_bits"] + c["mant_bits"]

    bpw = {m: wsum(lambda r: mech_bits(r, m)) / n_tot for m in MECHS}
    delta = {m: bpw["frozen"] - bpw[m] for m in MECHS if m != "frozen"}
    floor7_w = wsum(lambda r: r["floors"]["floor7"] * r["n"]) / n_tot
    floorrev_w = wsum(lambda r: r["floors"]["floor_rev"] * r["n"]) / n_tot
    bounds_w = {"frozen": floor7_w,
                "M1": wsum(lambda r: r["floors"]["bound_m1"] * r["n"]) / n_tot,
                "M2": floorrev_w,
                "M3": wsum(lambda r: r["floors"]["bound_m3"] * r["n"]) / n_tot}

    comp_keys = ("coded_bits", "pad_bits", "tab_bits", "mant_bits")
    comps = {m: {k: wsum(lambda r: r["cells"][m][k]) / n_tot
                 for k in comp_keys} for m in MECHS}
    for m in MECHS:
        comps[m]["misc_bits"] = wsum(
            lambda r: (r["cells"][m]["flag_bits"] + r["cells"][m]["rank_bits"]
                       + r["cells"][m]["cdir_bits"]
                       + r["cells"][m]["align_bits"] + HEAD_BITS)) / n_tot

    rt_ok = all(r["roundtrip"]["sha256_ok"] for r in recs)
    rt_blocks = {m: wsum(lambda r: r["roundtrip"][m]) for m in MECHS}

    # per-mechanism coder redundancy: measured emitted coded bits minus the
    # entropy-level information they carry, per coded symbol. The flush
    # emits FLUSH_BITS(=12) and carries the 12-bit L3 payload, so it is
    # information-neutral (payload counted at face value, 12*nb). Includes
    # 12-bit table quantization + bit-renorm/state-geometry losses -- this
    # is the number that attributes M2's loss to the frozen bit-by-bit
    # renorm coder (large per-symbol redundancy on near-fair binary
    # symbols) rather than to the phase bias itself.
    HKEY = {"frozen": "H_sym", "M1": "H_sym10", "M3": "H_sym11"}
    ph_cnt = np.full(7, float(W - 2))     # coded bits per phase per block:
    ph_cnt[5:] += 1.0                     # j+12 mod 7 -> phases 5,6 get W-1
    coder_red = {}
    for m in MECHS:
        emitted = wsum(lambda r: r["cells"][m]["coded_bits"])
        pay = wsum(lambda r: BORROW_BITS * r["nb"])
        if m == "M2":
            info = wsum(lambda r: r["n"] * r["H_sym"]
                        + r["nb"] * float((ph_cnt * np.array(
                            [hbin(p) for p in r["phase"]["p1"]])).sum()))
            nsym = wsum(lambda r: r["n"] + (7 * W - BORROW_BITS) * r["nb"])
        else:
            info = wsum(lambda r: r["n"] * r[HKEY[m]])
            nsym = n_tot
        coder_red[m] = {
            "coded_bpw": round(emitted / n_tot, 6),
            "info_bpw": round((info + pay) / n_tot, 6),
            "redundancy_bits_per_coded_symbol":
                round((emitted - pay - info) / nsym, 6),
            "coded_symbols_per_weight": round(nsym / n_tot, 6),
        }

    # per-tensor stability of the phase bias (entropy level, from stored
    # counts): per-tensor fitted probs vs pooled global constants
    ones_pool = np.zeros(7, np.float64)
    for r in recs:
        ones_pool += np.array(r["phase"]["ones"], np.float64)
    g = np.clip(ones_pool / n_tot, 1e-12, 1 - 1e-12)
    self_bits = wsum(lambda r: r["n"] * sum(hbin(p) for p in r["phase"]["p1"]))
    cross_bits = 0.0
    for r in recs:
        o = np.array(r["phase"]["ones"], np.float64)
        cross_bits += float((o * -np.log2(g)
                             + (r["n"] - o) * -np.log2(1 - g)).sum())
    stab_delta_bpw = (cross_bits - self_bits) / n_tot
    side_cost_bpw = wsum(lambda r: PHASE_TAB_BITS) / n_tot
    stab = {
        "global_p1": [round(float(x), 6) for x in ones_pool / n_tot],
        "per_tensor_entropy_bpw": round(self_bits / n_tot, 6),
        "global_crossentropy_bpw": round(cross_bits / n_tot, 6),
        "delta_global_minus_pertensor_bpw": round(stab_delta_bpw, 6),
        "per_tensor_probs_side_cost_bpw": round(side_cost_bpw, 6),
        "verdict": ("per-tensor fit pays for itself"
                    if stab_delta_bpw > side_cost_bpw else
                    "bias is per-tensor stable: global constants suffice "
                    "(per-tensor fit gains less than its 88-bit side cost)"),
        "note": ("entropy-level comparison from stored per-tensor phase "
                 "counts; the realized M2 cells use the per-tensor probs "
                 "with the 88-bit side cost charged either way"),
    }

    best_m = max(delta, key=lambda m: delta[m])
    best_delta = delta[best_m]
    fires = bool(best_delta >= GATE_BPW and rt_ok)
    proj_wm = FROZEN_WHOLE_MODEL_BPW - EXPERT_FRAC * best_delta
    scope = ("synthetic smoke -- evidential for coder behavior: synthetic "
             "mantissas carry the SAME log-uniform phase bias as real "
             "weights (pooled MSB p(1) ~ log2(4/3) ~ 0.4159), so an M2 "
             "loss here is a transferable coder-redundancy signal; only "
             "the M1/M3 table-amortization losses are synthetic artifacts "
             "(pad8(1024/2048 + nnz*12) over 3,072-weight tensors vanishes "
             "on multi-million-weight real tensors)"
             if synthetic else
             f"sampled experts on layers {layers}; frozen reference "
             f"recomputed on the same sample")
    verdict = (f"MANTISSA-PHASE GATE {'FIRES' if fires else 'does NOT fire'}: "
               f"best mechanism {best_m} realizes {best_delta:+.4f} b/w vs "
               f"frozen (gate >= {GATE_BPW}, all side costs charged, "
               f"round-trip {'proven' if rt_ok else 'FAILED'})")

    mode = "SYNTHETIC (smoke only)" if synthetic else "REAL (sampled)"
    print(f"\n=== candidate 0015 -- mantissa phase: realize the peel ceiling "
          f"inside the frozen format [{mode}] ===")
    print(f"sample: {len(recs)} tensors, {n_tot:,} params, layers {layers}; "
          f"acct stamp {ACCT_STAMP}")
    print(f"floors: H(sym)+7 = {floor7_w:.4f} | revised H(sym)+sum h(p_i) = "
          f"{floorrev_w:.4f} (headroom {floor7_w - floorrev_w:.4f} b/w; peel "
          f"ceiling was ~{PEEL_CEILING_BPW})")
    print(f"frozen recomputed on sample: {bpw['frozen']:.4f} b/w "
          f"(whole-model ref {FROZEN_WHOLE_MODEL_BPW})")
    print(f"round-trip: " + ", ".join(f"{m}={rt_blocks[m]}" for m in MECHS)
          + f" blocks; bits==accounted + SHA-256 exact: "
            f"{'PASS' if rt_ok else 'FAIL'}")

    print("\nrealized (b/w, weighted; ALL side costs charged; delta > 0 = "
          "saves vs frozen):")
    hdr = (f"{'mech':>8}{'bpw':>10}{'delta':>9}{'coded':>9}{'pad':>8}"
           f"{'tab':>8}{'mant':>8}{'misc':>8}{'bound':>10}{'gap':>8}")
    print(hdr)
    print("-" * len(hdr))
    for m in MECHS:
        c = comps[m]
        d = 0.0 if m == "frozen" else delta[m]
        print(f"{m:>8}{bpw[m]:>10.4f}{d:>+9.4f}"
              f"{c['coded_bits']:>9.4f}{c['pad_bits']:>8.4f}"
              f"{c['tab_bits']:>8.4f}{c['mant_bits']:>8.4f}"
              f"{c['misc_bits']:>8.4f}{bounds_w[m]:>10.4f}"
              f"{bpw[m] - bounds_w[m]:>8.4f}")
    print("  (bound: frozen H+7, M1 H(sym10)+6, M2 revised floor, "
          "M3 H(sym11)+5 -- entropy level, no side costs)")

    print("\ncoder redundancy (emitted coded bits minus entropy-level info, "
          "per coded symbol; flush is information-neutral):")
    for m in MECHS:
        cr = coder_red[m]
        print(f"{m:>8}  {cr['redundancy_bits_per_coded_symbol']:+.6f} b/sym "
              f"x {cr['coded_symbols_per_weight']:.4f} sym/w "
              f"(coded {cr['coded_bpw']:.4f} b/w vs info "
              f"{cr['info_bpw']:.4f} b/w)")
    print("  (M2's per-symbol redundancy on near-fair binary lanes is a "
          "property of the frozen bit-renorm coder [M=4096, bit-by-bit "
          "renorm]; an M2 loss falsifies per-bit binary lanes in THIS "
          "coder, not the mantissa-phase direction)")

    print("\nper-layer (b/w, weighted):")
    hdr = (f"{'layer':>6}{'floor7':>9}{'floorRev':>9}{'frozen':>9}"
           + "".join(f"{m:>9}" for m in MECHS if m != "frozen")
           + f"{'best':>7}{'delta':>9}")
    print(hdr)
    print("-" * len(hdr))
    per_layer = {}
    for L in layers:
        rs = [r for r in recs if r["layer"] == L]
        nn = sum(r["n"] for r in rs)
        row = {"tensors": len(rs), "params": int(nn),
               "floor7": round(sum(r["floors"]["floor7"] * r["n"]
                                   for r in rs) / nn, 6),
               "floor_rev": round(sum(r["floors"]["floor_rev"] * r["n"]
                                      for r in rs) / nn, 6)}
        for m in MECHS:
            row[m] = round(sum(mech_bits(r, m) for r in rs) / nn, 6)
        ds = {m: row["frozen"] - row[m] for m in MECHS if m != "frozen"}
        row["best"] = max(ds, key=lambda m: ds[m])
        row["best_delta"] = round(ds[row["best"]], 6)
        per_layer[f"L{L}"] = row
        print(f"{L:>6}{row['floor7']:>9.4f}{row['floor_rev']:>9.4f}"
              f"{row['frozen']:>9.4f}"
              + "".join(f"{row[m]:>9.4f}" for m in MECHS if m != "frozen")
              + f"{row['best']:>7}{row['best_delta']:>+9.4f}")
    print(f"{'ALL':>6}{floor7_w:>9.4f}{floorrev_w:>9.4f}{bpw['frozen']:>9.4f}"
          + "".join(f"{bpw[m]:>9.4f}" for m in MECHS if m != "frozen")
          + f"{best_m:>7}{best_delta:>+9.4f}")

    print(f"\nphase p(1) pooled over sample: "
          + " ".join(f"{x:.4f}" for x in stab["global_p1"]))
    print(f"per-tensor stability: global-constants cross-entropy costs "
          f"{stab_delta_bpw:+.6f} b/w vs per-tensor fit; per-tensor probs "
          f"side cost {side_cost_bpw:.6f} b/w -> {stab['verdict']}")
    print(f"\n{verdict}")
    print(f"projected whole-model at best mechanism: {proj_wm:.4f} b/w "
          f"(= {FROZEN_WHOLE_MODEL_BPW} - {EXPERT_FRAC} x {best_delta:+.4f}; "
          f"sample-selected mechanism -- selection-optimistic)")
    print(f"scope: {scope}")

    summary = {
        "mode": "synthetic" if synthetic else "real", "scope": scope,
        "acct_stamp": ACCT_STAMP, "acct": ACCT,
        "targets": len(recs), "total_params": int(n_tot),
        "layers": [int(x) for x in layers],
        "floor7_bpw_weighted": round(floor7_w, 6),
        "floor_revised_bpw_weighted": round(floorrev_w, 6),
        "floor_headroom_bpw": round(floor7_w - floorrev_w, 6),
        "peel_ceiling_ref_bpw": PEEL_CEILING_BPW,
        "frozen_whole_model_ref_bpw": FROZEN_WHOLE_MODEL_BPW,
        "realized_bpw": {m: round(bpw[m], 6) for m in MECHS},
        "delta_vs_frozen_bpw": {m: round(delta[m], 6) for m in delta},
        "entropy_bounds_bpw": {m: round(bounds_w[m], 6) for m in MECHS},
        "components_bpw": {m: {k: round(v, 6) for k, v in comps[m].items()}
                           for m in MECHS},
        "coder_redundancy": coder_red,
        "m2_attribution": (
            "M2 codes near-fair binary symbols through the frozen bit-renorm "
            "rANS (M=4096, state [M,2M), bit-by-bit renorm), whose measured "
            "per-symbol redundancy at p~0.416..0.5 (~+0.03 b/bit at the MSB, "
            "falling to ~0 at p=0.5) exceeds the MSB's entire entropy gain "
            "1-h(0.4159)=0.0205; an M2 failure therefore falsifies 'per-bit "
            "binary lanes in the frozen bit-renorm coder', NOT the "
            "mantissa-phase direction -- extended-symbol folding (M1) is the "
            "in-format path, a wider-state/multi-bit-per-symbol lane is the "
            "format-change path"),
        "per_layer": per_layer,
        "phase_stability": stab,
        "roundtrip": {"blocks": {m: int(rt_blocks[m]) for m in MECHS},
                      "all_ok": bool(rt_ok)},
        "gate": {"gate_bpw": GATE_BPW, "best_mech": best_m,
                 "best_delta_bpw": round(best_delta, 6),
                 "fires": fires},
        "projected_whole_model_bpw": round(proj_wm, 6),
        "verdict": verdict,
        "accounting_note": (
            "all cells use the frozen v2 mechanics exactly (W=128 bit-stride, "
            "DP T=4 tier budgets over measured emitted bits, P100 no-escape, "
            "L3 12-bit flush payload, per-block class flags, u32 rank anchors "
            "per 512 blocks, 96-bit class descriptors, 32-byte header, pad8 "
            "record align); mechanism-specific charges: M1 pad8(1024+nnz*12) "
            "table + pad8(6n-12nb) verbatim plane; M2 order-0 table + "
            "pad8(7*12)=88-bit phase probs, no verbatim plane (mantissa bits "
            "are inside the measured coded blocks); M3 pad8(2048+nnz*12) "
            "table + pad8(5n-12nb) plane; frozen parity asserted per tensor "
            "against probe_emission_peel.realized_cell; every mechanism "
            "round-trip proven per tensor on sampled block ranges with "
            "SHA-256-exact BF16 reconstruction"),
    }
    summaryp.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written: {summaryp}")
    return summary


# --------------------------------------------------------------------- main ---
def run(a, snap: Path, jsonl: Path, summaryp: Path):
    tg = ep.sample_targets(snap, a.synthetic, a.layer)
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
        print(f"\n[limit] {a.limit} tensors this invocation -- re-invoke to "
              f"resume.")
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
    jsonl = ART / f"mantissa_phase_results{tag}.jsonl"
    summaryp = ART / f"mantissa_phase_summary{tag}.json"

    lockp = ART / f"mantissa_phase{tag}.lock"
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
