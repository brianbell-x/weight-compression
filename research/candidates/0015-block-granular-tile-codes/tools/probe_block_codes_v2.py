"""probe_block_codes_v2.py -- candidate 0015 v2: attack the per-block overhead.

v1 (probe_block_codes.py, RESULTS.md) measured that the tile-fusible grid
(fixed stride, W <= 128) loses to realized stz (10.8822 b/w on the canonical
shard-7 layer-27 set) ENTIRELY on fixed per-block overhead, exactly decomposed
at W=128: floor 10.5583 + quant 0.0076 + coder excess 0.1328 (12-bit flush
0.094 + bit-renorm rounding 0.039) + byte-ceil pad slack 0.3027 + taxes 0.0086
+ escapes ~0.025 = 11.0349. Budget: combined per-block overhead < 0.316 b/w
beats stz at W=128; < 0.142 hits floor+0.15. v2 implements the four levers
named there, each priced separately and in combination, all preserving the
fusible contract: EVERY block is independently decodable starting from an
O(1)-computable address.

Field split (stz.py's exact convention): u16 LE, sym = u >> 7 (9-bit
sign+exponent), mant = u & 0x7F (7-bit mantissa, verbatim). Block sizes are
MEASURED emitted bits of the named coder (exact vectorized simulation of the
reference encoder, round-trip verified on samples) -- never ideal bits.

THE LEVERS
  L1  bit-granular fixed stride: slot budgets in BITS, not byte-ceiled bytes.
      Fixed bit stride is still O(1) address math (bit offset = base + rank *
      slot_bits). Kills the byte-ceil part of pad slack (~0.03-0.05 b/w).
  L2  multi-tier budgets (T in {1,2,4}): per-tensor, per-(W,P) the T tier
      budgets are chosen OPTIMALLY by dynamic programming over the distinct
      measured block sizes (minimizing total slot bits). Each block carries a
      ceil(log2(classes)) -bit flag (classes = nonempty tiers + escape class
      if present) in a fixed-width flag plane. Blocks of one class live at
      fixed stride in that class's region; the in-class rank of block i is
      anchor[class][i/512] + popcount over <=512 flags -- the same O(1)
      contract as v1's escape rank directory. Charged exactly: flag plane,
      one u32 rank anchor per class per 512 blocks, u64+u32 region descriptor
      per class, all in the accounting.
  L3  cheaper flush ("state carries payload"): the encoder's INITIAL state is
      seeded x0 = M + d where d = the first 12 bits of the block's own 7-bit
      mantissa stream, instead of the constant M. The decoder's FINAL state
      returns d (final-state check becomes payload recovery), so the 12-bit
      flush field is no longer dead weight: the mantissa plane shrinks to
      7W-12 bits per block (still fixed stride; the borrowed 12 bits sit at a
      fixed offset -- the flush field -- inside the block's own slot). No
      cross-block dependency. The residual cost (seeding from a larger state
      emits slightly more renorm bits, ~0-1 bit/block) is MEASURED, not
      modeled: the simulation and the round-trip both use the real seeds.
      Escaped blocks have no ANS state, so their raw slot carries the 12
      borrowed bits explicitly (slot = 9W + 12 bits) -- charged.
      Kernel-contract caveat: the borrowed 12 bits come back from the
      decoder's FINAL state, so weights 0 and 1 of a block finalize only
      after the full W-symbol decode. Block independence and O(1)
      addressing are unaffected, and a register-tile kernel that decodes
      the whole tile before the MMA is unaffected, but streaming /
      partial-tile consumers must decode the full block first.
  L4  column-conditioned ANS tables: table id = column group of the block's
      START address, group = start_col // gcd(W, C) -- purely address-derived,
      zero per-block side info. Side cost = the extra quantized tables,
      charged exactly (pad8(512 + nnz_g*12) per occupied group).
      GATE (runs FIRST, recorded either way): re-measure H(sym | column-group)
      vs H(sym) directly on the target tensors -- the motivating numbers
      (0014's H(exp|col) = 2.486) were EARLY-LAYER properties. If the
      numel-weighted conditional gain at the best W-grouping is < 0.05 b/w,
      L4 is dropped from the grid and the summary says so.

THE CODER (v1's coder, generalized): per-block single-lane bit-renormalizing
rANS over a 12-bit quantized table (M = 4096, deterministic largest-remainder,
every present symbol >= 1); state x in [M, 2M); encode consumes symbols in
reverse from initial state x0 (= M, or M + d under L3), renorm emits one bit
(x & 1) while x >= 2q; flush stores (x_final - M) in exactly 12 bits. The
decoder starts at M + flush, ends at x0, and under L3 reads d = x_end - M.
Under L4 the table is the block's group table. Block sizes are computed by an
exact vectorized simulation (bit-identical arithmetic); every tensor passes a
deterministic-sample round-trip gate per (W, coder-variant): pure-Python
encode -> serialized bytes -> decode, asserting (1) emitted bits == accounted
bits, (2) symbols exactly equal, (3) under L3 the recovered payload d exactly
equals the block's first 12 mantissa bits, (4) SHA-256 of the reconstructed
BF16 bytes (sym re-merged with the mantissa, L3 mantissa bits rebuilt from the
recovered payload) exactly equals the original. Any mismatch aborts.

GRID: W in {64,128,256} x T in {1,2,4} x P in {95,97,99,100} x L1/L3/L4
on/off (full factorial, so every lever's marginal is visible), plus the
full-stack best. W=256 rows are the non-fusible bracket (reported, never the
headline). Escape rule as v1: budget percentile P of measured sizes; blocks
over the top budget escape wholesale to raw 9 b/w slots in the escape class.

Pre-registered verdict rule (from the v1 handoff): if the best fusible
(W <= 128) v2 config beats 10.8822 -> Direction A FIRES at tile granularity;
if a competent v2 with all four levers cannot get under 10.88 -> the
tile-granular order-0 point is FALSIFIED for good (v1's storage-leaning
result stands regardless).

Baseline parity: identical to v1 (stz.plan_regroup imported, per-tensor
+/-0.01 b/w vs stz_tensor_stats.jsonl, weighted reference must reproduce
10.8822; synthetic mode gates exact realized-bits equality and >= 1 regroup
tensor). v1's verified loaders / JSONL resume / quantizer / bit-packing are
IMPORTED from probe_block_codes.py, not reimplemented; v1 stays untouched.

Simplification (asserted, holds for both target sets): n % W == 0 for all W
in the grid (layer-27 numel 4,988,928 and synthetic 3,072 are divisible by
256), so there are no partial tail blocks.

Usage:
  uv run python probe_block_codes_v2.py --synthetic    # smoke (fake snapshot)
  uv run python probe_block_codes_v2.py --gate-only    # L4 gate phase only
  uv run python probe_block_codes_v2.py                # real run (resumable)
  uv run python probe_block_codes_v2.py --summary      # table + gates + JSON
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import probe_block_codes as v1  # noqa: E402  -- verified v1 infrastructure

stz = v1.stz
REAL_SNAP, SYN_SNAP, STATS_JSONL, ART = v1.REAL_SNAP, v1.SYN_SNAP, v1.STATS_JSONL, v1.ART
M, M_LOG2, FLUSH_BITS = v1.M, v1.M_LOG2, v1.FLUSH_BITS
pad8, ceil_div, BITLEN, die = v1.pad8, v1.ceil_div, v1.BITLEN, v1.die

WS = (64, 128, 256)                  # block sizes; W<=128 = the fusible bracket
TIERS = (1, 2, 4)                    # L2 tier counts (1 = single budget, v1-like)
PS = (95, 97, 99, 100)               # top-budget percentiles of measured sizes
FUSIBLE_W_MAX = 128
RANK_GROUP, RANK_BITS = v1.RANK_GROUP, v1.RANK_BITS   # u32 anchor per 512 blocks
HEAD_BITS = v1.HEAD_BITS             # 32-byte per-tensor record header
CLASS_DIR_BITS = 96                  # per used class: u64 region base + u32 slot size
BORROW_BITS = 12                     # L3: mantissa bits carried by the ANS state
L4_GATE_MIN_GAIN = 0.05              # b/w; below this, drop L4 (pre-registered)
STZ_TARGET_BPW = 10.8822             # realized stz on this exact set (v2 G1)
WHOLE_MODEL_BPW = 10.8975
EXPERT_FRAC = 0.93
G2_SLACK = 0.15
PARITY_TOL = 0.01
REF_WEIGHTED_TOL = 0.001
BUDGET_BEAT_STZ = 0.316              # v1-measured overhead budget to beat stz @W=128
BUDGET_G2 = 0.142                    # v1-measured overhead budget for floor+0.15
V1_ANCHOR = ("W128_T1_P99_L10L30L40", 11.0349)  # v1 W128_P99 (real set) sanity anchor

CODER_SPEC = ("per-block single-lane bit-renorm rANS; M=4096 12-bit quantized "
              "table(s) (stores q-1); state in [M,2M); bit-by-bit renorm (emit "
              "x&1 while x>=2q); 12-bit flush = x_final - M; L3 seeds x0 = "
              "M + first-12-mantissa-bits (decoder final state returns them, "
              "so under L3 the first two weights of a block finalize only "
              "after the full W-symbol decode -- payload is the decoder's "
              "terminal state); "
              "L4 keys the table by column group = block_start_col // gcd(W,C); "
              "sizes are measured emitted bits, round-trip verified on samples")

ACCT = {"schema": 1, "v": 2, "M_LOG2": M_LOG2, "FLUSH_BITS": FLUSH_BITS,
        "WS": WS, "TIERS": TIERS, "PS": PS, "FUSIBLE_W_MAX": FUSIBLE_W_MAX,
        "RANK_GROUP": RANK_GROUP, "RANK_BITS": RANK_BITS,
        "HEAD_BITS": HEAD_BITS, "CLASS_DIR_BITS": CLASS_DIR_BITS,
        "BORROW_BITS": BORROW_BITS, "L4_GATE_MIN_GAIN": L4_GATE_MIN_GAIN,
        "CODER": CODER_SPEC}
ACCT_STAMP = hashlib.sha256(json.dumps(ACCT, sort_keys=True).encode()).hexdigest()[:12]


def check_stamp(rows: list[dict], jsonl: Path):
    bad = [r for r in rows if r.get("acct") != ACCT_STAMP]
    if bad:
        die(f"{len(bad)}/{len(rows)} rows in {jsonl} carry accounting stamp "
            f"{bad[0].get('acct')!r} != current {ACCT_STAMP!r} -- move that "
            f"file aside and re-run")


# ------------------------------------------------------------- coder (v2) ---
def rans_enc_block(syms: list, ql: list, cl: list, x0: int):
    """v1's reference encoder generalized to an arbitrary initial state
    x0 in [M, 2M). x0 = M reproduces v1's coder bit-for-bit."""
    assert M <= x0 < 2 * M
    x = x0
    bits = []
    ap = bits.append
    for s in reversed(syms):
        qq = ql[s]
        t = qq << 1
        while x >= t:
            ap(x & 1)
            x >>= 1
        x = M + cl[s] + (x - qq)
    return x - M, bits


def rans_dec_block(data: bytes, nbits: int, L: int, ql: list, cl: list, s2s: list):
    """v1's reference decoder generalized: returns (symbols, final_state) or
    None on inconsistency. Final state == encoder's x0; under L3 the caller
    reads the 12-bit payload as final_state - M."""
    if nbits < FLUSH_BITS:
        return None
    f = 0
    for i in range(FLUSH_BITS):
        f = (f << 1) | ((data[i >> 3] >> (7 - (i & 7))) & 1)
    x = M + f
    pos = FLUSH_BITS
    out = []
    ap = out.append
    for _ in range(L):
        slot = x - M
        s = s2s[slot]
        x = ql[s] + slot - cl[s]
        while x < M:
            if pos >= nbits:
                return None
            x = (x << 1) | ((data[pos >> 3] >> (7 - (pos & 7))) & 1)
            pos += 1
        ap(s)
    if pos != nbits or not (M <= x < 2 * M):
        return None
    return out, x


def rans_sim_blocks(qm: np.ndarray, cm: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """v1's exact vectorized encoder simulation generalized to per-block
    initial states x0 (int64, values in [M, 2M)). Bit-identical arithmetic;
    x0 == M everywhere reproduces v1.rans_sim_blocks exactly."""
    nbk, W = qm.shape
    x = x0.astype(np.int64).copy()
    bits = np.full(nbk, FLUSH_BITS, np.int64)
    for j in range(W - 1, -1, -1):
        qq = qm[:, j]
        thr1 = (qq << 1) - 1
        k = BITLEN[x] - BITLEN[thr1]
        np.maximum(k, 0, out=k)
        k += (x >> k) > thr1
        bits += k
        x = M + cm[:, j] + ((x >> k) - qq)
    return bits


# --------------------------------------------------------- L4 group tables ---
def block_groups(nb: int, W: int, C: int):
    """Address-derived column group of each block: group = start_col // g,
    g = gcd(W, C). Dense-remapped to occupied groups (unoccupied groups need
    no table). Returns (dense gid per block, n_groups)."""
    starts_col = (np.arange(nb, dtype=np.int64) * W) % C
    g = math.gcd(W, C)
    gid = starts_col // g
    uniq, dense = np.unique(gid, return_inverse=True)
    return dense.astype(np.int64), int(uniq.size)


def cond_entropy_bits(joint: np.ndarray, n: int) -> float:
    """H(sym | group) from a (G, 512) count matrix; exact empirical."""
    rows = joint.sum(1, keepdims=True).astype(np.float64)
    safe = np.where(rows > 0, rows, 1.0)
    p = joint / safe
    h = np.where(joint > 0, -p * np.log2(np.where(joint > 0, p, 1.0)), 0.0).sum(1)
    return float((rows[:, 0] / n * h).sum())


def build_table(hist: np.ndarray, n: int):
    """Quantized 12-bit table via v1's verified quantizer. Returns
    (q512, cum512, cl_q512 float, table_bits, nnz)."""
    q = v1.quantize_hist(hist, n)
    nnz = int((hist > 0).sum())
    present = np.flatnonzero(q)
    cum = np.zeros(512, np.int64)
    cum[present] = np.cumsum(q[present]) - q[present]
    cl_q = np.zeros(512)
    cl_q[q > 0] = M_LOG2 - np.log2(q[q > 0])
    return q, cum, cl_q, pad8(512 + nnz * 12), nnz


# ------------------------------------------------------------ L2 tier DP ---
def tier_dp(kept_rb: np.ndarray, l1: bool, t_max: int) -> dict:
    """Optimal tier budgets over the distinct measured kept-block sizes.
    Minimizes total slot bits when each block pays the smallest tier budget
    >= its size (slot cost = bits if l1 else byte-ceiled bits). Returns
    {T: (kept_slot_bits, budgets_bits, slot_sizes, class_counts)} for
    T in TIERS; fewer effective tiers used when optimal / when distinct
    sizes < T."""
    s, cnt = np.unique(kept_rb, return_counts=True)
    N = np.cumsum(cnt)                       # blocks with size <= s[d]
    D = s.size
    cost = (s if l1 else ((s + 7) // 8) * 8).astype(np.float64)
    F = np.empty((t_max, D))
    PAR = np.full((t_max, D), -1, np.int64)
    F[0] = N * cost
    idx = np.arange(D)
    for k in range(1, t_max):
        A = F[k - 1][None, :] - cost[:, None] * N[None, :]
        A[idx[None, :] >= idx[:, None]] = np.inf     # require d' < d
        best = A.argmin(1)
        split = cost * N + A[idx, best]
        skip = F[k - 1] <= split                     # empty tier allowed
        F[k] = np.where(skip, F[k - 1], split)
        PAR[k] = np.where(skip, -2, best)

    def backtrack(T: int):
        ths, k, d = [], T - 1, D - 1
        while True:
            if k == 0:
                ths.append(d)
                break
            if PAR[k][d] == -2:
                k -= 1
                continue
            ths.append(d)
            d = int(PAR[k][d])
            k -= 1
        ths = sorted(ths)
        budgets = [int(s[i]) for i in ths]
        slots = [int(cost[i]) for i in ths]
        counts, prev = [], 0
        for i in ths:
            counts.append(int(N[i]) - prev)
            prev = int(N[i])
        total = sum(c * sl for c, sl in zip(counts, slots))
        # integer recompute is authoritative; DP float value must agree
        if abs(total - F[T - 1][D - 1]) > 0.5:
            die(f"tier DP reconciliation failed: {total} vs {F[T - 1][D - 1]}")
        return total, budgets, slots, counts

    return {T: backtrack(min(T, D)) for T in TIERS if T <= t_max}


# ------------------------------------------------------------- L4 gate ---
def gate_tensor(sym: np.ndarray, n: int, C: int, H: float) -> dict:
    """Per-tensor L4 gate measurement: H(sym), H(sym|column) (exact,
    conditioning ceiling), and H(sym | block column group) for each W with
    the exact grouping the L4 coder would use."""
    cols = np.arange(n, dtype=np.int64) % C
    joint = np.bincount(cols * 512 + sym, minlength=C * 512).reshape(C, 512)
    h_col = cond_entropy_bits(joint, n)
    per_w = {}
    for W in WS:
        if n % W:
            die(f"n={n} not divisible by W={W} (v2 assumes no tail blocks)")
        nb = n // W
        gid, G = block_groups(nb, W, C)
        gw = np.repeat(gid, W)
        j = np.bincount(gw * 512 + sym, minlength=G * 512).reshape(G, 512)
        h_g = cond_entropy_bits(j, n)
        per_w[f"W{W}"] = {"G": G, "H_grp": round(h_g, 6),
                          "gain": round(H - h_g, 6)}
    return {"H_sym": round(H, 6), "H_col": round(h_col, 6),
            "gain_col": round(H - h_col, 6), "per_w": per_w}


def gate_summary_from_rows(rows: list[dict], synthetic: bool) -> dict:
    n_tot = sum(r["n"] for r in rows)
    w = lambda f: sum(f(r) * r["n"] for r in rows) / n_tot
    per_w = {}
    for W in WS:
        per_w[f"W{W}"] = {
            "H_grp_w": round(w(lambda r: r["gate"]["per_w"][f"W{W}"]["H_grp"]), 6),
            "gain_w": round(w(lambda r: r["gate"]["per_w"][f"W{W}"]["gain"]), 6),
            "G_mean": round(sum(r["gate"]["per_w"][f"W{W}"]["G"] for r in rows)
                            / len(rows), 2),
        }
    best_w = max(per_w, key=lambda k: per_w[k]["gain_w"])
    best_gain = per_w[best_w]["gain_w"]
    passed = best_gain >= L4_GATE_MIN_GAIN
    l4_active = True if synthetic else passed   # smoke exercises L4 regardless
    return {"tensors": len(rows), "total_params": int(n_tot),
            "H_sym_w": round(w(lambda r: r["gate"]["H_sym"]), 6),
            "H_col_w": round(w(lambda r: r["gate"]["H_col"]), 6),
            "gain_col_w": round(w(lambda r: r["gate"]["gain_col"]), 6),
            "per_w": per_w, "best_w": best_w, "best_gain_bpw": best_gain,
            "threshold_bpw": L4_GATE_MIN_GAIN, "gate_pass": bool(passed),
            "l4_active": bool(l4_active),
            "l4_forced": bool(synthetic and not passed),
            "note": ("synthetic smoke: L4 kept in the grid to exercise the "
                     "machinery even though the gate gain is below threshold"
                     if synthetic and not passed else
                     "gate decides L4 inclusion; recorded either way")}


# ------------------------------------------------------------- per-tensor ---
def analyze_tensor(raw: bytes, t: dict, synthetic: bool, stats_ref: dict,
                   l4_active: bool) -> dict:
    u = np.frombuffer(raw, "<u2")
    n = u.size
    R, C = t["shape"]
    assert n == R * C, (t["name"], n, R, C)
    sym = (u >> 7).astype(np.int64)
    mant = (u & 0x7F).astype(np.int64)
    hist = np.bincount(sym, minlength=512).astype(np.int64)
    p = hist[hist > 0] / n
    H = float(-(p * np.log2(p)).sum())
    floor_bpw = H + 7.0

    # ---- baseline parity (v1's exact gates, reused)
    plan = stz.plan_regroup(hist, n, R)
    base_bpw = plan["bits"] / n
    if synthetic:
        codec, chunks, st = stz.enc_tensor(raw, t["shape"])
        ref = st["bpw"]
        if codec == 1:
            realized = sum(len(c) for c in chunks) * 8
            if realized != plan["bits"]:
                die(f"PARITY: {t['name']} plan {plan['bits']} != realized {realized}")
    else:
        ref = stats_ref[t["name"]]
    diff = abs(round(base_bpw, 4) - ref)
    if diff > PARITY_TOL:
        die(f"PARITY FAILURE on {t['name']}: {base_bpw:.4f} vs {ref:.4f}")

    # ---- L4 gate stats (recorded per tensor either way)
    gate = gate_tensor(sym, n, C, H)

    # ---- base (order-0 per-tensor) table
    q0, cum0, clq0, table_bits0, nnz0 = build_table(hist, n)
    quant_delta = float((hist * clq0).sum() / n - H)

    l4_opts = (0, 1) if l4_active else (0,)
    measured, cells = {}, {}
    rt_blocks = rt_syms = 0
    sha_o, sha_r = hashlib.sha256(), hashlib.sha256()

    for W in WS:
        if n % W:
            die(f"n={n} not divisible by W={W}")
        nb = n // W
        starts = np.arange(nb, dtype=np.int64) * W
        # L3 seeds: first 12 bits of the block's 7-bit-per-weight mantissa stream
        d_seed = (mant[starts] << 5) | (mant[starts + 1] >> 2)
        x0_l3 = (M + d_seed).astype(np.int64)
        x0_l0 = np.full(nb, M, np.int64)

        # L4 grouping + per-group tables for this W
        gid, G = block_groups(nb, W, C)
        gw = np.repeat(gid, W)
        tabs = {0: {"bits": table_bits0, "G": 1}}
        if 1 in l4_opts:
            joint = np.bincount(gw * 512 + sym, minlength=G * 512).reshape(G, 512)
            Qm = np.zeros((G, 512), np.int64)
            CQm = np.zeros((G, 512), np.int64)
            CLQ = np.zeros((G, 512))
            tb4 = 0
            for g in range(G):
                qg, cg, clg, tbg, _ = build_table(joint[g], int(joint[g].sum()))
                Qm[g], CQm[g], CLQ[g] = qg, cg, clg
                tb4 += tbg
            tabs[1] = {"bits": int(tb4), "G": G}

        for l4 in l4_opts:
            if l4:
                qv, cv = Qm[gw, sym], CQm[gw, sym]
                per_q = CLQ[gw, sym]
            else:
                qv, cv = q0[sym], cum0[sym]
                per_q = clq0[sym]
            qm = qv.reshape(nb, W)
            cm = cv.reshape(nb, W)
            qent_b = per_q.reshape(nb, W).sum(1)      # per-block quantized entropy
            # table delta vs H(sym) for THIS row's tables: base-table quantization
            # loss when l4=0; group-table quantization loss net of the column-
            # conditioning gain when l4=1 (can be negative). Used by the summary
            # so the ovhd column is attributed to the tables actually in play.
            quant_row = float(qent_b.sum() / n - H)
            for l3 in (0, 1):
                x0v = x0_l3 if l3 else x0_l0
                rb = rans_sim_blocks(qm, cm, x0v)
                mkey = f"W{W}_L3{l3}L4{l4}"
                measured[mkey] = {
                    "coded_bits": int(rb.sum()),
                    "flush_bpw": round(nb * FLUSH_BITS / n, 6),
                    "quant_delta_bpw": round(quant_row, 6),
                    "excess_vs_qentropy_bpw":
                        round((int(rb.sum()) - float(qent_b.sum())) / n, 6),
                }

                # ---- round-trip gate: sampled blocks per (W, coder variant)
                gtabs = {}
                for i in sorted({0, nb - 1, int(np.argmin(rb)), int(np.argmax(rb))}):
                    g = int(gid[i]) if l4 else 0
                    if g not in gtabs:
                        if l4:
                            qq, cc = Qm[g], CQm[g]
                        else:
                            qq, cc = q0, cum0
                        pres = np.flatnonzero(qq)
                        gtabs[g] = (qq.tolist(), cc.tolist(),
                                    np.repeat(pres, qq[pres]).tolist())
                    ql, cl, s2s = gtabs[g]
                    s0, e0 = int(starts[i]), int(starts[i]) + W
                    seq = sym[s0:e0].tolist()
                    x0i = int(x0v[i])
                    ctx = f"{t['name']} {mkey} block {i}"
                    fl, bits = rans_enc_block(seq, ql, cl, x0i)
                    data, nbits = v1.pack_block(fl, bits)
                    if nbits != int(rb[i]):
                        die(f"ROUND-TRIP ({ctx}): emitted {nbits} != accounted {int(rb[i])}")
                    dec = rans_dec_block(data, nbits, W, ql, cl, s2s)
                    if dec is None or dec[0] != seq:
                        die(f"ROUND-TRIP ({ctx}): decode failed / symbols differ")
                    x_end = dec[1]
                    if x_end != x0i:
                        die(f"ROUND-TRIP ({ctx}): final state {x_end} != seed {x0i}")
                    mrec = mant[s0:e0].copy()
                    if l3:
                        d_rec = x_end - M
                        if d_rec != int(d_seed[i]):
                            die(f"ROUND-TRIP ({ctx}): payload {d_rec} != {int(d_seed[i])}")
                        # rebuild the two borrowed mantissa fields from payload
                        mrec[0] = d_rec >> 5
                        mrec[1] = ((d_rec & 31) << 2) | (mrec[1] & 3)
                    rec = ((np.array(dec[0], dtype=np.int64) << 7) | mrec).astype("<u2")
                    orig = raw[2 * s0:2 * e0]
                    if rec.tobytes() != orig:
                        die(f"ROUND-TRIP ({ctx}): reconstructed bytes != original")
                    sha_o.update(orig)
                    sha_r.update(rec.tobytes())
                    rt_blocks += 1
                    rt_syms += W

                # ---- grid cells: P x l1 x T (DP once per (P, l1), read all T)
                for P in PS:
                    B_top = int(v1.pct_higher(rb, P))
                    esc = rb > B_top
                    n_esc = int(esc.sum())
                    kept_rb = rb[~esc]
                    coded_kept = int(kept_rb.sum())
                    qent_kept = float(qent_b[~esc].sum())
                    for l1 in (0, 1):
                        dp = tier_dp(kept_rb, bool(l1), max(TIERS))
                        esc_content = 9 * W + (BORROW_BITS if l3 else 0)
                        esc_slot = esc_content if l1 else 8 * ceil_div(esc_content, 8)
                        for T in TIERS:
                            kept_slot, budgets, slots, counts = dp[T]
                            classes = len(budgets) + (1 if n_esc else 0)
                            flagb = int(math.ceil(math.log2(classes))) if classes > 1 else 0
                            flag_bits = nb * flagb if l1 else pad8(nb * flagb)
                            rank_bits = (classes * RANK_BITS * ceil_div(nb, RANK_GROUP)
                                         if classes > 1 else 0)
                            cdir = classes * CLASS_DIR_BITS if classes > 1 else 0
                            tab = tabs[l4]["bits"]
                            esc_bits = n_esc * esc_slot
                            sym_raw = (kept_slot + esc_bits + flag_bits + rank_bits
                                       + cdir + tab + HEAD_BITS)
                            align = (pad8(sym_raw) - sym_raw) if l1 else 0
                            sym_total = sym_raw + align
                            if l3:
                                if l1:
                                    mant_bits = pad8(7 * n - BORROW_BITS * nb)
                                else:
                                    mant_bits = nb * 8 * ceil_div(7 * W - BORROW_BITS, 8)
                            else:
                                mant_bits = pad8(7 * n)
                            pad_kept = kept_slot - coded_kept
                            assert pad_kept >= 0
                            xs = coded_kept - FLUSH_BITS * (nb - n_esc) - qent_kept
                            key = f"W{W}_T{T}_P{P}_L1{l1}L3{l3}L4{l4}"
                            cells[key] = {
                                "bpw": round((sym_total + mant_bits) / n, 6),
                                "sym": int(sym_total), "mant": int(mant_bits),
                                "kept": int(kept_slot), "cod": coded_kept,
                                "pad": int(pad_kept), "esc_n": n_esc,
                                "esc": int(esc_bits), "flag": int(flag_bits),
                                "rank": int(rank_bits), "cdir": int(cdir),
                                "tab": int(tab), "align": int(align),
                                "cls": int(classes), "xs": round(xs, 1),
                                "bud": slots,
                            }
                            # per-cell reconciliation: the stored sym-plane
                            # components must sum exactly to the charged total
                            c = cells[key]
                            if c["sym"] != (c["kept"] + c["esc"] + c["flag"]
                                            + c["rank"] + c["cdir"] + c["tab"]
                                            + HEAD_BITS + c["align"]):
                                die(f"CELL RECONCILIATION ({t['name']} {key}): "
                                    f"components do not sum to charged sym total")
                            mant_chk = ((pad8(7 * n - BORROW_BITS * nb) if l1 else
                                         nb * 8 * ceil_div(7 * W - BORROW_BITS, 8))
                                        if l3 else pad8(7 * n))
                            if c["mant"] != mant_chk:
                                die(f"CELL RECONCILIATION ({t['name']} {key}): "
                                    f"mant {c['mant']} != lever formula {mant_chk}")

    sha_ok = sha_o.digest() == sha_r.digest()
    if not sha_ok:
        die(f"ROUND-TRIP ({t['name']}): SHA-256 mismatch over sampled spans")

    return {
        "name": t["name"], "layer": t["layer"], "expert": t["expert"],
        "proj": t["proj"], "n": int(n), "R": int(R), "C": int(C),
        "acct": ACCT_STAMP, "l4_active": bool(l4_active),
        "H_sym": round(H, 6), "floor_bpw": round(floor_bpw, 6),
        "baseline": {"bpw": round(base_bpw, 6), "bits": int(plan["bits"]),
                     "variant": plan["variant"]},
        "parity": {"ref_bpw": ref, "abs_diff": round(diff, 6)},
        "quant": {"nnz": nnz0, "table_bits": int(table_bits0),
                  "delta_bpw": round(quant_delta, 6)},
        "gate": gate,
        "measured": measured,
        "roundtrip": {"blocks": rt_blocks, "block_syms": rt_syms,
                      "bits_ok": True, "sha256_ok": bool(sha_ok)},
        "cells": cells,
    }


# ------------------------------------------------------------------ summary ---
def lever_key(l1, l3, l4):
    return f"L1{l1}L3{l3}L4{l4}"


def summarize(tg: list[dict], jsonl: Path, summaryp: Path, gate_sum: dict,
              synthetic: bool):
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
    for nm in names:
        if rec[nm]["l4_active"] != gate_sum["l4_active"]:
            die(f"row {nm} was computed with l4_active={rec[nm]['l4_active']} "
                f"!= current gate decision {gate_sum['l4_active']}")
    n_tot = sum(rec[nm]["n"] for nm in names)
    wsum = lambda f: sum(f(rec[nm]) for nm in names)

    base_w = wsum(lambda r: r["baseline"]["bits"]) / n_tot
    ref_w = wsum(lambda r: r["parity"]["ref_bpw"] * r["n"]) / n_tot
    floor_w = wsum(lambda r: r["floor_bpw"] * r["n"]) / n_tot
    quant_w = wsum(lambda r: r["quant"]["delta_bpw"] * r["n"]) / n_tot
    parity_max = max(rec[nm]["parity"]["abs_diff"] for nm in names)
    if not synthetic and abs(ref_w - STZ_TARGET_BPW) > REF_WEIGHTED_TOL:
        die(f"weighted stz reference {ref_w:.4f} != canonical {STZ_TARGET_BPW}")

    rt_blocks = wsum(lambda r: r["roundtrip"]["blocks"])
    rt_ok = all(rec[nm]["roundtrip"]["bits_ok"] and rec[nm]["roundtrip"]["sha256_ok"]
                for nm in names)

    l4_opts = (0, 1) if gate_sum["l4_active"] else (0,)
    keys = [f"W{W}_T{T}_P{P}_L1{l1}L3{l3}L4{l4}"
            for W in WS for T in TIERS for P in PS
            for l1 in (0, 1) for l3 in (0, 1) for l4 in l4_opts]

    def agg(key):
        g = lambda f: wsum(lambda r: f(r["cells"][key]))
        sym, mant = g(lambda c: c["sym"]), g(lambda c: c["mant"])
        parts = key.split("_")
        W = int(parts[0][1:])
        lev = parts[3]                       # "L1{l1}L3{l3}L4{l4}"
        mkey = f"{parts[0]}_L3{lev[5]}L4{lev[8]}"
        nb = sum(rec[nm]["n"] // W for nm in names)
        esc_n = g(lambda c: c["esc_n"])
        return {
            "bpw": (sym + mant) / n_tot,
            # table delta of THIS row's tables vs H(sym): base-table quant loss
            # for L40 rows; group-table quant loss net of conditioning gain
            # (can be negative) for L41 rows
            "quant_bpw": wsum(lambda r: r["measured"][mkey]["quant_delta_bpw"]
                              * r["n"]) / n_tot,
            "pad_bpw": g(lambda c: c["pad"]) / n_tot,
            "esc_bpw": g(lambda c: c["esc"]) / n_tot,
            "esc_frac": esc_n / nb,
            "flag_bpw": g(lambda c: c["flag"]) / n_tot,
            "rank_bpw": (g(lambda c: c["rank"]) + g(lambda c: c["cdir"])) / n_tot,
            "tab_bpw": g(lambda c: c["tab"]) / n_tot,
            "align_bpw": g(lambda c: c["align"]) / n_tot,
            "flush_bpw": FLUSH_BITS * (nb - esc_n) / n_tot,
            "xs_bpw": g(lambda c: c["xs"]) / n_tot,
            "mant_credit_bpw": (wsum(lambda r: pad8(7 * r["n"]))
                                - mant) / n_tot,
        }

    grid = {k: agg(k) for k in keys}
    target = base_w if synthetic else STZ_TARGET_BPW

    # per-(W,T,levers) best P
    best_p = {}
    for W in WS:
        for T in TIERS:
            for l1 in (0, 1):
                for l3 in (0, 1):
                    for l4 in l4_opts:
                        cand = [f"W{W}_T{T}_P{P}_L1{l1}L3{l3}L4{l4}" for P in PS]
                        bk = min(cand, key=lambda k: grid[k]["bpw"])
                        best_p[f"W{W}_T{T}_{lever_key(l1, l3, l4)}"] = bk

    fus_keys = [k for k in keys if int(k.split("_")[0][1:]) <= FUSIBLE_W_MAX]
    best_fus_key = min(fus_keys, key=lambda k: grid[k]["bpw"])
    best_fus_bpw = grid[best_fus_key]["bpw"]
    best_any_key = min(keys, key=lambda k: grid[k]["bpw"])
    best_any_bpw = grid[best_any_key]["bpw"]

    # lever marginals on the fusible bracket: best with lever off vs on
    def best_where(pred):
        ks = [k for k in fus_keys if pred(k)]
        bk = min(ks, key=lambda k: grid[k]["bpw"])
        return bk, grid[bk]["bpw"]

    marginals = {}
    for lev, on, off in (("L1_bit_stride", "L11", "L10"),
                         ("L3_flush_payload", "L31", "L30"),
                         ("L4_column_tables", "L41", "L40")):
        if lev.startswith("L4") and not gate_sum["l4_active"]:
            marginals[lev] = {"dropped_by_gate": True}
            continue
        k_on, b_on = best_where(lambda k, o=on: o in k)
        k_off, b_off = best_where(lambda k, o=off: o in k)
        marginals[lev] = {"best_on": k_on, "bpw_on": round(b_on, 6),
                          "best_off": k_off, "bpw_off": round(b_off, 6),
                          "marginal_bpw": round(b_off - b_on, 6)}
    k_on, b_on = best_where(lambda k: "_T1_" not in k)
    k_off, b_off = best_where(lambda k: "_T1_" in k)
    marginals["L2_multi_tier"] = {"best_on": k_on, "bpw_on": round(b_on, 6),
                                  "best_off": k_off, "bpw_off": round(b_off, 6),
                                  "marginal_bpw": round(b_off - b_on, 6)}

    g1_fus = best_fus_bpw < target
    g2_fus = best_fus_bpw <= floor_w + G2_SLACK
    l4_margin_caveat = None
    if g1_fus:
        verdict = (f"DIRECTION A FIRES at tile granularity (fixed-stride "
                   f"W<={FUSIBLE_W_MAX} beats realized stz)")
    else:
        verdict = ("tile-granular order-0 point FALSIFIED for good: competent "
                   "v2 (all four levers, L4 " +
                   ("included" if gate_sum["l4_active"] else
                    "gate-dropped as pre-registered") +
                   ") cannot beat stz at W<=128; v1's storage-leaning result stands")
        # falsification honesty check: if L4 was gate-dropped and the miss is
        # smaller than the forgone conditional gain, the dropped lever could in
        # principle have bridged the gap -- say so instead of overclaiming
        if not gate_sum["l4_active"]:
            margin = best_fus_bpw - target
            gate_gain = gate_sum["best_gain_bpw"]
            if 0 < margin < gate_gain:
                l4_margin_caveat = {
                    "falsification_margin_bpw": round(margin, 6),
                    "forgone_l4_gate_gain_bpw": round(gate_gain, 6),
                    "note": ("falsification margin is smaller than the forgone "
                             "L4 conditional gain; the gate-dropped lever could "
                             "in principle bridge the gap"),
                }
                verdict += (f" [CAVEAT: miss vs target is only {margin:.4f} b/w, "
                            f"smaller than the forgone L4 gate gain "
                            f"{gate_gain:.4f} b/w -- the gate-dropped lever "
                            f"could in principle bridge the gap, so treat this "
                            f"falsification as provisional]")
    if not rt_ok:
        verdict = "PROVISIONAL: " + verdict
    scope = ("synthetic smoke -- carries no evidential weight" if synthetic
             else f"layer {v1.TARGET_LAYER} only; cross-layer transfer unvalidated")

    mode = "SYNTHETIC (smoke only)" if synthetic else "REAL layer-27"
    print(f"\n=== candidate 0015 v2 -- block-code overhead attack [{mode}] ===")
    print(f"targets: {len(names)} tensors, {n_tot:,} params; parity gate OK "
          f"(max |d bpw| = {parity_max:.6f}); acct stamp {ACCT_STAMP}")
    print(f"stz recomputed {base_w:.4f} | G1 target "
          f"{'recomputed (synthetic)' if synthetic else STZ_TARGET_BPW} | "
          f"floor H(sym)+7 = {floor_w:.4f} | G2 bar = {floor_w + G2_SLACK:.4f}")
    print(f"coder: {CODER_SPEC}")
    print(f"round-trip: {rt_blocks} blocks across {len(names)} tensors x all "
          f"coder variants, bits==accounted and SHA-256 exact: "
          f"{'PASS' if rt_ok else 'FAIL'}")
    print(f"12-bit quantization delta (base tables): +{quant_w:.4f} b/w")

    print(f"\nL4 GATE (measured first, on {'synthetic' if synthetic else 'layer-27'} "
          f"tensors): H(sym)={gate_sum['H_sym_w']:.4f}, "
          f"H(sym|col)={gate_sum['H_col_w']:.4f} (gain {gate_sum['gain_col_w']:.4f});")
    for W in WS:
        pw = gate_sum["per_w"][f"W{W}"]
        print(f"  block col-group W={W:<4} G~{pw['G_mean']:<6} "
              f"H(sym|grp)={pw['H_grp_w']:.4f}  gain={pw['gain_w']:.4f}")
    print(f"  best gain {gate_sum['best_gain_bpw']:.4f} b/w at {gate_sum['best_w']} "
          f"vs threshold {L4_GATE_MIN_GAIN} -> gate "
          f"{'PASS' if gate_sum['gate_pass'] else 'FAIL'}; L4 "
          + ("ACTIVE" if gate_sum["l4_active"] else "DROPPED (as pre-registered)")
          + (" [forced for smoke]" if gate_sum.get("l4_forced") else ""))

    print("\noverhead decomposition, best (T,P) per lever combo (b/w; "
          "ovhd = bpw - floor - quant, where quant is THAT row's table delta "
          "-- for L41 rows the group tables' delta incl. the conditioning "
          "gain; budgets: <0.316 beats stz, <0.142 = G2):")
    hdr = (f"{'config':>26}{'bpw':>9}{'save':>8}{'ovhd':>8}{'flush':>7}{'xs':>7}"
           f"{'pad':>7}{'esc':>7}{'flag':>7}{'rank':>7}{'tab':>7}{'mcr':>7}{'P':>5}")
    print(hdr)
    print("-" * len(hdr))
    combo_best = {}
    for W in WS:
        for l1 in (0, 1):
            for l3 in (0, 1):
                for l4 in l4_opts:
                    cand = [f"W{W}_T{T}_P{P}_L1{l1}L3{l3}L4{l4}"
                            for T in TIERS for P in PS]
                    bk = min(cand, key=lambda k: grid[k]["bpw"])
                    v = grid[bk]
                    combo_best[f"W{W}_{lever_key(l1, l3, l4)}"] = \
                        {"key": bk, "bpw": round(v["bpw"], 6)}
                    ov = v["bpw"] - floor_w - v["quant_bpw"]
                    label = f"W{W}_T{bk.split('_')[1][1:]}_{lever_key(l1, l3, l4)}"
                    print(f"{label:>26}{v['bpw']:>9.4f}{target - v['bpw']:>+8.4f}"
                          f"{ov:>8.4f}{v['flush_bpw']:>7.4f}{v['xs_bpw']:>7.4f}"
                          f"{v['pad_bpw']:>7.4f}{v['esc_bpw']:>7.4f}"
                          f"{v['flag_bpw']:>7.4f}{v['rank_bpw']:>7.4f}"
                          f"{v['tab_bpw']:>7.4f}{v['mant_credit_bpw']:>7.4f}"
                          f"{bk.split('_')[2][1:]:>5}")

    print("\nlever marginals (fusible W<=128, best config with lever on vs off):")
    for lev, m in marginals.items():
        if m.get("dropped_by_gate"):
            print(f"  {lev:>18}: dropped by gate")
            continue
        print(f"  {lev:>18}: {m['bpw_off']:.4f} -> {m['bpw_on']:.4f} "
              f"(marginal {m['marginal_bpw']:+.4f} b/w; on={m['best_on']})")

    print(f"\nbest fusible (W<={FUSIBLE_W_MAX}): {best_fus_key} = {best_fus_bpw:.4f} b/w "
          f"(save vs target {target - best_fus_bpw:+.4f}; over floor "
          f"{best_fus_bpw - floor_w:+.4f})")
    print(f"best any-W: {best_any_key} = {best_any_bpw:.4f} b/w (W=256 bracket is "
          f"not tile-fusible)")
    print(f"v2-G1 (< {'recomputed stz' if synthetic else STZ_TARGET_BPW}, W<="
          f"{FUSIBLE_W_MAX}): {'PASS' if g1_fus else 'FAIL'} "
          f"(d = {target - best_fus_bpw:+.4f})")
    print(f"v2-G2 (<= floor + {G2_SLACK}): {'PASS' if g2_fus else 'FAIL'} "
          f"(d = {best_fus_bpw - floor_w:+.4f})")
    ovh_best = best_fus_bpw - floor_w - grid[best_fus_key]["quant_bpw"]
    print(f"overhead of best fusible vs budgets: {ovh_best:.4f} b/w "
          f"(beat-stz budget {BUDGET_BEAT_STZ}: "
          f"{'UNDER' if ovh_best < BUDGET_BEAT_STZ else 'OVER'}; "
          f"G2 budget {BUDGET_G2}: {'UNDER' if ovh_best < BUDGET_G2 else 'OVER'})")
    if not synthetic and V1_ANCHOR[0] in grid:
        print(f"v1 anchor: {V1_ANCHOR[0]} = {grid[V1_ANCHOR[0]]['bpw']:.4f} vs "
              f"v1 W128_P99 = {V1_ANCHOR[1]} (not an exact v1 replica on two "
              f"counts: v2 escape slots are separate-region rather than v1's "
              f"slot+overflow, and v2 escapes rb > percentile (bit-granular) "
              f"while v1 escaped rb > 8*ceil(pct/8), so v1 kept the blocks in "
              f"the percentile-to-byte-ceil gap)")
    print(f"verdict: {verdict}  [{scope}]")
    proj = None
    if not synthetic:
        proj = WHOLE_MODEL_BPW - EXPERT_FRAC * (base_w - best_fus_bpw)
        print(f"projected whole-model at best fusible: {proj:.4f} b/w "
              f"(selected on this same set -- selection-optimistic; cross-layer "
              f"transfer unvalidated)")

    summary = {
        "mode": "synthetic" if synthetic else "real", "scope": scope,
        "acct_stamp": ACCT_STAMP, "acct": ACCT,
        "targets": len(names), "total_params": int(n_tot),
        "parity_max_abs_diff": parity_max,
        "baseline_recomputed_bpw": round(base_w, 6),
        "baseline_ref_weighted_bpw": round(ref_w, 6),
        "stz_target_bpw": STZ_TARGET_BPW,
        "floor_bpw_weighted": round(floor_w, 6),
        "quant_delta_bpw_weighted": round(quant_w, 6),
        "coder": CODER_SPEC,
        "l4_gate": gate_sum,
        "roundtrip": {"blocks": int(rt_blocks), "all_ok": bool(rt_ok)},
        "cells": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in grid.items()},
        "best_p_per_combo": best_p,
        "combo_best": combo_best,
        "lever_marginals": marginals,
        "overhead_budgets": {"beat_stz_bpw": BUDGET_BEAT_STZ, "g2_bpw": BUDGET_G2,
                             "best_fusible_overhead_bpw": round(ovh_best, 6)},
        "best": {"fusible": {"key": best_fus_key, "bpw": round(best_fus_bpw, 6)},
                 "any_w": {"key": best_any_key, "bpw": round(best_any_bpw, 6)}},
        "gates": {
            "G1_vs": "recomputed baseline (synthetic)" if synthetic else STZ_TARGET_BPW,
            "keyed_on": f"fixed-stride grid, W<={FUSIBLE_W_MAX}",
            "G1_pass": bool(g1_fus),
            "G1_delta_bpw": round(target - best_fus_bpw, 6),
            "G2_bar_bpw": round(floor_w + G2_SLACK, 6),
            "G2_pass": bool(g2_fus),
            "G2_delta_vs_floor_bpw": round(best_fus_bpw - floor_w, 6),
        },
        "prereg_verdict_rule": ("best fusible W<=128 < 10.8822 -> Direction A "
                                "FIRES at tile granularity; competent v2 (all "
                                "four levers) not under 10.88 -> tile-granular "
                                "order-0 point FALSIFIED for good"),
        "verdict": verdict,
        "verdict_l4_margin_caveat": l4_margin_caveat,
        "projected_whole_model_best_fusible_bpw": None if synthetic else round(proj, 6),
        "accounting_note": (
            "per-weight totals include the mantissa plane (pad8(7n), or "
            "7W-12 bits/block under L3 -- byte-ceiled per block when L1 off); "
            "block sizes are MEASURED emitted bits of the named coder incl. "
            "the real L3 seeds; kept blocks pay their DP-optimal tier slot, "
            "escaped blocks pay a raw 9W(+12 under L3)-bit slot in the escape "
            "class; charged side costs per cell: per-block class flags "
            "(ceil(log2(classes)) bits, pad8 when byte-granular), u32 rank "
            "anchor per class per 512 blocks, 96-bit region descriptor per "
            "class (classes>1), quantized table(s) pad8(512+nnz*12) (per "
            "column group under L4), 32-byte header, record-level pad8 when "
            "bit-granular; reconciliation checked per cell at build time "
            "(sym-plane components must sum exactly to the charged total and "
            "the mantissa plane must match the lever formula; any mismatch "
            "aborts the run)"),
    }
    summaryp.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written: {summaryp}")
    return summary


# ---------------------------------------------------------------------- main ---
def run(a, snap: Path, jsonl: Path, gate_jsonl: Path, gate_sump: Path,
        summaryp: Path):
    tg = v1.enum_targets(snap, a.synthetic)
    names = [t["name"] for t in tg]
    t0 = time.time()

    # ---- phase 1: L4 gate (always first; recorded either way)
    grows = v1.load_rows(gate_jsonl)
    check_stamp(grows, gate_jsonl)
    gdone = {r["name"] for r in grows}
    for t in tg:
        if t["name"] in gdone:
            continue
        if time.time() - t0 > a.budget_s:
            print(f"\n[budget] {a.budget_s:.0f}s reached during gate phase -- "
                  f"progress saved, re-invoke to resume.", flush=True)
            sys.exit(0)
        raw = v1.read_raw(snap, t)
        u = np.frombuffer(raw, "<u2")
        sym = (u >> 7).astype(np.int64)
        hist = np.bincount(sym, minlength=512)
        p = hist[hist > 0] / u.size
        H = float(-(p * np.log2(p)).sum())
        g = gate_tensor(sym, u.size, t["shape"][1], H)
        rec = {"name": t["name"], "n": int(u.size), "acct": ACCT_STAMP, "gate": g}
        with gate_jsonl.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        gdone.add(t["name"])
    grows = v1.load_rows(gate_jsonl)
    grows = [r for r in grows if r["name"] in set(names)]
    gate_sum = gate_summary_from_rows(grows, a.synthetic)
    gate_sump.write_text(json.dumps(gate_sum, indent=2))
    print(f"[gate] L4 gate complete over {len(grows)} tensors: best gain "
          f"{gate_sum['best_gain_bpw']:.4f} b/w at {gate_sum['best_w']} "
          f"(threshold {L4_GATE_MIN_GAIN}) -> "
          f"{'PASS' if gate_sum['gate_pass'] else 'FAIL'}; L4 "
          f"{'active' if gate_sum['l4_active'] else 'dropped'}"
          + (" [forced for smoke]" if gate_sum.get("l4_forced") else ""),
          flush=True)
    if a.gate_only:
        print(f"[gate] summary written: {gate_sump}")
        return
    if a.summary:
        return summarize(tg, jsonl, summaryp, gate_sum, a.synthetic)

    # ---- phase 2: main grid
    stats_ref = {}
    if not a.synthetic:
        if not STATS_JSONL.exists():
            die(f"missing parity reference {STATS_JSONL}")
        want = set(names)
        for line in STATS_JSONL.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["name"] in want:
                    stats_ref[r["name"]] = r["bpw"]
        missing = [nm for nm in names if nm not in stats_ref]
        if missing:
            die(f"{len(missing)} targets absent from stz stats "
                f"(first: {missing[0]})")

    prior = v1.load_rows(jsonl)
    check_stamp(prior, jsonl)
    done = {r["name"] for r in prior}
    processed = 0
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
        rec = analyze_tensor(raw, t, a.synthetic, stats_ref, gate_sum["l4_active"])
        with jsonl.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        done.add(t["name"])
        processed += 1
        if processed % 8 == 0:
            print(f"[{i + 1}/{len(tg)}] {processed} tensors, "
                  f"{time.time() - t0:.0f}s", flush=True)

    if a.limit and processed >= a.limit and len(done) < len(tg):
        print(f"\n[limit] {a.limit} tensors this invocation -- re-invoke to resume.")
        sys.exit(0)

    if a.synthetic:
        n_reg = sum(1 for r in v1.load_rows(jsonl)
                    if r["baseline"].get("variant") == "regroup")
        if n_reg == 0:
            die("synthetic strong parity gate never exercised: 0 regroup tensors")
        print(f"[gate] synthetic strong parity gate exercised on {n_reg}/{len(tg)} tensors")

    print(f"\nall {len(done)}/{len(tg)} tensors done ({time.time() - t0:.0f}s)")
    summarize(tg, jsonl, summaryp, gate_sum, a.synthetic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run against the synthetic tiny snapshot (smoke)")
    ap.add_argument("--summary", action="store_true",
                    help="summary + gates only (requires all tensors done)")
    ap.add_argument("--gate-only", action="store_true",
                    help="run/refresh the L4 gate phase only")
    ap.add_argument("--limit", type=int, default=0,
                    help="max tensors this invocation in the MAIN grid phase "
                         "(0 = no cap). The L4 gate phase intentionally ignores "
                         "it: the gate decision must be computed over the full "
                         "tensor set before any grid row is written (rows are "
                         "bound to that decision); the gate loop is bounded by "
                         "--budget-s and is checkpointed/resumable")
    ap.add_argument("--budget-s", type=float, default=420.0,
                    help="soft wall-clock budget; exits cleanly when exceeded")
    a = ap.parse_args()

    snap = SYN_SNAP if a.synthetic else REAL_SNAP
    tag = "_synthetic" if a.synthetic else ""
    ART.mkdir(parents=True, exist_ok=True)
    jsonl = ART / f"blockcodes_v2_results{tag}.jsonl"
    gate_jsonl = ART / f"blockcodes_v2_gate{tag}.jsonl"
    gate_sump = ART / f"blockcodes_v2_gate_summary{tag}.json"
    summaryp = ART / f"blockcodes_v2_summary{tag}.json"

    lockp = ART / f"blockcodes_v2{tag}.lock"
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
        run(a, snap, jsonl, gate_jsonl, gate_sump, summaryp)
    finally:
        try:
            lockp.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
