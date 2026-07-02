"""probe_block_codes.py -- candidate 0015: block-granular tile codes (Direction A).

Decisive first probe of NEXT_DIRECTIONS Direction A on the canonical layer-27
target set (128 experts x {up_proj, down_proj} = 256 tensors, wholly in shard
7). Candidate 0014 certified that per-WEIGHT fixed-width keying cannot beat the
realized stz baseline (10.8822 b/w numel-weighted on this set); this probe
relaxes the constraint to a fixed byte budget per BLOCK of W weights (fixed
stride => O(1) tile address), entropy-coded inside the block.

Per tensor (field split = stz's exact convention: sym = u >> 7 (9-bit
sign+exponent), mant = u & 0x7F (7-bit mantissa, verbatim, pad8(7n) bits)):

  FLOOR      per-tensor empirical sym distribution, order-0 H(sym);
             storage-floor reference = H(sym) + 7 b/w.
  IDEAL      per-block ideal code lengths (sum of -log2 p(sym) under the
             per-tensor distribution) for W in {32,...,16384}; persists
             p50/p90/p95/p97/p99/max and the full histogram at 1/16-b/w
             resolution -- sizing the padding tail is THE unknown here.
  FORMAT (b) padded fixed-stride blocks, exact accounting: per-block size =
             MEASURED emitted bits of the reference coder (below); byte budget
             B = P-th percentile (P in {90,95,97,99,100}) of measured sizes,
             ceil'd to bytes; blocks over 8B bits escape WHOLESALE to raw
             9 b/w (B-byte slot + fixed-stride overflow slot => exactly
             max(B, ceil(9W/8)) bytes); charged exactly: padding waste (kept
             slack AND escaped-slot slack), pad8(nb) escape bitmap,
             u32-per-512-blocks rank directory, quantized ANS table
             pad8(512 + nnz*12) (stores q-1 per present symbol, range 0..4095),
             32-byte header, pad8(7n) mantissa.
  FORMAT (a) superblock rANS: 4096-sym superblocks, 32 interleaved lanes of the
             same reference coder (lane l takes in-superblock positions
             l, l+32, ...), each superblock byte-padded; two-level offset index
             charged exactly (u32 per superblock from group base + u64 absolute
             per 64 superblocks); O(1) at SUPERBLOCK granularity only --
             storage-leaning bracket, never the fusible headline.

THE CODER (named exactly; all block sizes below are its measured output, not a
bound): per-block single-lane bit-renormalizing rANS over the per-tensor 12-bit
quantized table (M = 4096, deterministic largest-remainder, every present
symbol >= 1); state x in [M, 2M); encode consumes symbols in reverse, renorm
emits one bit (x & 1) while x >= 2q; flush stores (x_final - M) in exactly
FLUSH_BITS = 12 bits. Block sizes are computed by an exact vectorized
simulation of this coder (bit-identical arithmetic), and every tensor passes a
deterministic-sample round-trip gate: pure-Python encode -> serialized bytes ->
decode on sampled blocks and format-(a) lanes, asserting (1) emitted bits ==
the accounted per-block bits, (2) decoded symbols exactly equal, (3) SHA-256 of
the reconstructed raw BF16 bytes (sym re-merged with the verbatim mantissa)
exactly equals the original bytes. Any mismatch aborts the run. The measured
excess over the quantized entropy (renorm rounding + flush) is reported per W.

FUSIBILITY: random access inside a kept block still requires sequential ANS
decode of up to W symbols, so the fusible verdict is keyed to a tile-credible
cap W <= FUSIBLE_W_MAX = 128 (Direction A's stated 64-128 row-segment range).
Larger W and format (a) are reported as storage-leaning brackets; they can
never carry a CONFIRMED verdict.

Baseline parity: stz.plan_regroup imported (never reimplemented); each tensor
must match stz_tensor_stats.jsonl within +/-0.01 b/w, and the numel-weighted
reference must reproduce 10.8822 b/w. Synthetic mode gates harder: recomputed
bits must equal stz.enc_tensor's realized serialized bits exactly, and >= 1
tensor must exercise the regroup codec.

Gates (numel-weighted over the target set, keyed to the fusible W <= 128 grid):
  G1  best fixed-stride (W <= 128) < 10.8822 b/w (realized stz, same tensors);
  G2  best fixed-stride (W <= 128) <= weighted per-tensor (H(sym)+7) + 0.15.
Storage-leaning results (W > 128, format (a)) get their own labeled outcome.

Pure numpy, deterministic, one tensor in RAM at a time, resumable JSONL
(fsync'd appends, atomic truncated-tail self-repair, skip-done, accounting
stamp checked on resume -- constant changes refuse to mix with old rows),
self-limits each invocation to the time budget (default 420 s) and exits
cleanly for re-invocation.

Usage:
  uv run python probe_block_codes.py --synthetic    # smoke on the fake snapshot
  uv run python probe_block_codes.py                # real layer-27 run (resumable)
  uv run python probe_block_codes.py --summary      # table + gates + summary JSON
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "research/candidates/0009-fusible-exponent-codebook/tools"))
import stz  # noqa: E402  -- reuse plan_regroup / enc_tensor / st_header verbatim

REAL_SNAP = REPO / "models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot"
SYN_SNAP = REPO / "models/synthetic/nemotron_tiny/hf_snapshot"
STATS_JSONL = (REPO / "research/candidates/0009-fusible-exponent-codebook"
               / "tests/artifacts/stz/stz_tensor_stats.jsonl")
ART = HERE.parents[1] / "tests" / "artifacts"

TARGET_LAYER = 27                    # real mode: the expert layer wholly inside shard 7
WS = (32, 64, 128, 256, 512, 1024, 4096, 16384)   # block sizes (weights)
PS = (90, 95, 97, 99, 100)           # budget percentiles of measured block sizes
FUSIBLE_W_MAX = 128                  # tile-credible cap: sequential decode <= W syms
M_LOG2 = 12
M = 1 << M_LOG2                      # 4096-state quantized ANS table
FLUSH_BITS = 12                      # flush stores (x_final - M), x in [M, 2M)
SB = 4096                            # format (a) superblock symbols
LANES = 32                           # format (a) interleaved rANS lanes
L2_BITS = 32                         # per-superblock u32 offset from group base
L1_GROUP = 64                        # superblocks per absolute u64 anchor
L1_BITS = 64
RANK_GROUP = 512                     # blocks per u32 escape-rank anchor (format b)
RANK_BITS = 32
HEAD_BITS = 32 * 8                   # per-tensor record header (W,P,B,nb,n,R,C,ov,nnz,...)
HIST_BIN = 1.0 / 16                  # b/w resolution of persisted ideal-block histograms
HIST_BINS = 256                      # covers [0, 16) b/w
STZ_TARGET_BPW = 10.8822             # realized stz, numel-weighted on this exact set (G1)
WHOLE_MODEL_BPW = 10.8975            # realized stz whole-model (context only)
EXPERT_FRAC = 0.93                   # experts' share of whole-model BF16 numel
G2_SLACK = 0.15
PARITY_TOL = 0.01
REF_WEIGHTED_TOL = 0.001             # weighted stats-jsonl refs must reproduce 10.8822
PREREG_KEY = "W128_P97"              # pre-registered mid-grid config (honest projection)

CODER_SPEC = ("per-block single-lane bit-renorm rANS; M=4096 12-bit quantized "
              "table (stores q-1); state in [M,2M); bit-by-bit renorm (emit "
              "x&1 while x>=2q); 12-bit flush = x_final - M; sizes are "
              "measured emitted bits, round-trip verified on samples")

ACCT = {"schema": 2, "M_LOG2": M_LOG2, "FLUSH_BITS": FLUSH_BITS, "WS": WS,
        "PS": PS, "SB": SB, "LANES": LANES, "L2_BITS": L2_BITS,
        "L1_GROUP": L1_GROUP, "L1_BITS": L1_BITS, "RANK_GROUP": RANK_GROUP,
        "RANK_BITS": RANK_BITS, "HEAD_BITS": HEAD_BITS, "HIST_BIN": HIST_BIN,
        "HIST_BINS": HIST_BINS, "FUSIBLE_W_MAX": FUSIBLE_W_MAX,
        "CODER": CODER_SPEC}
ACCT_STAMP = hashlib.sha256(
    json.dumps(ACCT, sort_keys=True).encode()).hexdigest()[:12]

NAME_RE = re.compile(r"backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up|down)_proj\.weight$")

pad8 = lambda bits: int(bits) + (-int(bits) % 8)   # byte-alignment, stz's rule
ceil_div = lambda a, b: -(-a // b)

# bit-length lookup for the vectorized coder simulation (values < 2M = 8192)
BITLEN = np.zeros(2 * M, np.int64)
BITLEN[1:] = np.floor(np.log2(np.arange(1, 2 * M))).astype(np.int64) + 1


def die(msg: str):
    print(f"\n!!! ABORT: {msg}", flush=True)
    sys.exit(1)


# ------------------------------------------------------------------ targets ---
def enum_targets(snap: Path, synthetic: bool) -> list[dict]:
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    metas = {}
    for shard in sorted(set(idx["weight_map"].values())):
        ds, h = stz.st_header(snap / shard)
        for name, m in h.items():
            if name != "__metadata__":
                metas[name] = (shard, ds, m)
    out = []
    for name in idx["weight_map"]:
        m = NAME_RE.search(name)
        if not m:
            continue
        layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)
        if not synthetic and layer != TARGET_LAYER:
            continue
        shard, ds, meta = metas[name]
        assert meta["dtype"] == "BF16", (name, meta["dtype"])
        out.append({"name": name, "layer": layer, "expert": expert, "proj": proj,
                    "shard": shard, "ds": ds, "shape": tuple(meta["shape"]),
                    "off": tuple(meta["data_offsets"])})
    out.sort(key=lambda t: (t["layer"], t["proj"], t["expert"]))
    if not out:
        die(f"no expert target tensors found under {snap}")
    return out


def read_raw(snap: Path, t: dict) -> bytes:
    with (snap / t["shard"]).open("rb") as f:
        f.seek(t["ds"] + t["off"][0])
        return f.read(t["off"][1] - t["off"][0])


# ------------------------------------------------------------- jsonl helpers ---
def load_rows(jsonl: Path) -> list[dict]:
    """Parse the append-only JSONL. A truncated FINAL line (process killed
    mid-append) is repaired by truncating it away -- via a temp file +
    os.replace so a crash mid-repair can never corrupt completed rows. A
    missing trailing newline is fixed so a future append can never merge
    records. (0014's proven helper + stz.py's atomic-replace pattern.)"""
    if not jsonl.exists():
        return []
    text = jsonl.read_text()
    lines = text.split("\n")
    rows = []
    for i, l in enumerate(lines):
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                good = "\n".join(lines[:i])
                tmp = jsonl.with_name(jsonl.name + ".repair.tmp")
                tmp.write_text(good + ("\n" if good else ""))
                os.replace(tmp, jsonl)
                print(f"[warn] repaired truncated trailing record in {jsonl.name}; "
                      f"its tensor will be reprocessed", flush=True)
                return rows
            die(f"corrupt JSONL record mid-file ({jsonl}, line {i + 1}) -- refusing to guess")
    if text and not text.endswith("\n"):
        with jsonl.open("a") as f:
            f.write("\n")
    return rows


def check_stamp(rows: list[dict], jsonl: Path):
    """Refuse to mix rows written under different accounting constants."""
    bad = [r for r in rows if r.get("acct") != ACCT_STAMP]
    if bad:
        die(f"{len(bad)}/{len(rows)} rows in {jsonl} carry accounting stamp "
            f"{bad[0].get('acct')!r} != current {ACCT_STAMP!r} (accounting "
            f"constants or coder changed) -- move that file aside and re-run")


# ------------------------------------------------------------- quantization ---
def quantize_hist(hist: np.ndarray, n: int) -> np.ndarray:
    """Quantize the 512-bin sym histogram to counts summing to M=4096, every
    present symbol >= 1. Deterministic largest-remainder (ties: ascending sym).
    The shrink branch removes the FULL surplus per pass (vectorized, most
    over-allocated entries first), so hyper-concentrated histograms converge;
    a progress guard replaces the old fixed 64-pass cap."""
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
            order = np.lexsort((nz, -rem))          # largest remainder first
            q[order[:min(d, nz.size)]] += 1
        else:
            surplus = -d
            elig = np.flatnonzero(q > 1)
            if elig.size == 0:
                die("ANS quantization cannot reach the 4096 total")
            order = elig[np.lexsort((nz[elig], tgt[elig] - q[elig]))]  # most over-allocated first
            cap = q[order] - 1                       # each entry can give q-1
            ccap = np.cumsum(cap)
            k = int(np.searchsorted(ccap, surplus))
            if k >= order.size:                      # drain everything eligible
                q[order] = 1
            else:
                if k:
                    q[order[:k]] -= cap[:k]
                q[order[k]] -= surplus - (int(ccap[k - 1]) if k else 0)
    out = np.zeros(512, np.int64)
    out[nz] = q
    return out


def pct_higher(arr: np.ndarray, P: int):
    """Inverse-CDF ('higher') percentile: smallest value v with >= P% of arr <= v.
    Guarantees the budget keeps >= P% of blocks; P=100 -> max. Deterministic."""
    s = np.sort(arr)
    i = min(max(ceil_div(P * s.size, 100) - 1, 0), s.size - 1)
    return s[i]


# ------------------------------------------------------- reference rANS coder ---
def rans_enc_block(syms: list, ql: list, cl: list):
    """Reference encoder (pure Python), the coder named in CODER_SPEC.
    Returns (flush_value = x_final - M, renorm bits in emission order)."""
    x = M
    bits = []
    ap = bits.append
    for s in reversed(syms):
        qq = ql[s]
        t = qq << 1
        while x >= t:
            ap(x & 1)
            x >>= 1
        x = M + cl[s] + (x - qq)      # x//qq == 1 after renorm
    return x - M, bits


def pack_block(flush: int, bits: list) -> tuple[bytes, int]:
    """Serialize: FLUSH_BITS of (x_final - M) MSB-first, then renorm bits in
    REVERSE emission order (decoder pops LIFO). MSB-first byte packing."""
    stream = [(flush >> i) & 1 for i in range(FLUSH_BITS - 1, -1, -1)]
    stream.extend(reversed(bits))
    nbits = len(stream)
    by = bytearray(ceil_div(nbits, 8))
    for i, b in enumerate(stream):
        if b:
            by[i >> 3] |= 0x80 >> (i & 7)
    return bytes(by), nbits


def rans_dec_block(data: bytes, nbits: int, L: int, ql: list, cl: list, s2s: list):
    """Reference decoder. Returns the L symbols, or None on any inconsistency
    (bit starvation, leftover bits, final state != M)."""
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
        slot = x - M                   # x in [M, 2M) => x % M
        s = s2s[slot]
        x = ql[s] + slot - cl[s]
        while x < M:
            if pos >= nbits:
                return None
            x = (x << 1) | ((data[pos >> 3] >> (7 - (pos & 7))) & 1)
            pos += 1
        ap(s)
    if x != M or pos != nbits:
        return None
    return out


def rans_sim_blocks(qm: np.ndarray, cm: np.ndarray) -> np.ndarray:
    """Exact vectorized simulation of rans_enc_block's emitted bit count, one
    row per block (bit-identical arithmetic: same renorm, same flush). qm/cm:
    (nb, W) int64 per-symbol quantized counts / exclusive cumulative counts.
    Returns FLUSH_BITS + renorm bits per row."""
    nbk, W = qm.shape
    x = np.full(nbk, M, np.int64)
    bits = np.full(nbk, FLUSH_BITS, np.int64)
    for j in range(W - 1, -1, -1):
        qq = qm[:, j]
        thr1 = (qq << 1) - 1           # renorm while x > thr1  (== x >= 2q)
        k = BITLEN[x] - BITLEN[thr1]   # shifts to bring x under 2q ...
        np.maximum(k, 0, out=k)
        k += (x >> k) > thr1           # ... exact after one correction
        bits += k
        x = M + cm[:, j] + ((x >> k) - qq)
    return bits


def measured_block_bits(sym: np.ndarray, qv: np.ndarray, cv: np.ndarray,
                        W: int, ql: list, cl: list) -> np.ndarray:
    """Measured emitted bits for every W-weight block of the tensor: vectorized
    simulation for the full blocks, reference encoder for the tail block."""
    n = sym.size
    nb_full = n // W
    parts = []
    if nb_full:
        parts.append(rans_sim_blocks(qv[:nb_full * W].reshape(nb_full, W),
                                     cv[:nb_full * W].reshape(nb_full, W)))
    if n % W:
        fl, tb = rans_enc_block(sym[nb_full * W:].tolist(), ql, cl)
        parts.append(np.array([FLUSH_BITS + len(tb)], np.int64))
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def roundtrip_block(seq: list, expect_bits: int, ql: list, cl: list,
                    s2s: list, ctx: str):
    """Encode -> serialize -> decode one block; die unless the emitted bit
    count equals the accounted bits and the symbols round-trip exactly."""
    fl, bits = rans_enc_block(seq, ql, cl)
    data, nbits = pack_block(fl, bits)
    if nbits != expect_bits:
        die(f"ROUND-TRIP ({ctx}): emitted {nbits} bits != accounted {expect_bits}")
    dec = rans_dec_block(data, nbits, len(seq), ql, cl, s2s)
    if dec != seq:
        die(f"ROUND-TRIP ({ctx}): decoded symbols != original")


def sample_ids(nb: int, rb: np.ndarray, W: int) -> list[int]:
    """Deterministic block sample: extremes of the measured-size distribution
    plus fixed positions (more blocks at small W where they are cheap)."""
    ids = {0, nb - 1, int(np.argmin(rb)), int(np.argmax(rb))}
    if W <= 512:
        ids.add(nb // 3)
        ids.add((2 * nb) // 3)
    return sorted(ids)


# ------------------------------------------------------------- per-tensor ---
def analyze_tensor(raw: bytes, t: dict, synthetic: bool, stats_ref: dict) -> dict:
    u = np.frombuffer(raw, "<u2")
    n = u.size
    R, C = t["shape"]
    assert n == R * C, (t["name"], n, R, C)
    sym = (u >> 7).astype(np.uint16)             # 9-bit sign+exponent (stz's split)
    hist = np.bincount(sym, minlength=512).astype(np.int64)
    p = hist[hist > 0] / n
    H = float(-(p * np.log2(p)).sum())
    floor_bpw = H + 7.0
    mant_bits = pad8(7 * n)                      # 7-bit mantissa plane, verbatim

    # ---- baseline: stz recomputed + parity gate
    plan = stz.plan_regroup(hist, n, R)
    base_bpw = plan["bits"] / n
    if synthetic:   # strong gate: exact equality with the realized encoder
        codec, chunks, st = stz.enc_tensor(raw, t["shape"])
        ref = st["bpw"]
        if codec == 1:
            realized = sum(len(c) for c in chunks) * 8
            if realized != plan["bits"]:
                die(f"PARITY: {t['name']} plan {plan['bits']} != realized {realized} bits")
    else:
        ref = stats_ref[t["name"]]
    diff = abs(round(base_bpw, 4) - ref)
    if diff > PARITY_TOL:
        die(f"PARITY FAILURE on {t['name']}: recomputed {base_bpw:.4f} vs "
            f"stz reference {ref:.4f} (|d|={diff:.4f} > {PARITY_TOL})")

    # ---- quantized 12-bit ANS table (shared by both formats and the coder)
    q = quantize_hist(hist, n)
    nnz = int((hist > 0).sum())
    table_bits = pad8(512 + nnz * 12)            # presence bitmap + 12b (q-1) per sym
    present = np.flatnonzero(q)
    cumq = np.zeros(512, np.int64)
    cumq[present] = np.cumsum(q[present]) - q[present]
    slot2sym = np.repeat(present, q[present])    # M slots -> symbol
    ql, cl, s2s = q.tolist(), cumq.tolist(), slot2sym.tolist()

    cl_i = np.zeros(512)
    cl_i[hist > 0] = -np.log2(hist[hist > 0] / n)
    cl_q = np.zeros(512)
    cl_q[q > 0] = M_LOG2 - np.log2(q[q > 0])
    quant_delta = float((hist * cl_q).sum() / n - H)   # b/w cost of quantization
    per_i = cl_i[sym]                            # per-weight ideal bits (float64)
    per_q = cl_q[sym]                            # per-weight quantized-entropy bits
    qv = q[sym]                                  # per-weight quantized counts (int64)
    cv = cumq[sym]                               # per-weight cumulative counts (int64)

    ideal_blocks, format_b, measured = {}, {}, {}
    rt_blocks = rt_syms = 0
    sha_o, sha_r = hashlib.sha256(), hashlib.sha256()
    for W in WS:
        starts = np.arange(0, n, W, dtype=np.int64)
        ends = np.append(starts[1:], n)
        lens_i = ends - starts                   # int64 block lengths
        s_i = np.add.reduceat(per_i, starts)
        s_q = np.add.reduceat(per_q, starts)
        nb = starts.size

        # -- ideal per-block code lengths: THE unknown this probe measures
        bl_bw = s_i / lens_i
        pc = np.percentile(bl_bw, [50, 90, 95, 97, 99, 100])
        hcounts = np.bincount(
            np.minimum((bl_bw / HIST_BIN).astype(np.int64), HIST_BINS - 1),
            minlength=HIST_BINS)
        ideal_blocks[f"W{W}"] = {
            "nb": int(nb),
            "p50": round(float(pc[0]), 4), "p90": round(float(pc[1]), 4),
            "p95": round(float(pc[2]), 4), "p97": round(float(pc[3]), 4),
            "p99": round(float(pc[4]), 4), "max": round(float(pc[5]), 4),
            "hist_bin_bw": HIST_BIN,
            "hist": {str(i): int(c) for i, c in enumerate(hcounts) if c},
        }

        # -- MEASURED per-block emitted bits of the reference coder
        rb = measured_block_bits(sym, qv, cv, W, ql, cl)
        assert rb.size == nb
        sq_sum = float(s_q.sum())
        measured[f"W{W}"] = {
            "coded_bits": int(rb.sum()),
            "flush_bpw": round(nb * FLUSH_BITS / n, 6),
            "excess_vs_qentropy_bpw": round((int(rb.sum()) - sq_sum) / n, 6),
        }

        # -- round-trip gate on a deterministic sample of blocks
        for i in sample_ids(nb, rb, W):
            s0, e0 = int(starts[i]), int(ends[i])
            seq = sym[s0:e0].tolist()
            roundtrip_block(seq, int(rb[i]), ql, cl, s2s,
                            f"{t['name']} W{W} block {i}")
            rec = ((np.array(seq, dtype="<u2") << 7) | (u[s0:e0] & 0x7F))
            orig = raw[2 * s0:2 * e0]
            if rec.astype("<u2").tobytes() != orig:
                die(f"ROUND-TRIP ({t['name']} W{W} block {i}): reconstructed "
                    f"bytes != original")
            sha_o.update(orig)
            sha_r.update(rec.astype("<u2").tobytes())
            rt_blocks += 1
            rt_syms += len(seq)

        # -- format (b): fixed-stride slots budgeted from MEASURED block bits
        raw_block_bytes = ceil_div(9 * W, 8)     # wholesale-escape slot (raw 9 b/w)
        for P in PS:
            B = int(ceil_div(int(pct_higher(rb, P)), 8))     # byte budget (stride)
            esc = rb > 8 * B
            n_esc = int(esc.sum())
            ov_bytes = max(0, raw_block_bytes - B)
            kept_bits = (nb - n_esc) * 8 * B
            esc_bits = n_esc * 8 * (B + ov_bytes)
            pad_kept = int((8 * B - rb[~esc]).sum())
            pad_esc = int((8 * (B + ov_bytes) - 9 * lens_i[esc]).sum())
            assert pad_kept >= 0 and pad_esc >= 0
            bitmap_bits = pad8(nb)
            rank_bits = RANK_BITS * ceil_div(nb, RANK_GROUP)
            tax_bits = bitmap_bits + rank_bits + table_bits + HEAD_BITS
            sym_bits = kept_bits + esc_bits + tax_bits
            format_b[f"W{W}_P{P}"] = {
                "bpw": round((sym_bits + mant_bits) / n, 6),
                "sym_bits": int(sym_bits), "B_bytes": B, "ov_bytes": int(ov_bytes),
                "nb": int(nb), "esc_blocks": n_esc,
                "esc_frac": round(n_esc / nb, 6),
                "pad_bits": pad_kept + pad_esc, "tax_bits": int(tax_bits),
            }

    # ---- format (a): 4096-sym superblocks, 32 interleaved lanes (measured)
    n_sb_full = n // SB
    payload_parts, lane_bits_full, tail_lane_bits = [], None, []
    if n_sb_full:
        lq = (qv[:n_sb_full * SB].reshape(n_sb_full, SB // LANES, LANES)
              .transpose(0, 2, 1).reshape(n_sb_full * LANES, SB // LANES))
        lc = (cv[:n_sb_full * SB].reshape(n_sb_full, SB // LANES, LANES)
              .transpose(0, 2, 1).reshape(n_sb_full * LANES, SB // LANES))
        lane_bits_full = rans_sim_blocks(np.ascontiguousarray(lq),
                                         np.ascontiguousarray(lc))
        payload_parts.append(lane_bits_full.reshape(n_sb_full, LANES).sum(1))
    tail_start = n_sb_full * SB
    if tail_start < n:
        for l in range(LANES):
            seq = sym[tail_start:][l::LANES]
            if seq.size == 0:
                continue
            fl, tb = rans_enc_block(seq.tolist(), ql, cl)
            tail_lane_bits.append(FLUSH_BITS + len(tb))
        payload_parts.append(np.array([sum(tail_lane_bits)], np.int64))
    payload = (payload_parts[0] if len(payload_parts) == 1
               else np.concatenate(payload_parts))
    sb_bits = payload + (-payload % 8)           # byte-align each superblock
    n_sb = int(sb_bits.size)
    lanes_total = n_sb_full * LANES + len(tail_lane_bits)
    lane_flush_bits = lanes_total * FLUSH_BITS   # inside the payload, NOT tax
    index_bits = L2_BITS * n_sb + L1_BITS * ceil_div(n_sb, L1_GROUP)
    pad_bits_a = int((sb_bits - payload).sum())
    tax_bits_a = index_bits + table_bits + HEAD_BITS   # side structures only
    sym_bits_a = int(sb_bits.sum()) + tax_bits_a
    # reconciliation: printed components must sum to the charged total
    assert sym_bits_a == int(payload.sum()) + pad_bits_a + tax_bits_a
    format_a = {
        "bpw": round((sym_bits_a + mant_bits) / n, 6),
        "sym_bits": int(sym_bits_a), "n_sb": n_sb,
        "payload_bits": int(payload.sum()),
        "lane_flush_bits": int(lane_flush_bits), "lanes": int(lanes_total),
        "pad_bits": pad_bits_a, "index_bits": int(index_bits),
        "tax_bits": int(tax_bits_a),
    }

    # -- round-trip gate on sampled format-(a) lanes
    rt_lanes = 0
    if lane_bits_full is not None:
        for g in sorted({0, n_sb_full * LANES - 1}):
            k, l = divmod(g, LANES)
            seq = sym[k * SB:(k + 1) * SB][l::LANES].tolist()
            roundtrip_block(seq, int(lane_bits_full[g]), ql, cl, s2s,
                            f"{t['name']} format(a) lane {g}")
            rt_lanes += 1
    if tail_lane_bits:
        seq = sym[tail_start:][0::LANES].tolist()
        roundtrip_block(seq, int(tail_lane_bits[0]), ql, cl, s2s,
                        f"{t['name']} format(a) tail lane 0")
        rt_lanes += 1

    sha_ok = sha_o.digest() == sha_r.digest()
    if not sha_ok:
        die(f"ROUND-TRIP ({t['name']}): SHA-256 mismatch over sampled spans")

    return {
        "name": t["name"], "layer": t["layer"], "expert": t["expert"],
        "proj": t["proj"], "n": int(n), "R": int(R), "C": int(C),
        "acct": ACCT_STAMP,
        "H_sym": round(H, 6), "floor_bpw": round(floor_bpw, 6),
        "mant_bits": int(mant_bits),
        "baseline": {"bpw": round(base_bpw, 6), "bits": int(plan["bits"]),
                     "variant": plan["variant"], "b": plan.get("b"), "k": plan.get("k")},
        "parity": {"ref_bpw": ref, "abs_diff": round(diff, 6)},
        "quant": {"nnz": nnz, "table_bits": int(table_bits),
                  "delta_bpw": round(quant_delta, 6)},
        "measured": measured,
        "roundtrip": {"blocks": rt_blocks, "block_syms": rt_syms,
                      "lanes": rt_lanes, "bits_ok": True, "sha256_ok": bool(sha_ok)},
        "ideal_blocks": ideal_blocks,
        "format_b": format_b,
        "format_a": format_a,
    }


# ------------------------------------------------------------------- summary ---
def summarize(tg: list[dict], jsonl: Path, summaryp: Path, synthetic: bool):
    rows = load_rows(jsonl)
    check_stamp(rows, jsonl)
    rec = {}
    for r in rows:                                # first record per name wins
        rec.setdefault(r["name"], r)
    names = [t["name"] for t in tg]
    miss = [nm for nm in names if nm not in rec]
    if miss:
        die(f"summary requires all tensors done; {len(miss)} missing "
            f"(first: {miss[0]}) -- re-invoke without --summary to resume")
    n_tot = sum(rec[nm]["n"] for nm in names)
    wsum = lambda f: sum(f(rec[nm]) for nm in names)

    base_w = wsum(lambda r: r["baseline"]["bits"]) / n_tot
    ref_w = wsum(lambda r: r["parity"]["ref_bpw"] * r["n"]) / n_tot
    floor_w = wsum(lambda r: r["floor_bpw"] * r["n"]) / n_tot
    quant_w = wsum(lambda r: r["quant"]["delta_bpw"] * r["n"]) / n_tot
    parity_max = max(rec[nm]["parity"]["abs_diff"] for nm in names)
    if not synthetic and abs(ref_w - STZ_TARGET_BPW) > REF_WEIGHTED_TOL:
        die(f"weighted stz reference {ref_w:.4f} != canonical {STZ_TARGET_BPW} "
            f"(tol {REF_WEIGHTED_TOL}) -- wrong target set?")

    rt_blocks = wsum(lambda r: r["roundtrip"]["blocks"])
    rt_lanes = wsum(lambda r: r["roundtrip"]["lanes"])
    rt_ok = all(rec[nm]["roundtrip"]["bits_ok"] and rec[nm]["roundtrip"]["sha256_ok"]
                for nm in names)

    def agg_b(key):
        sym = wsum(lambda r: r["format_b"][key]["sym_bits"])
        mant = wsum(lambda r: r["mant_bits"])
        nb = wsum(lambda r: r["format_b"][key]["nb"])
        esc = wsum(lambda r: r["format_b"][key]["esc_blocks"])
        pad = wsum(lambda r: r["format_b"][key]["pad_bits"])
        tax = wsum(lambda r: r["format_b"][key]["tax_bits"])
        return {"bpw": (sym + mant) / n_tot, "esc_frac": esc / nb,
                "pad_bpw": pad / n_tot, "tax_bpw": tax / n_tot}

    grid = {f"W{W}_P{P}": agg_b(f"W{W}_P{P}") for W in WS for P in PS}
    a_sym = wsum(lambda r: r["format_a"]["sym_bits"])
    a = {"bpw": (a_sym + wsum(lambda r: r["mant_bits"])) / n_tot,
         "esc_frac": 0.0,
         "pad_bpw": wsum(lambda r: r["format_a"]["pad_bits"]) / n_tot,
         "tax_bpw": wsum(lambda r: r["format_a"]["tax_bits"]) / n_tot,
         "lane_flush_bpw": wsum(lambda r: r["format_a"]["lane_flush_bits"]) / n_tot}

    # measured coder excess over quantized entropy per W (numel-weighted)
    excess = {f"W{W}": wsum(lambda r: r["measured"][f"W{W}"]["excess_vs_qentropy_bpw"]
                            * r["n"]) / n_tot for W in WS}

    # ideal-tail diagnostics per W (numel-weighted means of per-tensor stats)
    tails = {}
    for W in WS:
        g = lambda f: wsum(lambda r: f(r["ideal_blocks"][f"W{W}"]) * r["n"]) / n_tot
        tails[f"W{W}"] = {k: round(g(lambda ib, kk=k: ib[kk]), 4)
                          for k in ("p50", "p90", "p95", "p97", "p99", "max")}

    target = base_w if synthetic else STZ_TARGET_BPW
    fus_keys = [f"W{W}_P{P}" for W in WS if W <= FUSIBLE_W_MAX for P in PS]
    best_fus_key = min(fus_keys, key=lambda k: grid[k]["bpw"])
    best_fus_bpw = grid[best_fus_key]["bpw"]
    best_grid_key = min(grid, key=lambda k: grid[k]["bpw"])
    candidates = {**grid, "A_4096x32": a}
    best_any_key = min(candidates, key=lambda k: candidates[k]["bpw"])
    best_any_bpw = candidates[best_any_key]["bpw"]

    # gates and verdict are keyed on the tile-credible fixed-stride grid ONLY;
    # W > FUSIBLE_W_MAX and format (a) are storage-leaning brackets
    g1_fus = best_fus_bpw < target
    g2_fus = best_fus_bpw <= floor_w + G2_SLACK
    g1_any = best_any_bpw < target
    g2_any = best_any_bpw <= floor_w + G2_SLACK
    if g1_fus and g2_fus:
        verdict = (f"CONFIRMED (fixed-stride, W<={FUSIBLE_W_MAX}, "
                   f"measured coder bits)")
    elif g1_fus:
        verdict = f"positive (G1 only, fixed-stride W<={FUSIBLE_W_MAX})"
    elif g1_any:
        verdict = ("positive (storage-leaning only: needs W>"
                   f"{FUSIBLE_W_MAX} or superblock format (a) -- not tile-fusible)")
    else:
        verdict = "FALSIFIED at this operating point"
    if not rt_ok:
        verdict = "PROVISIONAL: " + verdict     # round-trip gate incomplete
    scope = ("synthetic smoke -- carries no evidential weight" if synthetic
             else f"layer {TARGET_LAYER} only; cross-layer transfer unvalidated")

    mode = "SYNTHETIC (smoke only)" if synthetic else "REAL layer-27"
    print(f"\n=== candidate 0015 block-granular tile codes -- summary [{mode}] ===")
    print(f"targets: {len(names)} tensors, {n_tot:,} params; "
          f"parity gate OK (max |d bpw| vs stz reference = {parity_max:.6f})")
    print(f"stz recomputed {base_w:.4f} b/w | stz G1 target "
          f"{'n/a (synthetic)' if synthetic else STZ_TARGET_BPW} | "
          f"floor H(sym)+7 = {floor_w:.4f} b/w | G2 bar = {floor_w + G2_SLACK:.4f} b/w")
    print(f"coder: {CODER_SPEC}")
    print(f"round-trip gate: {rt_blocks} blocks + {rt_lanes} lanes across "
          f"{len(names)} tensors, bits==accounted and SHA-256 exact: "
          f"{'PASS' if rt_ok else 'FAIL'}")
    print(f"12-bit ANS quantization delta (stated separately): +{quant_w:.4f} b/w")
    print("measured coder excess over quantized entropy (b/w, incl. flush): "
          + "  ".join(f"W{W}={excess[f'W{W}']:.4f}" for W in WS))

    print("\nideal per-block code length (b/w, numel-weighted mean of per-tensor stats):")
    hdr = f"{'W':>7}{'p50':>9}{'p90':>9}{'p95':>9}{'p97':>9}{'p99':>9}{'max':>9}{'p99/p50':>9}"
    print(hdr); print("-" * len(hdr))
    for W in WS:
        tt = tails[f"W{W}"]
        ratio = tt["p99"] / tt["p50"] if tt["p50"] else float("inf")
        print(f"{W:>7}{tt['p50']:>9.3f}{tt['p90']:>9.3f}{tt['p95']:>9.3f}"
              f"{tt['p97']:>9.3f}{tt['p99']:>9.3f}{tt['max']:>9.3f}{ratio:>9.3f}")

    print("\nrealized formats (MEASURED coder bits; all side costs charged; "
          "+ mantissa pad8(7n)):")
    print("columns: save_stz = target - bpw (positive = win); "
          "over_floor = bpw - floor (positive = above floor)")
    hdr = (f"{'format':>14}{'bpw':>10}{'save_stz':>10}{'over_floor':>11}"
           f"{'esc%':>8}{'pad b/w':>10}{'tax b/w':>10}{'  fusible':>9}")
    print(hdr); print("-" * len(hdr))
    for W in WS:
        for P in PS:
            k = f"W{W}_P{P}"
            v = grid[k]
            fus = "yes" if W <= FUSIBLE_W_MAX else "no"
            print(f"{k:>14}{v['bpw']:>10.4f}{target - v['bpw']:>+10.4f}"
                  f"{v['bpw'] - floor_w:>+11.4f}{100 * v['esc_frac']:>8.3f}"
                  f"{v['pad_bpw']:>10.4f}{v['tax_bpw']:>10.4f}{fus:>9}")
    print(f"{'A_4096x32':>14}{a['bpw']:>10.4f}{target - a['bpw']:>+10.4f}"
          f"{a['bpw'] - floor_w:>+11.4f}{0.0:>8.3f}{a['pad_bpw']:>10.4f}"
          f"{a['tax_bpw']:>10.4f}{'no':>9}")

    print("\nper-W best (best P per W; sequential decode inside a kept block = W syms):")
    hdr = (f"{'W':>7}{'best P':>8}{'bpw':>10}{'save_stz':>10}"
           f"{'G1':>6}{'fusible (W<=' + str(FUSIBLE_W_MAX) + ')':>18}")
    print(hdr); print("-" * len(hdr))
    per_w_best = {}
    for W in WS:
        bk = min((f"W{W}_P{P}" for P in PS), key=lambda k: grid[k]["bpw"])
        v = grid[bk]["bpw"]
        per_w_best[f"W{W}"] = {"key": bk, "bpw": round(v, 6),
                               "g1_pass": bool(v < target),
                               "fusible": bool(W <= FUSIBLE_W_MAX)}
        print(f"{W:>7}{bk.split('_')[1]:>8}{v:>10.4f}{target - v:>+10.4f}"
              f"{'PASS' if v < target else 'FAIL':>6}"
              f"{'yes' if W <= FUSIBLE_W_MAX else 'no':>18}")

    print(f"\nbest fixed-stride (fusible, W<={FUSIBLE_W_MAX}): {best_fus_key} "
          f"at {best_fus_bpw:.4f} b/w")
    print(f"best storage-leaning bracket: {best_any_key} at {best_any_bpw:.4f} b/w "
          f"(full grid + format (a); NOT the fusible headline)")
    print(f"G1 (< {'recomputed stz' if synthetic else STZ_TARGET_BPW} b/w, fixed-stride "
          f"W<={FUSIBLE_W_MAX}): {'PASS' if g1_fus else 'FAIL'} "
          f"(d = {target - best_fus_bpw:+.4f}; any-format: "
          f"{'PASS' if g1_any else 'FAIL'} at {target - best_any_bpw:+.4f})")
    print(f"G2 (<= floor + {G2_SLACK}, fixed-stride W<={FUSIBLE_W_MAX}): "
          f"{'PASS' if g2_fus else 'FAIL'} "
          f"(d = {best_fus_bpw - floor_w:+.4f} vs bar {G2_SLACK})")
    print(f"verdict: {verdict}  [{scope}]")
    proj_pre = proj_fus = None
    if not synthetic:
        proj_fus = WHOLE_MODEL_BPW - EXPERT_FRAC * (base_w - best_fus_bpw)
        proj_pre = WHOLE_MODEL_BPW - EXPERT_FRAC * (base_w - grid[PREREG_KEY]["bpw"])
        print(f"projected whole-model, pre-registered {PREREG_KEY}: {proj_pre:.4f} b/w "
              f"(honest expectation; cross-layer transfer unvalidated)")
        print(f"projected whole-model, best fixed-stride {best_fus_key}: "
              f"{proj_fus:.4f} b/w (best-of-grid SELECTED ON THIS SAME layer-27 "
              f"set -- selection-optimistic; cross-layer transfer unvalidated)")

    summary = {
        "mode": "synthetic" if synthetic else "real",
        "scope": scope,
        "acct_stamp": ACCT_STAMP, "acct": ACCT,
        "targets": len(names), "total_params": int(n_tot),
        "parity_max_abs_diff": parity_max,
        "baseline_recomputed_bpw": round(base_w, 6),
        "baseline_ref_weighted_bpw": round(ref_w, 6),
        "stz_target_bpw": STZ_TARGET_BPW,
        "whole_model_baseline_bpw": WHOLE_MODEL_BPW, "expert_frac": EXPERT_FRAC,
        "floor_bpw_weighted": round(floor_w, 6),
        "quant_delta_bpw_weighted": round(quant_w, 6),
        "coder": CODER_SPEC,
        "coder_excess_vs_qentropy_bpw": {k: round(v, 6) for k, v in excess.items()},
        "roundtrip": {"blocks": int(rt_blocks), "lanes": int(rt_lanes),
                      "all_ok": bool(rt_ok)},
        "ideal_tails_bw": tails,
        "format_b": {k: {kk: round(vv, 6) for kk, vv in v.items()}
                     for k, v in grid.items()},
        "format_a": {kk: round(vv, 6) for kk, vv in a.items()},
        "per_w_best": per_w_best,
        "fusible_w_max": FUSIBLE_W_MAX,
        "best": {
            "fixed_stride_fusible": {"key": best_fus_key,
                                     "bpw": round(best_fus_bpw, 6)},
            "fixed_stride_any_w": {"key": best_grid_key,
                                   "bpw": round(grid[best_grid_key]["bpw"], 6)},
            "storage_leaning_any": {"key": best_any_key,
                                    "bpw": round(best_any_bpw, 6)},
        },
        "gates": {
            "G1_vs": "recomputed baseline (synthetic)" if synthetic else STZ_TARGET_BPW,
            "keyed_on": f"fixed-stride grid, W<={FUSIBLE_W_MAX}",
            "G1_pass": bool(g1_fus),
            "G1_delta_bpw": round(target - best_fus_bpw, 6),
            "G1_pass_any_format": bool(g1_any),
            "G1_delta_any_format_bpw": round(target - best_any_bpw, 6),
            "G2_bar_bpw": round(floor_w + G2_SLACK, 6), "G2_pass": bool(g2_fus),
            "G2_delta_vs_floor_bpw": round(best_fus_bpw - floor_w, 6),
            "G2_pass_any_format": bool(g2_any),
        },
        "verdict": verdict,
        "projected_whole_model_prereg_bpw":
            None if synthetic else round(proj_pre, 6),
        "projected_whole_model_best_fixed_stride_bpw":
            None if synthetic else round(proj_fus, 6),
        "projection_caveat": ("pre-registered = W128_P97 chosen before any real "
                              "run; 'best' is selected on this same layer-27 set "
                              "(selection-optimistic); cross-layer transfer "
                              "unvalidated either way"),
        "accounting_note": (
            "per-weight totals include the pad8(7n) mantissa plane; block sizes "
            "are MEASURED emitted bits of the named coder (" + CODER_SPEC + "), "
            "not an entropy bound -- no modeled component remains; format (b) "
            "charges kept blocks a full B-byte fixed-stride slot, escaped blocks "
            "max(B, ceil(9W/8)) bytes wholesale-raw, pad8(nb) escape bitmap, "
            "u32/512-blocks rank directory, pad8(512+nnz*12) quantized-table "
            "(12-bit field stores q-1, range 0..4095, so q=4096 fits), 32B "
            "header; pad_bits includes BOTH kept-slot slack and escaped-slot "
            "slack; format (a) payload = measured lane bits incl. 32x12-bit "
            "lane flushes (reported separately as lane_flush_bits, inside the "
            "payload); format (a) tax = offset index + table + header only "
            "(no double count; reconciliation asserted per tensor)"),
    }
    summaryp.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written: {summaryp}")


# ---------------------------------------------------------------------- main ---
def run(a, snap: Path, jsonl: Path, summaryp: Path):
    tg = enum_targets(snap, a.synthetic)
    names = [t["name"] for t in tg]
    if a.summary:
        return summarize(tg, jsonl, summaryp, a.synthetic)

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
            die(f"{len(missing)} targets absent from stz_tensor_stats.jsonl "
                f"(first: {missing[0]})")

    prior = load_rows(jsonl)
    check_stamp(prior, jsonl)
    done = {r["name"] for r in prior}
    t0, processed = time.time(), 0
    for i, t in enumerate(tg):
        if t["name"] in done:
            continue
        if a.limit and processed >= a.limit:
            break
        if time.time() - t0 > a.budget_s:
            print(f"\n[budget] {a.budget_s:.0f}s reached after {processed} tensors -- "
                  f"progress saved, re-invoke to resume.", flush=True)
            sys.exit(0)
        raw = read_raw(snap, t)
        rec = analyze_tensor(raw, t, a.synthetic, stats_ref)
        with jsonl.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        done.add(t["name"])
        processed += 1
        if processed % 16 == 0:
            print(f"[{i + 1}/{len(tg)}] {processed} tensors, "
                  f"{time.time() - t0:.0f}s", flush=True)

    if a.limit and processed >= a.limit and len(done) < len(tg):
        print(f"\n[limit] {a.limit} tensors this invocation -- re-invoke to resume.")
        sys.exit(0)

    if a.synthetic:   # the strong parity gate must actually have been exercised
        n_reg = sum(1 for r in load_rows(jsonl)
                    if r["baseline"].get("variant") == "regroup")
        if n_reg == 0:
            die("synthetic strong parity gate never exercised: 0 tensors chose "
                "the regroup codec")
        print(f"[gate] synthetic strong parity gate exercised on {n_reg}/{len(tg)} tensors")

    print(f"\nall {len(done)}/{len(tg)} tensors done ({time.time() - t0:.0f}s)")
    summarize(tg, jsonl, summaryp, a.synthetic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run against the synthetic tiny snapshot (smoke)")
    ap.add_argument("--summary", action="store_true",
                    help="summary + gates only (requires all tensors done)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max tensors this invocation (0 = no cap)")
    ap.add_argument("--budget-s", type=float, default=420.0,
                    help="soft wall-clock budget; exits cleanly when exceeded")
    a = ap.parse_args()

    snap = SYN_SNAP if a.synthetic else REAL_SNAP
    tag = "_synthetic" if a.synthetic else ""
    ART.mkdir(parents=True, exist_ok=True)
    jsonl = ART / f"blockcodes_results{tag}.jsonl"
    summaryp = ART / f"blockcodes_summary{tag}.json"

    # exclusive run lock (concurrent appends would interleave JSONL records)
    lockp = ART / f"blockcodes{tag}.lock"
    try:
        fd = os.open(lockp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = lockp.read_text().strip() or "?"
        except OSError:
            holder = "?"
        die(f"lock file {lockp} exists (written by pid {holder}); another "
            f"invocation may be running -- if none is, delete the lock and retry")
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
