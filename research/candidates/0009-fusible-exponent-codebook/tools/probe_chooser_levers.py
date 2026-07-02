"""probe_chooser_levers.py -- E-scope stz tool: pre-probe of three CHOOSER-SCALE levers.

NOT a new candidate. Each lever is a per-tensor OPTION a future .stz min-envelope
chooser could adopt where it wins, so the deliverable per lever is its
adoption-aware envelope gain vs the REALIZED stz baseline, exactly priced, on the
MoE expert tensors (128 experts x {up,down}_proj) of three layers (1 anomalous
early / 13 mid / 27 canonical late; see NEXT_DIRECTIONS.md "New leads").

Baseline: stz.plan_regroup (imported, never reimplemented). Parity gate: the
recomputed baseline must match stz_tensor_stats.jsonl EXACTLY per tensor
(bpw, b, k, n_esc, codec) -- loud abort otherwise. In --synthetic mode the gate
is stronger: stz.enc_tensor is run and realized bits must equal plan bits
exactly, and the run aborts if zero tensors exercised the regroup path.

The three levers (each priced separately; joint = per-tensor min over all):

  V1 per-row(-group) second-level escape k. stz picks ONE second-level escape
     width k per tensor; the up_proj escape mask is row-overdispersed
     (Fano ~2.3, candidate 0014 forensics), so let k vary per row group
     (group in {1, 8, 64} rows). One per-tensor second-level ordered table of
     L_cap = 2^cap - 1 syms is stored; a group choosing k uses its length-
     (2^k - 1) PREFIX, so all groups share one table. Side stream: the chosen
     k index, ceil(log2(len(KS))) = 3 bits per group. Exact cost = min over
     first-level b in {2,3,4,5} x cap in {0,3,4,5,6} of
       pad8(n*b) + pad8(3*ng) + pad8(sum_g k_g*n_esc_g) + pad8(9*n_raw)
       + pad8(R*pw) + pad8(R*pw2) + pad8(7n) + (K + L_cap)*16 + head + 7*u64,
     the per-group k_g argmin of k*n_esc_g + 9*n_raw_g(k) (k=0 -> 9*n_esc_g),
     i.e. priced through the (k-b)-style conversion rule, never "9 bits saved
     per escape". Row prefix tables (pw / pw2) are kept, so nothing is
     untransmitted -- but note the random-access caveat: baseline stz finds
     row r's escape codes at bit k*prefix[r] in O(1), while with per-group k
     the code bit offset of row r is sum over previous groups of k_g*cnt_g,
     an O(ng) weighted prefix scan over the stored count prefix + k side
     stream (or a load-time-derived offset table). A strictly-O(1) STORED
     per-group code-offset table is NOT charged here; it would cost roughly
     _pw(bits)*ng extra bits (~0.01 b/w at group=1 on a 5M-param tensor) --
     the same order as V1's expected ceiling, so V1 gains at group=1 should
     be read with that fusibility-honesty caveat.

  V2 per-column(-group) BASE re-centering as a chooser option. Subtract an
     address-derived per-column-group (group in {1, 16, 64} columns) base
     exponent -- the group's modal 8-bit exponent, stored as one int8 per
     group -- from the exponent field (mod 256, sign untouched), then run the
     UNMODIFIED stz chooser (plan_regroup) on the transformed syms. Fires only
     if column exponent distributions are shifted copies rather than
     shape-varying. Exact cost = plan_regroup(transformed) + 16 (g field)
     + 64 (side-stream length prefix) + pad8(8*ng).

  V3 fractional-m repricing of the index plane. Non-power-of-2 alphabet
     M in {3..32}: codes 0..M-2 = top-(M-1) syms, code M-1 = escape, packed by
     grouped radix coding: G = max digits with M^G <= 2^64, groups restart at
     every row (random access survives at group granularity, w <= 64 bits per
     group word), full groups cost w = bitlen(M^G - 1) bits, the row-remainder
     group costs bitlen(M^rem - 1); group padding charged exactly. Escape
     side: stz's own second-level k in {0,3,4,5,6} envelope, mirrored
     stream-for-stream. This prices the 2026-07 vetting salvage claim
     (+0.03-0.05 b/w on concentrated tensors) against the REALIZED baseline.

All three cost models are WRITER-VERIFIED: enc_v1/enc_v2/enc_v3 serialize a
real record whose byte length must equal the priced bits exactly, and
dec_v1/dec_v2/dec_v3 reconstruct the tensor bytes from the record alone,
SHA-256 bit-exact -- the standard stz itself was held to. At least one
configuration per lever is round-trip-proven per run (the max-gain tensor;
`adopted` is recorded -- a lever priced but never decoded is not evidence).

Envelope pricing is adoption-aware and exact by construction: every lever
option carries only per-tensor side costs (no cross-tensor tables), so the
per-tensor min over {baseline, options} with all side costs fully charged IS
the achievable serialized total. Lever composition (e.g. V2+V3 in one tensor)
is NOT priced; the joint envelope is the min over single-lever options.

Pre-registered bar (report all three regardless): a lever enters the chooser
recommendation only if its decay-weighted model-wide projection is
>= +0.01 b/w. Projection: per-layer expert-numel-weighted gain, interpolated
across the measured layers (1, 13, 27) over the 23 real MoE layers
(early-layer gains decay to ~0 by layer 13 -- the 0014 pattern), times the
experts' share of whole-model BF16 numel (computed from stz_tensor_stats).

Deterministic, pure numpy, one tensor in RAM at a time, per-tensor resumable
JSONL (append + fsync, dedupe-on-load, truncated-tail self-repair).

Usage:
  uv run python probe_chooser_levers.py --synthetic      # smoke on the fake snapshot
  uv run python probe_chooser_levers.py --layer 27       # real run (resumable)
  uv run python probe_chooser_levers.py --layer 1
  uv run python probe_chooser_levers.py --layer 13
  uv run python probe_chooser_levers.py --summary        # aggregate all layers + projection
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, re, struct, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE.parent))
import stz  # noqa: E402  -- reuse plan_regroup/_pw/pack_width/unpack_width/enc_tensor/dec_tensor

REAL_SNAP = REPO / "models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot"
SYN_SNAP = REPO / "models/synthetic/nemotron_tiny/hf_snapshot"
STATS_JSONL = HERE.parents[1] / "tests/artifacts/stz/stz_tensor_stats.jsonl"
ART = HERE.parents[1] / "tests/artifacts/chooser_levers"

KS = (0, 3, 4, 5, 6)          # stz's exact second-level escape envelope
BS = (2, 3, 4, 5)             # stz's exact first-level index-bit envelope
V1_GROUPS = (1, 8, 64)        # rows per k-choice group
V2_GROUPS = (1, 16, 64)       # columns per base group
V3_MS = tuple(range(3, 33))   # fractional alphabet sizes (powers of 2 included: envelope)
KSEL_BITS = 3                 # ceil(log2(len(KS))) bits per group in the V1 side stream
HEAD_V1 = 30 * 8              # '<BBHIIQQBB': b,cap,group,R,C,n,n_esc,pw,pw2
HEAD_V3 = 30 * 8              # '<BBHIIQQBB': M,k,G,R,C,n,n_esc,pw,pw2
V2_SIDE_HEAD = 16 + 64        # u16 g + u64 base-stream length prefix
GATE_MODEL_BPW = 0.01         # pre-registered recommendation bar (model-wide b/w)
MEASURE_LAYERS = (1, 13, 27)  # planned real measurement layers
LEVERS = ("v1", "v2", "v3")

assert (1 << KSEL_BITS) >= len(KS)
NAME_RE = re.compile(r"backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up|down)_proj\.weight$")
KS_ARR = np.array(KS, np.int64)

pad8 = lambda bits: bits + (-bits % 8)   # per-stream byte-alignment, stz's rule


def die(msg: str):
    print(f"\n!!! ABORT: {msg}", flush=True)
    sys.exit(1)


# ------------------------------------------------------------------ targets ---
def target_names(snap: Path, synthetic: bool, layer: int) -> list[str]:
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    out = []
    for name in idx["weight_map"]:
        m = NAME_RE.search(name)
        if m and (synthetic or int(m.group(1)) == layer):
            out.append(name)
    return sorted(out)


def enum_targets(snap: Path, synthetic: bool, layer: int) -> list[dict]:
    wanted = set(target_names(snap, synthetic, layer))
    if not wanted:
        die(f"no expert target tensors for layer {layer} under {snap}")
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    shards = sorted({idx["weight_map"][n] for n in wanted})
    out = []
    for shard in shards:
        ds, h = stz.st_header(snap / shard)
        for name, meta in h.items():
            if name not in wanted:
                continue
            m = NAME_RE.search(name)
            assert meta["dtype"] == "BF16", (name, meta["dtype"])
            assert len(meta["shape"]) == 2, (name, meta["shape"])
            out.append({"name": name, "layer": int(m.group(1)), "expert": int(m.group(2)),
                        "proj": m.group(3), "shard": shard, "ds": ds,
                        "shape": tuple(meta["shape"]), "off": tuple(meta["data_offsets"])})
    out.sort(key=lambda t: (t["layer"], t["proj"], t["expert"]))
    if len(out) != len(wanted):
        die(f"index lists {len(wanted)} targets but shard headers yielded {len(out)}")
    return out


def read_raw(snap: Path, t: dict) -> bytes:
    with (snap / t["shard"]).open("rb") as f:
        f.seek(t["ds"] + t["off"][0])
        return f.read(t["off"][1] - t["off"][0])


# ---------------------------------------------------- variable-width packing ---
def pack_varwidth(vals: np.ndarray, widths: np.ndarray) -> bytes:
    """MSB-first concatenation of vals[i] in widths[i] bits, byte-padded."""
    total = int(widths.sum())
    if total == 0:
        return b""
    bits = np.zeros(total, np.uint8)
    ends = np.cumsum(widths.astype(np.int64))
    starts = ends - widths
    for w in np.unique(widths):
        w = int(w)
        if w == 0:
            continue
        m = widths == w
        v = vals[m].astype(np.uint64)
        sh = np.arange(w - 1, -1, -1, dtype=np.uint64)
        bb = ((v[:, None] >> sh[None, :]) & np.uint64(1)).astype(np.uint8)
        pos = starts[m][:, None] + np.arange(w, dtype=np.int64)[None, :]
        bits[pos.ravel()] = bb.ravel()
    return np.packbits(bits).tobytes()


def unpack_varwidth(buf: bytes, widths: np.ndarray) -> np.ndarray:
    """Inverse of pack_varwidth; returns uint64 values."""
    total = int(widths.sum())
    out = np.zeros(widths.size, np.uint64)
    if total == 0:
        return out
    bits = np.unpackbits(np.frombuffer(buf, np.uint8), count=total)
    ends = np.cumsum(widths.astype(np.int64))
    starts = ends - widths
    for w in np.unique(widths):
        w = int(w)
        if w == 0:
            continue
        m = widths == w
        pos = starts[m][:, None] + np.arange(w, dtype=np.int64)[None, :]
        bb = bits[pos.ravel()].reshape(-1, w).astype(np.uint64)
        weights = np.uint64(1) << np.arange(w - 1, -1, -1, dtype=np.uint64)
        out[m] = (bb * weights[None, :]).sum(1, dtype=np.uint64)
    return out


def row_prefix_bytes(counts: np.ndarray, R: int, width: int) -> bytes:
    """stz's per-row cumulative-count prefix table."""
    prefix = np.zeros(R, np.uint64)
    if R > 1:
        prefix[1:] = np.cumsum(counts.astype(np.uint64))[:-1]
    return stz.pack_width(prefix, width)


def check_prefix(counts: np.ndarray, R: int, prefix: np.ndarray, what: str):
    chk = np.zeros(R, np.uint64)
    if R > 1:
        chk[1:] = np.cumsum(counts.astype(np.uint64))[:-1]
    assert np.array_equal(chk, prefix.astype(np.uint64)), f"{what} prefix corrupt"


# ------------------------------------------------------------------ lever V1 ---
def sym_order(hist: np.ndarray) -> np.ndarray:
    """stz's codebook order: descending count, ties ascending sym."""
    return np.lexsort((np.arange(512), -hist))


def v1_row_buckets(sym2d: np.ndarray, hist: np.ndarray, b: int):
    """Escape structure for first-level b: per-row counts of escapes bucketed by
    second-level rank (<7, <15, <31, <63, rest) -- everything v1 pricing needs."""
    R, C = sym2d.shape
    order = sym_order(hist)
    K = (1 << b) - 1
    topk = order[:K]
    esc_hist = hist.copy()
    esc_hist[topk] = 0
    order2 = sym_order(esc_hist)
    rank = np.empty(512, np.int64)
    rank[order2] = np.arange(512)
    intop = np.zeros(512, bool)
    intop[topk] = True
    esc_mask = ~intop[sym2d]
    pos = np.flatnonzero(esc_mask.ravel())
    rows = pos // C
    ranks = rank[sym2d.ravel()[pos]]
    bucket = np.digitize(ranks, (7, 15, 31, 63))
    rb5 = np.bincount(rows * 5 + bucket, minlength=R * 5).reshape(R, 5)
    return {"order": order, "order2": order2, "rank": rank, "topk": topk,
            "rb5": rb5, "pos": pos, "rows": rows, "ranks": ranks,
            "n_esc": int(pos.size)}


def v1_group_agg(rb5: np.ndarray, R: int, group: int) -> np.ndarray:
    ng = -(-R // group)
    padrows = ng * group - R
    if padrows:
        rb5 = np.vstack([rb5, np.zeros((padrows, 5), rb5.dtype)])
    return rb5.reshape(ng, group, 5).sum(1)


def v1_choice(grp5: np.ndarray, cap: int):
    """Per-group k argmin under cap. Returns (k index per group, total code bits,
    total raw escapes). Ties prefer k=0 (argmin first index; allowed[0] = 0)."""
    n_esc_g = grp5.sum(1)
    b7 = grp5[:, 0]
    b15 = b7 + grp5[:, 1]
    b31 = b15 + grp5[:, 2]
    b63 = b31 + grp5[:, 3]
    below = np.stack([np.zeros_like(b7), b7, b15, b31, b63])
    costs = np.stack([9 * n_esc_g,
                      3 * n_esc_g + 9 * (n_esc_g - b7),
                      4 * n_esc_g + 9 * (n_esc_g - b15),
                      5 * n_esc_g + 9 * (n_esc_g - b31),
                      6 * n_esc_g + 9 * (n_esc_g - b63)])
    allowed = [0] + [i for i in range(1, 5) if KS[i] <= cap]
    ki = np.array(allowed)[np.argmin(costs[allowed], 0)]
    kv = KS_ARR[ki]
    bits_codes = int((kv * n_esc_g).sum())
    n_raw = int((n_esc_g - below[ki, np.arange(ki.size)]).sum())
    return ki, kv, bits_codes, n_raw


def v1_total(n: int, R: int, ng: int, b: int, cap: int,
             bits_codes: int, n_raw: int, n_esc: int) -> int:
    K = (1 << b) - 1
    Lcap = (1 << cap) - 1
    pw, pw2 = stz._pw(n_esc), stz._pw(n_raw)
    return (pad8(n * b) + pad8(KSEL_BITS * ng) + pad8(bits_codes) + pad8(9 * n_raw)
            + pad8(R * pw) + pad8(R * pw2) + pad8(n * 7)
            + (K + Lcap) * 16 + HEAD_V1 + 7 * 64)


def v1_price(sym2d: np.ndarray, hist: np.ndarray, n: int, R: int, C: int,
             group: int, bstructs: dict) -> dict:
    best = None
    for b in BS:
        st = bstructs[b]
        grp5 = v1_group_agg(st["rb5"], R, group)
        ng = grp5.shape[0]
        for cap in KS:
            ki, kv, bits_codes, n_raw = v1_choice(grp5, cap)
            total = v1_total(n, R, ng, b, cap, bits_codes, n_raw, st["n_esc"])
            if best is None or total < best["bits"]:
                kh = {str(KS[i]): int(c) for i, c in enumerate(np.bincount(ki, minlength=5)) if c}
                best = {"bits": int(total), "bpw": round(total / n, 6), "b": b, "cap": cap,
                        "ng": int(ng), "n_esc": st["n_esc"], "n_raw": int(n_raw),
                        "k_hist": kh}
    return best


def enc_v1(raw: bytes, shape, group: int, b: int, cap: int) -> bytes:
    """Serialize one tensor in the V1 format; asserts written bits == v1_total."""
    u = np.frombuffer(raw, "<u2")
    n = u.size
    R, C = shape
    sym = (u >> 7).astype(np.uint16)
    mant = (u & 0x7F).astype(np.uint8)
    hist = np.bincount(sym, minlength=512)
    st = v1_row_buckets(sym.reshape(R, C), hist, b)
    K = (1 << b) - 1
    Lcap = (1 << cap) - 1
    topk = st["topk"].astype(np.uint16)
    table2 = st["order2"][:Lcap].astype(np.uint16)
    codemap = np.full(512, K, np.uint8)
    codemap[topk] = np.arange(K, dtype=np.uint8)
    idx = codemap[sym]
    pos, rows, ranks, n_esc = st["pos"], st["rows"], st["ranks"], st["n_esc"]
    grp5 = v1_group_agg(st["rb5"], R, group)
    ng = grp5.shape[0]
    ki, kv, bits_codes, n_raw_plan = v1_choice(grp5, cap)
    plan_bits = v1_total(n, R, ng, b, cap, bits_codes, n_raw_plan, n_esc)

    kk = kv[rows // group]                       # per-escape second-level width
    Lk = (np.int64(1) << kk) - 1                 # k=0 -> 0 -> everything raw
    raw_sel = ranks >= Lk
    coded = kk > 0
    codes = np.minimum(ranks, Lk)[coded].astype(np.uint64)
    esc_codes = pack_varwidth(codes, kk[coded])
    assert len(esc_codes) * 8 == pad8(bits_codes)
    esc_syms = sym[pos]
    n_raw = int(raw_sel.sum())
    assert n_raw == n_raw_plan
    pw, pw2 = stz._pw(n_esc), stz._pw(n_raw)
    streams = [stz.pack_width(idx, b),
               stz.pack_width(ki.astype(np.uint8), KSEL_BITS),
               esc_codes,
               stz.pack_width(esc_syms[raw_sel], 9),
               row_prefix_bytes(np.bincount(rows, minlength=R), R, pw),
               row_prefix_bytes(np.bincount(rows[raw_sel], minlength=R), R, pw2),
               stz.pack_width(mant, 7)]
    head = struct.pack("<BBHIIQQBB", b, cap, group, R, C, n, n_esc, pw, pw2)
    head += topk.astype("<u2").tobytes() + table2.astype("<u2").tobytes()
    buf = bytearray(head)
    for s in streams:
        buf += struct.pack("<Q", len(s)) + s
    assert len(buf) * 8 == plan_bits, (len(buf) * 8, plan_bits)
    return bytes(buf)


def dec_v1(buf: bytes) -> bytes:
    b, cap, group, R, C, n, n_esc, pw, pw2 = struct.unpack_from("<BBHIIQQBB", buf, 0)
    off = 30
    K = (1 << b) - 1
    Lcap = (1 << cap) - 1
    topk = np.frombuffer(buf, "<u2", K, off); off += K * 2
    table2 = np.frombuffer(buf, "<u2", Lcap, off); off += Lcap * 2

    def stream():
        nonlocal off
        (ln,) = struct.unpack_from("<Q", buf, off)
        off += 8
        s = buf[off:off + ln]
        off += ln
        return s

    idx = stz.unpack_width(stream(), n, b)
    ng = -(-R // group)
    ki = stz.unpack_width(stream(), ng, KSEL_BITS)
    kv = KS_ARR[ki.astype(np.int64)]
    esc_mask = idx == K
    pos = np.flatnonzero(esc_mask)
    assert pos.size == n_esc
    rows = pos // C
    kk = kv[rows // group]
    Lk = (np.int64(1) << kk) - 1
    coded = kk > 0
    codes = unpack_varwidth(stream(), kk[coded])
    raw_from_coded = codes == Lk[coded].astype(np.uint64)
    raw_sel = ~coded
    idxc = np.flatnonzero(coded)
    raw_sel[idxc[raw_from_coded]] = True
    esc_raw = stz.unpack_width(stream(), int(raw_sel.sum()), 9)
    prefix = stz.unpack_width(stream(), R, pw)
    prefix2 = stz.unpack_width(stream(), R, pw2)
    esc_syms = np.empty(n_esc, np.uint16)
    esc_syms[idxc[~raw_from_coded]] = table2[codes[~raw_from_coded].astype(np.int64)]
    esc_syms[raw_sel] = esc_raw.astype(np.uint16)
    cb = np.zeros(K + 1, np.uint16)
    cb[:K] = topk
    sym = cb[idx]
    sym[pos] = esc_syms
    check_prefix(np.bincount(rows, minlength=R), R, prefix, "v1 escape")
    check_prefix(np.bincount(rows[raw_sel], minlength=R), R, prefix2, "v1 raw-escape")
    mant = stz.unpack_width(stream(), n, 7)
    assert off == len(buf), (off, len(buf))
    out = ((sym.astype(np.uint16) << 7) | mant).astype("<u2").tobytes()
    assert len(out) == n * 2
    return out


# ------------------------------------------------------------------ lever V2 ---
def v2_transform(sym2d: np.ndarray, g: int):
    """Per-column-group modal-exponent base (uint8) and the re-centered syms."""
    R, C = sym2d.shape
    ng = -(-C // g)
    gid = (np.arange(C) // g).astype(np.int64)
    keys = gid[None, :] * 512 + sym2d.astype(np.int64)
    hg = np.bincount(keys.ravel(), minlength=ng * 512).reshape(ng, 2, 256)
    base = hg.sum(1).argmax(1).astype(np.uint8)     # modal exponent per group
    baseb = base[gid].astype(np.int64)
    e2 = ((sym2d & 0xFF).astype(np.int64) - baseb[None, :]) & 0xFF
    sym2 = ((sym2d & 0x100).astype(np.int64) | e2).astype(np.uint16)
    return base, sym2


def v2_price(sym2d: np.ndarray, n: int, R: int, g: int) -> dict:
    base, sym2 = v2_transform(sym2d, g)
    plan = stz.plan_regroup(np.bincount(sym2.ravel(), minlength=512), n, R)
    side = V2_SIDE_HEAD + pad8(8 * base.size)
    total = plan["bits"] + side
    return {"bits": int(total), "bpw": round(total / n, 6), "g": g, "ng": int(base.size),
            "inner_variant": plan["variant"], "inner_b": plan.get("b"),
            "inner_k": plan.get("k"), "n_esc": int(plan.get("n_esc", 0))}


def enc_v2(raw: bytes, shape, g: int) -> bytes:
    u = np.frombuffer(raw, "<u2")
    R, C = shape
    base, sym2 = v2_transform(((u >> 7).astype(np.uint16)).reshape(R, C), g)
    u2 = ((sym2.ravel().astype(np.uint16) << 7) | (u & 0x7F)).astype("<u2")
    codec, chunks, _ = stz.enc_tensor(u2.tobytes(), shape)
    assert codec == 1, "v2 round-trip requires a regroup inner plan (raw16 is never adopted)"
    base_bytes = base.astype(np.uint8).tobytes()
    rec = struct.pack("<H", g) + struct.pack("<Q", len(base_bytes)) + base_bytes
    rec += b"".join(chunks)
    return rec


def dec_v2(buf: bytes, nbytes: int) -> bytes:
    (g,) = struct.unpack_from("<H", buf, 0)
    (ln,) = struct.unpack_from("<Q", buf, 2)
    base = np.frombuffer(buf, np.uint8, ln, 10)
    off = 10 + ln
    # peek R, n from the embedded stz record header '<BBIQQBB'
    (R,) = struct.unpack_from("<I", buf, off + 2)
    (n,) = struct.unpack_from("<Q", buf, off + 6)
    C = n // R
    assert -(-C // g) == ln, "v2 base-table length inconsistent with g"
    f = io.BytesIO(buf[off:])
    u2 = np.frombuffer(stz.dec_tensor(f, nbytes), "<u2")
    assert f.read(1) == b"", "v2 trailing bytes after inner record"
    gid = (np.arange(C) // g).astype(np.int64)
    baseb = base[gid].astype(np.int64)
    e2 = ((u2 >> 7) & 0xFF).astype(np.int64).reshape(R, C)
    e = (e2 + baseb[None, :]) & 0xFF
    u = (u2 & np.uint16(0x807F)) | (e.astype(np.uint16).ravel() << 7)
    return u.astype("<u2").tobytes()


# ------------------------------------------------------------------ lever V3 ---
def v3_G(M: int) -> int:
    G = 1
    while M ** (G + 1) <= (1 << 64):
        G += 1
    return G


def v3_geometry(M: int, C: int):
    G = v3_G(M)
    w = (M ** G - 1).bit_length()
    n_full = C // G
    rem = C - n_full * G
    wl = (M ** rem - 1).bit_length() if rem else 0
    return G, w, n_full, rem, wl


def v3_price(hist: np.ndarray, n: int, R: int, C: int) -> dict:
    order = sym_order(hist)
    counts = hist[order]
    cum = np.cumsum(counts)
    best = None
    for M in V3_MS:
        Kc = M - 1
        n_esc = int(n - cum[Kc - 1])
        G, w, n_full, rem, wl = v3_geometry(M, C)
        idx_bits = R * (n_full * w + wl)
        pw = stz._pw(n_esc)
        esc_counts = counts[Kc:]
        for k in KS:
            if k == 0:
                L = 0
                esc_bits = pad8(n_esc * 9) + pad8(R * pw)
                ns = 4
            else:
                L = (1 << k) - 1
                n_raw = n_esc - int(esc_counts[:L].sum())
                pw2 = stz._pw(n_raw)
                esc_bits = pad8(n_esc * k) + pad8(n_raw * 9) + pad8(R * pw) + pad8(R * pw2)
                ns = 6
            total = pad8(idx_bits) + esc_bits + pad8(n * 7) + (Kc + L) * 16 + HEAD_V3 + ns * 64
            if best is None or total < best["bits"]:
                best = {"bits": int(total), "bpw": round(total / n, 6), "M": M, "G": G,
                        "k": k, "n_esc": n_esc}
    return best


def enc_v3(raw: bytes, shape, M: int, k: int) -> bytes:
    u = np.frombuffer(raw, "<u2")
    n = u.size
    R, C = shape
    sym = (u >> 7).astype(np.uint16)
    mant = (u & 0x7F).astype(np.uint8)
    hist = np.bincount(sym, minlength=512)
    order = sym_order(hist)
    Kc = M - 1
    topk = order[:Kc].astype(np.uint16)
    codemap = np.full(512, Kc, np.uint8)
    codemap[topk] = np.arange(Kc, dtype=np.uint8)
    idx = codemap[sym]
    idx2d = idx.reshape(R, C)
    G, w, n_full, rem, wl = v3_geometry(M, C)
    powers = np.array([M ** i for i in range(G)], dtype=np.uint64)
    parts = []
    if n_full:
        parts.append((idx2d[:, :n_full * G].reshape(R, n_full, G).astype(np.uint64)
                      * powers[None, None, :]).sum(2, dtype=np.uint64))
    if rem:
        vr = (idx2d[:, n_full * G:].astype(np.uint64) * powers[None, :rem]).sum(1, dtype=np.uint64)
        parts.append(vr[:, None])
    vals = np.concatenate(parts, 1).ravel() if len(parts) > 1 else parts[0].ravel()
    wrow = ([w] * n_full) + ([wl] if rem else [])
    widths = np.tile(np.array(wrow, np.int64), R)
    idx_stream = pack_varwidth(vals, widths)
    assert len(idx_stream) * 8 == pad8(R * (n_full * w + wl))

    esc_mask = idx == Kc
    esc_syms = sym[esc_mask]
    n_esc = int(esc_syms.size)
    pw = stz._pw(n_esc)
    rows = np.flatnonzero(esc_mask) // C
    streams = [idx_stream]
    if k:
        esc_hist = hist.copy()
        esc_hist[topk] = 0
        order2 = sym_order(esc_hist)
        L = (1 << k) - 1
        codebook2 = order2[:L].astype(np.uint16)
        codemap2 = np.full(512, L, np.uint16)
        codemap2[codebook2] = np.arange(L, dtype=np.uint16)
        esc_codes = codemap2[esc_syms]
        raw_in_esc = esc_codes == L
        esc_raw = esc_syms[raw_in_esc]
        pw2 = stz._pw(int(esc_raw.size))
        streams += [stz.pack_width(esc_codes, k), stz.pack_width(esc_raw, 9),
                    row_prefix_bytes(np.bincount(rows, minlength=R), R, pw),
                    row_prefix_bytes(np.bincount(rows[raw_in_esc], minlength=R), R, pw2)]
    else:
        pw2 = 0
        codebook2 = np.empty(0, np.uint16)
        streams += [stz.pack_width(esc_syms, 9),
                    row_prefix_bytes(np.bincount(rows, minlength=R), R, pw)]
    streams.append(stz.pack_width(mant, 7))
    head = struct.pack("<BBHIIQQBB", M, k, G, R, C, n, n_esc, pw, pw2)
    head += topk.astype("<u2").tobytes() + codebook2.astype("<u2").tobytes()
    buf = bytearray(head)
    for s in streams:
        buf += struct.pack("<Q", len(s)) + s
    return bytes(buf)


def dec_v3(buf: bytes) -> bytes:
    M, k, G, R, C, n, n_esc, pw, pw2 = struct.unpack_from("<BBHIIQQBB", buf, 0)
    off = 30
    Kc = M - 1
    topk = np.frombuffer(buf, "<u2", Kc, off); off += Kc * 2
    L = (1 << k) - 1 if k else 0
    codebook2 = np.frombuffer(buf, "<u2", L, off); off += L * 2

    def stream():
        nonlocal off
        (ln,) = struct.unpack_from("<Q", buf, off)
        off += 8
        s = buf[off:off + ln]
        off += ln
        return s

    w = (M ** G - 1).bit_length()
    n_full = C // G
    rem = C - n_full * G
    wl = (M ** rem - 1).bit_length() if rem else 0
    wrow = ([w] * n_full) + ([wl] if rem else [])
    widths = np.tile(np.array(wrow, np.int64), R)
    vals = unpack_varwidth(stream(), widths).reshape(R, n_full + (1 if rem else 0))
    idx2d = np.empty((R, C), np.uint8)
    if n_full:
        v = vals[:, :n_full].copy()
        d = np.empty((R, n_full, G), np.uint8)
        for i in range(G):
            d[..., i] = (v % M).astype(np.uint8)
            v //= M
        idx2d[:, :n_full * G] = d.reshape(R, n_full * G)
    if rem:
        v = vals[:, -1].copy()
        dr = np.empty((R, rem), np.uint8)
        for i in range(rem):
            dr[:, i] = (v % M).astype(np.uint8)
            v //= M
        idx2d[:, n_full * G:] = dr
    idx = idx2d.ravel()
    cb = np.zeros(Kc + 1, np.uint16)
    cb[:Kc] = topk
    sym = cb[idx]
    esc_mask = idx == Kc
    pos = np.flatnonzero(esc_mask)
    assert pos.size == n_esc
    rows = pos // C
    if k:
        esc_codes = stz.unpack_width(stream(), n_esc, k)
        in2 = esc_codes < L
        esc_raw = stz.unpack_width(stream(), int((~in2).sum()), 9)
        esc_syms = np.empty(n_esc, np.uint16)
        esc_syms[in2] = codebook2[esc_codes[in2].astype(np.int64)]
        esc_syms[~in2] = esc_raw.astype(np.uint16)
        prefix = stz.unpack_width(stream(), R, pw)
        prefix2 = stz.unpack_width(stream(), R, pw2)
        check_prefix(np.bincount(rows, minlength=R), R, prefix, "v3 escape")
        check_prefix(np.bincount(rows[~in2], minlength=R), R, prefix2, "v3 raw-escape")
    else:
        esc_syms = stz.unpack_width(stream(), n_esc, 9).astype(np.uint16)
        prefix = stz.unpack_width(stream(), R, pw)
        check_prefix(np.bincount(rows, minlength=R), R, prefix, "v3 escape")
    sym[pos] = esc_syms
    mant = stz.unpack_width(stream(), n, 7)
    assert off == len(buf), (off, len(buf))
    out = ((sym.astype(np.uint16) << 7) | mant).astype("<u2").tobytes()
    assert len(out) == n * 2
    return out


# ------------------------------------------------------------- jsonl helpers ---
def load_rows(jsonl: Path) -> list[dict]:
    """Append-only JSONL with truncated-tail self-repair and first-wins dedupe."""
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
                jsonl.write_text(good + ("\n" if good else ""))
                print(f"[warn] repaired truncated trailing record in {jsonl.name}", flush=True)
                break
            die(f"corrupt JSONL record mid-file ({jsonl}, line {i + 1}) -- refusing to guess")
    seen, out = set(), []
    for r in rows:
        key = (r.get("stage"), r.get("name") if r.get("stage") == "levers" else r.get("lever"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def append(jsonl: Path, rec: dict):
    with jsonl.open("a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------ pricing driver ---
def price_tensor(raw: bytes, t: dict) -> dict:
    u = np.frombuffer(raw, "<u2")
    n = u.size
    R, C = t["shape"]
    sym = (u >> 7).astype(np.uint16)
    sym2d = sym.reshape(R, C)
    hist = np.bincount(sym, minlength=512)
    plan = stz.plan_regroup(hist, n, R)
    bstructs = {b: v1_row_buckets(sym2d, hist, b) for b in BS}
    v1 = {f"g{g}": v1_price(sym2d, hist, n, R, C, g, bstructs) for g in V1_GROUPS}
    v2 = {f"g{g}": v2_price(sym2d, n, R, g) for g in V2_GROUPS}
    v3 = v3_price(hist, n, R, C)
    base_bits = int(plan["bits"])
    opts = {"baseline": base_bits}
    opts.update({f"v1_{k}": v["bits"] for k, v in v1.items()})
    opts.update({f"v2_{k}": v["bits"] for k, v in v2.items()})
    opts["v3"] = v3["bits"]
    pick = min(opts, key=lambda k: (opts[k], k != "baseline"))
    return {
        "stage": "levers", "name": t["name"], "layer": t["layer"], "expert": t["expert"],
        "proj": t["proj"], "n": int(n), "R": int(R), "C": int(C),
        "baseline": {"bits": base_bits, "bpw": round(base_bits / n, 6),
                     "variant": plan["variant"], "b": plan.get("b"), "k": plan.get("k"),
                     "n_esc": int(plan.get("n_esc", 0))},
        "v1": v1, "v2": v2, "v3": v3,
        "chooser": {"pick": pick, "bits": opts[pick],
                    "gain_bits": base_bits - opts[pick],
                    "gain_bpw": round((base_bits - opts[pick]) / n, 6)},
    }, plan


def parity_gate(rec: dict, plan: dict, raw: bytes, t: dict, stats_ref: dict | None):
    """Real: exact match vs the recorded realized stz stats (bpw/b/k/n_esc/codec).
    Synthetic: stz.enc_tensor realized bits must equal plan bits exactly."""
    n = rec["n"]
    if stats_ref is not None:
        ref = stats_ref[t["name"]]
        if ref["codec"] == 0:
            ok = plan["variant"] == "raw16" and ref["bpw"] == 16.0
        else:
            ok = (plan["variant"] == "regroup"
                  and round(plan["bits"] / n, 4) == ref["bpw"]
                  and plan["b"] == ref["b"] and plan["k"] == ref["k"]
                  and int(plan["n_esc"]) == ref["n_esc"])
        if not ok:
            die(f"PARITY FAILURE on {t['name']}: recomputed "
                f"{{bpw {round(plan['bits'] / n, 4)}, b {plan.get('b')}, k {plan.get('k')}, "
                f"n_esc {plan.get('n_esc')}}} != recorded {ref}")
        return {"mode": "stats-exact", "ref_bpw": ref["bpw"], "match": True}
    codec, chunks, st = stz.enc_tensor(raw, t["shape"])
    if codec == 1:
        realized = sum(len(c) for c in chunks) * 8
        if realized != plan["bits"]:
            die(f"PARITY FAILURE on {t['name']}: plan {plan['bits']} != realized {realized} bits")
    else:
        if plan["variant"] != "raw16":
            die(f"PARITY FAILURE on {t['name']}: enc chose raw16, plan chose regroup")
    return {"mode": "enc-exact", "ref_bpw": st["bpw"], "match": True,
            "regroup": codec == 1}


# --------------------------------------------------------------- round trips ---
def lever_best(rec: dict, lever: str):
    """(bits, config) of the lever's best option for one tensor record.
    For v2, only regroup inner plans are encodable (raw16 is never adopted)."""
    if lever == "v1":
        key = min(rec["v1"], key=lambda k: rec["v1"][k]["bits"])
        v = rec["v1"][key]
        return v["bits"], {"group": int(key[1:]), "b": v["b"], "cap": v["cap"]}
    if lever == "v2":
        cands = {k: v for k, v in rec["v2"].items() if v["inner_variant"] == "regroup"}
        if not cands:
            return None, None
        key = min(cands, key=lambda k: cands[k]["bits"])
        return cands[key]["bits"], {"g": cands[key]["g"]}
    v = rec["v3"]
    return v["bits"], {"M": v["M"], "k": v["k"]}


def roundtrip_lever(lever: str, rec: dict, raw: bytes, shape) -> dict:
    bits, cfg = lever_best(rec, lever)
    sha0 = hashlib.sha256(raw).hexdigest()
    if lever == "v1":
        blob = enc_v1(raw, shape, cfg["group"], cfg["b"], cfg["cap"])
        dec = dec_v1(blob)
    elif lever == "v2":
        blob = enc_v2(raw, shape, cfg["g"])
        dec = dec_v2(blob, len(raw))
    else:
        blob = enc_v3(raw, shape, cfg["M"], cfg["k"])
        dec = dec_v3(blob)
    if len(blob) * 8 != bits:
        die(f"ROUNDTRIP {lever} on {rec['name']}: serialized {len(blob) * 8} bits "
            f"!= priced {bits} (cost model is not exact)")
    if hashlib.sha256(dec).hexdigest() != sha0:
        die(f"ROUNDTRIP {lever} on {rec['name']}: reconstruction is not bit-exact")
    gain = rec["baseline"]["bits"] - bits
    return {"stage": "roundtrip", "lever": lever, "name": rec["name"], "config": cfg,
            "bits": bits, "bits_match_plan": True, "sha256_match": True,
            "gain_bits": int(gain), "adopted": bool(gain > 0)}


def run_roundtrips(tg: list[dict], rows: list[dict], jsonl: Path, snap: Path):
    lever_rows = {r["name"]: r for r in rows if r["stage"] == "levers"}
    done = {r["lever"] for r in rows if r["stage"] == "roundtrip"}
    by_name = {t["name"]: t for t in tg}
    for lever in LEVERS:
        if lever in done:
            continue
        best_name, best_gain = None, None
        for name, r in lever_rows.items():
            bits, _ = lever_best(r, lever)
            if bits is None:
                continue
            gain = r["baseline"]["bits"] - bits
            if best_gain is None or gain > best_gain:
                best_name, best_gain = name, gain
        if best_name is None:
            die(f"no encodable configuration found for lever {lever} "
                f"(v2: every inner plan was raw16?)")
        t = by_name[best_name]
        rt = roundtrip_lever(lever, lever_rows[best_name], read_raw(snap, t), t["shape"])
        append(jsonl, rt)
        print(f"[rt {lever}] {best_name}: {rt['config']} bits==plan, SHA-256 OK, "
              f"gain={rt['gain_bits']} bits, adopted={rt['adopted']}", flush=True)


# ------------------------------------------------------------------- summary ---
def moe_layer_numels(synthetic: bool):
    """{MoE layer: expert numel} and total BF16 numel, for the projection weights."""
    lays, tot = {}, 0
    if synthetic:
        for shard in sorted(set(json.loads(
                (SYN_SNAP / "model.safetensors.index.json").read_text())["weight_map"].values())):
            _, h = stz.st_header(SYN_SNAP / shard)
            for name, m in h.items():
                if name == "__metadata__" or m["dtype"] != "BF16":
                    continue
                numel = int(np.prod(m["shape"]))
                tot += numel
                mm = NAME_RE.search(name)
                if mm:
                    lays[int(mm.group(1))] = lays.get(int(mm.group(1)), 0) + numel
        return lays, tot
    if not STATS_JSONL.exists():
        die(f"missing {STATS_JSONL} (needed for projection weights)")
    for line in STATS_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["dtype"] != "BF16":
            continue
        numel = r["nbytes"] // 2
        tot += numel
        m = NAME_RE.search(r["name"])
        if m:
            lays[int(m.group(1))] = lays.get(int(m.group(1)), 0) + numel
    return lays, tot


def layer_summary(lever_rows: list[dict]) -> dict:
    n_tot = sum(r["n"] for r in lever_rows)
    base_bits = sum(r["baseline"]["bits"] for r in lever_rows)
    out = {"tensors": len(lever_rows), "params": int(n_tot),
           "baseline_bpw": round(base_bits / n_tot, 6), "levers": {}, "variants": {}}
    variant_keys = ([("v1", f"g{g}") for g in V1_GROUPS]
                    + [("v2", f"g{g}") for g in V2_GROUPS] + [("v3", None)])
    for lever, vk in variant_keys:
        bits = env = adopt = adopt_n = 0
        for r in lever_rows:
            vb = r[lever][vk]["bits"] if vk else r[lever]["bits"]
            bits += vb
            env += min(r["baseline"]["bits"], vb)
            if vb < r["baseline"]["bits"]:
                adopt += 1
                adopt_n += r["n"]
        key = f"{lever}_{vk}" if vk else lever
        out["variants"][key] = {
            "bpw": round(bits / n_tot, 6),
            "envelope_gain_bpw": round((base_bits - env) / n_tot, 6),
            "adopt_tensors": adopt, "adopt_frac": round(adopt / len(lever_rows), 4),
            "adopt_numel_frac": round(adopt_n / n_tot, 4)}
    joint_env = 0
    for lever in LEVERS:
        env = adopt = adopt_n = 0
        for r in lever_rows:
            lb = (min(v["bits"] for v in r[lever].values()) if lever != "v3"
                  else r["v3"]["bits"])
            env += min(r["baseline"]["bits"], lb)
            if lb < r["baseline"]["bits"]:
                adopt += 1
                adopt_n += r["n"]
        out["levers"][lever] = {
            "envelope_bpw": round(env / n_tot, 6),
            "envelope_gain_bpw": round((base_bits - env) / n_tot, 6),
            "adopt_tensors": adopt, "adopt_frac": round(adopt / len(lever_rows), 4),
            "adopt_numel_frac": round(adopt_n / n_tot, 4)}
    for r in lever_rows:
        opts = [r["baseline"]["bits"], r["v3"]["bits"]]
        opts += [v["bits"] for v in r["v1"].values()]
        opts += [v["bits"] for v in r["v2"].values()]
        joint_env += min(opts)
    out["joint"] = {"envelope_bpw": round(joint_env / n_tot, 6),
                    "envelope_gain_bpw": round((base_bits - joint_env) / n_tot, 6)}
    return out


def summarize(synthetic: bool):
    snap = SYN_SNAP if synthetic else REAL_SNAP
    layers_data = {}   # layer -> (summary dict, roundtrip records)
    if synthetic:
        files = [(1, ART / "levers_synthetic.jsonl")]
    else:
        files = []
        for p in sorted(ART.glob("levers_layer*.jsonl")):
            files.append((int(p.stem.replace("levers_layer", "")), p))
    for layer, p in files:
        if not p.exists():
            continue
        rows = load_rows(p)
        lever_rows = [r for r in rows if r["stage"] == "levers"]
        expected = set(target_names(snap, synthetic, layer))
        have = {r["name"] for r in lever_rows}
        if not expected <= have:
            print(f"[warn] layer {layer}: {len(expected - have)} tensors not yet priced -- "
                  f"excluded from the summary (re-invoke to resume)", flush=True)
            continue
        lever_rows = [r for r in lever_rows if r["name"] in expected]
        rts = [r for r in rows if r["stage"] == "roundtrip"]
        layers_data[layer] = (layer_summary(lever_rows), rts)
    if not layers_data:
        die("no complete layer results to summarize -- run the probe first")

    mode = "SYNTHETIC (smoke only -- projections carry no evidential weight)" \
        if synthetic else "REAL"
    print(f"\n=== chooser-levers pre-probe -- summary [{mode}] ===")
    for layer in sorted(layers_data):
        ls, rts = layers_data[layer]
        print(f"\n-- layer {layer}: {ls['tensors']} tensors, {ls['params']:,} params, "
              f"baseline {ls['baseline_bpw']:.4f} b/w --")
        hdr = f"{'option':<12}{'bpw':>10}{'env gain b/w':>14}{'adopted':>12}"
        print(hdr)
        print("-" * len(hdr))
        for key, v in ls["variants"].items():
            print(f"{key:<12}{v['bpw']:>10.4f}{v['envelope_gain_bpw']:>+14.6f}"
                  f"{v['adopt_tensors']:>7}/{ls['tensors']}")
        for lever in LEVERS:
            lv = ls["levers"][lever]
            print(f"{lever + ' (best)':<12}{lv['envelope_bpw']:>10.4f}"
                  f"{lv['envelope_gain_bpw']:>+14.6f}{lv['adopt_tensors']:>7}/{ls['tensors']}")
        print(f"{'joint':<12}{ls['joint']['envelope_bpw']:>10.4f}"
              f"{ls['joint']['envelope_gain_bpw']:>+14.6f}")
        for r in rts:
            print(f"  roundtrip {r['lever']}: {r['name'].split('.experts.')[-1]} "
                  f"{r['config']} bits==plan={r['bits_match_plan']} "
                  f"sha={r['sha256_match']} adopted={r['adopted']}")

    # decay-weighted model-wide projection over the MoE layers
    lays, tot_bf16 = moe_layer_numels(synthetic)
    measured = sorted(layers_data)
    moe_layers = np.array(sorted(lays), np.float64)
    weights = np.array([lays[int(l)] for l in moe_layers], np.float64)
    projection = {}
    for what in list(LEVERS) + ["joint"]:
        g = np.array([(layers_data[l][0]["levers"][what]["envelope_gain_bpw"]
                       if what != "joint" else layers_data[l][0]["joint"]["envelope_gain_bpw"])
                      for l in measured], np.float64)
        interp = np.interp(moe_layers, np.array(measured, np.float64), g)
        model_gain = float((interp * weights).sum() / tot_bf16)
        projection[what] = {
            "measured_layers": measured,
            "measured_gains_bpw": [round(float(x), 6) for x in g],
            "model_wide_gain_bpw": round(model_gain, 6),
            "passes_bar": bool(model_gain >= GATE_MODEL_BPW)}
    complete = set(measured) >= set(MEASURE_LAYERS)
    rt_evidence = {}
    for lever in LEVERS:
        recs = [r for layer in layers_data for r in layers_data[layer][1]
                if r["lever"] == lever]
        rt_evidence[lever] = {
            "roundtrips": len(recs),
            "adopted_roundtrip": any(r["adopted"] for r in recs),
            "note": ("OK" if any(r["adopted"] for r in recs) else
                     "priced but no ADOPTED configuration decoded yet -- not full evidence")}

    print(f"\n-- decay-weighted model-wide projection "
          f"({len(moe_layers)} MoE layers, expert/total BF16 = "
          f"{weights.sum() / tot_bf16:.4f}) --")
    if not complete:
        print(f"   [provisional: measured layers {measured} of planned {list(MEASURE_LAYERS)}]")
    for what, pr in projection.items():
        bar = "PASSES" if pr["passes_bar"] else "below"
        print(f"  {what:<6} gains@{measured} = {pr['measured_gains_bpw']} "
              f"-> model-wide {pr['model_wide_gain_bpw']:+.6f} b/w  "
              f"[{bar} the >= +{GATE_MODEL_BPW} bar]")
    for lever, ev in rt_evidence.items():
        print(f"  evidence {lever}: {ev['roundtrips']} roundtrip(s), "
              f"adopted={ev['adopted_roundtrip']} ({ev['note']})")

    summary = {
        "mode": "synthetic" if synthetic else "real",
        "scope": ("synthetic smoke -- no evidential weight" if synthetic else
                  f"measured layers {measured}"
                  + ("" if complete else f" (planned {list(MEASURE_LAYERS)}; provisional)")),
        "baseline": "realized stz (plan_regroup, parity-gated exact per tensor)",
        "gate_model_bpw": GATE_MODEL_BPW,
        "layers": {str(l): layers_data[l][0] for l in sorted(layers_data)},
        "projection": projection,
        "roundtrip_evidence": rt_evidence,
        "notes": [
            "adoption-aware envelope: all lever side costs are per-tensor, so the "
            "per-tensor min with full side-cost charging is the achievable total",
            "joint envelope = per-tensor min over single-lever options; lever "
            "composition (e.g. V2+V3) is not priced",
            "V1 priced through the per-group k*n_esc + 9*n_raw conversion rule, "
            "never 9 or 16 bits per escape avoided",
            "V1 random-access caveat: per-group k makes row r's escape-code bit "
            "offset an O(ng) weighted prefix (sum k_g*cnt_g) derivable from the "
            "stored count prefix + k side stream, not baseline stz's O(1) "
            "k*prefix[r]; a strictly-O(1) stored per-group code-offset table is "
            "NOT charged and would cost ~_pw(bits)*ng extra bits (~0.01 b/w at "
            "group=1 on a 5M-param tensor) -- same order as V1's expected gain, "
            "so read group=1 V1 gains with that caveat (or assume a load-time-"
            "derived offset table)",
        ],
    }
    outp = ART / ("summary_synthetic.json" if synthetic else "summary.json")
    outp.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written: {outp}")
    return summary


# ---------------------------------------------------------------------- main ---
def run(a):
    snap = SYN_SNAP if a.synthetic else REAL_SNAP
    ART.mkdir(parents=True, exist_ok=True)
    jsonl = ART / ("levers_synthetic.jsonl" if a.synthetic else f"levers_layer{a.layer}.jsonl")
    tg = enum_targets(snap, a.synthetic, a.layer)
    names = [t["name"] for t in tg]

    stats_ref = None
    if not a.synthetic:
        if not STATS_JSONL.exists():
            die(f"missing parity reference {STATS_JSONL}")
        stats_ref = {}
        for line in STATS_JSONL.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["name"] in set(names):
                    stats_ref[r["name"]] = r
        missing = [n for n in names if n not in stats_ref]
        if missing:
            die(f"{len(missing)} targets absent from stz_tensor_stats.jsonl "
                f"(first: {missing[0]})")

    rows = load_rows(jsonl)
    done = {r["name"] for r in rows if r["stage"] == "levers"}
    t0, processed, n_regroup = time.time(), 0, 0
    for i, t in enumerate(tg):
        if t["name"] in done:
            continue
        if a.limit and processed >= a.limit:
            print(f"[limit] {processed} tensors this invocation -- re-invoke to resume")
            return
        if time.time() - t0 > a.budget_s:
            print(f"[budget] {processed} tensors in {time.time() - t0:.0f}s -- "
                  f"re-invoke to resume")
            return
        raw = read_raw(snap, t)
        rec, plan = price_tensor(raw, t)
        gate = parity_gate(rec, plan, raw, t, stats_ref)
        if gate.get("regroup"):
            n_regroup += 1
        rec["parity"] = gate
        append(jsonl, rec)
        done.add(t["name"])
        processed += 1
        if processed % 16 == 0:
            print(f"[{i + 1}/{len(tg)}] {processed} tensors, {time.time() - t0:.0f}s",
                  flush=True)

    rows = load_rows(jsonl)
    if a.synthetic:
        n_reg = sum(1 for r in rows if r["stage"] == "levers" and r["parity"].get("regroup"))
        if n_reg == 0:
            die("synthetic strong parity gate never exercised: 0 tensors chose the "
                "regroup codec")
        print(f"[gate] synthetic strong parity gate exercised on {n_reg}/{len(tg)} tensors")
    else:
        print(f"[gate] parity exact vs stz_tensor_stats.jsonl on {len(tg)}/{len(tg)} tensors")

    run_roundtrips(tg, rows, jsonl, snap)
    print(f"\npricing + roundtrips complete for "
          f"{'synthetic' if a.synthetic else f'layer {a.layer}'} "
          f"({time.time() - t0:.0f}s)")
    summarize(a.synthetic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="smoke on the synthetic tiny snapshot")
    ap.add_argument("--layer", type=int, default=27,
                    help="target real MoE layer (plan: 1, 13, 27)")
    ap.add_argument("--summary", action="store_true",
                    help="aggregate all completed layers + model-wide projection")
    ap.add_argument("--limit", type=int, default=0, help="max tensors this invocation")
    ap.add_argument("--budget-s", type=float, default=400.0,
                    help="soft wall-clock budget; exits cleanly when exceeded")
    a = ap.parse_args()
    if a.summary:
        summarize(a.synthetic)
    else:
        run(a)


if __name__ == "__main__":
    main()
