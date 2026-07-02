"""probe_column_codebooks.py -- candidate 0014: column-keyed codebooks (Direction D).

Decisive, exactly-accounted probe over the layer-27 expert tensors (128 experts x
{up_proj, down_proj}, wholly inside shard 7). Per tensor it computes the realized
serialized cost of:

  (a) BASELINE  -- the current stz chooser, recomputed via stz.plan_regroup
      (imported, never reimplemented). Parity gate: must match the recorded
      realized bpw in stz_tensor_stats.jsonl within +/-0.01 b/w on every target
      tensor; loud abort otherwise. In --synthetic mode the gate is stronger:
      stz.enc_tensor is run and our bits must equal its realized bits exactly
      (enc_tensor itself asserts plan bits == written bits), and the run aborts
      if zero tensors exercised the regroup path (the gate cannot pass vacuously).
  (b) PER-TENSOR COLUMN-KEYED -- one top-K codebook per group of g columns
      (group id = col // g, address-derived -> fusible), sweep
      g in {1,4,16,64} x index bits b in {3,4}; second-level escape codebook
      stays per-tensor/global with stz's k in {0,3,4,5,6} envelope.
  (c) SHARED COLUMN-KEYED -- same sweep, first-level tables built from the
      aggregated per-group histograms over all experts of the layer (per
      projection type), charged once per layer (+ a 128-bit shared-table frame)
      and amortized over the tensors that share them.
  (d) ESCAPE FORENSICS on the winning variant -- per-row/per-column escape
      densities (Fano vs binomial), spatial adjacency lift, H(sign|col).

The bit accounting is WRITER-VERIFIED, not analytic-only: enc_colkey/dec_colkey
serialize and decode the column-keyed record for every (g, b) variant on a
deterministic tensor sample (all tensors in synthetic mode; expert 0 of each
projection in real mode, for both per-tensor and shared tables), asserting
serialized bits == colkey_plan bits exactly and SHA-256 bit-exact reconstruction
-- the same standard stz itself was held to.

Column axis: axis 1 (the contiguous in-row axis; col = flat_index % C). The
recon recomputed the quoted NEXT_DIRECTIONS numbers both ways and axis 1
reproduces them exactly, so a both-orientations sweep is unnecessary.

Cost model mirrors stz byte-for-byte: per-stream byte padding, u64 length
prefix per stream (4 streams k=0, 6 streams k>0), u16 per codebook entry
(first level ng*K*16 bits, second level L*16), per-row escape prefixes at
stz._pw widths, fixed 30-byte header '<BBHIIQQBB' (b,k,g,R,C,n,n_esc,pw,pw2).
Shared tables additionally carry a 128-bit frame (u64 length + packed
layer/proj/g/b id) charged across their sharers -- nothing is waved off.

Envelopes: env(base+pt) is exact (per-tensor tables fully charged). The
base+all envelope is ADOPTION-AWARE: a fixed-point per-tensor assignment where
every adopted shared table (payload + frame) is charged fully, once, across
its actual adopters -- the earlier uniformly-amortized env_all systematically
undercharged partially-adopted tables and has been removed.

Hardening (2026-07-01 adversarial review): single-file atomic aggregate
checkpoint (histograms + membership in one npz, one os.replace, plus a
count-consistency check on load), exclusive run lock, truncated-JSONL-tail
self-repair, fsync'd appends, verdict scoped to layer 27, per-variant
first-level table bytes surfaced (winner ties prefer the largest g / smallest
table), summary refuses shared-winner forensics on an incomplete aggregate.

Pure numpy, deterministic, one tensor in RAM at a time, resumable (JSONL append
+ checkpointed aggregate histograms), self-limits each invocation to the time
budget (default 400 s) and exits cleanly for re-invocation.

Usage:
  uv run python probe_column_codebooks.py --synthetic     # smoke on the fake snapshot
  uv run python probe_column_codebooks.py                  # real layer-27 run (resumable)
  uv run python probe_column_codebooks.py --summary        # table + forensics + summary JSON
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, struct, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "research/candidates/0009-fusible-exponent-codebook/tools"))
import stz  # noqa: E402  -- reuse plan_regroup/_pw/enc_tensor/st_header/pack_width/unpack_width

REAL_SNAP = REPO / "models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot"
SYN_SNAP = REPO / "models/synthetic/nemotron_tiny/hf_snapshot"
STATS_JSONL = (REPO / "research/candidates/0009-fusible-exponent-codebook"
               / "tests/artifacts/stz/stz_tensor_stats.jsonl")
ART = HERE.parents[1] / "tests" / "artifacts"

TARGET_LAYER = 27           # real mode: the one expert layer wholly inside shard 7
GS = (1, 4, 16, 64)         # column-group sizes (columns per first-level codebook)
BS = (3, 4)                 # first-level index bits
KS = (0, 3, 4, 5, 6)        # second-level escape bits (stz's exact envelope)
HEAD_BITS = 30 * 8          # '<BBHIIQQBB': b,k,g,R,C,n,n_esc,pw,pw2 (ng derivable from C,g)
SHARED_FRAME_BITS = 128     # per shared table: u64 length + packed (layer,proj,g,b) id
WHOLE_MODEL_BPW = 10.8975   # realized stz, BF16 numel-weighted (repriced baseline)
EXPERT_FRAC = 0.93          # experts' share of whole-model BF16 numel
PARITY_TOL = 0.01
GATE_BPW = 0.09             # brief.md success gate (~ +0.5 whole-model pt)

NAME_RE = re.compile(r"backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up|down)_proj\.weight$")

pad8 = lambda bits: bits + (-bits % 8)   # per-stream byte-alignment, stz's rule


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


# ---------------------------------------------------------------- histograms ---
def col_hists(sym2d: np.ndarray, g: int) -> np.ndarray:
    """(ng, 512) histogram of the 9-bit sym per column group (group = col // g)."""
    R, C = sym2d.shape
    ng = -(-C // g)
    gid = (np.arange(C) // g).astype(np.int64)
    keys = gid[None, :] * 512 + sym2d.astype(np.int64)
    return np.bincount(keys.ravel(), minlength=ng * 512).reshape(ng, 512)


def group_order(h2d: np.ndarray) -> np.ndarray:
    """Per-group sym order: descending count, ties ascending sym -- exactly stz's
    lexsort((arange, -hist)) rule, vectorized over groups via a composite key."""
    comp = np.arange(512, dtype=np.int64)[None, :] - h2d.astype(np.int64) * 512
    return np.argsort(comp, axis=1)


def escapes_under(h2d: np.ndarray, topk: np.ndarray):
    """n_esc and the 512-bin histogram of escaped syms, given per-group top-K ids."""
    covered = np.take_along_axis(h2d, topk, 1)
    n_esc = int(h2d.sum() - covered.sum())
    eh = h2d.copy()
    np.put_along_axis(eh, topk, 0, 1)
    return n_esc, eh.sum(0)


# ------------------------------------------------------------- colkey costing ---
def colkey_plan(n: int, R: int, ng: int, n_esc: int, esc_hist: np.ndarray,
                b: int, table_share: int = 1) -> dict:
    """Exact serialized bits of one column-keyed record. Mirrors stz.plan_regroup's
    accounting: byte padding per stream, u64 length prefixes, u16 table entries,
    stz._pw prefix widths, min over the k in {0,3,4,5,6} escape envelope.
    table_share > 1 amortizes the first-level table (payload + a 128-bit shared
    frame: u64 length + packed layer/proj/g/b id), charged once per layer."""
    K = (1 << b) - 1
    order2 = np.lexsort((np.arange(512), -esc_hist))     # stz's second-level rule
    c2 = esc_hist[order2]
    pw = stz._pw(n_esc)
    table_bits_full = ng * K * 16                        # u16 per entry, per group
    frame = SHARED_FRAME_BITS if table_share > 1 else 0
    table_bits = (table_bits_full + frame) / table_share
    best = None
    for k in KS:
        if k == 0:
            L = 0
            esc_bits = pad8(n_esc * 9) + pad8(R * pw)
            n_streams = 4
        else:
            L = (1 << k) - 1
            n_raw = int(n_esc - c2[:L].sum())
            pw2 = stz._pw(n_raw)
            esc_bits = pad8(n_esc * k) + pad8(n_raw * 9) + pad8(R * pw) + pad8(R * pw2)
            n_streams = 6
        total = (pad8(n * b) + esc_bits + pad8(n * 7)
                 + table_bits + L * 16 + HEAD_BITS + n_streams * 64)
        if best is None or total < best["bits"]:
            best = {"bits": float(total), "b": b, "k": k, "n_esc": n_esc,
                    "table_bits_full": int(table_bits_full),
                    "table_bits_charged": float(table_bits)}
    return best


def variant_row(n, R, h2d, order, b, table_share=1):
    K = (1 << b) - 1
    n_esc, eh = escapes_under(h2d, order[:, :K])
    p = colkey_plan(n, R, h2d.shape[0], n_esc, eh, b, table_share)
    return {"bpw": round(p["bits"] / n, 6), "bits": round(p["bits"], 3),
            "b": b, "k": p["k"],
            "n_esc": n_esc, "esc_rate": round(n_esc / n, 6),
            "table_bits_full": p["table_bits_full"],
            "table_bits_charged": round(p["table_bits_charged"], 3)}


# ------------------------------------------------- colkey encoder / decoder ---
def enc_colkey(raw: bytes, shape, g: int, b: int, order: np.ndarray,
               h2d: np.ndarray) -> tuple[bytes, dict]:
    """Serialize one tensor in the column-keyed format (full table, share=1).
    Mirrors stz.enc_tensor stream-for-stream; proves colkey_plan's accounting is
    real: asserts written*8 == plan bits."""
    u = np.frombuffer(raw, "<u2")
    n = u.size
    R, C = shape
    K = (1 << b) - 1
    ng = h2d.shape[0]
    sym2d = (u >> 7).astype(np.uint16).reshape(R, C)
    mant = (u & 0x7F).astype(np.uint8)
    n_esc, eh = escapes_under(h2d, order[:, :K])
    plan = colkey_plan(n, R, ng, n_esc, eh, b, table_share=1)
    k, pw = plan["k"], stz._pw(n_esc)
    codemap = np.full((ng, 512), K, np.uint8)            # per-group sym -> code
    np.put_along_axis(codemap, order[:, :K], np.arange(K, dtype=np.uint8)[None, :], 1)
    gid = np.arange(C, dtype=np.int64) // g
    idx = codemap[gid[None, :], sym2d].ravel()           # b-bit index plane
    esc_mask = idx == K
    esc_syms = sym2d.ravel()[esc_mask]
    assert int(esc_syms.size) == n_esc

    def row_prefix(mask, width):                          # cumulative counts, stz's rule
        prefix = np.zeros(R, np.uint64)
        if R > 1:
            per_row = mask.reshape(R, -1).sum(1, dtype=np.uint64)
            prefix[1:] = np.cumsum(per_row)[:-1]
        return stz.pack_width(prefix, width)

    streams = [stz.pack_width(idx, b)]
    if k:
        L = (1 << k) - 1
        order2 = np.lexsort((np.arange(512), -eh))
        codebook2 = order2[:L].astype(np.uint16)
        codemap2 = np.full(512, L, np.uint16)
        codemap2[codebook2] = np.arange(L, dtype=np.uint16)
        esc_codes = codemap2[esc_syms]
        raw_in_esc = esc_codes == L
        esc_raw = esc_syms[raw_in_esc]
        pw2 = stz._pw(int(esc_raw.size))
        raw_mask = np.zeros(n, bool)
        raw_mask[np.flatnonzero(esc_mask)[raw_in_esc]] = True
        streams += [stz.pack_width(esc_codes, k), stz.pack_width(esc_raw, 9),
                    row_prefix(esc_mask, pw), row_prefix(raw_mask, pw2)]
    else:
        pw2 = 0
        codebook2 = np.empty(0, np.uint16)
        streams += [stz.pack_width(esc_syms, 9), row_prefix(esc_mask, pw)]
    streams.append(stz.pack_width(mant, 7))
    head = struct.pack("<BBHIIQQBB", b, k, g, R, C, n, n_esc, pw, pw2)
    head += order[:, :K].astype("<u2").tobytes() + codebook2.astype("<u2").tobytes()
    buf = bytearray(head)
    for s in streams:
        buf += struct.pack("<Q", len(s)) + s
    assert len(buf) * 8 == int(round(plan["bits"])), (len(buf) * 8, plan["bits"])
    return bytes(buf), plan


def dec_colkey(buf: bytes) -> bytes:
    """Decode one column-keyed record from its own bytes alone; returns the
    original BF16 tensor bytes. Validates the row prefixes (the random-access
    side structure) like stz.dec_tensor."""
    b, k, g, R, C, n, n_esc, pw, pw2 = struct.unpack_from("<BBHIIQQBB", buf, 0)
    off = 30
    K = (1 << b) - 1
    ng = -(-C // g)
    table = np.frombuffer(buf, "<u2", ng * K, off).reshape(ng, K)
    off += ng * K * 2
    L = (1 << k) - 1 if k else 0
    codebook2 = np.frombuffer(buf, "<u2", L, off)
    off += L * 2

    def stream():
        nonlocal off
        (ln,) = struct.unpack_from("<Q", buf, off)
        off += 8
        s = buf[off:off + ln]
        off += ln
        return s

    idx = stz.unpack_width(stream(), n, b).reshape(R, C)
    cb = np.zeros((ng, K + 1), np.uint16)
    cb[:, :K] = table
    gid = np.arange(C, dtype=np.int64) // g
    sym = cb[gid[None, :], idx].ravel()
    esc_mask = (idx == K).ravel()
    del idx
    if k:
        esc_codes = stz.unpack_width(stream(), n_esc, k)
        in2 = esc_codes < L
        esc_raw = stz.unpack_width(stream(), int((~in2).sum()), 9)
        esc_syms = np.empty(n_esc, np.uint16)
        esc_syms[in2] = codebook2[esc_codes[in2].astype(np.int64)]
        esc_syms[~in2] = esc_raw.astype(np.uint16)
        prefix = stz.unpack_width(stream(), R, pw)
        prefix2 = stz.unpack_width(stream(), R, pw2)
    else:
        esc_syms = stz.unpack_width(stream(), n_esc, 9).astype(np.uint16)
        prefix = stz.unpack_width(stream(), R, pw)
        prefix2 = None
    sym[esc_mask] = esc_syms
    if R > 1:
        per_row = esc_mask.reshape(R, -1).sum(1, dtype=np.uint64)
        chk = np.zeros(R, np.uint64)
        chk[1:] = np.cumsum(per_row)[:-1]
        assert np.array_equal(chk, prefix.astype(np.uint64)), "row escape-prefix corrupt"
        if k:
            raw_rows = np.flatnonzero(esc_mask)[~in2] // (n // R)
            per_row2 = np.bincount(raw_rows, minlength=R).astype(np.uint64)
            chk2 = np.zeros(R, np.uint64)
            chk2[1:] = np.cumsum(per_row2)[:-1]
            assert np.array_equal(chk2, prefix2.astype(np.uint64)), "raw-escape prefix corrupt"
    mant = stz.unpack_width(stream(), n, 7)
    assert off == len(buf), (off, len(buf))
    out = ((sym.astype(np.uint16) << 7) | mant).astype("<u2").tobytes()
    assert len(out) == n * 2
    return out


def roundtrip_variants(raw: bytes, shape, orders: dict, h2ds: dict) -> dict:
    """Encode + decode every (g, b) variant with the given first-level orders.
    Proves (1) serialized bits == colkey_plan bits exactly (table_share=1) and
    (2) SHA-256 bit-exact reconstruction -- the standard stz was held to."""
    sha0 = hashlib.sha256(raw).hexdigest()
    out = {"sha256": sha0, "table_share_encoded": 1, "variants": {}}
    for g in GS:
        for b in BS:
            blob, plan = enc_colkey(raw, shape, g, b, orders[g], h2ds[g])
            dec = dec_colkey(blob)
            if hashlib.sha256(dec).hexdigest() != sha0:
                die(f"ROUNDTRIP FAILURE: g{g} b{b} reconstruction is not bit-exact")
            out["variants"][f"g{g}_b{b}"] = {"bits": len(blob) * 8, "k": plan["k"],
                                             "bits_match_plan": True, "sha256_match": True}
    return out


# ------------------------------------------------------- aggregate hist state ---
def agg_key(t: dict, g: int) -> str:
    return f"L{t['layer']}_{t['proj']}_g{g}"


def load_agg(aggp: Path):
    """Single-file atomic checkpoint: histograms and the included-tensor
    membership live in the SAME npz, committed by one os.replace, so they can
    never diverge (fixes the crash-window double-count)."""
    if not aggp.exists():
        return {}, set()
    with np.load(aggp, allow_pickle=False) as z:
        if "__included__" not in z.files:
            die(f"aggregate checkpoint {aggp} is in the old two-file format "
                f"(no __included__ member) -- delete it (and any .state.json) and re-run")
        included = {str(x) for x in z["__included__"]}
        agg = {kk: z[kk].astype(np.int64) for kk in z.files if kk != "__included__"}
    return agg, included


def save_agg(aggp: Path, agg: dict, included: set):
    tmp = aggp.with_name(aggp.stem + ".tmp.npz")
    inc = np.array(sorted(included)) if included else np.empty(0, dtype="<U1")
    np.savez_compressed(tmp, __included__=inc, **agg)
    os.replace(tmp, aggp)                                 # one atomic commit for both


def check_agg_consistency(agg: dict, included: set, tg: list[dict], aggp: Path):
    """Die loudly if the checkpointed histograms disagree with the membership
    list (sum of counts per key must equal the numel of the included tensors)."""
    by_name = {t["name"]: t for t in tg}
    unknown = included - set(by_name)
    if unknown:
        die(f"aggregate checkpoint {aggp} lists unknown tensors "
            f"(first: {sorted(unknown)[0]}) -- delete it and re-run")
    n_by = {}
    for nm in included:
        t = by_name[nm]
        key = (t["layer"], t["proj"])
        n_by[key] = n_by.get(key, 0) + t["shape"][0] * t["shape"][1]
    for (layer, proj), expect in n_by.items():
        for g in GS:
            kk = f"L{layer}_{proj}_g{g}"
            if kk not in agg:
                die(f"aggregate checkpoint {aggp} is missing key {kk} for included "
                    f"tensors -- delete it and re-run")
            got = int(agg[kk].sum())
            if got != expect:
                die(f"aggregate checkpoint inconsistent: {kk} holds {got} counts but "
                    f"the included tensors account for {expect} (possible double-count)"
                    f" -- delete {aggp} and re-run")


# ------------------------------------------------------------- jsonl helpers ---
def load_rows(jsonl: Path) -> list[dict]:
    """Parse the append-only JSONL. A truncated FINAL line (process killed
    mid-append) is repaired by truncating it away -- safe, since done-sets derive
    from parsed rows and the tensor is simply reprocessed. A missing trailing
    newline is fixed so a future append can never merge records."""
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
            if i == len(lines) - 1:  # partial trailing record from a killed append
                good = "\n".join(lines[:i])
                jsonl.write_text(good + ("\n" if good else ""))
                print(f"[warn] repaired truncated trailing record in {jsonl.name}; "
                      f"its tensor will be reprocessed", flush=True)
                return rows
            die(f"corrupt JSONL record mid-file ({jsonl}, line {i + 1}) -- refusing to guess")
    if text and not text.endswith("\n"):
        with jsonl.open("a") as f:
            f.write("\n")
    return rows


# ----------------------------------------------------------------- forensics ---
def h2_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def escape_forensics(snap, tg, winner: str, agg: dict) -> list[dict]:
    """Per-row/per-column escape densities, spatial structure, H(sign|col) on a
    deterministic 4-tensor sample under the winning variant's tables."""
    kind, gs, bs = winner.split("_")
    g, b = int(gs[1:]), int(bs[1:])
    K = (1 << b) - 1
    by = {}
    for t in tg:
        by.setdefault((t["layer"], t["proj"]), []).append(t)
    samples = []
    for key in sorted(by):
        lst = by[key]
        samples += [lst[0], lst[len(lst) // 2]]
    out = []
    for t in samples[:4]:
        raw = read_raw(snap, t)
        u = np.frombuffer(raw, "<u2")
        R, C = t["shape"]
        sym2d = (u >> 7).astype(np.uint16).reshape(R, C)
        h2d = col_hists(sym2d, g)
        order = group_order(agg[agg_key(t, g)]) if kind == "sh" else group_order(h2d)
        ng = h2d.shape[0]
        topk_mask = np.zeros((ng, 512), bool)
        np.put_along_axis(topk_mask, order[:, :K], True, 1)
        gid = np.arange(C) // g
        esc = ~topk_mask[gid[None, :], sym2d]            # (R, C) escape mask
        rate = float(esc.mean())
        row_c, col_c = esc.sum(1), esc.sum(0)
        fano = lambda c: float(c.var() / c.mean()) if c.mean() > 0 else 0.0
        base = max(rate, 1e-12)
        h_lift = float((esc[:, :-1] & esc[:, 1:]).sum() / max(int(esc[:, :-1].sum()), 1) / base)
        v_lift = float((esc[:-1, :] & esc[1:, :]).sum() / max(int(esc[:-1, :].sum()), 1) / base)
        sign2d = ((u >> 15) & 1).astype(np.uint8).reshape(R, C)
        out.append({
            "name": t["name"], "variant": winner, "esc_rate": round(rate, 6),
            "row_density": {"mean": float(row_c.mean()), "std": float(row_c.std()),
                            "fano": round(fano(row_c), 4)},
            "col_density": {"mean": float(col_c.mean()), "std": float(col_c.std()),
                            "fano": round(fano(col_c), 4)},
            "binomial_fano_ref": round(1 - rate, 4),
            "adjacency_lift": {"horizontal": round(h_lift, 4), "vertical": round(v_lift, 4)},
            "H_sign": round(float(h2_bits(np.array([sign2d.mean()]))[0]), 6),
            "H_sign_given_col": round(float(h2_bits(sign2d.mean(0)).mean()), 6),
        })
    return out


# ------------------------------------------------------ adoption-aware envelope ---
def adoption_envelope(names, pt, sh):
    """Adoption-aware 'base + all variants' envelope (fixes the amortization
    undercharge): iterate the per-tensor argmin assignment against per-table
    adopter counts, then price the FINAL assignment exactly -- every adopted
    shared table (payload + 128-bit frame) charged fully, once, split over its
    actual adopters. The result is an achievable serialized total, never an
    undercharge. Returns (total_bits, detail)."""
    opts, table_full = {}, {}
    for nm in names:
        o = [("baseline", None, float(pt[nm]["baseline"]["bits"]))]
        for vk, v in pt[nm]["variants"].items():
            o.append((vk, None, float(v["bits"])))
        r = sh[nm]
        for vk, v in r["variants"].items():
            tkey = (r["layer"], r["proj"], vk)
            table_full[tkey] = float(v["table_bits_full"]) + SHARED_FRAME_BITS
            o.append((vk, tkey, float(v["bits"]) - float(v["table_bits_charged"])))
        opts[nm] = o

    def pick(nm, adopters, cur_tkey):
        best = None
        for opt in opts[nm]:
            _, tkey, bits = opt
            if tkey is None:
                cost = bits
            else:
                m = adopters.get(tkey, 0)
                denom = m if (tkey == cur_tkey and m > 0) else m + 1
                cost = bits + table_full[tkey] / denom
            if best is None or cost < best[0]:
                best = (cost, opt)
        return best[1]

    # init: the optimistic uniform-adoption charge (what the records assume)
    assign = {}
    for nm in names:
        share = max(int(sh[nm].get("n_share", 1)), 1)
        best = None
        for opt in opts[nm]:
            _, tkey, bits = opt
            cost = bits if tkey is None else bits + table_full[tkey] / share
            if best is None or cost < best[0]:
                best = (cost, opt)
        assign[nm] = best[1]
    iters = 0
    for iters in range(1, 65):
        adopters = {}
        for opt in assign.values():
            if opt[1] is not None:
                adopters[opt[1]] = adopters.get(opt[1], 0) + 1
        new = {nm: pick(nm, adopters, assign[nm][1]) for nm in names}
        stable = all(new[nm][0] == assign[nm][0] and new[nm][1] == assign[nm][1]
                     for nm in names)
        assign = new
        if stable:
            break
    adopters = {}
    for opt in assign.values():
        if opt[1] is not None:
            adopters[opt[1]] = adopters.get(opt[1], 0) + 1
    total = sum(opt[2] for opt in assign.values())
    total += sum(table_full[tk] for tk in adopters)
    counts = {}
    for opt in assign.values():
        counts[opt[0]] = counts.get(opt[0], 0) + 1
    detail = {
        "iterations": iters,
        "choice_counts": dict(sorted(counts.items())),
        "adopted_shared_tables": {f"L{tk[0]}_{tk[1]}_{tk[2]}": c
                                  for tk, c in sorted(adopters.items())},
        "note": (f"adoption-aware exact pricing: each adopted shared table (payload "
                 f"+ {SHARED_FRAME_BITS}-bit frame) charged fully once across its "
                 f"actual adopters; replaces the invalid uniformly-amortized env_all"),
    }
    return total, detail


# ------------------------------------------------------------------- summary ---
def summarize(tg, jsonl: Path, aggp: Path, summaryp: Path, snap: Path, synthetic: bool):
    rows = load_rows(jsonl)
    pt = {r["name"]: r for r in rows if r["stage"] == "per_tensor"}
    sh = {r["name"]: r for r in rows if r["stage"] == "shared"}
    rt = [r for r in rows if r["stage"] == "roundtrip"]
    names = [t["name"] for t in tg]
    miss = [n for n in names if n not in pt] + [n for n in names if n not in sh]
    if miss:
        die(f"summary requires all stages complete; {len(miss)} records missing "
            f"(first: {miss[0]}) -- re-invoke without --summary to resume")
    n_tot = sum(pt[n]["n"] for n in names)
    wmean = lambda f: sum(f(n) * pt[n]["n"] for n in names) / n_tot
    base = wmean(lambda n: pt[n]["baseline"]["bpw"])
    variants = {}
    for g in GS:
        for b in BS:
            variants[f"pt_g{g}_b{b}"] = wmean(lambda n, k=f"pt_g{g}_b{b}": pt[n]["variants"][k]["bpw"])
            variants[f"sh_g{g}_b{b}"] = wmean(lambda n, k=f"sh_g{g}_b{b}": sh[n]["variants"][k]["bpw"])
    env_pt = wmean(lambda n: min([pt[n]["baseline"]["bpw"]]
                                 + [v["bpw"] for v in pt[n]["variants"].values()]))
    env_all_bits, env_detail = adoption_envelope(names, pt, sh)
    env_all = env_all_bits / n_tot
    parity_max = max(pt[n]["parity"]["abs_diff"] for n in names)

    def tbytes(vk):                                       # first-level table payload
        src = pt if vk.startswith("pt_") else sh
        return max(src[n]["variants"][vk]["table_bits_full"] for n in names) // 8

    # winner: min bpw; ties (within 1e-6 b/w) prefer the largest g -- the smallest
    # first-level table, i.e. the most runtime-credible variant (review fix)
    best_bpw = min(variants.values())
    near = [k for k, v in variants.items() if v - best_bpw <= 1e-6]
    winner = sorted(near, key=lambda k: (-int(k.split("_")[1][1:]), k))[0]
    win_kind = "sh" if winner.startswith("sh") else "pt"

    proj = lambda d: WHOLE_MODEL_BPW - EXPERT_FRAC * d          # projected whole-model b/w
    pts = lambda d: EXPERT_FRAC * d / 16 * 100                  # pts of the original 16 b/w

    mode = ("SYNTHETIC (smoke only -- projections meaningless)" if synthetic
            else f"REAL layer-{TARGET_LAYER}")
    print(f"\n=== candidate 0014 column-keyed codebooks -- summary [{mode}] ===")
    print(f"targets: {len(names)} tensors, {n_tot:,} params; "
          f"parity gate OK (max |d bpw| vs stz reference = {parity_max:.6f})")
    hdr = (f"{'variant':<16}{'bpw':>10}{'d vs stz':>11}{'proj model b/w':>16}"
           f"{'d pts of 16':>13}{'tableKB':>9}")
    print(hdr); print("-" * len(hdr))
    print(f"{'stz baseline':<16}{base:>10.4f}{'-':>11}{WHOLE_MODEL_BPW:>16.4f}{'-':>13}{'<0.1':>9}")
    for k in sorted(variants):
        d = base - variants[k]
        print(f"{k:<16}{variants[k]:>10.4f}{d:>+11.4f}{proj(d):>16.4f}{pts(d):>+13.3f}"
              f"{tbytes(k) / 1024:>9.1f}")
    for lbl, v in (("env(base+pt)", env_pt), ("env(base+all)*", env_all)):
        d = base - v
        print(f"{lbl:<16}{v:>10.4f}{d:>+11.4f}{proj(d):>16.4f}{pts(d):>+13.3f}{'':>9}")
    print("* adoption-aware exact pricing (see envelope note in the summary JSON)")

    d_win = base - variants[winner]
    verdict = ("CONFIRMED" if d_win >= GATE_BPW else
               "weak positive" if d_win > 0 else "FALSIFIED at this operating point")
    scope = ("synthetic smoke -- carries no evidential weight" if synthetic else
             f"layer {TARGET_LAYER} only; cross-layer transfer unvalidated")
    # serializer coverage gate (review fix: no CONFIRMED/weak-positive headline
    # without writer-verified bits for the winning variant kind on every proj)
    projs_needed = sorted({t["proj"] for t in tg})
    covered = sorted({r["proj"] for r in rt if r["kind"] == win_kind})
    serializer_ok = set(projs_needed) <= set(covered)
    if d_win > 0 and not serializer_ok:
        verdict = (f"PROVISIONAL {verdict} -- serializer round-trip missing for the "
                   f"winning variant kind ({win_kind}) on projections "
                   f"{sorted(set(projs_needed) - set(covered))}")
    print(f"\nwinner: {winner}  d={d_win:+.4f} b/w  gate(>={GATE_BPW}): {verdict}  [{scope}]")
    print(f"serializer round-trips: {len(rt)} records; winning-kind coverage "
          f"{covered} of {projs_needed} -> {'OK' if serializer_ok else 'INCOMPLETE'}")

    agg, included = load_agg(aggp)
    if win_kind == "sh":
        if not set(names) <= included:
            die("winning variant uses shared tables but the aggregate-histogram "
                "checkpoint is missing/incomplete -- re-run the probe (without "
                "--summary) to rebuild it, then retry")
        check_agg_consistency(agg, included, tg, aggp)
    forensics = escape_forensics(snap, tg, winner, agg)
    print("\nescape forensics (winning variant):")
    for f in forensics:
        print(f"  {f['name']}: esc={f['esc_rate']:.4f} "
              f"fano(row)={f['row_density']['fano']} fano(col)={f['col_density']['fano']} "
              f"(binomial~{f['binomial_fano_ref']}) "
              f"adj h={f['adjacency_lift']['horizontal']} v={f['adjacency_lift']['vertical']} "
              f"H(sign)={f['H_sign']} H(sign|col)={f['H_sign_given_col']}")

    summary = {
        "mode": "synthetic" if synthetic else "real",
        "scope": scope,
        "targets": len(names), "total_params": int(n_tot),
        "parity_max_abs_diff": parity_max,
        "baseline_bpw": round(base, 6),
        "whole_model_baseline_bpw": WHOLE_MODEL_BPW, "expert_frac": EXPERT_FRAC,
        "variants": {k: {"bpw": round(v, 6), "delta_bpw": round(base - v, 6),
                         "projected_whole_model_bpw": round(proj(base - v), 6),
                         "delta_pts_of_16": round(pts(base - v), 4),
                         "first_level_table_bytes_max": int(tbytes(k))}
                     for k, v in sorted(variants.items())},
        "variants_note": ("sh_* rows assume uniform adoption by all sharers of a "
                          "(layer, proj): the (payload + frame)/share amortization "
                          "is exact only then"),
        "envelope": {"base_plus_pt": round(env_pt, 6),
                     "base_plus_all_adoption_aware": round(env_all, 6),
                     "adoption_detail": env_detail},
        "serializer_roundtrip": {
            "records": len(rt),
            "coverage": {kind: sorted({r["proj"] for r in rt if r["kind"] == kind})
                         for kind in ("pt", "sh")},
            "winner_kind_covered_all_projections": serializer_ok,
            "note": ("every record asserts serialized bits == colkey_plan bits "
                     "exactly (table_share=1) and SHA-256 bit-exact reconstruction"),
        },
        "winner": winner, "winner_delta_bpw": round(d_win, 6), "gate_bpw": GATE_BPW,
        "verdict": verdict, "forensics": forensics,
    }
    summaryp.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written: {summaryp}")


# ---------------------------------------------------------------------- main ---
def run(a, snap: Path, jsonl: Path, aggp: Path, summaryp: Path):
    tg = enum_targets(snap, a.synthetic)
    names = [t["name"] for t in tg]
    n_share = {}
    for t in tg:
        n_share[(t["layer"], t["proj"])] = n_share.get((t["layer"], t["proj"]), 0) + 1

    if a.summary:
        return summarize(tg, jsonl, aggp, summaryp, snap, a.synthetic)

    # reference bpw for the parity gate (real mode: recorded realized stz stats)
    stats_ref = {}
    if not a.synthetic:
        if not STATS_JSONL.exists():
            die(f"missing parity reference {STATS_JSONL}")
        for line in STATS_JSONL.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["name"] in set(names):
                    stats_ref[r["name"]] = r["bpw"]
        missing = [n for n in names if n not in stats_ref]
        if missing:
            die(f"{len(missing)} targets absent from stz_tensor_stats.jsonl (first: {missing[0]})")

    rows = load_rows(jsonl)
    done_pt = {r["name"] for r in rows if r["stage"] == "per_tensor"}
    done_sh = {r["name"] for r in rows if r["stage"] == "shared"}
    done_rt_pt = {r["name"] for r in rows if r["stage"] == "roundtrip" and r["kind"] == "pt"}
    done_rt_sh = {r["name"] for r in rows if r["stage"] == "roundtrip" and r["kind"] == "sh"}
    agg, included = load_agg(aggp)
    check_agg_consistency(agg, included, tg, aggp)

    t0, processed = time.time(), 0

    def checkpoint_exit(where: str):
        save_agg(aggp, agg, included)
        print(f"\n[{where}] budget/limit reached after {processed} tensors "
              f"({time.time()-t0:.0f}s) -- progress saved, re-invoke to resume.", flush=True)
        sys.exit(0)

    def append(rec: dict):
        with jsonl.open("a") as f:
            f.write(json.dumps(rec) + "\n")     # single write per record ...
            f.flush()
            os.fsync(f.fileno())                # ... durably on disk before we move on

    # ---- stage A: per-tensor variants + baseline parity + shared-hist accumulation
    #      + writer verification (enc/dec round-trip) on the deterministic sample
    for i, t in enumerate(tg):
        rt_eligible = a.synthetic or t["expert"] == 0
        need_row = t["name"] not in done_pt
        need_agg = t["name"] not in included
        need_rt = rt_eligible and t["name"] not in done_rt_pt
        if not (need_row or need_agg or need_rt):
            continue
        if a.limit and processed >= a.limit:
            checkpoint_exit("stage A")
        if time.time() - t0 > a.budget_s:
            checkpoint_exit("stage A")
        raw = read_raw(snap, t)
        u = np.frombuffer(raw, "<u2")
        n = u.size
        R, C = t["shape"]
        sym = (u >> 7).astype(np.uint16)          # 9-bit sign+exponent, stz's exact split
        sym2d = sym.reshape(R, C)
        h2ds = {g: col_hists(sym2d, g) for g in GS}
        if need_agg:
            for g in GS:
                k = agg_key(t, g)
                agg[k] = agg.get(k, 0) + h2ds[g]
            included.add(t["name"])
        orders = {g: group_order(h2ds[g]) for g in GS} if (need_row or need_rt) else {}
        if need_row:
            plan = stz.plan_regroup(np.bincount(sym, minlength=512), n, R)
            base_bpw = plan["bits"] / n
            if a.synthetic:  # strong gate: exact equality with the realized encoder
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
            variants = {}
            for g in GS:
                for b in BS:
                    variants[f"pt_g{g}_b{b}"] = variant_row(n, R, h2ds[g], orders[g], b)
            append({"stage": "per_tensor", "name": t["name"], "layer": t["layer"],
                    "expert": t["expert"], "proj": t["proj"], "n": int(n), "R": int(R),
                    "C": int(C),
                    "baseline": {"bpw": round(base_bpw, 6), "bits": int(plan["bits"]),
                                 "variant": plan["variant"], "b": plan.get("b"),
                                 "k": plan.get("k"), "n_esc": int(plan.get("n_esc", 0))},
                    "parity": {"ref_bpw": ref, "abs_diff": round(diff, 6)},
                    "variants": variants})
            done_pt.add(t["name"])
        if need_rt:
            rt = roundtrip_variants(raw, t["shape"], orders, h2ds)
            append({"stage": "roundtrip", "kind": "pt", "name": t["name"],
                    "layer": t["layer"], "expert": t["expert"], "proj": t["proj"],
                    "n": int(n), **rt})
            done_rt_pt.add(t["name"])
            print(f"[rt pt] {t['name']}: {len(rt['variants'])} variants serialized, "
                  f"bits==plan and SHA-256 round-trip OK", flush=True)
        processed += 1
        if processed % 16 == 0:
            print(f"[A {i+1}/{len(tg)}] {processed} tensors, {time.time()-t0:.0f}s", flush=True)
    save_agg(aggp, agg, included)

    if a.synthetic:  # the strong gate must actually have been exercised (review fix)
        n_reg = sum(1 for r in load_rows(jsonl) if r["stage"] == "per_tensor"
                    and r["baseline"].get("variant") == "regroup")
        if n_reg == 0:
            die("synthetic strong parity gate never exercised: 0 tensors chose the "
                "regroup codec -- the synthetic snapshot is supposed to exercise it")
        print(f"[gate] synthetic strong parity gate exercised on {n_reg}/{len(tg)} tensors")

    # ---- stage B: shared tables (aggregate must be complete first)
    if not set(names) <= included:
        die("aggregate histograms incomplete after stage A -- should not happen")
    shared_orders = {}
    for t in tg:
        rt_eligible = a.synthetic or t["expert"] == 0
        need_sh_row = t["name"] not in done_sh
        need_rt = rt_eligible and t["name"] not in done_rt_sh
        if not (need_sh_row or need_rt):
            continue
        if a.limit and processed >= a.limit:
            checkpoint_exit("stage B")
        if time.time() - t0 > a.budget_s:
            checkpoint_exit("stage B")
        raw = read_raw(snap, t)
        u = np.frombuffer(raw, "<u2")
        n = u.size
        R, C = t["shape"]
        sym2d = (u >> 7).astype(np.uint16).reshape(R, C)
        share = n_share[(t["layer"], t["proj"])]
        h2ds = {g: col_hists(sym2d, g) for g in GS}
        orders = {}
        for g in GS:
            k = agg_key(t, g)
            if k not in shared_orders:
                shared_orders[k] = group_order(agg[k])
            orders[g] = shared_orders[k]
        if need_sh_row:
            variants = {}
            for g in GS:
                for b in BS:
                    variants[f"sh_g{g}_b{b}"] = variant_row(n, R, h2ds[g], orders[g], b,
                                                            table_share=share)
            append({"stage": "shared", "name": t["name"], "layer": t["layer"],
                    "expert": t["expert"], "proj": t["proj"], "n": int(n), "R": int(R),
                    "C": int(C), "n_share": share, "variants": variants})
            done_sh.add(t["name"])
        if need_rt:
            rt = roundtrip_variants(raw, t["shape"], orders, h2ds)
            append({"stage": "roundtrip", "kind": "sh", "name": t["name"],
                    "layer": t["layer"], "expert": t["expert"], "proj": t["proj"],
                    "n": int(n), **rt})
            done_rt_sh.add(t["name"])
            print(f"[rt sh] {t['name']}: {len(rt['variants'])} variants serialized, "
                  f"bits==plan and SHA-256 round-trip OK", flush=True)
        processed += 1
        if processed % 16 == 0:
            print(f"[B] {processed} tensors, {time.time()-t0:.0f}s", flush=True)

    print(f"\nstages complete: {len(done_pt)}/{len(tg)} per_tensor, "
          f"{len(done_sh)}/{len(tg)} shared ({time.time()-t0:.0f}s)")
    summarize(tg, jsonl, aggp, summaryp, snap, a.synthetic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run against the synthetic tiny snapshot (smoke)")
    ap.add_argument("--summary", action="store_true",
                    help="summary + forensics only (requires completed stages)")
    ap.add_argument("--limit", type=int, default=0, help="max tensors read this invocation")
    ap.add_argument("--budget-s", type=float, default=400.0,
                    help="soft wall-clock budget; checkpoints and exits when exceeded")
    ap.add_argument("--layer", type=int, default=27,
                    help="target expert layer (cross-layer closure rider; artifacts "
                         "for non-default layers get a _layer<N> suffix)")
    a = ap.parse_args()

    global TARGET_LAYER
    TARGET_LAYER = a.layer
    snap = SYN_SNAP if a.synthetic else REAL_SNAP
    tag = "_synthetic" if a.synthetic else ("" if a.layer == 27 else f"_layer{a.layer}")
    ART.mkdir(parents=True, exist_ok=True)
    jsonl = ART / f"colkey_results{tag}.jsonl"
    aggp = ART / f"colkey_shared_hists{tag}.npz"
    summaryp = ART / f"colkey_summary{tag}.json"

    # exclusive run lock: overlapping invocations would double-accumulate the
    # shared aggregate (review fix). Stale locks must be deleted by hand.
    lockp = ART / f"colkey{tag}.lock"
    try:
        fd = os.open(lockp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = lockp.read_text().strip() or "?"
        except OSError:
            holder = "?"
        die(f"lock file {lockp} exists (written by pid {holder}); another invocation "
            f"may be running -- if you are certain none is, delete the lock and retry")
    with os.fdopen(fd, "w") as lf:
        lf.write(str(os.getpid()))
    try:
        run(a, snap, jsonl, aggp, summaryp)
    finally:
        try:
            lockp.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
