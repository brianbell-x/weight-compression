"""stz: a REAL serialized container for the 0009 fixed-width lossless codec.

Upgrades the whole-model 30.03% claim from bit-accounting (whole_model_lossless.py
counts bits but never writes a stream) to a demonstrated artifact: every shard is
encoded to a .stz file on disk, and `verify` decodes the .stz alone and proves the
reconstructed shard is SHA-256-identical to the original file, byte for byte
(safetensors header included).

Codec = candidate 0009 "regroup": per BF16 tensor the 9-bit sym = sign+exponent
field is coded as a fixed-width b-bit index into a per-tensor top-K codebook
(K = 2^b - 1, code K = escape) plus an in-order escape stream; the 7-bit mantissa
is packed verbatim. Per-row escape prefix tables keep every stream random-access
at row granularity (the fusible property). On top of the accounting model this
adds two levers the vetting pass identified as free wins:
  - per-tensor min-envelope chooser: index_bits b in {2,3,4,5} x escape coding,
    with raw16 fallback (kills any tensor the codec would inflate);
  - second-level escape codebook: escape values coded in k bits against the
    next-L most frequent syms (L = 2^k - 1), reserved code -> raw 9-bit stream,
    with its own per-row prefix so raw escapes stay row-addressable.

Hardened per adversarial review (2026-07-01): exact chooser cost model (incl.
per-stream byte padding), atomic per-shard writes + resume, streamed verify
(no shard-sized buffer; peak = one tensor's decode transients), coverage check
against model.safetensors.index.json, monotone-offset guard, version/name/length
asserts. Run with plain `uv run python` (never -O): the guards are asserts.

Format v2 (little-endian):
  file:   magic 'STZ1' | u16 ver=2 | u16 reserved | u32 name_len | name utf-8
          | 32B sha256(original file) | u64 original file size
          | u64 header_block_len | header block bytes (verbatim: 8B len + JSON)
          | u32 n_gaps | gaps (u64 off, u64 len, bytes)   <- before records so
          | u32 n_records | records...                       verify can stream
  record: u32 name_len | name | u8 codec (0 verbatim | 1 regroup) | u64 nbytes
          codec 0: payload bytes
          codec 1: u8 b | u8 k | u32 R | u64 n | u64 n_esc | u8 pw | u8 pw2
                   | K u16 codebook | (L u16 codebook2 if k>0)
                   | length-prefixed streams (u64 len each):
                     idx_plane, [esc_codes, esc_raw9, row_prefix, row_prefix2]
                     (k=0: esc_raw9, row_prefix), mantissa_plane

Usage:
  uv run python research/candidates/0009-fusible-exponent-codebook/tools/stz.py compress <snapshot_dir> <out_dir> [--shards N]
  uv run python research/candidates/0009-fusible-exponent-codebook/tools/stz.py verify <out_dir> <snapshot_dir> [--emit shard.st]
"""
from __future__ import annotations
import argparse, hashlib, json, mmap, os, re, struct, time
from pathlib import Path
import numpy as np

MAGIC = b"STZ1"
VERSION = 2
IS_EXPERT = re.compile(r"mixer\.experts\.\d+\.(up|down)_proj\.weight$")
GB = 1024 ** 3
CH = 1 << 23  # packing chunk (elements); multiple of 8 so chunk bit-counts stay byte-aligned


# ------------------------------------------------------------- bit packing ---
def pack_width(vals: np.ndarray, width: int) -> bytes:
    """Pack integer values < 2^width into a contiguous MSB-first bitstream."""
    n = vals.size
    if n == 0:
        return b""
    if width == 8:
        return vals.astype(np.uint8).tobytes()
    if width == 16:
        return vals.astype("<u2").tobytes()
    if width == 4:
        v = vals.astype(np.uint8)
        if n % 2:
            v = np.append(v, np.uint8(0))
        return ((v[0::2] << 4) | v[1::2]).tobytes()
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    out = bytearray()
    for s in range(0, n, CH):
        c = vals[s:s + CH].astype(np.uint64)
        bits = ((c[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)
        out += np.packbits(bits.ravel()).tobytes()
    return bytes(out)


def unpack_width(buf: bytes, n: int, width: int) -> np.ndarray:
    """Inverse of pack_width. Chunked (no full bit-expansion in RAM); returns the
    narrowest dtype that holds `width` bits."""
    dt = np.uint8 if width <= 8 else (np.uint16 if width <= 16 else np.uint64)
    if n == 0:
        return np.empty(0, dt)
    if width == 8:
        return np.frombuffer(buf, np.uint8)[:n].copy()
    if width == 16:
        return np.frombuffer(buf, "<u2")[:n].copy()
    if width == 4:
        b = np.frombuffer(buf, np.uint8)
        v = np.empty(b.size * 2, np.uint8)
        v[0::2] = b >> 4
        v[1::2] = b & 0xF
        return v[:n]
    raw = np.frombuffer(buf, np.uint8)
    w = (np.uint64(1) << np.arange(width - 1, -1, -1, dtype=np.uint64))
    out = np.empty(n, dt)
    for s in range(0, n, CH):  # CH*width % 8 == 0 -> every chunk starts byte-aligned
        e = min(s + CH, n)
        bits = np.unpackbits(raw[s * width // 8:(e * width + 7) // 8],
                             count=(e - s) * width)
        out[s:e] = (bits.reshape(-1, width).astype(np.uint64) @ w).astype(dt)
    return out


# ------------------------------------------------------------------ codec ---
def _pw(count: int) -> int:
    return max(1, int(np.ceil(np.log2(count + 1)))) if count else 1


def plan_regroup(hist: np.ndarray, n: int, R: int):
    """Exact serialized cost (bits) of every (b, k) variant; returns the argmin plan.

    hist: 512-bin histogram of the 9-bit sym field. Costs mirror the writer
    byte-for-byte, including per-stream byte-alignment padding. Ties break toward
    k=0 (the strictly row-addressable variant) then smaller b.
    """
    order = np.lexsort((np.arange(512), -hist))  # desc count, sym ties ascending
    counts = hist[order]
    best = {"variant": "raw16", "bits": n * 16}
    pad = lambda bits: -bits % 8  # byte alignment loss per stream
    padded = lambda bits: bits + pad(bits)
    head_bits = (1 + 1 + 4 + 8 + 8 + 1 + 1) * 8
    for b in (2, 3, 4, 5):
        K = (1 << b) - 1
        n_esc = int(n - counts[:K].sum())
        pw = _pw(n_esc)
        esc_counts = counts[K:]
        for k in (0, 3, 4, 5, 6):
            if k == 0:
                L = 0
                esc_bits = padded(n_esc * 9) + padded(R * pw)
                n_streams = 4
            else:
                L = (1 << k) - 1
                n_raw = n_esc - int(esc_counts[:L].sum())
                pw2 = _pw(n_raw)
                esc_bits = (padded(n_esc * k) + padded(n_raw * 9)
                            + padded(R * pw) + padded(R * pw2))
                n_streams = 6
            total = (padded(n * b) + esc_bits + padded(n * 7)
                     + (K + L) * 16 + head_bits + n_streams * 64)
            if total < best["bits"]:
                best = {"variant": "regroup", "bits": total, "b": b, "k": k,
                        "codebook": order[:K].astype(np.uint16),
                        "codebook2": order[K:K + L].astype(np.uint16) if k else np.empty(0, np.uint16),
                        "n_esc": n_esc, "pw": pw,
                        "pw2": _pw(n_raw) if k else 0}
    return best


def enc_tensor(raw: bytes, shape) -> tuple[int, list[bytes], dict]:
    """Encode one BF16 tensor. Returns (codec, chunks-to-write, stats)."""
    u = np.frombuffer(raw, "<u2")
    n = u.size
    sym = (u >> 7).astype(np.uint16)          # 9-bit sign+exponent field
    mant = (u & 0x7F).astype(np.uint8)        # 7-bit mantissa, fully live
    R = shape[0] if len(shape) >= 2 else 1
    plan = plan_regroup(np.bincount(sym, minlength=512), n, R)
    if plan["variant"] == "raw16":
        return 0, [bytes(raw)], {"bpw": 16.0}
    b, k, pw, pw2 = plan["b"], plan["k"], plan["pw"], plan["pw2"]
    K = (1 << b) - 1
    codemap = np.full(512, K, np.uint8)
    codemap[plan["codebook"]] = np.arange(K, dtype=np.uint8)
    idx = codemap[sym]
    esc_mask = idx == K
    esc_syms = sym[esc_mask]
    n_esc = int(esc_syms.size)
    assert n_esc == plan["n_esc"]

    def row_prefix(mask_counts_source, width):
        prefix = np.zeros(R, np.uint64)
        if R > 1:
            per_row = mask_counts_source.reshape(R, -1).sum(1, dtype=np.uint64)
            prefix[1:] = np.cumsum(per_row)[:-1]
        return pack_width(prefix, width)

    streams = [pack_width(idx, b)]
    if k:
        L = (1 << k) - 1
        codemap2 = np.full(512, L, np.uint16)
        codemap2[plan["codebook2"]] = np.arange(L, dtype=np.uint16)
        esc_codes = codemap2[esc_syms]
        raw_in_esc = esc_codes == L
        esc_raw = esc_syms[raw_in_esc]
        # raw-escape positions per row, for row-addressable raw stream
        raw_mask = np.zeros(n, bool)
        raw_mask[np.flatnonzero(esc_mask)[raw_in_esc]] = True
        streams += [pack_width(esc_codes, k), pack_width(esc_raw, 9),
                    row_prefix(esc_mask, pw), row_prefix(raw_mask, pw2)]
    else:
        streams += [pack_width(esc_syms, 9), row_prefix(esc_mask, pw)]
    streams.append(pack_width(mant, 7))
    head = struct.pack("<BBIQQBB", b, k, R, n, n_esc, pw, pw2)
    head += plan["codebook"].astype("<u2").tobytes() + plan["codebook2"].astype("<u2").tobytes()
    chunks = [head]
    for s in streams:
        chunks.append(struct.pack("<Q", len(s)))
        chunks.append(s)
    written = sum(len(c) for c in chunks)
    assert written * 8 == plan["bits"], (written * 8, plan["bits"])  # cost model is exact
    return 1, chunks, {"bpw": round(written * 8 / n, 4), "b": b, "k": k, "n_esc": n_esc}


def dec_tensor(f, nbytes: int) -> bytes:
    """Decode one regroup record from the stream; returns the original payload bytes."""
    b, k, R, n, n_esc, pw, pw2 = struct.unpack("<BBIQQBB", f.read(24))
    K = (1 << b) - 1
    codebook = np.frombuffer(f.read(K * 2), "<u2")
    L = (1 << k) - 1 if k else 0
    codebook2 = np.frombuffer(f.read(L * 2), "<u2") if k else np.empty(0, "<u2")

    def stream():
        (ln,) = struct.unpack("<Q", f.read(8))
        return f.read(ln)

    idx = unpack_width(stream(), n, b)                     # uint8
    cb = np.zeros(K + 1, np.uint16)
    cb[:K] = codebook
    sym = cb[idx]
    esc_mask = idx == K
    del idx
    if k:
        esc_codes = unpack_width(stream(), n_esc, k)       # uint8
        in2 = esc_codes < L
        esc_raw = unpack_width(stream(), int((~in2).sum()), 9)
        esc_syms = np.empty(n_esc, np.uint16)
        esc_syms[in2] = codebook2[esc_codes[in2].astype(np.int64)]
        esc_syms[~in2] = esc_raw.astype(np.uint16)
        prefix = unpack_width(stream(), R, pw)
        prefix2 = unpack_width(stream(), R, pw2)
    else:
        esc_syms = unpack_width(stream(), n_esc, 9).astype(np.uint16)
        prefix = unpack_width(stream(), R, pw)
        prefix2 = None
    sym[esc_mask] = esc_syms
    if R > 1:  # prefix tables well-formed (the random-access side structure)
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
    del esc_mask, prefix, prefix2
    mant = unpack_width(stream(), n, 7)                    # uint8
    u = (sym.astype(np.uint16) << 7) | mant
    out = u.astype("<u2").tobytes()
    assert len(out) == nbytes
    return out


# -------------------------------------------------------------- container ---
def st_header(p: Path):
    with p.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return 8 + n, json.loads(f.read(n))


def tensor_order(h: dict):
    metas = [(name, m) for name, m in h.items() if name != "__metadata__"]
    return sorted(metas, key=lambda kv: kv[1]["data_offsets"][0])


def compress_shard(shard: Path, out: Path):
    ds, h = st_header(shard)
    fsize = shard.stat().st_size
    sha = hashlib.sha256()
    with shard.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            sha.update(blk)
    orig_sha = sha.hexdigest()
    f = shard.open("rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    metas = tensor_order(h)
    # pre-scan gaps so they can sit BEFORE the records in the container
    pos, gaps = 0, []
    for name, meta in metas:
        b0, e0 = meta["data_offsets"]
        assert b0 >= pos and e0 >= b0, ("non-monotone data_offsets", name, b0, e0, pos)
        if b0 > pos:
            gaps.append((pos, bytes(mm[ds + pos:ds + b0])))
        pos = e0
    if fsize - ds - pos > 0:
        gaps.append((pos, bytes(mm[ds + pos:fsize])))
    stats = []
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as w:
        name_b = shard.name.encode()
        w.write(MAGIC + struct.pack("<HHI", VERSION, 0, len(name_b)) + name_b)
        w.write(bytes.fromhex(orig_sha) + struct.pack("<QQ", fsize, ds))
        w.write(mm[:ds])
        w.write(struct.pack("<I", len(gaps)))
        for off, data in gaps:
            w.write(struct.pack("<QQ", off, len(data)) + data)
        w.write(struct.pack("<I", len(metas)))
        for name, meta in metas:
            b0, e0 = meta["data_offsets"]
            raw = mm[ds + b0:ds + e0]  # bytes copy: one tensor at a time in RAM
            nb = e0 - b0
            if meta["dtype"] == "BF16" and nb >= 2:
                codec, chunks, st = enc_tensor(raw, meta["shape"])
            else:
                codec, chunks, st = 0, [raw], {"bpw": 16.0}
            nm = name.encode()
            w.write(struct.pack("<I", len(nm)) + nm + struct.pack("<BQ", codec, nb))
            for c in chunks:
                w.write(c)
            stats.append({"name": name, "dtype": meta["dtype"], "nbytes": nb, "codec": codec,
                          "expert": bool(IS_EXPERT.search(name)), **st})
    mm.close(); f.close()
    os.replace(tmp, out)  # atomic: a crash never leaves a plausible partial .stz
    return {"shard": shard.name, "orig_sha256": orig_sha, "orig_bytes": fsize,
            "stz_bytes": out.stat().st_size, "tensors": stats}


def verify_stz(stz: Path, snapshot: Path, emit: Path | None = None):
    """Decode a .stz using ONLY its own bytes, streaming (peak RAM = one tensor);
    SHA-256 the reconstruction and compare against both the hash stored at encode
    time and a fresh hash of the original file."""
    with stz.open("rb") as f:
        assert f.read(4) == MAGIC
        ver, _, nl = struct.unpack("<HHI", f.read(8))
        assert ver == VERSION, f"format v{ver}, expected v{VERSION}"
        shard_name = f.read(nl).decode()
        stored_sha = f.read(32).hex()
        fsize, ds = struct.unpack("<QQ", f.read(16))
        hdr = f.read(ds)
        (n_gaps,) = struct.unpack("<I", f.read(4))
        gaps = []
        for _ in range(n_gaps):
            off, ln = struct.unpack("<QQ", f.read(16))
            gaps.append((off, f.read(ln)))
        (n_rec,) = struct.unpack("<I", f.read(4))
        metas = tensor_order(json.loads(hdr[8:].decode()))
        assert len(metas) == n_rec
        sha = hashlib.sha256()
        sha.update(hdr)
        total = len(hdr)
        out_f = emit.open("wb") if emit else None
        if out_f:
            out_f.write(hdr)

        def put(data):
            nonlocal total
            sha.update(data)
            total += len(data)
            if out_f:
                out_f.write(data)

        cursor, gi = 0, 0
        for meta_name, meta in metas:
            (nl,) = struct.unpack("<I", f.read(4))
            name = f.read(nl).decode()
            assert name == meta_name, (name, meta_name)
            codec, nb = struct.unpack("<BQ", f.read(9))
            b0, e0 = meta["data_offsets"]
            assert nb == e0 - b0
            if b0 > cursor:
                off, g = gaps[gi]; gi += 1
                assert off == cursor and len(g) == b0 - cursor
                put(g)
            put(f.read(nb) if codec == 0 else dec_tensor(f, nb))
            cursor = e0
        if gi < len(gaps):
            off, g = gaps[gi]; gi += 1
            assert off == cursor
            put(g)
        assert gi == len(gaps)
        if out_f:
            out_f.close()
    assert total == fsize, (total, fsize)
    rec_sha = sha.hexdigest()
    fresh = hashlib.sha256()
    with (snapshot / shard_name).open("rb") as f2:
        for blk in iter(lambda: f2.read(1 << 24), b""):
            fresh.update(blk)
    return {"shard": shard_name, "sha256_reconstructed": rec_sha,
            "sha256_stored_at_encode": stored_sha, "sha256_original_now": fresh.hexdigest(),
            "MATCH": rec_sha == stored_sha == fresh.hexdigest()}


# --------------------------------------------------------------------- cli ---
def index_shards(snap: Path) -> list[str]:
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    return sorted(set(idx["weight_map"].values()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compress"); c.add_argument("snapshot"); c.add_argument("out")
    c.add_argument("--shards", type=int, default=0, help="limit (0 = all)")
    v = sub.add_parser("verify"); v.add_argument("out"); v.add_argument("snapshot")
    v.add_argument("--emit", default=None, help="also write one reconstructed shard to this path")
    a = ap.parse_args()
    snap, outd = Path(a.snapshot), Path(a.out)
    all_shards = index_shards(snap)
    if a.cmd == "compress":
        outd.mkdir(parents=True, exist_ok=True)
        shards = all_shards[:a.shards] if a.shards else all_shards
        manifest, t0 = [], time.time()
        for i, sh in enumerate(shards, 1):
            stz_p, meta_p = outd / (sh + ".stz"), outd / (sh + ".stz.meta.json")
            if stz_p.exists() and meta_p.exists():  # resume: skip completed shards
                slim = json.loads(meta_p.read_text())
                manifest.append(slim)
                print(f"[{i}/{len(shards)}] {sh}: resume-skip ({slim['reduction_pct']}%)", flush=True)
                continue
            r = compress_shard(snap / sh, stz_p)
            slim = {k: r[k] for k in ("shard", "orig_sha256", "orig_bytes", "stz_bytes")}
            slim["reduction_pct"] = round(100 * (1 - r["stz_bytes"] / r["orig_bytes"]), 2)
            meta_p.write_text(json.dumps(slim))
            (outd / (sh + ".stz.stats.jsonl")).write_text(
                "\n".join(json.dumps(t) for t in r["tensors"]) + "\n")
            manifest.append(slim)
            done = {"shards_expected": len(all_shards), "shards_done": len(manifest),
                    "complete": len(manifest) == len(all_shards), "shards": manifest,
                    "total_orig_bytes": sum(m["orig_bytes"] for m in manifest),
                    "total_stz_bytes": sum(m["stz_bytes"] for m in manifest)}
            done["whole_model_reduction_pct"] = round(
                100 * (1 - done["total_stz_bytes"] / done["total_orig_bytes"]), 2)
            (outd / "stz_manifest.json").write_text(json.dumps(done, indent=2))
            print(f"[{i}/{len(shards)}] {sh}: {slim['reduction_pct']}% "
                  f"({slim['stz_bytes']/GB:.2f} GiB) elapsed={time.time()-t0:.0f}s", flush=True)
        print(json.dumps({k: v for k, v in done.items() if k != "shards"}, indent=2))
    else:
        stzs = sorted(outd.glob("*.stz"))
        found = {s.name[:-4] for s in stzs}
        expected = set(all_shards)
        results, t0 = [], time.time()
        for i, s in enumerate(stzs, 1):
            emit = Path(a.emit) if (a.emit and i == 1) else None
            r = verify_stz(s, snap, emit)
            results.append(r)
            print(f"[{i}/{len(stzs)}] {r['shard']}: MATCH={r['MATCH']} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        report = {
            "WHOLE_MODEL_VERIFIED": found == expected and all(r["MATCH"] for r in results),
            "coverage_complete": found == expected,
            "shards_expected": len(expected), "shards_verified": len(results),
            "missing": sorted(expected - found),
            "all_present_match": all(r["MATCH"] for r in results),
            "shards": results,
        }
        (outd / "stz_verify.json").write_text(json.dumps(report, indent=2))
        print(f"WHOLE_MODEL_VERIFIED: {report['WHOLE_MODEL_VERIFIED']} "
              f"(coverage {len(results)}/{len(expected)})")


if __name__ == "__main__":
    main()
