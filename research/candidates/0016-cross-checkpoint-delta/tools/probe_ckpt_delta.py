"""Candidate 0016 probe: exact cross-checkpoint delta coding (Direction C).

Direction of the encoding (matches the claim): TARGET = the LOCAL checkpoint
(the release, fully on disk) coded GIVEN BASE = the SIBLING checkpoint
(base/pre-RL revision on HF, streamed one shard at a time and deleted after
use).  Measures, per aligned BF16 tensor pair:

  1. %-bit-identical uint16 words + run-length structure of the match mask
     (concrete RLE cost: best of raw mask / Elias-gamma run lengths /
     Rice-coded gaps, chooser + count fields charged).
  2. XOR plane split.  Field split for BF16 (u16 LE):  sym = u >> 7 (9 bits,
     sign+exponent), mant = u & 0x7F (7 bits).  Reports H0 of the XOR high
     byte, XOR sym and XOR mantissa, over all words and over non-match words.
  3. An exact delta-coding cost model from those measurements, ALL side costs
     charged:  match-mask RLE + non-match XOR syms coded at H0 (full histogram
     table charged) + non-match mantissas verbatim (7 b each) + per-tensor
     name/shape header.  => projected b/w for the RELEASE checkpoint given the
     base.  A second variant prices the non-match XOR mantissas at H0 (with the
     same histogram-table charge) so the real run can distinguish "delta
     mantissas are incompressible" from "model refused to price them".  The
     field-split decomposition is mechanically reconstructed and asserted
     bit-exact for every tensor (lossless by construction, checked).
  4. Baselines on the same bytes:
       - zstd --patch-from semantics.  Neither the `zstd` nor `xdelta3` CLI is
         installed on this Windows box (checked at runtime and recorded), so
         patch-from is implemented with the python `zstandard` package: the
         base bytes are attached as a DICT_TYPE_RAWCONTENT dictionary
         (prefix-dictionary semantics, same mechanism `--patch-from` uses),
         long-distance matching enabled, window_log sized to cover
         dict+target (capped at 31 = zstd's own limit, which is also why the
         whole-shard variant is skipped for real ~5 GB shards; the per-tensor
         variant is exactly aligned and always runs).  Level 19.  Every
         baseline compression is round-trip verified.
       - xdelta3-class: tried via python module then CLI; availability is
         recorded.  If absent, zstd-patch-from-with-LDM is the delta-class
         baseline of record and the summary documents that.
  5. Sanity, no-base reference: standalone stz-class cost of the TARGET
     (release) tensors (candidate 0009 regroup-K15 accounting, ~11.2-11.3 b/w
     on the real model vs 10.90 realized .stz) and a streaming standalone zstd
     (level 9, multithreaded — reference number only, no vetting weight) of
     the whole release shard bytes.

Resumable: every result is one JSONL line in <work>/results.jsonl keyed by
(pair, type, name); reruns skip finished keys.  <work>/summary.json is
rewritten after every shard;  --summarize rebuilds it from the JSONL alone.

Examples
--------
  # synthetic smoke (no network): fake a sibling by flipping mantissa LSBs on
  # a controlled fraction of words of the synthetic snapshot
  uv run python research/candidates/0016-cross-checkpoint-delta/tools/probe_ckpt_delta.py --synthetic

  # exercise the sym-coding path too (also flip the low exponent bit on 0.2%)
  uv run python research/candidates/0016-cross-checkpoint-delta/tools/probe_ckpt_delta.py --synthetic --flip-sym-frac 0.002 --work-tag symflip

  # real run: base shard 1 (embedding+layers 0-3) and shard 7 (expert-heavy),
  # ~10 GB download, one shard on disk at a time, deleted after processing
  uv run python research/candidates/0016-cross-checkpoint-delta/tools/probe_ckpt_delta.py \
      --shards model-00001-of-00013.safetensors model-00007-of-00013.safetensors

  # summary only (no processing)
  uv run python research/candidates/0016-cross-checkpoint-delta/tools/probe_ckpt_delta.py --summarize
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

# xet downloads die on connection resets with no in-flight retry; plain HTTP
# resumes.  Must be set before huggingface_hub is imported (real mode imports
# it lazily, but keep the guarantee unconditional).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
CAND_DIR = SCRIPT_DIR.parent
DEFAULT_WORK = CAND_DIR / "tests" / "artifacts" / "probe_work"

DEFAULT_LOCAL_SNAPSHOT = Path(
    "C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot")
DEFAULT_REPO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
DEFAULT_REVISION = "97ab8012882a655dc38df4fee47422aca9caca07"  # pinned; weights never modified
SYNTH_SNAPSHOT = Path("C:/dev/compression/models/synthetic/nemotron_tiny/hf_snapshot")

K = 15  # candidate 0009 regroup codebook size (stz-class accounting)


# --------------------------------------------------------------------------- io
def read_header(path: Path):
    """safetensors header: returns (data_start, header_dict)."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


class LocalSnapshot:
    """Random access to tensors of the on-disk checkpoint via its index."""

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        idx = json.loads((snapshot / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = idx["weight_map"]
        self._open: dict[str, tuple] = {}

    def _shard(self, shard_name: str):
        if shard_name not in self._open:
            p = self.snapshot / shard_name
            ds, hdr = read_header(p)
            f = p.open("rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._open[shard_name] = (f, mm, ds, hdr)
        return self._open[shard_name]

    def get(self, name: str):
        """-> (meta, raw_bytes) or None if the tensor is not in the local model."""
        shard_name = self.weight_map.get(name)
        if shard_name is None:
            return None
        _, mm, ds, hdr = self._shard(shard_name)
        meta = hdr[name]
        b, e = meta["data_offsets"]
        return meta, mm[ds + b: ds + e]

    def close(self):
        for f, mm, _, _ in self._open.values():
            mm.close()
            f.close()
        self._open.clear()


# ------------------------------------------------------------------ entropy/rle
def h0_bits(vals: np.ndarray, minlength: int) -> float:
    """Zeroth-order entropy in bits/symbol of an integer array."""
    if vals.size == 0:
        return 0.0
    c = np.bincount(vals, minlength=minlength).astype(np.float64)
    p = c[c > 0] / vals.size
    return float(-(p * np.log2(p)).sum())


def run_lengths(mask: np.ndarray):
    change = np.flatnonzero(mask[1:] != mask[:-1])
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [mask.size]))
    return (ends - starts).astype(np.int64)


def gamma_bits_total(vals: np.ndarray) -> int:
    """Total Elias-gamma length of positive ints (2*floor(log2(v))+1)."""
    if vals.size == 0:
        return 0
    return int((2 * np.floor(np.log2(vals.astype(np.float64))) + 1).sum())


def rice_bits_total(gaps: np.ndarray) -> tuple[int, int]:
    """Best Rice-coded total bits for gaps>=1 (coded as gap-1) and the chosen k.
    Cost per value v: (v>>k) + 1 + k.  +5 bits charged for storing k."""
    g = (gaps - 1).astype(np.int32)  # gaps < 2^31; sums accumulated in int64
    best_bits, best_k = None, 0
    for k in range(17):
        bits = int((g >> k).sum(dtype=np.int64)) + g.size * (1 + k)
        if best_bits is None or bits < best_bits:
            best_bits, best_k = bits, k
    return best_bits + 5, best_k


def mask_cost_bits(match: np.ndarray, nonmatch_idx: np.ndarray):
    """Concrete, decodable cost of the match mask.  Chooser (2 b) + count (40 b)
    always charged.  Options: raw bitmap; Elias-gamma RLE (1 b first value);
    Rice-coded gaps between non-match positions."""
    n = match.size
    options: dict[str, int] = {"raw": n}
    runs = run_lengths(match)
    options["gamma_rle"] = 1 + gamma_bits_total(runs)
    if nonmatch_idx.size:
        gaps = np.diff(nonmatch_idx, prepend=-1)  # first gap = idx0 + 1 >= 1
        rb, _k = rice_bits_total(gaps)
        options["rice_gaps"] = rb
    choice = min(options, key=options.get)  # type: ignore[arg-type]
    return options[choice] + 2 + 40, choice, {k: int(v) for k, v in options.items()}, int(runs.size)


# ------------------------------------------------------- stz-class (0009) model
def bits_regroup(raw: bytes, R: int) -> int:
    """Candidate 0009 regroup-K15 standalone accounting (verbatim from
    stream_validate.py): 4-bit index plane + 9-bit escapes + escape row index +
    codebook + 7-bit mantissa verbatim.  ~11.2-11.3 b/w on the real model."""
    u = np.frombuffer(raw, np.uint16)
    sym = (((u >> 15).astype(np.uint32) << 8) | ((u >> 7) & 0xFF).astype(np.uint32))
    n = u.size
    hist = np.bincount(sym, minlength=512)
    n_esc = n - int(np.sort(hist)[::-1][:K].sum())
    return n * 4 + n_esc * 9 + R * max(1, int(np.ceil(np.log2(n_esc + 1)))) + K * 9 + n * 7


# ------------------------------------------------------------------- delta model
def tensor_delta_record(name: str, meta: dict, base_raw: bytes, target_raw: bytes,
                        baseline_level: int) -> dict:
    """All per-tensor measurements + the exact cost model, side costs charged.

    Direction: base_raw = the GIVEN checkpoint (sibling/pre-RL base on the real
    run), target_raw = the checkpoint being ENCODED (the local release).  Casts
    are int32 (values <= 511) and intermediates freed sequentially to keep the
    peak reasonable on the ~700 MB tensors."""
    a = np.frombuffer(base_raw, np.uint16)    # base checkpoint (given)
    b = np.frombuffer(target_raw, np.uint16)  # target = what we must encode
    n = int(b.size)
    x = a ^ b
    match = x == 0
    nonmatch_idx = np.flatnonzero(~match).astype(np.int32)  # n < 2^31 always
    n_nm = int(nonmatch_idx.size)

    # H0 stats, computed one field at a time (int32, freed before the next)
    xhi = (x >> 8).astype(np.int32)             # high BYTE of the XOR
    h0_hi_all = h0_bits(xhi, 256)
    del xhi
    xs = (x >> 7).astype(np.int32)              # 9-bit sym field of the XOR
    h0_sym_all = h0_bits(xs, 512)
    xs_nm = xs[nonmatch_idx]
    del xs
    xm = (x & 0x7F).astype(np.int32)            # 7-bit mantissa field of the XOR
    h0_mant_all = h0_bits(xm, 128)
    xm_nm = xm[nonmatch_idx]
    del xm

    # (1) match mask
    mask_bits, mask_choice, mask_options, n_runs = mask_cost_bits(match, nonmatch_idx)
    del match

    # (3) cost model: mask + coded non-match syms (+ table) + verbatim mantissas
    if n_nm:
        d = int(np.count_nonzero(np.bincount(xs_nm, minlength=512)))
        h0_sym_nm = h0_bits(xs_nm, 512)
        sym_payload_bits = int(math.ceil(n_nm * h0_sym_nm))
        sym_table_bits = 16 + d * (9 + 32)      # distinct-count + (value,count) rows
        dm = int(np.count_nonzero(np.bincount(xm_nm, minlength=128)))
        h0_mant_nm = h0_bits(xm_nm, 128)
        mant_h0_payload_bits = int(math.ceil(n_nm * h0_mant_nm))
        mant_h0_table_bits = 16 + dm * (7 + 32)
    else:
        d = dm = 0
        h0_sym_nm = h0_mant_nm = 0.0
        sym_payload_bits = mant_h0_payload_bits = 0
        sym_table_bits = mant_h0_table_bits = 16
    mant_bits = 7 * n_nm
    header_bits = len(name.encode()) * 8 + 96   # name + numel/shape/offsets fields
    common_bits = mask_bits + sym_payload_bits + sym_table_bits + header_bits
    total_bits = common_bits + mant_bits                      # mantissas verbatim
    total_bits_h0mant = common_bits + mant_h0_payload_bits + mant_h0_table_bits

    # mechanical exactness check of the split-field decomposition
    rec = a.copy()
    rec[nonmatch_idx] = a[nonmatch_idx] ^ ((xs_nm.astype(np.uint16) << 7)
                                           | xm_nm.astype(np.uint16))
    recon_exact = bool(np.array_equal(rec, b))
    del rec, x, xs_nm, xm_nm, nonmatch_idx

    # (4) per-tensor zstd patch-from baseline (base tensor bytes as raw dict);
    #     target (release) coded given base (sibling) — same direction as model
    pf_bytes, pf_desc = zstd_patch_from(target_raw, base_raw, baseline_level)

    # (5) standalone stz-class no-base reference on the TARGET (release) tensor
    shape = meta.get("shape") or [1]
    stz_bits = bits_regroup(target_raw, shape[0] if shape else 1)

    return {
        "type": "tensor", "name": name, "shape": meta.get("shape"), "numel": n,
        "pct_match": round(100.0 * (n - n_nm) / n, 4),
        "n_nonmatch": n_nm, "n_runs": n_runs,
        "h0_xor_hi_byte_all": round(h0_hi_all, 4),
        "h0_xor_sym_all": round(h0_sym_all, 4),
        "h0_xor_sym_nonmatch": round(h0_sym_nm, 4),
        "h0_xor_mant_all": round(h0_mant_all, 4),
        "h0_xor_mant_nonmatch": round(h0_mant_nm, 4),
        "mask_bits": int(mask_bits), "mask_choice": mask_choice,
        "mask_options_bits": mask_options,
        "sym_distinct": d, "sym_payload_bits": sym_payload_bits,
        "sym_table_bits": sym_table_bits, "mant_bits": mant_bits,
        "mant_h0_payload_bits": mant_h0_payload_bits,
        "mant_h0_table_bits": mant_h0_table_bits, "mant_distinct": dm,
        "header_bits": header_bits,
        "model_total_bits": int(total_bits),
        "model_bw": round(total_bits / n, 4),
        "model_total_bits_h0mant": int(total_bits_h0mant),
        "model_bw_h0mant": round(total_bits_h0mant / n, 4),
        "recon_exact": recon_exact,
        "patchfrom_tensor_bytes": pf_bytes, "patchfrom_desc": pf_desc,
        "patchfrom_tensor_bw": round(pf_bytes * 8 / n, 4),
        "stz_class_bits": int(stz_bits),
        "stz_class_bw": round(stz_bits / n, 4),
    }


# -------------------------------------------------------------------- baselines
def zstd_patch_from(target: bytes, base: bytes, level: int):
    """zstd --patch-from semantics via python zstandard: base bytes as a
    DICT_TYPE_RAWCONTENT dictionary, LDM on, window covering dict+target
    (capped at zstd's 2^31 limit).  Round-trip verified."""
    import zstandard as zstd
    need = max(len(base) + len(target), 1 << 10)
    wl = min(31, max(10, need.bit_length()))
    d = zstd.ZstdCompressionDict(base, dict_type=zstd.DICT_TYPE_RAWCONTENT)
    params = zstd.ZstdCompressionParameters.from_level(
        level, window_log=wl, enable_ldm=True)
    comp = zstd.ZstdCompressor(compression_params=params, dict_data=d).compress(target)
    dec = zstd.ZstdDecompressor(dict_data=d, max_window_size=1 << wl).decompress(
        comp, max_output_size=len(target))
    if dec != target:
        raise RuntimeError("zstd patch-from round-trip mismatch")
    desc = (f"python-zstandard {zstd.__version__} level={level} window_log={wl} "
            f"enable_ldm=True dict=DICT_TYPE_RAWCONTENT(base) round_trip=verified")
    return len(comp), desc


def zstd_standalone_stream(mm, level: int = 9):
    """Streaming, multithreaded standalone zstd over shard-sized bytes (a
    no-base REFERENCE number only — carries no vetting weight, hence level 9
    threaded instead of a ~40-min single-shot level 19).  Chunked round-trip
    verify against the source, nothing shard-sized copied into RAM."""
    import zstandard as zstd
    ch = 1 << 26  # 64 MiB
    n = len(mm)
    cobj = zstd.ZstdCompressor(level=level, threads=-1).compressobj()
    parts = [cobj.compress(mm[off:off + ch]) for off in range(0, n, ch)]
    parts.append(cobj.flush())
    comp = b"".join(parts)
    del parts
    dobj = zstd.ZstdDecompressor().decompressobj()
    pos = 0
    for off in range(0, len(comp), ch):
        out = dobj.decompress(comp[off:off + ch])
        if out:
            if mm[pos:pos + len(out)] != out:
                raise RuntimeError("zstd standalone round-trip mismatch")
            pos += len(out)
    if pos != n:
        raise RuntimeError(f"zstd standalone round-trip short: {pos} != {n}")
    return len(comp), f"python-zstandard streaming level={level} threads=-1 round_trip=verified"


def xdelta3_status():
    """Record what delta-class tooling exists on this box."""
    try:
        import xdelta3  # type: ignore
        return {"available": True, "via": "python module xdelta3"}
    except Exception:
        pass
    cli = shutil.which("xdelta3")
    if cli:
        return {"available": True, "via": f"CLI {cli}"}
    return {"available": False,
            "note": "no xdelta3 CLI or python module on this box; "
                    "zstd patch-from (raw-content dict + LDM) is the "
                    "delta-class baseline of record"}


def xdelta3_encode(target: bytes, base: bytes):
    """Only called when xdelta3 is available (module or CLI)."""
    try:
        import xdelta3  # type: ignore
        patch = xdelta3.encode(base, target)
        if xdelta3.decode(base, patch) != target:
            raise RuntimeError("xdelta3 round-trip mismatch")
        return len(patch), "python module xdelta3, round_trip=verified"
    except ImportError:
        pass
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "base").write_bytes(base)
        (tdp / "tgt").write_bytes(target)
        subprocess.run(["xdelta3", "-e", "-9", "-f", "-s", str(tdp / "base"),
                        str(tdp / "tgt"), str(tdp / "patch")], check=True)
        subprocess.run(["xdelta3", "-d", "-f", "-s", str(tdp / "base"),
                        str(tdp / "patch"), str(tdp / "rec")], check=True)
        if (tdp / "rec").read_bytes() != target:
            raise RuntimeError("xdelta3 CLI round-trip mismatch")
        return (tdp / "patch").stat().st_size, "xdelta3 CLI -e -9, round_trip=verified"


# ------------------------------------------------------------------- shard loop
def process_sibling_shard(sib_path: Path, shard_name: str, pair: str,
                          local: LocalSnapshot, done: set, jsonl, args):
    ds, hdr = read_header(sib_path)
    f = sib_path.open("rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    names = [k for k in hdr if k != "__metadata__"]
    names.sort(key=lambda k: hdr[k]["data_offsets"][0])
    t0 = time.time()
    n_done = 0
    for name in names:
        key = f"{pair}::tensor::{name}"
        skey = f"{pair}::skip::{name}"
        if key in done or skey in done:
            continue
        meta = hdr[name]
        b, e = meta["data_offsets"]
        loc = local.get(name)
        rec = None
        if loc is None:
            rec = {"type": "skip", "name": name, "reason": "not in local checkpoint",
                   "bytes": e - b}
        else:
            lmeta, lraw = loc
            lb, le = lmeta["data_offsets"]
            if meta["dtype"] != "BF16" or lmeta["dtype"] != "BF16":
                rec = {"type": "skip", "name": name,
                       "reason": f"dtype pair {lmeta['dtype']}(local)/{meta['dtype']}(sibling), probe is BF16-only",
                       "bytes": e - b, "local_bytes": le - lb}
            elif lmeta["shape"] != meta["shape"]:
                rec = {"type": "skip", "name": name,
                       "reason": f"shape mismatch {lmeta['shape']} vs {meta['shape']}",
                       "bytes": e - b, "local_bytes": le - lb}
            else:
                # direction matches the claim: TARGET = local release,
                # BASE = downloaded sibling
                rec = tensor_delta_record(name, lmeta, mm[ds + b: ds + e], lraw,
                                          args.baseline_level)
        rec.update({"pair": pair, "sibling_shard": shard_name})
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        done.add(key if rec["type"] == "tensor" else skey)
        n_done += 1
        if rec["type"] == "tensor" and (n_done % 50 == 0 or rec["numel"] > 10_000_000):
            print(f"  [{pair}/{shard_name}] {n_done} tensors, last={name} "
                  f"match={rec['pct_match']}% bw={rec['model_bw']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # shard-level record: byte baselines on the full shard pair.  Direction as
    # everywhere: TARGET = local (release) shard, BASE = sibling shard.
    shard_key = f"{pair}::shard::{shard_name}"
    if shard_key not in done:
        rec = {"type": "shard", "pair": pair, "sibling_shard": shard_name,
               "sibling_shard_bytes": len(mm),
               "sibling_header_bytes": ds,
               "cli_zstd": shutil.which("zstd") or "not installed",
               "cli_xdelta3": shutil.which("xdelta3") or "not installed",
               "xdelta3": xdelta3_status()}
        local_shard_path = local.snapshot / shard_name
        if local_shard_path.exists():
            lsize = local_shard_path.stat().st_size  # no 5 GB read for a size check
            rec["local_shard_bytes"] = lsize
            rec["local_header_bytes"] = read_header(local_shard_path)[0]
            if max(lsize, len(mm)) <= args.max_whole_shard_bytes:
                tgt_bytes = local_shard_path.read_bytes()  # target = release shard
                base_bytes = mm[:]                         # base   = sibling shard
                n_c, desc = zstd_patch_from(tgt_bytes, base_bytes, args.baseline_level)
                rec["whole_shard_patchfrom"] = {
                    "compressed_bytes": n_c, "desc": desc,
                    "target": "local(release) shard", "base": "sibling shard"}
                if rec["xdelta3"]["available"]:
                    n_x, xdesc = xdelta3_encode(tgt_bytes, base_bytes)
                    rec["whole_shard_xdelta3"] = {
                        "compressed_bytes": n_x, "desc": xdesc,
                        "target": "local(release) shard", "base": "sibling shard"}
                del tgt_bytes, base_bytes
            else:
                rec["whole_shard_patchfrom"] = {
                    "skipped": f"shard exceeds --max-whole-shard-bytes="
                               f"{args.max_whole_shard_bytes} (zstd window limit is "
                               f"2^31; per-tensor patch-from is the aligned baseline)"}
        else:
            rec["whole_shard_patchfrom"] = {
                "skipped": "no same-named local shard (boundary shift); "
                           "per-tensor patch-from is the aligned baseline"}
        # standalone no-base reference on the TARGET (release) shard when it
        # exists (falls back to the sibling shard bytes otherwise)
        if local_shard_path.exists():
            with local_shard_path.open("rb") as lf:
                lmm = mmap.mmap(lf.fileno(), 0, access=mmap.ACCESS_READ)
                nz, zdesc = zstd_standalone_stream(lmm, level=9)
                lmm.close()
            rec["standalone_zstd"] = {"compressed_bytes": nz,
                                      "desc": zdesc + " target=local(release) shard",
                                      "shard_bytes": lsize}
        else:
            nz, zdesc = zstd_standalone_stream(mm, level=9)
            rec["standalone_zstd"] = {"compressed_bytes": nz,
                                      "desc": zdesc + " target=sibling shard (no same-named local)",
                                      "shard_bytes": len(mm)}
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        done.add(shard_key)
    mm.close()
    f.close()


# --------------------------------------------------------------------- summary
def read_results(results: Path) -> list[dict]:
    """Decode results.jsonl tolerantly: a truncated TRAILING line (what a
    process kill mid-write leaves behind) is dropped and truncated off the
    file so both resume and --summarize keep working; a decode error anywhere
    before the final line still raises (that is real corruption)."""
    recs: list[dict] = []
    if not results.exists():
        return recs
    raw = results.read_bytes()
    off, n = 0, len(raw)
    while off < n:
        nl = raw.find(b"\n", off)
        end = n if nl == -1 else nl
        line = raw[off:end].strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                if raw[end:].strip():
                    raise  # mid-file corruption: do not silently drop records
                print(f"[repair] dropping truncated trailing line "
                      f"({end - off} bytes) in {results} and truncating file",
                      flush=True)
                with results.open("r+b") as f:
                    f.truncate(off)
                break
        off = end + 1
    return recs


def summarize(work: Path) -> dict:
    results = work / "results.jsonl"
    pairs: dict[str, dict] = {}
    for rec in read_results(results):
                p = pairs.setdefault(rec["pair"], {
                    "tensors": 0, "skips": [], "numel": 0, "model_bits": 0,
                    "model_bits_h0mant": 0,
                    "stz_bits": 0, "pf_bytes": 0, "nm": 0, "runs": 0,
                    "recon_all_exact": True, "wsum_h0_sym_nm": 0.0,
                    "wsum_h0_hi_all": 0.0, "wsum_h0_mant_all": 0.0,
                    "wsum_h0_mant_nm": 0.0, "nm_numel": 0, "shards": {},
                    "worst_match_pct": None, "best_match_pct": None})
                if rec["type"] == "tensor":
                    p["tensors"] += 1
                    p["numel"] += rec["numel"]
                    p["model_bits"] += rec["model_total_bits"]
                    p["model_bits_h0mant"] += rec.get("model_total_bits_h0mant",
                                                      rec["model_total_bits"])
                    p["stz_bits"] += rec["stz_class_bits"]
                    p["pf_bytes"] += rec["patchfrom_tensor_bytes"]
                    p["nm"] += rec["n_nonmatch"]
                    p["runs"] += rec["n_runs"]
                    p["recon_all_exact"] &= rec["recon_exact"]
                    p["wsum_h0_sym_nm"] += rec["h0_xor_sym_nonmatch"] * rec["n_nonmatch"]
                    p["wsum_h0_hi_all"] += rec["h0_xor_hi_byte_all"] * rec["numel"]
                    p["wsum_h0_mant_all"] += rec.get("h0_xor_mant_all", 0.0) * rec["numel"]
                    p["wsum_h0_mant_nm"] += rec.get("h0_xor_mant_nonmatch", 0.0) * rec["n_nonmatch"]
                    p["nm_numel"] += rec["n_nonmatch"]
                    m = rec["pct_match"]
                    p["worst_match_pct"] = m if p["worst_match_pct"] is None else min(p["worst_match_pct"], m)
                    p["best_match_pct"] = m if p["best_match_pct"] is None else max(p["best_match_pct"], m)
                elif rec["type"] == "skip":
                    p["skips"].append({"name": rec["name"], "reason": rec["reason"],
                                       "bytes": rec["bytes"],
                                       "local_bytes": rec.get("local_bytes")})
                elif rec["type"] == "shard":
                    p["shards"][rec["sibling_shard"]] = {
                        k: rec.get(k) for k in
                        ("sibling_shard_bytes", "sibling_header_bytes",
                         "local_shard_bytes", "local_header_bytes",
                         "whole_shard_patchfrom", "whole_shard_xdelta3",
                         "standalone_zstd", "standalone_zstd19_bytes", "xdelta3",
                         "cli_zstd", "cli_xdelta3")}

    out: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "results_jsonl": str(results), "pairs": {}}
    for name, p in pairs.items():
        n = p["numel"]
        if n == 0:
            out["pairs"][name] = {"tensors": 0, "skips": p["skips"], "shards": p["shards"]}
            continue
        model_bw = p["model_bits"] / n
        model_bw_h0m = p["model_bits_h0mant"] / n
        pf_bw = p["pf_bytes"] * 8 / n
        stz_bw = p["stz_bits"] / n
        entry = {
            "tensors": p["tensors"],
            "paired_numel": n,
            "pct_bit_identical_words": round(100.0 * (n - p["nm"]) / n, 4),
            "match_pct_range_per_tensor": [p["worst_match_pct"], p["best_match_pct"]],
            "match_mask_runs": p["runs"],
            "h0_xor_sym_nonmatch_weighted": round(p["wsum_h0_sym_nm"] / p["nm_numel"], 4) if p["nm_numel"] else 0.0,
            "h0_xor_hi_byte_all_weighted": round(p["wsum_h0_hi_all"] / n, 4),
            "h0_xor_mant_all_weighted": round(p["wsum_h0_mant_all"] / n, 4),
            "h0_xor_mant_nonmatch_weighted": round(p["wsum_h0_mant_nm"] / p["nm_numel"], 4) if p["nm_numel"] else 0.0,
            "all_reconstructions_exact": p["recon_all_exact"],
            "delta_model_bw": round(model_bw, 4),
            "delta_model_bw_h0mant": round(model_bw_h0m, 4),
            "delta_model_reduction_vs_16bw_pct": round(100 * (1 - model_bw / 16), 2),
            "baseline_patchfrom_per_tensor_bw": round(pf_bw, 4),
            "standalone_stz_class_bw": round(stz_bw, 4),
            "model_beats_patchfrom": bool(model_bw < pf_bw),
            "model_beats_xdelta3": None,  # filled below if a whole-shard xdelta3 ran
            "skips": p["skips"],
            "shards": p["shards"],
        }
        # xdelta3 comparison only exists where a whole-shard xdelta3 ran.
        # Equal byte scope on BOTH sides of the inequality: xdelta3 covers the
        # whole target (release) shard — header + skipped tensors + paired
        # tensors — so the model side is charged model bits + skipped tensors
        # verbatim (8 b/B) + the safetensors header verbatim over the same
        # shards, and b/w is per shard uint16 word, not per paired word.
        xd_shards = {sh: s for sh, s in p["shards"].items()
                     if isinstance(s.get("whole_shard_xdelta3"), dict)
                     and "compressed_bytes" in s["whole_shard_xdelta3"]}
        if xd_shards:
            xd_bytes = sum(s["whole_shard_xdelta3"]["compressed_bytes"]
                           for s in xd_shards.values())
            scope_bytes = sum((s.get("local_shard_bytes")
                               or s["sibling_shard_bytes"]) for s in xd_shards.values())
            hdr_bits = 8 * sum((s.get("local_header_bytes")
                                or s.get("sibling_header_bytes") or 0)
                               for s in xd_shards.values())
            skip_bits = 8 * sum((sk.get("local_bytes") or sk["bytes"])
                                for sk in p["skips"])
            scope_words = scope_bytes / 2
            model_scope_bits = p["model_bits"] + skip_bits + hdr_bits
            entry["xdelta3_scope_note"] = (
                "whole-target-shard scope: model side = model bits + skipped "
                "tensors verbatim + safetensors header verbatim; b/w per shard word")
            entry["baseline_xdelta3_whole_shard_bw"] = round(xd_bytes * 8 / scope_words, 4)
            entry["model_same_scope_bw"] = round(model_scope_bits / scope_words, 4)
            entry["model_beats_xdelta3"] = bool(model_scope_bits < xd_bytes * 8)
        else:
            entry["model_beats_xdelta3"] = "n/a (xdelta3 unavailable; see xdelta3 note in shards)"
        out["pairs"][name] = entry

    (work / "summary.json").write_text(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------- synthetic sibling
def make_synthetic_sibling(snapshot: Path, out_dir: Path, flip_frac: float,
                           flip_sym_frac: float, seed: int):
    """Fake a training-time sibling: flip the mantissa LSB on a controlled
    fraction of BF16 words (optionally also the lowest exponent bit on a much
    smaller fraction, to exercise the sym-coding path).  Deterministic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = json.loads((snapshot / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx["weight_map"].values()))
    rng = np.random.default_rng(seed)
    manifest = {}
    for sh in shards:
        data = bytearray((snapshot / sh).read_bytes())
        ds, hdr = read_header(snapshot / sh)
        flips = symflips = 0
        for name, meta in hdr.items():
            if name == "__metadata__" or meta["dtype"] != "BF16":
                continue
            b, e = meta["data_offsets"]
            u = np.frombuffer(bytes(data[ds + b: ds + e]), np.uint16).copy()
            m = rng.random(u.size) < flip_frac
            u[m] ^= 1                        # mantissa LSB
            flips += int(m.sum())
            if flip_sym_frac > 0:
                m2 = rng.random(u.size) < flip_sym_frac
                u[m2] ^= 1 << 7              # lowest sym (exponent) bit
                symflips += int(m2.sum())
            data[ds + b: ds + e] = u.tobytes()
        (out_dir / sh).write_bytes(bytes(data))
        manifest[sh] = {"mant_lsb_flips": flips, "sym_bit_flips": symflips}
    (out_dir / "sibling_manifest.json").write_text(json.dumps(
        {"snapshot": str(snapshot), "flip_frac": flip_frac,
         "flip_sym_frac": flip_sym_frac, "seed": seed, "shards": manifest}, indent=2))
    return shards


# ------------------------------------------------------------------------ hf dl
def dl_retry(*args, **kw):
    """hf_hub_download with backoff: a connection reset must not kill a
    multi-hour streamed run.  (HF_HUB_DISABLE_XET=1 was set before import.)
    Non-retryable errors (bad repo/revision/filename, gated repo) re-raise
    immediately instead of burning ~22 min of backoff."""
    from huggingface_hub import hf_hub_download
    try:
        from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError,
                                            RepositoryNotFoundError,
                                            RevisionNotFoundError)
    except ImportError:  # older hub layout
        from huggingface_hub.utils import (EntryNotFoundError, GatedRepoError,  # type: ignore
                                           RepositoryNotFoundError,
                                           RevisionNotFoundError)
    fatal = (KeyboardInterrupt, RepositoryNotFoundError, RevisionNotFoundError,
             EntryNotFoundError, GatedRepoError)
    for attempt in range(1, 11):
        try:
            return hf_hub_download(*args, **kw)
        except fatal:
            raise
        except Exception as ex:
            if attempt == 10:
                raise
            wait = min(30 * attempt, 300)
            print(f"[retry] download failed ({type(ex).__name__}); "
                  f"attempt {attempt}/10, sleeping {wait}s", flush=True)
            time.sleep(wait)


def http_range_fetch(url: str, start: int, end_excl: int, tries: int = 8) -> bytes:
    """Small ranged GET with backoff (for straggler tensors that live in a
    base shard we are not downloading in full)."""
    import requests
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end_excl - 1}"},
                             timeout=120)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            data = r.content if r.status_code == 206 else r.content[start:end_excl]
            if len(data) != end_excl - start:
                raise RuntimeError(f"short read {len(data)} != {end_excl - start}")
            return data
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            if attempt == tries:
                raise
            wait = min(15 * attempt, 120)
            print(f"[retry] range fetch failed ({type(ex).__name__}); "
                  f"attempt {attempt}/{tries}, sleeping {wait}s", flush=True)
            time.sleep(wait)


def process_extra_tensor(repo: str, revision: str, spec: str, pair: str,
                         local: LocalSnapshot, done: set, jsonl, args):
    """Range-fetch a single tensor from a base shard (shard::tensor spec) and
    record it under the same pair.  Covers the boundary-shift stragglers
    without downloading their whole ~5 GB shards."""
    sh, _, tname = spec.partition("::")
    key = f"{pair}::tensor::{tname}"
    skey = f"{pair}::skip::{tname}"
    if key in done or skey in done:
        print(f"[skip] extra {tname} already recorded", flush=True)
        return
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{sh}"
    hlen = struct.unpack("<Q", http_range_fetch(url, 0, 8))[0]
    hdr = json.loads(http_range_fetch(url, 8, 8 + hlen))
    if tname not in hdr:
        raise KeyError(f"{tname} not in sibling shard {sh}")
    meta = hdr[tname]
    ds = 8 + hlen
    b, e = meta["data_offsets"]
    sraw = http_range_fetch(url, ds + b, ds + e)
    loc = local.get(tname)
    if loc is None:
        rec = {"type": "skip", "name": tname, "reason": "not in local checkpoint",
               "bytes": e - b}
    else:
        lmeta, lraw = loc
        lb, le = lmeta["data_offsets"]
        if meta["dtype"] != "BF16" or lmeta["dtype"] != "BF16":
            rec = {"type": "skip", "name": tname,
                   "reason": f"dtype pair {lmeta['dtype']}(local)/{meta['dtype']}(sibling), probe is BF16-only",
                   "bytes": e - b, "local_bytes": le - lb}
        elif lmeta["shape"] != meta["shape"]:
            rec = {"type": "skip", "name": tname,
                   "reason": f"shape mismatch {lmeta['shape']} vs {meta['shape']}",
                   "bytes": e - b, "local_bytes": le - lb}
        else:
            rec = tensor_delta_record(tname, lmeta, sraw, lraw, args.baseline_level)
    rec.update({"pair": pair, "sibling_shard": sh, "via": "http_range_fetch"})
    jsonl.write(json.dumps(rec) + "\n")
    jsonl.flush()
    done.add(key if rec["type"] == "tensor" else skey)
    print(f"[extra] {tname} from {sh}: type={rec['type']}"
          + (f" match={rec['pct_match']}% bw={rec['model_bw']}"
             if rec["type"] == "tensor" else ""), flush=True)


# -------------------------------------------------------------------------- main
def load_done(work: Path) -> set:
    done = set()
    for rec in read_results(work / "results.jsonl"):
        tag = rec["name"] if rec["type"] in ("tensor", "skip") else rec["sibling_shard"]
        done.add(f"{rec['pair']}::{rec['type']}::{tag}")
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true",
                    help="smoke on the synthetic snapshot with a faked sibling (no network)")
    ap.add_argument("--flip-frac", type=float, default=0.03,
                    help="synthetic: fraction of BF16 words whose mantissa LSB is flipped")
    ap.add_argument("--flip-sym-frac", type=float, default=0.0,
                    help="synthetic: fraction additionally flipping the lowest exponent bit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--local-snapshot", default=str(DEFAULT_LOCAL_SNAPSHOT))
    ap.add_argument("--shards", nargs="*", default=[
        "model-00001-of-00013.safetensors", "model-00007-of-00013.safetensors"],
        help="sibling shard filenames to stream (real mode)")
    ap.add_argument("--extra-tensors", nargs="*", default=[
        "model-00006-of-00013.safetensors::backbone.layers.24.norm.weight",
        "model-00008-of-00013.safetensors::backbone.layers.29.mixer.experts.95.up_proj.weight"],
        help="real mode: shard::tensor stragglers range-fetched individually "
             "(boundary-shift tensors of local shard 7 that live in adjacent "
             "base shards); pass nothing to disable")
    ap.add_argument("--work", default=None,
                    help=f"work/output dir (default {DEFAULT_WORK} or _synthetic variant)")
    ap.add_argument("--work-tag", default=None,
                    help="suffix for the default work dir (separate smoke variants)")
    ap.add_argument("--keep", action="store_true", help="do not delete sibling shards")
    ap.add_argument("--baseline-level", type=int, default=19)
    ap.add_argument("--max-whole-shard-bytes", type=int, default=1 << 30,
                    help="whole-shard patch-from only below this size (zstd 2^31 window cap)")
    ap.add_argument("--summarize", action="store_true",
                    help="rebuild summary.json from results.jsonl, no processing")
    args = ap.parse_args()

    if args.work:
        work = Path(args.work)
    else:
        work = Path(str(DEFAULT_WORK) + ("_synthetic" if args.synthetic else ""))
        if args.work_tag:
            work = Path(str(work) + f"_{args.work_tag}")
    work.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        print(json.dumps(summarize(work), indent=2))
        return

    # refuse to resume a work dir created with different parameters (a rerun
    # with other flip params would otherwise silently mix two configurations)
    stamp = ({"mode": "synthetic", "flip_frac": args.flip_frac,
              "flip_sym_frac": args.flip_sym_frac, "seed": args.seed}
             if args.synthetic else
             {"mode": "real", "repo": args.repo, "revision": args.revision,
              "local_snapshot": str(args.local_snapshot)})
    stamp_p = work / "config_stamp.json"
    if stamp_p.exists():
        prev = json.loads(stamp_p.read_text())
        if prev != stamp:
            sys.exit(f"[abort] {work} holds results for a different config:\n"
                     f"  existing: {prev}\n  current:  {stamp}\n"
                     f"Use --work/--work-tag for a fresh dir.")
    else:
        stamp_p.write_text(json.dumps(stamp, indent=2))

    done = load_done(work)
    if done:
        print(f"[resume] {len(done)} records already in {work / 'results.jsonl'}", flush=True)
    jsonl = (work / "results.jsonl").open("a")

    if args.synthetic:
        local = LocalSnapshot(SYNTH_SNAPSHOT)
        sib_dir = work / "sibling"
        shards = make_synthetic_sibling(SYNTH_SNAPSHOT, sib_dir, args.flip_frac,
                                        args.flip_sym_frac, args.seed)
        pair = "synthetic"
        print(f"[synthetic] sibling faked at {sib_dir} "
              f"(flip_frac={args.flip_frac}, flip_sym_frac={args.flip_sym_frac}, "
              f"seed={args.seed})", flush=True)
        for sh in shards:
            process_sibling_shard(sib_dir / sh, sh, pair, local, done, jsonl, args)
            if not args.keep:
                (sib_dir / sh).unlink(missing_ok=True)
            summarize(work)
            print(f"[shard done] {sh}", flush=True)
    else:
        local = LocalSnapshot(Path(args.local_snapshot))
        pair = f"{args.repo}@{(args.revision or 'main')[:12]}"
        dl_dir = work / "dl" / args.repo.replace("/", "__")  # fresh per-repo dir
        dl_dir.mkdir(parents=True, exist_ok=True)
        for sh in args.shards:
            if f"{pair}::shard::{sh}" in done:
                print(f"[skip] {sh} already fully processed", flush=True)
                continue
            t0 = time.time()
            p = Path(dl_retry(args.repo, sh, revision=args.revision, local_dir=dl_dir))
            print(f"[dl] {sh}: {p.stat().st_size / 2**30:.2f} GiB in "
                  f"{time.time() - t0:.0f}s", flush=True)
            process_sibling_shard(p, sh, pair, local, done, jsonl, args)
            if not args.keep:
                p.unlink(missing_ok=True)
            summarize(work)
            print(f"[shard done] {sh} (sibling shard deleted={not args.keep})", flush=True)
        for spec in args.extra_tensors:
            process_extra_tensor(args.repo, args.revision, spec, pair, local,
                                 done, jsonl, args)
        if args.extra_tensors:
            summarize(work)

    local.close()
    jsonl.close()
    out = summarize(work)
    print(json.dumps(out, indent=2))
    print(f"\n[done] results -> {work / 'results.jsonl'}\n"
          f"[done] summary -> {work / 'summary.json'}")


if __name__ == "__main__":
    main()
