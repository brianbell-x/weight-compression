"""Stream-validate the fusible codec on a HuggingFace BF16 model without holding it on disk.

Pulls one safetensors shard at a time from the Hub, runs the published fixed-width
codebook encoding, exact inverse check, and byte accounting, checkpoints running
totals, then DELETES the shard before pulling the next. Peak disk ~= one shard
(~5 GB), so a 240 GB (Super) or 1.1 TB (Ultra) model can be validated on a laptop.

Use --shards N for a cheap probe: e.g. --shards 1 downloads ~5 GB, tells you whether the
sign+exponent concentration (and thus the ~30% reduction) holds at this scale, and whether
every BF16 tensor round-trips losslessly on that shard.

Examples
--------
  # ~5 GB probe: does the 30% transfer to Super? does it round-trip lossless?
  uv run python tools/stream_validate.py nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --shards 1

  # full streamed validation (bounded disk), delete each shard as we go
  uv run python tools/stream_validate.py nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

Auth: gated repos need a token -- run `uv run hf auth login` or set HF_TOKEN first.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path
import numpy as np

# xet downloads die on connection resets with no in-flight retry; plain HTTP
# resumes. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import hf_hub_download

K = 15
GB = 1024**3
CODEC_CHUNK_WORDS = 1 << 22
IS_EXPERT = re.compile(r"mixer\.experts\.\d+\.(up|down)_proj\.weight$")


# --- codec ------------------------------------------------------------------------
def bounded_bincount(values, minlength):
    histogram = np.zeros(minlength, np.uint64)
    for start in range(0, values.size, CODEC_CHUNK_WORDS):
        chunk = values[start : start + CODEC_CHUNK_WORDS]
        histogram += np.bincount(chunk, minlength=minlength).astype(np.uint64)
    return histogram


def enc_bytesplit_verify(raw, R):
    a = np.frombuffer(raw, np.uint8)
    high = a[1::2]
    n = high.size
    hist = bounded_bincount(high, 256)
    top = np.argsort(hist)[::-1][:K].astype(np.uint8)
    code_map = np.full(256, K, np.uint8)
    code_map[top] = np.arange(K, dtype=np.uint8)
    cb = np.zeros(16, np.uint8)
    cb[:K] = top
    cb[K] = top[0]
    n_esc = 0
    ok = True
    for start in range(0, n, CODEC_CHUNK_WORDS):
        chunk = high[start : start + CODEC_CHUNK_WORDS]
        idx = code_map[chunk]
        esc = idx == K
        n_esc += int(np.count_nonzero(esc))
        rebuilt = cb[idx]
        rebuilt[esc] = chunk[esc]
        ok = ok and bool(np.array_equal(rebuilt, chunk))
    bits = (
        n * 4 + n_esc * 8 + R * max(1, int(np.ceil(np.log2(n_esc + 1)))) + K * 8 + n * 8
    )
    return bits, ok, n_esc


def bits_regroup(raw, R):
    u = np.frombuffer(raw, np.uint16)
    n = u.size
    hist = np.zeros(512, np.uint64)
    for start in range(0, n, CODEC_CHUNK_WORDS):
        words = u[start : start + CODEC_CHUNK_WORDS]
        sym = ((words >> 15) << 8) | ((words >> 7) & 0xFF)
        hist += bounded_bincount(sym, 512)
    n_esc = n - int(np.sort(hist)[::-1][:K].sum())
    return (
        n * 4 + n_esc * 9 + R * max(1, int(np.ceil(np.log2(n_esc + 1)))) + K * 9 + n * 7
    )


def header(p):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def process_shard(
    path, acc, stats_fh=None, repo=None, revision=None, run_contract_sha256=None
):
    ds, h = header(path)
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        for name, meta in h.items():
            if name == "__metadata__":
                continue
            b, e = meta["data_offsets"]
            nbytes = e - b
            acc["total_raw"] += nbytes
            acc["dtype_raw"][meta["dtype"]] = (
                acc["dtype_raw"].get(meta["dtype"], 0) + nbytes
            )
            if meta["dtype"] == "BF16" and nbytes >= 2:
                shape = meta["shape"]
                R = shape[0] if shape else 1
                raw = memoryview(mm)[ds + b : ds + e]
                try:
                    bs_bits, ok, n_esc = enc_bytesplit_verify(raw, R)
                    rg_bits = bits_regroup(raw, R)
                finally:
                    raw.release()
                acc["bf16_raw"] += nbytes
                acc["bf16_enc_bs"] += bs_bits / 8
                acc["bf16_enc_rg"] += rg_bits / 8
                acc["n_bf16"] += 1
                acc["n_esc_total"] += n_esc
                acc["all_lossless"] = acc["all_lossless"] and ok
                if stats_fh is not None:
                    stats_fh.write(
                        json.dumps(
                            {
                                "repo": repo,
                                "revision": revision,
                                "run_contract_sha256": run_contract_sha256,
                                "shard": path.name,
                                "name": name,
                                "shape": shape,
                                "nbytes": nbytes,
                                "lossless_ok": ok,
                                "n_esc": n_esc,
                                "bpw_bytesplit": round(bs_bits / (nbytes // 2), 4),
                                "bpw_regroup": round(rg_bits / (nbytes // 2), 4),
                                "expert": bool(IS_EXPERT.search(name)),
                            }
                        )
                        + "\n"
                    )
                if IS_EXPERT.search(name):
                    acc["n_expert"] += 1
                    acc["expert_raw"] += nbytes
                    acc["expert_enc_bs"] += bs_bits / 8
                    acc["expert_enc_rg"] += rg_bits / 8
            else:
                acc["other_raw"] += nbytes
    finally:
        mm.close()
        f.close()


def summarize(a, meta):
    comp_bs = a["other_raw"] + a["bf16_enc_bs"]
    comp_rg = a["other_raw"] + a["bf16_enc_rg"]

    def gb(x):
        return round(x / GB, 3)

    def pct(part, whole):
        return round(100 * (1 - part / whole), 2) if whole else 0

    return {
        "repo": meta["repo"],
        "revision": meta.get("revision"),
        "run_contract_sha256": meta.get("run_contract_sha256"),
        "run_contract": meta.get("run_contract"),
        "shards_processed": meta["done"],
        "shards_total": meta["total"],
        "is_partial_estimate": meta["done"] < meta["total"],
        "ALL_BF16_TENSORS_LOSSLESS": a["all_lossless"],
        "n_bf16_tensors": a["n_bf16"],
        "n_expert_tensors": a["n_expert"],
        "total_escapes": a["n_esc_total"],
        "seen_GB": {
            "total_raw": gb(a["total_raw"]),
            "bf16": gb(a["bf16_raw"]),
            "experts": gb(a["expert_raw"]),
            "non_bf16_other": gb(a["other_raw"]),
        },
        "non_bf16_by_dtype_GB": {
            k: gb(v) for k, v in sorted(a["dtype_raw"].items()) if k != "BF16"
        },
        "bf16_share_of_seen_pct": round(100 * a["bf16_raw"] / a["total_raw"], 1)
        if a["total_raw"]
        else 0,
        "expert_share_of_seen_pct": round(100 * a["expert_raw"] / a["total_raw"], 1)
        if a["total_raw"]
        else 0,
        "byte_split_K15_12bw": {
            "compressed_GB": gb(comp_bs),
            "reduction_pct": pct(comp_bs, a["total_raw"]),
            "expert_only_reduction_pct": pct(a["expert_enc_bs"], a["expert_raw"])
            if a["expert_raw"]
            else 0,
        },
        "regroup_K15_11p3bw": {
            "compressed_GB": gb(comp_rg),
            "reduction_pct": pct(comp_rg, a["total_raw"]),
            "expert_only_reduction_pct": pct(a["expert_enc_rg"], a["expert_raw"])
            if a["expert_raw"]
            else 0,
        },
    }


def dl_retry(*args, **kw):
    """hf_hub_download with backoff: a connection reset must not kill a
    multi-hour streamed run."""
    for attempt in range(1, 11):
        try:
            return hf_hub_download(*args, **kw)
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            if attempt == 10:
                raise
            wait = min(30 * attempt, 300)
            print(
                f"[retry] download failed ({type(ex).__name__}); "
                f"attempt {attempt}/10, sleeping {wait}s",
                flush=True,
            )
            time.sleep(wait)


def write_json_atomic(path, value):
    """Publish a small resume manifest without leaving a partial JSON file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_contract(args):
    """Bind resumable state to code, environment lock, snapshot, and selection."""
    lock = None
    if args.environment_lock:
        lock_path = Path(args.environment_lock)
        if not lock_path.is_file():
            raise FileNotFoundError(
                f"Environment lock was not found at {lock_path}. Stage the exact "
                "requirements lock or omit --environment-lock for an unbound ad-hoc run."
            )
        lock = {"path": lock_path.name, "sha256": file_sha256(lock_path)}
    contract = {
        "schema_version": 1,
        "repo": args.repo,
        "revision": args.revision,
        "start": args.start,
        "shards": args.shards,
        "K": K,
        "stats_jsonl": args.stats_jsonl,
        "python": sys.version,
        "evaluator_sha256": file_sha256(__file__),
        "environment_lock": lock,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    contract["sha256"] = hashlib.sha256(encoded).hexdigest()
    return contract


def ensure_stats_identity(stats_path, contract):
    """Refuse to append per-tensor rows from a different immutable snapshot."""
    if not stats_path:
        return None
    stats = Path(stats_path)
    identity_path = Path(str(stats) + ".manifest.json")
    expected = {
        "schema_version": 2,
        "repo": contract["repo"],
        "revision": contract["revision"],
        "run_contract_sha256": contract["sha256"],
        "run_contract": contract,
    }
    if identity_path.exists():
        found = json.loads(identity_path.read_text())
        if found != expected:
            raise RuntimeError(
                f"Stats manifest {identity_path} belongs to run contract "
                f"{found.get('run_contract_sha256')!r} for "
                f"{found.get('repo')!r}@{found.get('revision')!r}, but this run requested "
                f"{contract['sha256']!r} for "
                f"{contract['repo']!r}@{contract['revision']!r}. Use a new "
                "--stats-jsonl path or restore the matching evaluator, lock, config, "
                "and revision before resuming."
            )
    elif stats.exists() and stats.stat().st_size:
        raise RuntimeError(
            f"Stats file {stats} already contains rows but has no revision manifest. "
            "Move it aside and rerun so a repo/revision-bound manifest can be created."
        )
    else:
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(identity_path, expected)
    return identity_path


def stats_part_path(stats_path, shard):
    shard_key = hashlib.sha256(shard.encode()).hexdigest()
    return Path(str(stats_path) + ".parts") / f"{shard_key}.jsonl"


def prepare_stats_part(stats_path, shard):
    final = stats_part_path(stats_path, shard)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(final.name + ".tmp")
    temporary.unlink(missing_ok=True)
    return temporary, final


def commit_stats_part(temporary, final):
    Path(temporary).replace(final)
    return file_sha256(final)


def validate_stats_parts(stats_path, done_shards, part_hashes):
    if not stats_path:
        if part_hashes:
            raise RuntimeError(
                "Checkpoint contains per-shard stats hashes, but this run has no "
                "--stats-jsonl output. Restore the matching run configuration."
            )
        return
    if set(done_shards) != set(part_hashes):
        missing = sorted(set(done_shards) - set(part_hashes))
        extra = sorted(set(part_hashes) - set(done_shards))
        raise RuntimeError(
            "Checkpoint and transactional stats parts disagree: "
            f"missing hashes for {missing}, unexpected hashes for {extra}. "
            "Use a fresh --work and --stats-jsonl destination."
        )
    for shard, expected in sorted(part_hashes.items()):
        part = stats_part_path(stats_path, shard)
        if not part.is_file():
            raise RuntimeError(
                f"Checkpoint marks {shard} complete, but its stats part {part} is "
                "missing. Restore the part or restart with fresh output paths."
            )
        actual = file_sha256(part)
        if actual != expected:
            raise RuntimeError(
                f"Stats part {part} has SHA-256 {actual}, but the checkpoint requires "
                f"{expected}. Restore it or restart with fresh output paths."
            )


def materialize_stats(stats_path, selected_shards, done_shards, part_hashes):
    if not stats_path:
        return
    if set(selected_shards) != set(done_shards):
        missing = sorted(set(selected_shards) - set(done_shards))
        extra = sorted(set(done_shards) - set(selected_shards))
        raise RuntimeError(
            "Cannot publish per-tensor stats without exact selected-shard coverage: "
            f"missing {missing}, unexpected {extra}. Resume the matching run first."
        )
    validate_stats_parts(stats_path, done_shards, part_hashes)
    stats = Path(stats_path)
    stats.parent.mkdir(parents=True, exist_ok=True)
    temporary = stats.with_name(stats.name + ".tmp")
    with temporary.open("wb") as output:
        for shard in selected_shards:
            if shard in done_shards:
                with stats_part_path(stats, shard).open("rb") as source:
                    shutil.copyfileobj(source, output, length=1 << 20)
    temporary.replace(stats)


def restore_checkpoint(checkpoint, contract, acc):
    """Restore only totals computed from the exact requested Hub snapshot."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return set(), {}
    saved = json.loads(checkpoint.read_text())
    saved_revision = saved.get("revision")
    saved_contract = saved.get("run_contract_sha256")
    if (
        saved.get("repo") != contract["repo"]
        or saved_revision != contract["revision"]
        or saved_contract != contract["sha256"]
    ):
        raise RuntimeError(
            f"Checkpoint {checkpoint} belongs to run contract {saved_contract!r} for "
            f"{saved.get('repo')!r}@{saved_revision!r}, but this run requested "
            f"{contract['sha256']!r} for "
            f"{contract['repo']!r}@{contract['revision']!r}. Use a new --work "
            "directory or restore the matching evaluator, lock, config, and revision."
        )
    acc.update(saved["acc"])
    done_shards = set(saved["done_shards"])
    print(
        f"[resume] checkpoint for {contract['repo']}@{contract['revision']} "
        f"contract={contract['sha256']}: "
        f"{len(done_shards)} shard(s) already processed",
        flush=True,
    )
    return done_shards, saved.get("stats_parts_sha256", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "repo", help="HF repo id, e.g. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    )
    ap.add_argument(
        "--shards",
        type=int,
        default=0,
        help="process only N shards from --start (0 = all)",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="0-based index of the first shard to process (for range-partitioned parallel runs; give each range its own --work dir)",
    )
    ap.add_argument(
        "--work",
        default="_stream_work",
        help="scratch dir for the current shard",
    )
    ap.add_argument(
        "--keep", action="store_true", help="do not delete shards after processing"
    )
    ap.add_argument("--revision", default=None)
    ap.add_argument(
        "--environment-lock",
        default=None,
        help="exact requirements lock whose SHA-256 joins the resume run contract",
    )
    ap.add_argument(
        "--stats-jsonl",
        default=None,
        help="append one json line per BF16 tensor (per-tensor bit accounting)",
    )
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"stream_result_{args.repo.split('/')[-1]}.json"
    contract = build_run_contract(args)
    ensure_stats_identity(args.stats_jsonl, contract)

    idx_path = dl_retry(
        args.repo,
        "model.safetensors.index.json",
        revision=args.revision,
        local_dir=work,
    )
    index_payload = json.loads(Path(idx_path).read_text())
    weight_map = index_payload["weight_map"]
    shards = sorted(set(weight_map.values()))
    total = len(shards)
    if args.start:
        shards = shards[args.start :]
    if args.shards:
        shards = shards[: args.shards]

    # Use the retained index total; never fan out one metadata request per shard.
    full_bytes = index_payload.get("metadata", {}).get("total_size")
    if isinstance(full_bytes, int) and full_bytes >= 0:
        print(
            f"[info] {args.repo}: {total} shards, full download ~{full_bytes / GB:.1f} GB "
            f"| processing {len(shards)} shard(s), peak disk ~one shard",
            flush=True,
        )
    else:
        print(
            f"[info] {args.repo}: {total} shards | processing {len(shards)} "
            "shard(s), peak disk ~one shard (index has no total_size)",
            flush=True,
        )

    acc = dict(
        total_raw=0,
        bf16_raw=0,
        bf16_enc_bs=0.0,
        bf16_enc_rg=0.0,
        expert_raw=0,
        expert_enc_bs=0.0,
        expert_enc_rg=0.0,
        other_raw=0,
        n_bf16=0,
        n_expert=0,
        n_esc_total=0,
        all_lossless=True,
        dtype_raw={},
    )
    ckpt = work / f"checkpoint_{args.repo.split('/')[-1]}.json"
    done_shards, stats_parts = restore_checkpoint(ckpt, contract, acc)
    validate_stats_parts(args.stats_jsonl, done_shards, stats_parts)
    t0 = time.time()
    for i, sh in enumerate(shards, 1):
        if sh in done_shards:
            continue
        td = time.time()
        p = Path(dl_retry(args.repo, sh, revision=args.revision, local_dir=work))
        dl = time.time() - td
        stats_temporary = stats_final = None
        if args.stats_jsonl:
            stats_temporary, stats_final = prepare_stats_part(args.stats_jsonl, sh)
        stats_fh = open(stats_temporary, "w") if stats_temporary else None
        try:
            process_shard(
                p,
                acc,
                stats_fh,
                repo=args.repo,
                revision=args.revision,
                run_contract_sha256=contract["sha256"],
            )
        finally:
            if stats_fh:
                stats_fh.close()
        if stats_temporary:
            stats_parts[sh] = commit_stats_part(stats_temporary, stats_final)
        if not args.keep:
            p.unlink(missing_ok=True)
        done_shards.add(sh)
        write_json_atomic(
            ckpt,
            {
                "schema_version": 2,
                "repo": args.repo,
                "revision": args.revision,
                "run_contract_sha256": contract["sha256"],
                "run_contract": contract,
                "acc": acc,
                "done_shards": sorted(done_shards),
                "stats_parts_sha256": stats_parts,
            },
        )
        meta = {
            "repo": args.repo,
            "revision": args.revision,
            "run_contract_sha256": contract["sha256"],
            "run_contract": contract,
            "done": i,
            "total": total,
        }
        res = summarize(acc, meta)
        res["_progress"] = {
            "shard": i,
            "of_selected": len(shards),
            "of_total": total,
            "shard_dl_s": round(dl, 1),
            "elapsed_s": round(time.time() - t0, 1),
        }
        out.write_text(json.dumps(res, indent=2))
        bs = res["byte_split_K15_12bw"]["reduction_pct"]
        print(
            f"[shard {i}/{len(shards)}] lossless={acc['all_lossless']} "
            f"bf16_tensors={acc['n_bf16']} reduction={bs}% (byte-split) "
            f"dl={dl:.0f}s elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    materialize_stats(args.stats_jsonl, shards, done_shards, stats_parts)
    final = summarize(
        acc,
        {
            "repo": args.repo,
            "revision": args.revision,
            "run_contract_sha256": contract["sha256"],
            "run_contract": contract,
            "done": len(shards),
            "total": total,
        },
    )
    out.write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))
    print(f"\n[done] result -> {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        msg = str(ex)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            print(
                "\n[auth] This repo looks gated/private. Accept the license on the model page, "
                "then run `uv run hf auth login` or set HF_TOKEN, and retry.",
                file=sys.stderr,
            )
        raise
