"""Stream-validate the fusible codec on a HuggingFace BF16 model without holding it on disk.

Pulls one safetensors shard at a time from the Hub, runs the SAME fixed-width codebook
encoding + bit-exact lossless verify + byte accounting used in candidate 0009's
whole_model_lossless.py, checkpoints running totals, then DELETES the shard before pulling
the next. Peak disk ~= one shard (~5 GB), so a 240 GB (Super) or 1.1 TB (Ultra) model can be
validated on a laptop.

Use --shards N for a cheap probe: e.g. --shards 1 downloads ~5 GB, tells you whether the
sign+exponent concentration (and thus the ~30% reduction) holds at this scale, and whether
every BF16 tensor round-trips losslessly on that shard.

Examples
--------
  # ~5 GB probe: does the 30% transfer to Super? does it round-trip lossless?
  uv run python research/candidates/0009-fusible-exponent-codebook/tools/stream_validate.py nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --shards 1

  # full streamed validation (bounded disk), delete each shard as we go
  uv run python research/candidates/0009-fusible-exponent-codebook/tools/stream_validate.py nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

Auth: gated repos need a token -- run `uv run hf auth login` or set HF_TOKEN first.
"""
from __future__ import annotations
import argparse, json, struct, mmap, re, time, os, sys
from pathlib import Path
import numpy as np
# xet downloads die on connection resets with no in-flight retry; plain HTTP
# resumes. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url

K = 15
GB = 1024 ** 3
IS_EXPERT = re.compile(r"mixer\.experts\.\d+\.(up|down)_proj\.weight$")


# --- codec (verbatim from candidate 0009 whole_model_lossless.py) -----------------
def enc_bytesplit_verify(raw, R):
    a = np.frombuffer(raw, np.uint8)
    low, high = a[0::2], a[1::2]
    n = high.size
    hist = np.bincount(high, minlength=256)
    top = np.argsort(hist)[::-1][:K].astype(np.uint8)
    code_map = np.full(256, K, np.uint8)
    code_map[top] = np.arange(K, dtype=np.uint8)
    idx = code_map[high]
    esc = idx == K
    n_esc = int(esc.sum())
    cb = np.zeros(16, np.uint8); cb[:K] = top; cb[K] = top[0]
    high_rec = cb[idx]
    high_rec[esc] = high[esc]
    ok = bool(np.array_equal(high_rec, high))  # low byte stored verbatim -> full lossless
    bits = n * 4 + n_esc * 8 + R * max(1, int(np.ceil(np.log2(n_esc + 1)))) + K * 8 + n * 8
    return bits, ok, n_esc


def bits_regroup(raw, R):
    u = np.frombuffer(raw, np.uint16)
    sym = (((u >> 15).astype(np.uint32) << 8) | ((u >> 7) & 0xFF).astype(np.uint32))
    n = u.size
    hist = np.bincount(sym, minlength=512)
    n_esc = n - int(np.sort(hist)[::-1][:K].sum())
    return n * 4 + n_esc * 9 + R * max(1, int(np.ceil(np.log2(n_esc + 1)))) + K * 9 + n * 7


def header(p):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def process_shard(path, acc):
    ds, h = header(path)
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    for name, meta in h.items():
        if name == "__metadata__":
            continue
        b, e = meta["data_offsets"]
        nbytes = e - b
        acc["total_raw"] += nbytes
        acc["dtype_raw"][meta["dtype"]] = acc["dtype_raw"].get(meta["dtype"], 0) + nbytes
        if meta["dtype"] == "BF16" and nbytes >= 2:
            raw = mm[ds + b: ds + e]
            shape = meta["shape"]
            R = shape[0] if shape else 1
            bs_bits, ok, n_esc = enc_bytesplit_verify(raw, R)
            rg_bits = bits_regroup(raw, R)
            acc["bf16_raw"] += nbytes
            acc["bf16_enc_bs"] += bs_bits / 8
            acc["bf16_enc_rg"] += rg_bits / 8
            acc["n_bf16"] += 1
            acc["n_esc_total"] += n_esc
            acc["all_lossless"] = acc["all_lossless"] and ok
            if IS_EXPERT.search(name):
                acc["n_expert"] += 1
                acc["expert_raw"] += nbytes
                acc["expert_enc_bs"] += bs_bits / 8
                acc["expert_enc_rg"] += rg_bits / 8
        else:
            acc["other_raw"] += nbytes
    mm.close(); f.close()


def summarize(a, meta):
    comp_bs = a["other_raw"] + a["bf16_enc_bs"]
    comp_rg = a["other_raw"] + a["bf16_enc_rg"]
    gb = lambda x: round(x / GB, 3)
    pct = lambda part, whole: round(100 * (1 - part / whole), 2) if whole else 0
    return {
        "repo": meta["repo"],
        "shards_processed": meta["done"], "shards_total": meta["total"],
        "is_partial_estimate": meta["done"] < meta["total"],
        "ALL_BF16_TENSORS_LOSSLESS": a["all_lossless"],
        "n_bf16_tensors": a["n_bf16"], "n_expert_tensors": a["n_expert"],
        "total_escapes": a["n_esc_total"],
        "seen_GB": {"total_raw": gb(a["total_raw"]), "bf16": gb(a["bf16_raw"]),
                    "experts": gb(a["expert_raw"]), "non_bf16_other": gb(a["other_raw"])},
        "non_bf16_by_dtype_GB": {k: gb(v) for k, v in sorted(a["dtype_raw"].items()) if k != "BF16"},
        "bf16_share_of_seen_pct": round(100 * a["bf16_raw"] / a["total_raw"], 1) if a["total_raw"] else 0,
        "expert_share_of_seen_pct": round(100 * a["expert_raw"] / a["total_raw"], 1) if a["total_raw"] else 0,
        "byte_split_K15_12bw": {
            "compressed_GB": gb(comp_bs),
            "reduction_pct": pct(comp_bs, a["total_raw"]),
            "expert_only_reduction_pct": pct(a["expert_enc_bs"], a["expert_raw"]) if a["expert_raw"] else 0},
        "regroup_K15_11p3bw": {
            "compressed_GB": gb(comp_rg),
            "reduction_pct": pct(comp_rg, a["total_raw"]),
            "expert_only_reduction_pct": pct(a["expert_enc_rg"], a["expert_raw"]) if a["expert_raw"] else 0},
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
            print(f"[retry] download failed ({type(ex).__name__}); "
                  f"attempt {attempt}/10, sleeping {wait}s", flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="HF repo id, e.g. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
    ap.add_argument("--shards", type=int, default=0, help="process only the first N shards (0 = all)")
    ap.add_argument("--work", default="research/candidates/0009-fusible-exponent-codebook/tools/_stream_work", help="scratch dir for the current shard")
    ap.add_argument("--keep", action="store_true", help="do not delete shards after processing")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    out = work / f"stream_result_{args.repo.split('/')[-1]}.json"

    idx_path = dl_retry(args.repo, "model.safetensors.index.json",
                        revision=args.revision, local_dir=work)
    weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))
    total = len(shards)
    if args.shards:
        shards = shards[:args.shards]

    # peek at total download size so the user knows the full cost before committing
    try:
        full_bytes = 0
        for sh in sorted(set(weight_map.values())):
            md = get_hf_file_metadata(hf_hub_url(args.repo, sh, revision=args.revision))
            full_bytes += md.size or 0
        print(f"[info] {args.repo}: {total} shards, full download ~{full_bytes/GB:.1f} GB "
              f"| processing {len(shards)} shard(s), peak disk ~one shard", flush=True)
    except Exception as ex:
        print(f"[info] {args.repo}: {total} shards (size peek failed: {ex})", flush=True)

    acc = dict(total_raw=0, bf16_raw=0, bf16_enc_bs=0.0, bf16_enc_rg=0.0,
               expert_raw=0, expert_enc_bs=0.0, expert_enc_rg=0.0,
               other_raw=0, n_bf16=0, n_expert=0, n_esc_total=0,
               all_lossless=True, dtype_raw={})
    ckpt = work / f"checkpoint_{args.repo.split('/')[-1]}.json"
    done_shards: set[str] = set()
    if ckpt.exists():
        saved = json.loads(ckpt.read_text())
        if saved.get("repo") == args.repo:
            acc.update(saved["acc"])
            done_shards = set(saved["done_shards"])
            print(f"[resume] checkpoint: {len(done_shards)} shard(s) already processed", flush=True)
    t0 = time.time()
    for i, sh in enumerate(shards, 1):
        if sh in done_shards:
            continue
        td = time.time()
        p = Path(dl_retry(args.repo, sh, revision=args.revision, local_dir=work))
        dl = time.time() - td
        process_shard(p, acc)
        if not args.keep:
            p.unlink(missing_ok=True)
        done_shards.add(sh)
        ckpt.write_text(json.dumps({"repo": args.repo, "acc": acc,
                                    "done_shards": sorted(done_shards)}))
        meta = {"repo": args.repo, "done": i, "total": total}
        res = summarize(acc, meta)
        res["_progress"] = {"shard": i, "of_selected": len(shards), "of_total": total,
                            "shard_dl_s": round(dl, 1), "elapsed_s": round(time.time() - t0, 1)}
        out.write_text(json.dumps(res, indent=2))
        bs = res["byte_split_K15_12bw"]["reduction_pct"]
        print(f"[shard {i}/{len(shards)}] lossless={acc['all_lossless']} "
              f"bf16_tensors={acc['n_bf16']} reduction={bs}% (byte-split) "
              f"dl={dl:.0f}s elapsed={time.time()-t0:.0f}s", flush=True)

    final = summarize(acc, {"repo": args.repo, "done": len(shards), "total": total})
    out.write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))
    print(f"\n[done] result -> {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        msg = str(ex)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            print("\n[auth] This repo looks gated/private. Accept the license on the model page, "
                  "then run `uv run hf auth login` or set HF_TOKEN, and retry.", file=sys.stderr)
        raise
