from __future__ import annotations

"""Similarity survey: parse every tensor and mechanically note what is similar.

Three cuts of "similar", computed exhaustively over a whole model:
  1. exact-duplicate bytes        (sha256 per tensor -> identical twins)
  2. value / structural similarity (block-mean signature cosine within a shape group,
                                    incl. the cross-LAYER same-role cut)
  3. byte-layout / distribution   (per-tensor high-byte histogram + an EXACT global
                                    BF16 value histogram -> the codebook/entropy floor)

Pure numpy. BF16 is decoded losslessly as (u16 << 16).view(f32); the raw 16-bit
pattern is the exact stored value, so every count here is exact, not sampled.

Usage:
  uv run python research/candidates/0010-similarity-survey/tools/survey.py fingerprint --shard <path.safetensors> --out <dir>
  uv run python research/candidates/0010-similarity-survey/tools/survey.py merge --in <dir> --out <dir>/report
A whole model = one fingerprint call per shard (parallelisable) + one merge.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4,
               "I16": 2, "I8": 1, "U8": 1, "U16": 2, "U32": 4, "BOOL": 1}

SIG_K = 256  # block-mean signature length (fixed -> same-shape tensors comparable)


def read_header(path: Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def sha256_stream(mm, begin, end):
    h = hashlib.sha256()
    step = 8 * 1024 * 1024
    for s in range(begin, end, step):
        h.update(mm[s:min(s + step, end)].tobytes())
    return h.hexdigest()


def byte_entropy(counts, total):
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def block_mean_sig(vals, k=SIG_K):
    """Fixed-length linear fingerprint: mean of k contiguous blocks of the flattened
    tensor. Same map for every tensor of a given length -> cosine is comparable."""
    n = vals.size
    if n == 0:
        return np.zeros(k, dtype=np.float64)
    if n < k:
        out = np.zeros(k, dtype=np.float64)
        out[:n] = vals.astype(np.float64)
        return out
    trim = n - (n % k)
    return vals[:trim].astype(np.float64).reshape(k, -1).mean(axis=1)


def fingerprint_shard(shard: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    data_start, header = read_header(shard)
    mm = np.memmap(shard, dtype=np.uint8, mode="r")

    global_u16 = np.zeros(65536, dtype=np.int64)   # exact BF16 value histogram
    global_byte = np.zeros(256, dtype=np.int64)
    records = []

    for name, meta in header.items():
        if name == "__metadata__":
            continue
        b, e = meta["data_offsets"]
        ab, ae = data_start + b, data_start + e
        dtype, shape = meta["dtype"], meta["shape"]
        numel = prod(shape)
        raw = mm[ab:ae]  # uint8 view, no copy

        rec = {
            "name": name, "dtype": dtype, "shape": shape, "numel": numel,
            "nbytes": int(ae - ab), "shard": shard.name,
            "sha256": sha256_stream(mm, ab, ae),
        }
        global_byte += np.bincount(raw, minlength=256).astype(np.int64)

        if dtype == "BF16" and numel > 0:
            u16 = raw.view(np.uint16)
            vh = np.bincount(u16, minlength=65536).astype(np.int64)
            global_u16 += vh
            tot = int(u16.size)
            order = np.sort(vh)[::-1]
            rec["distinct"] = int((vh > 0).sum())
            rec["top1_frac"] = float(order[0] / tot)
            rec["top16_frac"] = float(order[:16].sum() / tot)
            rec["value_entropy"] = byte_entropy(vh, tot)          # order-0, bits/weight
            hi = (u16 >> 8).astype(np.uint8)                      # sign+exp byte
            hi_hist = np.bincount(hi, minlength=256).astype(np.int64)
            rec["hi_entropy"] = byte_entropy(hi_hist, tot)
            rec["lo_entropy"] = byte_entropy(np.bincount(u16 & 0xFF, minlength=256), tot)
            rec["hi_hist"] = hi_hist.tolist()
            f = (u16.astype(np.uint32) << 16).view(np.float32)
            finite = np.isfinite(f)
            ff = f[finite]
            rec["mean"] = float(ff.mean()) if ff.size else 0.0
            rec["std"] = float(ff.std()) if ff.size else 0.0
            rec["absmax"] = float(np.abs(ff).max()) if ff.size else 0.0
            rec["frac_zero"] = float((u16 == 0).mean())
            rec["frac_nonfinite"] = float((~finite).mean())
            rec["sig"] = block_mean_sig(f).tolist()
        elif dtype == "F32" and numel > 0:
            f = raw.view(np.float32)
            finite = np.isfinite(f)
            ff = f[finite]
            rec["mean"] = float(ff.mean()) if ff.size else 0.0
            rec["std"] = float(ff.std()) if ff.size else 0.0
            rec["absmax"] = float(np.abs(ff).max()) if ff.size else 0.0
            rec["frac_zero"] = float((f == 0).mean())
            rec["sig"] = block_mean_sig(f).tolist()

        records.append(rec)

    stem = shard.stem
    (outdir / f"{stem}.fp.json").write_text(
        json.dumps({"shard": shard.name, "records": records}), encoding="utf-8")
    np.savez(outdir / f"{stem}.hist.npz", global_u16=global_u16, global_byte=global_byte)
    print(json.dumps({"shard": shard.name, "tensors": len(records),
                      "bf16": sum(1 for r in records if r["dtype"] == "BF16")}))


def cosine_matrix(sigs):
    m = sigs / (np.linalg.norm(sigs, axis=1, keepdims=True) + 1e-30)
    return m @ m.T


def merge(indir: Path, outstem: Path):
    fps = sorted(indir.glob("*.fp.json"))
    hists = sorted(indir.glob("*.hist.npz"))
    records = []
    for fp in fps:
        records.extend(json.loads(fp.read_text(encoding="utf-8"))["records"])
    gu16 = np.zeros(65536, dtype=np.int64)
    gbyte = np.zeros(256, dtype=np.int64)
    for hp in hists:
        z = np.load(hp)
        gu16 += z["global_u16"]
        gbyte += z["global_byte"]

    # ---- exact-duplicate groups ----
    by_hash = {}
    for r in records:
        by_hash.setdefault(r["sha256"], []).append(r["name"])
    dup_groups = [{"sha256": h, "names": ns, "n": len(ns)}
                  for h, ns in by_hash.items() if len(ns) > 1]
    dup_groups.sort(key=lambda g: -g["n"])

    # ---- exact global BF16 value census (the codebook / entropy floor) ----
    tot = int(gu16.sum())
    nz = gu16[gu16 > 0]
    p = nz / tot
    order = np.sort(gu16)[::-1]
    cover = np.cumsum(order) / tot
    def k_for(frac):
        return int(np.searchsorted(cover, frac) + 1)
    census = {
        "total_bf16_values": tot,
        "distinct_values": int((gu16 > 0).sum()),
        "order0_entropy_bits": float(-(p * np.log2(p)).sum()),
        "top16_coverage": float(cover[15]) if cover.size > 15 else 1.0,
        "top256_coverage": float(cover[255]) if cover.size > 255 else 1.0,
        "k_for_0.98": k_for(0.98), "k_for_0.999": k_for(0.999),
        "k_for_0.9999": k_for(0.9999),
    }

    # ---- value/structural similarity within (dtype, shape) groups ----
    groups = {}
    for i, r in enumerate(records):
        if "sig" in r:
            groups.setdefault((r["dtype"], tuple(r["shape"])), []).append(i)
    sim_report = []
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        sigs = np.array([records[i]["sig"] for i in idxs], dtype=np.float64)
        C = cosine_matrix(sigs)
        n = len(idxs)
        iu = np.triu_indices(n, k=1)
        off = C[iu]
        # strongest non-trivial pair in this shape group
        j = int(np.argmax(np.abs(off)))
        a, bx = idxs[iu[0][j]], idxs[iu[1][j]]
        sim_report.append({
            "dtype": key[0], "shape": list(key[1]), "n_tensors": n,
            "mean_abs_cos": float(np.abs(off).mean()),
            "max_abs_cos": float(np.abs(off).max()),
            "top_pair": [records[a]["name"], records[bx]["name"]],
            "top_pair_cos": float(off[j]),
        })
    sim_report.sort(key=lambda s: -s["max_abs_cos"])

    report = {
        "n_tensors": len(records),
        "n_bf16": sum(1 for r in records if r["dtype"] == "BF16"),
        "shape_groups": len(groups),
        "exact_dup_groups": dup_groups,
        "value_census": census,
        "similarity_by_shape": sim_report,
        "global_byte_entropy": byte_entropy(gbyte, int(gbyte.sum())),
    }
    Path(f"{outstem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # full per-tensor table (sigs/hists dropped for size) for downstream lenses
    slim = [{k: v for k, v in r.items() if k not in ("sig",)} for r in records]
    with Path(f"{outstem}.records.jsonl").open("w", encoding="utf-8") as f:
        for r in slim:
            f.write(json.dumps(r) + "\n")
    np.savez(f"{outstem}.global.npz", global_u16=gu16, global_byte=gbyte)

    print(json.dumps({
        "tensors": report["n_tensors"], "bf16": report["n_bf16"],
        "exact_dup_groups": len(dup_groups),
        "dup_tensors": sum(g["n"] for g in dup_groups),
        "global_value_entropy_bits": round(census["order0_entropy_bits"], 4),
        "distinct_bf16_values": census["distinct_values"],
        "k_for_0.999": census["k_for_0.999"],
        "max_shape_group_cos": round(sim_report[0]["max_abs_cos"], 4) if sim_report else None,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("fingerprint")
    fp.add_argument("--shard", type=Path, required=True)
    fp.add_argument("--out", type=Path, required=True)
    mg = sub.add_parser("merge")
    mg.add_argument("--in", dest="indir", type=Path, required=True)
    mg.add_argument("--out", dest="outstem", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "fingerprint":
        fingerprint_shard(args.shard, args.out)
    else:
        merge(args.indir, args.outstem)


if __name__ == "__main__":
    main()
