from __future__ import annotations
import json, struct, hashlib, csv, io, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

SNAP = Path(r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot")
ART = Path(__file__).resolve().parent
ROWS, COLS = 131072, 2688
ROW_BYTES = COLS * 2  # BF16


def tensor_slice(shard_name, tname):
    p = SNAP / shard_name
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data_start = 8 + n
        b, e = hdr[tname]["data_offsets"]
        f.seek(data_start + b)
        raw = f.read(e - b)
    assert len(raw) == ROWS * ROW_BYTES, (len(raw), ROWS * ROW_BYTES)
    return raw


def bf16_to_f32(u16):
    # u16: uint16 array of bf16 bits -> float32
    return (u16.astype(np.uint32) << 16).view(np.float32)


def analyze(name, raw):
    out = {}
    arr = np.frombuffer(raw, dtype=np.uint16).reshape(ROWS, COLS)  # bit patterns
    # ---- 1. exact duplicate rows by raw bytes ----
    groups = defaultdict(list)
    rb = raw  # bytes
    mv = memoryview(rb)
    for i in range(ROWS):
        h = hashlib.blake2b(mv[i * ROW_BYTES:(i + 1) * ROW_BYTES], digest_size=16).digest()
        groups[h].append(i)
    dup_groups = {h: idxs for h, idxs in groups.items() if len(idxs) > 1}
    total_dup_rows = sum(len(v) for v in dup_groups.values())
    redundant_rows = sum(len(v) - 1 for v in dup_groups.values())  # rows removable
    unique_rows = len(groups)
    largest = max((len(v) for v in dup_groups.values()), default=0)
    largest_grp = max(dup_groups.values(), key=len) if dup_groups else []
    out["unique_rows"] = unique_rows
    out["dup_group_count"] = len(dup_groups)
    out["total_rows_in_dup_groups"] = total_dup_rows
    out["redundant_rows_removable"] = redundant_rows
    out["largest_dup_group_size"] = largest
    out["largest_dup_group_min_id"] = int(min(largest_grp)) if largest_grp else None
    out["largest_dup_group_sample_ids"] = sorted(largest_grp)[:10]

    # ---- 2. constant rows (all 2688 vals identical bits) & zero rows ----
    first_col = arr[:, 0:1]
    const_mask = np.all(arr == first_col, axis=1)
    zero_mask = np.all(arr == 0, axis=1)
    out["constant_rows"] = int(const_mask.sum())
    out["zero_rows"] = int(zero_mask.sum())
    const_ids = np.nonzero(const_mask)[0]
    out["constant_row_min_id"] = int(const_ids.min()) if const_ids.size else None
    # value of constant rows (distinct bit patterns among constant rows)
    if const_ids.size:
        cv = arr[const_ids, 0]
        uvals, ucnts = np.unique(cv, return_counts=True)
        out["constant_row_distinct_values"] = int(uvals.size)
        out["constant_row_value_breakdown"] = [
            {"bits": int(v), "f32": float(bf16_to_f32(np.array([v], np.uint16))[0]), "count": int(c)}
            for v, c in sorted(zip(uvals.tolist(), ucnts.tolist()), key=lambda x: -x[1])[:10]
        ]

    # ---- 3. per-row L2 norm + distinct value count vs id ----
    f32 = bf16_to_f32(arr)  # ROWS x COLS float32 (~1.4GB) -- ok once
    l2 = np.sqrt((f32.astype(np.float64) ** 2).sum(axis=1))
    # distinct bit patterns per row (vectorized via sort)
    sorted_rows = np.sort(arr, axis=1)
    distinct = 1 + (np.diff(sorted_rows, axis=1) != 0).sum(axis=1)
    out["l2_min"] = float(l2.min()); out["l2_max"] = float(l2.max()); out["l2_mean"] = float(l2.mean())
    return out, l2, distinct, const_mask, zero_mask, arr, groups, dup_groups


def write_csv(name, l2, distinct, const_mask, zero_mask):
    # downsample stride for plotting CSV but keep tail dense
    path = ART / f"{name}_per_row.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token_id", "l2_norm", "distinct_values", "is_constant", "is_zero"])
        stride = 64
        for i in range(0, ROWS, stride):
            w.writerow([i, f"{l2[i]:.6f}", int(distinct[i]), int(const_mask[i]), int(zero_mask[i])])
    return path


def main():
    results = {}
    raws = {}
    EMB = ("model-00001-of-00013.safetensors", "backbone.embeddings.weight")
    LMH = ("model-00013-of-00013.safetensors", "lm_head.weight")

    for tag, (shard, tname) in [("embeddings", EMB), ("lm_head", LMH)]:
        print(f"[{tag}] reading {tname} ...", flush=True)
        raw = tensor_slice(shard, tname)
        raws[tag] = raw
        # original sha for reconstruction check
        orig_sha = hashlib.sha256(raw).hexdigest()
        print(f"[{tag}] analyzing ...", flush=True)
        res, l2, distinct, const_mask, zero_mask, arr, groups, dup_groups = analyze(tag, raw)
        res["orig_sha256"] = orig_sha
        res["orig_bytes"] = len(raw)

        # ---- 5. dedup encode + byte-exact reconstruction ----
        # unique payloads in first-seen order; index per row
        seen = {}
        order = []  # list of row indices that are unique payload representatives
        index = np.empty(ROWS, dtype=np.uint32)
        for i in range(ROWS):
            key = bytes(memoryview(raw)[i * ROW_BYTES:(i + 1) * ROW_BYTES])
            j = seen.get(key)
            if j is None:
                j = len(order)
                seen[key] = j
                order.append(i)
            index[i] = j
        n_unique = len(order)
        # reconstruct
        unique_block = bytearray(n_unique * ROW_BYTES)
        for slot, i in enumerate(order):
            unique_block[slot * ROW_BYTES:(slot + 1) * ROW_BYTES] = memoryview(raw)[i * ROW_BYTES:(i + 1) * ROW_BYTES]
        recon = bytearray(ROWS * ROW_BYTES)
        ub = memoryview(unique_block)
        for i in range(ROWS):
            slot = int(index[i])
            recon[i * ROW_BYTES:(i + 1) * ROW_BYTES] = ub[slot * ROW_BYTES:(slot + 1) * ROW_BYTES]
        recon_sha = hashlib.sha256(bytes(recon)).hexdigest()
        res["recon_sha256"] = recon_sha
        res["recon_exact"] = (recon_sha == orig_sha)
        res["n_unique_payloads"] = n_unique
        # storage accounting
        idx_bytes = ROWS * 4  # uint32 index
        # could be smaller: ceil(log2(n_unique)/8). compute tight too
        import math
        bits = max(1, math.ceil(math.log2(max(2, n_unique))))
        idx_bytes_tight = math.ceil(ROWS * bits / 8)
        dedup_bytes = n_unique * ROW_BYTES + idx_bytes_tight
        res["dedup_total_bytes"] = dedup_bytes
        res["dedup_index_bytes_tight"] = idx_bytes_tight
        res["dedup_index_bits_per_row"] = bits
        res["bytes_saved_standalone"] = len(raw) - dedup_bytes
        res["pct_saved_standalone"] = 100.0 * (len(raw) - dedup_bytes) / len(raw)

        csvp = write_csv(tag, l2, distinct, const_mask, zero_mask)
        res["per_row_csv"] = str(csvp)
        results[tag] = res
        print(f"[{tag}] done. unique={n_unique} redundant={res['redundant_rows_removable']} zero={res['zero_rows']} const={res['constant_rows']} recon_exact={res['recon_exact']}", flush=True)

    # ---- 6. shared rows between embeddings and lm_head ----
    emb_set = {}
    raw_e = raws["embeddings"]; raw_l = raws["lm_head"]
    for i in range(ROWS):
        emb_set[bytes(memoryview(raw_e)[i * ROW_BYTES:(i + 1) * ROW_BYTES])] = i
    shared = 0
    shared_ids = []
    for i in range(ROWS):
        key = bytes(memoryview(raw_l)[i * ROW_BYTES:(i + 1) * ROW_BYTES])
        if key in emb_set:
            shared += 1
            if len(shared_ids) < 20:
                shared_ids.append(i)
    results["shared_emb_lmhead_rows"] = {"count": shared, "sample_lmhead_ids": shared_ids}

    (ART / "results.json").write_text(json.dumps(results, indent=2, default=int), encoding="utf-8")
    print(json.dumps(results, indent=2, default=int))


if __name__ == "__main__":
    main()
