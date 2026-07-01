from __future__ import annotations

import csv
import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import xxhash
from safetensors.torch import safe_open

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot"
SHARD = SNAPSHOT / "model-00001-of-00013.safetensors"
OUT = ROOT / "research/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/shard_00001"
PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F16": 2, "F64": 8, "I64": 8, "I32": 4, "U8": 1}


def read_header():
    with SHARD.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return n, 8 + n, header


def prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def classify(name):
    layer = block = expert = proj = None
    if name == "backbone.embeddings.weight":
        return "embedding", layer, block, expert, "embedding"
    m = re.search(r"backbone\.layers\.(\d+)\.", name)
    if m:
        layer = int(m.group(1))
        block = {"M": "mamba", "E": "moe", "*": "attention"}[PATTERN[layer]]
    m = re.search(r"\.experts\.(\d+)\.", name)
    if m:
        expert = int(m.group(1))
    for token in ("up_proj", "down_proj", "in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate", "norm", "conv1d", "A_log", "D", "dt_bias"):
        if token in name:
            proj = token
            break
    group = "global"
    if block == "moe" and expert is not None:
        group = f"moe_routed_expert_{proj}"
    elif block == "moe" and "shared_experts" in name:
        group = f"moe_shared_expert_{proj}"
    elif block:
        group = f"{block}_{proj or 'other'}"
    return group, layer, block, expert, proj


def stream_hashes(rows):
    with SHARD.open("rb") as f:
        for row in rows:
            f.seek(row["absolute_begin"])
            remaining = row["byte_count"]
            sha = hashlib.sha256()
            xxh = xxhash.xxh3_128()
            while remaining:
                chunk = f.read(min(8 * 1024 * 1024, remaining))
                remaining -= len(chunk)
                sha.update(chunk)
                xxh.update(chunk)
            row["sha256"] = sha.hexdigest()
            row["xxh3_128"] = xxh.hexdigest()


def matrix_csv(path, array):
    np.savetxt(path, np.asarray(array, dtype=np.float32), delimiter=",", fmt="%.9g")


def line_csv(path, values):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "value"])
        for i, v in enumerate(np.asarray(values, dtype=np.float32).reshape(-1)):
            w.writerow([i, float(v)])


def bf16_bytes_csv(path, tensor, n=512):
    values = tensor.flatten()[:n].to(torch.float32).numpy()
    raw = tensor.flatten()[:n].contiguous().view(torch.uint8).numpy().reshape(-1, 2)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["flat_index", "float32_value", "bf16_hex", "stored_low_byte", "stored_high_byte"])
        for i, ((lo, hi), val) in enumerate(zip(raw, values, strict=True)):
            w.writerow([i, float(val), f"0x{int(hi):02x}{int(lo):02x}", int(lo), int(hi)])


def byte_hist_csv(path, tensor):
    raw = tensor.contiguous().view(torch.uint8).numpy().reshape(-1, 2)
    low = np.bincount(raw[:, 0], minlength=256)
    high = np.bincount(raw[:, 1], minlength=256)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["byte_value", "stored_low_byte_count", "stored_high_byte_count"])
        for i in range(256):
            w.writerow([i, int(low[i]), int(high[i])])


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = OUT / "visual_samples"
    samples.mkdir(exist_ok=True)
    header_len, data_start, header = read_header()
    rows = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        group, layer, block, expert, projection = classify(name)
        shape = meta["shape"]
        dtype = meta["dtype"]
        numel = prod(shape)
        rows.append({
            "name": name,
            "dtype": dtype,
            "shape": "x".join(map(str, shape)),
            "rank": len(shape),
            "numel": numel,
            "byte_count": end - begin,
            "data_begin": begin,
            "data_end": end,
            "absolute_begin": data_start + begin,
            "absolute_end": data_start + end,
            "layer": "" if layer is None else layer,
            "block_type": block or "",
            "expert": "" if expert is None else expert,
            "projection": projection or "",
            "group": group,
            "expected_bytes_match": (end - begin) == numel * DTYPE_BYTES.get(dtype, 0),
        })
    rows.sort(key=lambda r: r["absolute_begin"])
    stream_hashes(rows)

    fields = list(rows[0])
    write_csv(OUT / "manifest.csv", rows, fields)
    with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row["name"])
    duplicate_hashes = [v for v in by_hash.values() if len(v) > 1]
    group_bytes = Counter()
    block_bytes = Counter()
    layer_bytes = Counter()
    shape_counts = Counter()
    dtype_counts = Counter()
    for row in rows:
        group_bytes[row["group"]] += row["byte_count"]
        block_bytes[row["block_type"] or "global"] += row["byte_count"]
        layer_bytes[str(row["layer"])] += row["byte_count"]
        shape_counts[f'{row["dtype"]} {row["shape"]}'] += 1
        dtype_counts[row["dtype"]] += 1
    summary = {
        "shard": str(SHARD.relative_to(ROOT)),
        "file_size": SHARD.stat().st_size,
        "header_length": header_len,
        "data_start": data_start,
        "tensor_count": len(rows),
        "dtype_counts": dict(dtype_counts),
        "group_bytes": dict(group_bytes.most_common()),
        "block_bytes": dict(block_bytes.most_common()),
        "layer_bytes": dict(layer_bytes.most_common()),
        "same_shape_counts": dict(shape_counts.most_common(20)),
        "exact_duplicate_tensor_groups": duplicate_hashes,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with safe_open(SHARD, framework="pt", device="cpu") as f:
        matrix_csv(samples / "embedding_rows_0_63_cols_0_63.csv", f.get_slice("backbone.embeddings.weight")[:64, :64].to(torch.float32).numpy())
        matrix_csv(samples / "layer0_mamba_in_proj_0_127x0_127.csv", f.get_slice("backbone.layers.0.mixer.in_proj.weight")[:128, :128].to(torch.float32).numpy())
        line_csv(samples / "layer0_norm_weight.csv", f.get_tensor("backbone.layers.0.norm.weight").to(torch.float32).numpy())
        line_csv(samples / "layer0_mamba_A_log.csv", f.get_tensor("backbone.layers.0.mixer.A_log").numpy())
        line_csv(samples / "layer1_router_bias.csv", f.get_tensor("backbone.layers.1.mixer.gate.e_score_correction_bias").numpy())
        e0_up = f.get_tensor("backbone.layers.1.mixer.experts.0.up_proj.weight")
        matrix_csv(samples / "layer1_expert000_up_proj_0_127x0_127.csv", e0_up[:128, :128].to(torch.float32).numpy())
        matrix_csv(samples / "layer1_expert000_down_proj_0_127x0_127.csv", f.get_slice("backbone.layers.1.mixer.experts.0.down_proj.weight")[:128, :128].to(torch.float32).numpy())
        matrix_csv(samples / "layer1_expert001_up_proj_0_127x0_127.csv", f.get_slice("backbone.layers.1.mixer.experts.1.up_proj.weight")[:128, :128].to(torch.float32).numpy())
        bf16_bytes_csv(samples / "bf16_inside_layer1_expert000_up_first512.csv", e0_up)
        byte_hist_csv(samples / "byte_hist_layer1_expert000_up.csv", e0_up)
        stats = []
        for i in range(128):
            row = {"expert": i}
            for side in ("up", "down"):
                name = f"backbone.layers.1.mixer.experts.{i}.{side}_proj.weight"
                t = f.get_tensor(name).to(torch.float32)
                row |= {f"{side}_mean": float(t.mean()), f"{side}_std": float(t.std()), f"{side}_min": float(t.min()), f"{side}_max": float(t.max())}
            stats.append(row)
        write_csv(samples / "layer1_expert_stats.csv", stats, list(stats[0]))

    print(json.dumps({"out": str(OUT), "tensors": len(rows), "duplicate_groups": len(duplicate_hashes)}, indent=2))


if __name__ == "__main__":
    main()
