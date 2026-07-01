from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path

DTYPE_BYTES = {"BF16": 2, "F32": 4, "F16": 2, "F64": 8, "I64": 8, "I32": 4, "U8": 1}


def prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def read_header(path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, 8 + n, json.loads(f.read(n))


def hash_payload(path, begin, size):
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(begin)
        while size:
            chunk = f.read(min(size, 8 * 1024 * 1024))
            size -= len(chunk)
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    header_len, data_start, header = read_header(args.shard)
    rows = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        shape, dtype = meta["shape"], meta["dtype"]
        rows.append({
            "name": name,
            "dtype": dtype,
            "shape": "x".join(map(str, shape)),
            "rank": len(shape),
            "numel": prod(shape),
            "byte_count": end - begin,
            "absolute_begin": data_start + begin,
            "absolute_end": data_start + end,
            "expected_bytes_match": (end - begin) == prod(shape) * DTYPE_BYTES.get(dtype, 0),
            "sha256": hash_payload(args.shard, data_start + begin, end - begin),
        })
    rows.sort(key=lambda r: r["absolute_begin"])
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "shard": str(args.shard),
        "file_size": args.shard.stat().st_size,
        "header_length": header_len,
        "data_start": data_start,
        "tensor_count": len(rows),
        "payload_bytes": sum(r["byte_count"] for r in rows),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "tensors": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
