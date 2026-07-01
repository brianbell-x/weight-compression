"""Generalized 'dead precision' probe.

0002's original claim (F32 low-16 mantissa bits are constant-zero) is the special
case of a general mechanism: any bit POSITION that is identical across every
element of a tensor carries zero information and can be dropped losslessly with no
coding (store the constant mask once per tensor). This scans the WHOLE model
(all shards, BF16 + F32) and reports the model-wide free-byte ceiling.

For each tensor viewed as uint words (u2 for BF16, u4 for F32):
  AND = bitwise-and of all words   -> bits that are always 1
  OR  = bitwise-or  of all words   -> bits that are always 0 where OR bit is 0
  constant positions = bits where AND == OR (always-0 or always-1)
free_bits = popcount of constant mask; free_bytes = numel * free_bits / 8.
Also breaks out the sign bit and, for F32, the low-16 (original 0002) result.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_header(path: Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return 8 + n, header


DT = {"BF16": ("<u2", 16, 2), "F32": ("<u4", 32, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    tot = {}  # dtype -> [bytes, free_bytes, sign_free_bytes]
    for shard in args.shards:
        data_start, header = read_header(shard)
        with shard.open("rb") as f:
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                dt = meta.get("dtype")
                if dt not in DT:
                    continue
                npdt, width, esz = DT[dt]
                begin, end = meta["data_offsets"]
                f.seek(data_start + begin)
                raw = f.read(end - begin)
                w = np.frombuffer(raw, dtype=npdt)
                if w.size == 0:
                    continue
                # incremental not needed: tensors fit; use reduce
                a = np.bitwise_and.reduce(w)
                o = np.bitwise_or.reduce(w)
                const_mask = ~(a ^ o) & ((1 << width) - 1)  # 1 where bit constant
                free_bits = int(bin(int(const_mask)).count("1"))
                numel = int(w.size)
                free_bytes = numel * free_bits / 8.0
                sign_const = int((const_mask >> (width - 1)) & 1)  # MSB = sign
                sign_free_bytes = numel * sign_const / 8.0
                low16_const = None
                if dt == "F32":
                    low16_const = int((const_mask & 0xFFFF) == 0xFFFF)
                t = tot.setdefault(dt, [0, 0.0, 0.0])
                t[0] += len(raw); t[1] += free_bytes; t[2] += sign_free_bytes
                rows.append({
                    "shard": shard.name, "name": name, "dtype": dt,
                    "numel": numel, "bytes": len(raw),
                    "free_bits_per_elem": free_bits,
                    "free_bytes": free_bytes,
                    "sign_constant": sign_const,
                    "const_mask_hex": hex(int(const_mask)),
                    "f32_low16_all_const": low16_const,
                })

    summary = {}
    for dt, (b, fb, sfb) in tot.items():
        summary[dt] = {
            "total_bytes": b,
            "free_bytes_constbit": fb,
            "free_frac_constbit": fb / b if b else 0,
            "free_bytes_signonly": sfb,
        }
    grand_bytes = sum(v["total_bytes"] for v in summary.values())
    grand_free = sum(v["free_bytes_constbit"] for v in summary.values())
    summary["MODEL"] = {
        "total_bytes": grand_bytes,
        "free_bytes_constbit": grand_free,
        "free_frac_constbit": grand_free / grand_bytes if grand_bytes else 0,
    }
    out = {"summary": summary, "tensors": rows}
    (args.out / "constant_bits.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
