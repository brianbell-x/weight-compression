"""Scan F32 tensors for dead low-16 mantissa bits (BF16-representable) and prove
an exact 50% truncation round-trip via SHA-256 hashing.

For each F32 tensor: read raw little-endian bytes, view as uint32, test
(word & 0x0000FFFF) == 0 for ALL elements. If a tensor is 100% clean, drop the
low 2 bytes of every element (keep the high 2 = the BF16 form), then reconstruct
by zero-padding the low 2 bytes back and hash-compare to the original bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def read_header(path: Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return 8 + n, header


def read_bytes(path: Path, begin: int, end: int) -> bytes:
    with path.open("rb") as f:
        f.seek(begin)
        return f.read(end - begin)


def truncate_drop_low2(raw: bytes) -> bytes:
    """Keep the high 2 bytes (bits 16..31) of every little-endian F32 word."""
    a = np.frombuffer(raw, dtype="<u2").reshape(-1, 2)  # [lo16, hi16] per word
    return a[:, 1].tobytes()  # high half = the BF16 representation


def reconstruct_zero_pad(trunc: bytes, n_words: int) -> bytes:
    hi = np.frombuffer(trunc, dtype="<u2")
    out = np.zeros((n_words, 2), dtype="<u2")
    out[:, 1] = hi  # low half stays zero
    return out.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    total_f32_bytes = 0
    total_saved = 0
    clean_count = 0
    f32_count = 0
    worst_nonzero = 0

    for shard in args.shards:
        data_start, header = read_header(shard)
        for name, meta in header.items():
            if name == "__metadata__" or meta.get("dtype") != "F32":
                continue
            f32_count += 1
            begin, end = meta["data_offsets"]
            raw = read_bytes(shard, data_start + begin, data_start + end)
            n_words = len(raw) // 4
            words = np.frombuffer(raw, dtype="<u4")
            low = words & np.uint32(0x0000FFFF)
            nonzero = int(np.count_nonzero(low))
            is_clean = nonzero == 0
            worst_nonzero = max(worst_nonzero, nonzero)
            total_f32_bytes += len(raw)

            roundtrip_ok = None
            if is_clean:
                clean_count += 1
                trunc = truncate_drop_low2(raw)
                rebuilt = reconstruct_zero_pad(trunc, n_words)
                roundtrip_ok = (
                    hashlib.sha256(rebuilt).hexdigest()
                    == hashlib.sha256(raw).hexdigest()
                )
                total_saved += len(raw) - len(trunc)

            results.append({
                "shard": shard.name,
                "name": name,
                "shape": meta["shape"],
                "numel": n_words,
                "bytes": len(raw),
                "nonzero_low16": nonzero,
                "clean": is_clean,
                "roundtrip_exact": roundtrip_ok,
            })

    summary = {
        "f32_tensor_count": f32_count,
        "clean_tensor_count": clean_count,
        "clean_fraction": (clean_count / f32_count) if f32_count else None,
        "worst_nonzero_low16": worst_nonzero,
        "total_f32_bytes": total_f32_bytes,
        "bytes_saved_truncation": total_saved,
        "all_clean_roundtrips_exact": all(
            r["roundtrip_exact"] for r in results if r["clean"]
        ),
        "any_clean_roundtrip_failed": any(
            r["clean"] and r["roundtrip_exact"] is False for r in results
        ),
    }
    out = {"summary": summary, "tensors": results}
    (args.out / "scan_results.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
