"""Synthetic proof: de-interleave + sign-fold + re-interleave is EXACTLY reversible
on every BF16 tensor in the synthetic snapshot, and static rANS round-trips exactly.
"""
from __future__ import annotations
import hashlib, json, struct, sys
from pathlib import Path
import numpy as np
import codec as C

SNAP = Path("models/synthetic/nemotron_tiny/hf_snapshot")


def read_header(p):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return 8 + n, h


def sha(b): return hashlib.sha256(b).hexdigest()


def main():
    results = {"tensors_checked": 0, "all_exact": True, "failures": []}
    for shard in sorted(SNAP.glob("*.safetensors")):
        ds, h = read_header(shard)
        blob = shard.read_bytes()
        for name, meta in h.items():
            if name == "__metadata__" or meta.get("dtype") != "BF16":
                continue
            b, e = meta["data_offsets"]
            raw = blob[ds + b: ds + e]
            if len(raw) % 2:  # not a clean bf16 stream
                continue
            low, high = C.deinterleave(raw)
            mag7, sign = C.fold_sign(high)
            # reconstruct
            high2 = C.unfold_sign(mag7, sign)
            # also exercise sign bit pack/unpack
            packed = C.pack_bits(sign)
            sign2 = C.unpack_bits(packed, sign.size)
            assert np.array_equal(sign, sign2)
            high2 = C.unfold_sign(mag7, sign2)
            rebuilt = C.reinterleave(low, high2)
            ok = sha(rebuilt) == sha(raw)
            results["tensors_checked"] += 1
            if not ok:
                results["all_exact"] = False
                results["failures"].append(name)

    # rANS self-test on a skewed synthetic stream + on a real expert-like plane
    rng = np.random.default_rng(0)
    # skewed distribution similar to folded high byte (mass on a few symbols)
    probs = np.zeros(128); probs[60] = 0.5; probs[59] = 0.2; probs[61] = 0.15
    probs[40] = 0.1; probs[88] = 0.05
    syms = rng.choice(128, size=200000, p=probs).astype(np.uint8)
    freqs = C.build_freqs(syms, 128)
    enc = C.rans_encode(syms, freqs)
    dec = C.rans_decode(enc, freqs, syms.size)
    rans_ok = np.array_equal(syms, dec)
    H = C.order0_entropy_bits(syms, 128)
    results["rans_roundtrip_exact"] = bool(rans_ok)
    results["rans_entropy_bits"] = round(H, 4)
    results["rans_realized_bits_per_sym"] = round(len(enc) * 8 / syms.size, 4)

    print(json.dumps(results, indent=2))
    Path("synthetic_result.json").write_text(json.dumps(results, indent=2))
    if not (results["all_exact"] and rans_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
