"""Real shard-1 measurements for the BF16 exponent-plane codec.

- entropy of low/high/folded-mag7/sign planes across N layer-1 experts (up & down)
- cross-expert KL of high-byte histograms (up vs down separately)
- one real expert folded-high plane: actual static-rANS encode/decode, EXACT hash round-trip
- headline bytes: method (rANS-floor) vs zstd-19 raw interleaved vs zstd-19 on planes
"""
from __future__ import annotations
import csv, hashlib, json, struct, sys, time, re
from pathlib import Path
import numpy as np
import zstandard as zstd
import codec as C

SHARD = Path("C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot/model-00001-of-00013.safetensors")
N_EXPERTS = 16          # layer-1 experts sampled for entropy/KL/headline
TABLE_FROM = 4          # build the shared rANS table from this many experts
ART = Path(__file__).resolve().parent


def read_header(p):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return 8 + n, h


def slice_raw(mm, ds, meta):
    b, e = meta["data_offsets"]
    return bytes(mm[ds + b: ds + e])


def planes(raw):
    low, high = C.deinterleave(raw)
    mag7, sign = C.fold_sign(high)
    return low, high, mag7, sign


def zc(data, level=19):
    return zstd.ZstdCompressor(level=level).compress(data)


def main():
    ds, h = read_header(SHARD)
    ups = sorted([k for k in h if re.search(r"layers\.1\.mixer\.experts\.\d+\.up_proj\.weight$", k)],
                 key=lambda k: int(re.search(r"experts\.(\d+)\.", k).group(1)))[:N_EXPERTS]
    dns = sorted([k for k in h if re.search(r"layers\.1\.mixer\.experts\.\d+\.down_proj\.weight$", k)],
                 key=lambda k: int(re.search(r"experts\.(\d+)\.", k).group(1)))[:N_EXPERTS]

    import mmap
    f = open(SHARD, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    rows = []           # per-tensor entropy rows
    hi_hists = {"up": [], "down": []}   # high-byte hists for KL
    # accumulators for headline (rANS-floor)
    agg = {"up": {}, "down": {}}
    raw_cache = {}

    def measure(keys, kind):
        for k in keys:
            raw = slice_raw(mm, ds, h[k])
            raw_cache[(kind, k)] = raw
            low, high, mag7, sign = planes(raw)
            n = low.size
            H_low = C.order0_entropy_bits(low, 256)
            H_high = C.order0_entropy_bits(high, 256)
            H_mag7 = C.order0_entropy_bits(mag7, 128)
            H_sign = C.order0_entropy_bits(sign, 2)
            hi_hists[kind].append(C.histogram(high, 256))
            rows.append({"kind": kind, "tensor": k, "numel": n,
                         "H_low": round(H_low, 4), "H_high": round(H_high, 4),
                         "H_mag7": round(H_mag7, 4), "H_sign": round(H_sign, 4)})
            a = agg[kind]
            a["n"] = a.get("n", 0) + n
            # method high-plane realized bits (rANS-floor): min of full-high8 vs fold(mag7 rANS + 1bit raw sign)
            fold_bits = H_mag7 * n + 1.0 * n
            no_fold_bits = H_high * n
            a["high_bits"] = a.get("high_bits", 0.0) + min(fold_bits, no_fold_bits)
            a["fold_bits"] = a.get("fold_bits", 0.0) + fold_bits
            a["nofold_bits"] = a.get("nofold_bits", 0.0) + no_fold_bits

    measure(ups, "up")
    measure(dns, "down")

    # ---- cross-expert KL (high-byte hist), up vs down separately ----
    kl = {}
    for kind in ("up", "down"):
        hs = hi_hists[kind]
        ref = hs[0]
        vals = [C.kl_divergence(hi, ref) for hi in hs[1:]]
        # also worst pairwise vs the mean dist
        mean = np.mean(hs, axis=0)
        vals_mean = [C.kl_divergence(hi, mean) for hi in hs]
        kl[kind] = {"vs_expert0_mean_bits": round(float(np.mean(vals)), 6),
                    "vs_expert0_max_bits": round(float(np.max(vals)), 6),
                    "vs_meanhist_max_bits": round(float(np.max(vals_mean)), 6)}

    # ---- actual static-rANS exact round-trip on ONE real expert folded-high plane ----
    k0 = ups[0]
    raw0 = raw_cache[("up", k0)]
    low0, high0, mag70, sign0 = planes(raw0)
    # build table from first TABLE_FROM up experts (shared)
    samp = np.concatenate([C.fold_sign(C.deinterleave(raw_cache[("up", k)])[1])[0] for k in ups[:TABLE_FROM]])
    freqs = C.build_freqs(samp, 128)
    t0 = time.time()
    enc = C.rans_encode(mag70, freqs)
    t1 = time.time()
    dec = C.rans_decode(enc, freqs, mag70.size)
    t2 = time.time()
    rans_exact_mag = bool(np.array_equal(mag70, dec))
    # full exact reconstruction of the ORIGINAL tensor bytes from coded parts
    sign_packed = C.pack_bits(sign0)
    sign_rt = C.unpack_bits(sign_packed, sign0.size)
    high_rt = C.unfold_sign(dec, sign_rt)
    rebuilt = C.reinterleave(low0, high_rt)
    full_exact = hashlib.sha256(rebuilt).hexdigest() == hashlib.sha256(raw0).hexdigest()
    rans_realized_bits = len(enc) * 8 / mag70.size

    # ---- headline bytes over the sampled experts ----
    def block(kind):
        a = agg[kind]
        n = a["n"]
        method_high_bytes = a["high_bits"] / 8.0
        method_low_bytes = n  # low plane raw, 8 bits/elem
        method_total = method_high_bytes + method_low_bytes
        raw_bytes = n * 2
        return {"numel": n, "raw_bytes": int(raw_bytes),
                "method_high_bytes(rANS-floor)": int(method_high_bytes),
                "method_low_bytes(raw)": int(method_low_bytes),
                "method_total_bytes": int(method_total),
                "method_ratio": round(method_total / raw_bytes, 4),
                "fold_total_bytes": int(a["fold_bits"] / 8 + n),
                "nofold_total_bytes": int(a["nofold_bits"] / 8 + n)}

    headline = {"up": block("up"), "down": block("down")}

    # ---- zstd baselines on the sampled experts (concatenated) ----
    def zstd_block(kind, keys):
        raw_all = b"".join(raw_cache[(kind, k)] for k in keys)
        lows, highs = [], []
        for k in keys:
            lo, hi = C.deinterleave(raw_cache[(kind, k)])
            lows.append(lo); highs.append(hi)
        low_all = np.concatenate(lows).tobytes()
        high_all = np.concatenate(highs).tobytes()
        z_raw = len(zc(raw_all))
        z_planes = len(zc(high_all)) + len(zc(low_all))
        return {"zstd19_raw_interleaved_bytes": z_raw,
                "zstd19_planes_bytes": z_planes,
                "raw_bytes": len(raw_all),
                "zstd19_raw_ratio": round(z_raw / len(raw_all), 4),
                "zstd19_planes_ratio": round(z_planes / len(raw_all), 4)}

    baselines = {"up": zstd_block("up", ups), "down": zstd_block("down", dns)}

    mm.close(); f.close()

    # write per-tensor csv
    with (ART / "real_entropy.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    out = {
        "shard": SHARD.name,
        "n_experts_sampled": N_EXPERTS,
        "per_tensor_entropy_means": {
            kind: {col: round(float(np.mean([r[col] for r in rows if r["kind"] == kind])), 4)
                   for col in ("H_low", "H_high", "H_mag7", "H_sign")}
            for kind in ("up", "down")},
        "cross_expert_KL_bits": kl,
        "rans_real_expert": {
            "tensor": k0, "numel": int(mag70.size),
            "mag7_roundtrip_exact": rans_exact_mag,
            "full_tensor_bytes_exact": full_exact,
            "rans_realized_bits_per_sym": round(rans_realized_bits, 4),
            "mag7_entropy_bits": round(C.order0_entropy_bits(mag70, 128), 4),
            "encode_sec": round(t1 - t0, 1), "decode_sec": round(t2 - t1, 1),
            "table_built_from_experts": TABLE_FROM},
        "headline_method": headline,
        "zstd_baselines": baselines,
    }
    print(json.dumps(out, indent=2))
    (ART / "real_result.json").write_text(json.dumps(out, indent=2))
    if not full_exact:
        sys.exit(1)


if __name__ == "__main__":
    main()
