"""Probe V2: bit-REGROUPED fixed-width codebook.

The byte split leaves exp_lsb in the raw 'low byte', so the raw field is 8 bits.
Regroup bit-wise instead:
  symbol = sign(1) | exponent(8)   -> 9-bit field, LOW ENTROPY, codebook it (K + escape)
  raw    = mantissa(7)             -> 7 pure-random bits stored raw
If the (sign,exp) field stays concentrated, raw drops 8->7 b/w (save 1 bit on EVERY
weight) and we may match/beat the variable-length 32% floor while staying fixed-width.

Also re-confirm exact lossless round-trip from (codebook, fixed index, in-order escape,
per-row offsets, raw 7-bit mantissa).
"""
from __future__ import annotations
import struct, json, re, mmap, hashlib, math
from pathlib import Path
import numpy as np

SHARD = Path("C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot/model-00001-of-00013.safetensors")
N_EXPERTS = 8


def read_header(p):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return 8 + n, h


def order0_bits(arr, alphabet):
    c = np.bincount(arr.astype(np.int64), minlength=alphabet).astype(np.float64)
    n = c.sum(); p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def fw_bits(symbols, raw_bits_per_weight, shape, K, alphabet):
    n = symbols.size; R, Cc = shape
    hist = np.bincount(symbols, minlength=alphabet)
    n_escape = int(n - np.sort(hist)[::-1][:K].sum())
    idx_width = max(1, math.ceil(math.log2(K + 1)))
    esc_val_bits = math.ceil(math.log2(alphabet))      # raw escaped symbol
    off_bits = R * max(1, math.ceil(math.log2(n_escape + 1)))
    cb_bits = K * math.ceil(math.log2(alphabet))
    total = n * idx_width + n_escape * esc_val_bits + off_bits + cb_bits + n * raw_bits_per_weight
    return {"K": K, "idx_width": idx_width, "escape_rate": n_escape / n,
            "bits_per_weight": total / n, "reduction": 1 - (total / n) / 16}


def roundtrip_regroup(raw, K):
    """Exact lossless: codebook (sign|exp) 9-bit, raw 7-bit mantissa."""
    u = np.frombuffer(raw, dtype=np.uint16).copy()
    sign = (u >> 15).astype(np.uint32)
    exp = ((u >> 7) & 0xFF).astype(np.uint32)
    mant = (u & 0x7F).astype(np.uint16)
    sym = ((sign << 8) | exp).astype(np.int32)          # 0..511
    hist = np.bincount(sym, minlength=512)
    top = np.argsort(hist)[::-1][:K].astype(np.int32)
    code_of = {int(v): i for i, v in enumerate(top)}
    ESC = K
    idx = np.array([code_of.get(int(s), ESC) for s in sym], dtype=np.int32)
    escapes = sym[idx == ESC].astype(np.int32)
    # decode
    esc_ptr = 0; sym_rt = np.empty(sym.size, dtype=np.int32)
    topl = top.tolist()
    for i in range(sym.size):
        c = int(idx[i])
        if c == ESC:
            sym_rt[i] = escapes[esc_ptr]; esc_ptr += 1
        else:
            sym_rt[i] = topl[c]
    sign_rt = (sym_rt >> 8).astype(np.uint16)
    exp_rt = (sym_rt & 0xFF).astype(np.uint16)
    u_rt = ((sign_rt << 15) | (exp_rt << 7) | mant).astype(np.uint16)
    rebuilt = u_rt.tobytes()
    ok = hashlib.sha256(rebuilt).hexdigest() == hashlib.sha256(raw).hexdigest()
    return ok, int(escapes.size)


def main():
    ds, h = read_header(SHARD)
    def pick(proj):
        ks = [k for k in h if re.search(rf"layers\.1\.mixer\.experts\.\d+\.{proj}_proj\.weight$", k)]
        return sorted(ks, key=lambda k: int(re.search(r"experts\.(\d+)\.", k).group(1)))[:N_EXPERTS]
    f = open(SHARD, "rb"); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    def slice_raw(meta):
        b, e = meta["data_offsets"]; return bytes(mm[ds + b: ds + e])

    KS = [7, 15, 31]
    out = {}
    for kind, keys in (("up", pick("up")), ("down", pick("down"))):
        supp, Hs = [], []
        fw = {K: [] for K in KS}; rt = None
        topk = {k: [] for k in [8, 16, 32]}
        for ki, k in enumerate(keys):
            raw = slice_raw(h[k]); shape = tuple(h[k]["shape"])
            u = np.frombuffer(raw, dtype=np.uint16)
            sign = (u >> 15).astype(np.uint32); exp = ((u >> 7) & 0xFF).astype(np.uint32)
            sym = ((sign << 8) | exp).astype(np.int64)
            n = sym.size
            hist = np.bincount(sym, minlength=512)
            supp.append(int((hist > 0).sum())); Hs.append(order0_bits(sym.astype(np.int64), 512))
            srt = np.sort(hist)[::-1]; cum = np.cumsum(srt)
            for kk in [8, 16, 32]:
                topk[kk].append(float(cum[kk - 1] / n))
            for K in KS:
                fw[K].append(fw_bits(sym.astype(np.int32), 7, shape, K, 512))
            if ki == 0:
                ok, nesc = roundtrip_regroup(raw, 15)
                rt = {"tensor": k, "K": 15, "exact_roundtrip": ok, "n_escape": nesc}
        def m(K):
            rows = fw[K]
            return {"K": K, "idx_width": rows[0]["idx_width"],
                    "mean_escape_rate": round(float(np.mean([r["escape_rate"] for r in rows])), 6),
                    "mean_bits_per_weight": round(float(np.mean([r["bits_per_weight"] for r in rows])), 4),
                    "reduction": round(float(np.mean([r["reduction"] for r in rows])), 4)}
        out[kind] = {"support_signexp_mean": round(float(np.mean(supp)), 1),
                     "support_signexp_max": int(np.max(supp)),
                     "H_signexp_mean": round(float(np.mean(Hs)), 4),
                     "rANS_floor_regroup_b_per_w": round(float(np.mean(Hs)) + 7, 4),
                     "topk_signexp_cum_mass": {f"top{kk}": round(float(np.mean(topk[kk])), 6) for kk in [8, 16, 32]},
                     "fixed_width": {f"K{K}": m(K) for K in KS},
                     "exact_roundtrip_check": rt}
    mm.close(); f.close()
    print(json.dumps(out, indent=2))
    (Path(__file__).resolve().parent / "regroup_result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
