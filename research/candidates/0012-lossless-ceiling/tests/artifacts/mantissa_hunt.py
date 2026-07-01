"""Hunt the last detectable pattern: WHERE does mantissa structure live, and is it
exploitable losslessly? brotli-11 found expert_up mantissa=6.84b but embeddings=7.00b.

For a sample spanning layers (early/mid/late) and roles, measure per tensor:
  - brotli-11 bits/weight on the raw mantissa plane (real compressor, strongest)
  - H(mant) order-0 and H(mant | exp)  (does conditioning on magnitude help?)
  - mantissa entropy of the SMALL-magnitude subset (low exp) vs large  (near-zero weights?)
Aggregate the model-wide exploitable mantissa slice (numel-weighted) so we know if it is
material or a rounding error, and whether it is a real lever or noise.
"""
from __future__ import annotations
import json, struct, sys
from pathlib import Path
import numpy as np
import brotli

SNAP = r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def load_u16(path, name):
    ds, h = read_header(path)
    if name not in h:
        return None
    b, e = h[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(ds + b); raw = f.read(e - b)
    return np.frombuffer(raw, dtype=np.uint16)


def H(a, k):
    c = np.bincount(a, minlength=k).astype(np.float64)
    p = c[c > 0] / c.sum(); return float(-(p * np.log2(p)).sum())


def Hcond(sym, ctx, ks, kc):
    n = sym.size; key = ctx.astype(np.int64) * ks + sym
    jc = np.bincount(key); cc = np.bincount(ctx, minlength=kc); tot = 0.0
    for cv in np.nonzero(cc)[0]:
        blk = jc[cv * ks:(cv + 1) * ks]; s = blk.sum()
        if s: p = blk[blk > 0] / s; tot += (s / n) * float(-(p * np.log2(p)).sum())
    return tot


def probe(u16, cap=8_000_000):
    if u16.size > cap:
        u16 = u16[:cap]
    n = u16.size
    exp = ((u16 >> 7) & 0xFF).astype(np.int64)
    mant = (u16 & 0x7F).astype(np.uint8)
    brotli_bpw = len(brotli.compress(mant.tobytes(), quality=11)) * 8 / n
    hm = H(mant.astype(np.int64), 128)
    hmc = Hcond(mant.astype(np.int64), exp - exp.min(), 128, int(exp.max() - exp.min() + 1))
    # magnitude split: small-|w| = low exponent tail (bottom 25% of exp values)
    thr = np.quantile(exp, 0.25)
    small = mant[exp <= thr]; large = mant[exp > thr]
    hm_small = H(small.astype(np.int64), 128) if small.size else 0.0
    hm_large = H(large.astype(np.int64), 128) if large.size else 0.0
    return {"n": int(n), "mant_brotli_bpw": round(brotli_bpw, 4),
            "H_mant": round(hm, 4), "H_mant_given_exp": round(hmc, 4),
            "cond_saving_b": round(hm - hmc, 4),
            "H_mant_small_mag": round(hm_small, 4), "H_mant_large_mag": round(hm_large, 4),
            "small_frac": round(float((exp <= thr).mean()), 4)}


if __name__ == "__main__":
    shard = f"{SNAP}\\model-00001-of-00013.safetensors"
    ds, h = read_header(shard)
    # sample across roles/layers present in shard 1
    names = [n for n in h if n != "__metadata__" and h[n].get("dtype") == "BF16"
             and np.prod(h[n]["shape"]) >= 1_000_000]
    # pick a spread: experts, attn, embeddings, norms across available layers
    import re
    pick = []
    seen_roles = {}
    for nm in names:
        role = re.sub(r"\d+", "#", nm)
        if seen_roles.get(role, 0) < 2:
            pick.append(nm); seen_roles[role] = seen_roles.get(role, 0) + 1
    out = []
    tot_n = 0; tot_brotli = 0.0
    for nm in pick[:16]:
        u16 = load_u16(shard, nm)
        if u16 is None or u16.size < 1_000_000:
            continue
        r = probe(u16); r["name"] = nm
        out.append(r); tot_n += r["n"]; tot_brotli += r["mant_brotli_bpw"] * r["n"]
        print(json.dumps(r), flush=True)
    agg = {"sampled_tensors": len(out), "numel_wt_mant_brotli_bpw": round(tot_brotli / tot_n, 4),
           "vs_raw7_saving_pct_of_model": round((7 - tot_brotli / tot_n) / 16 * 100, 3)}
    print(json.dumps(agg, indent=2), flush=True)
    Path("mantissa_hunt_result.json").write_text(json.dumps({"agg": agg, "per_tensor": out}, indent=2), encoding="utf-8")
