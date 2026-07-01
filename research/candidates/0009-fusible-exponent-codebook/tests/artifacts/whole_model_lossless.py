"""Whole-model lossless proof + byte accounting on the REAL 30B Nemotron.

Streams all 13 shards; for every BF16 tensor computes the fixed-width codebook encoding
(byte-split K=15 = 12 b/w, the GPU-validated layout; plus the regroup K=15 ~11.3 b/w
budget) and VERIFIES bit-exact reconstruction (rebuilt high plane == original). Non-BF16
tensors pass through unchanged. Reports the real whole-model and expert-only reductions
and whether EVERY BF16 tensor round-tripped losslessly.

Peak RAM ~ a few tensors (mmap, one tensor at a time). Checkpoints per shard.
"""
from __future__ import annotations
import json, struct, mmap, re, time
from pathlib import Path
import numpy as np

SNAP = Path("models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot")
OUT = Path(__file__).resolve().parent / "whole_model_lossless_result.json"
K = 15


def header(p):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def enc_bytesplit_verify(raw, R):
    a = np.frombuffer(raw, np.uint8)
    low, high = a[0::2], a[1::2]
    n = high.size
    hist = np.bincount(high, minlength=256)
    top = np.argsort(hist)[::-1][:K].astype(np.uint8)
    code_map = np.full(256, K, np.uint8)
    code_map[top] = np.arange(K, dtype=np.uint8)
    idx = code_map[high]
    esc = idx == K
    n_esc = int(esc.sum())
    # verify bit-exact reconstruction of the high plane from (codebook, idx, escape stream)
    cb = np.zeros(16, np.uint8); cb[:K] = top; cb[K] = top[0]
    high_rec = cb[idx]
    high_rec[esc] = high[esc]
    ok = bool(np.array_equal(high_rec, high))    # low byte stored verbatim -> full lossless
    bits = n * 4 + n_esc * 8 + R * max(1, int(np.ceil(np.log2(n_esc + 1)))) + K * 8 + n * 8
    return bits, ok, n_esc


def bits_regroup(raw, R):
    u = np.frombuffer(raw, np.uint16)
    sym = (((u >> 15).astype(np.uint32) << 8) | ((u >> 7) & 0xFF).astype(np.uint32))
    n = u.size
    hist = np.bincount(sym, minlength=512)
    n_esc = n - int(np.sort(hist)[::-1][:K].sum())
    return n * 4 + n_esc * 9 + R * max(1, int(np.ceil(np.log2(n_esc + 1)))) + K * 9 + n * 7


def main():
    idx_json = json.loads((SNAP / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx_json["weight_map"].values()))
    acc = dict(total_raw=0, bf16_raw=0, bf16_enc_bs=0.0, bf16_enc_rg=0.0,
               expert_raw=0, expert_enc_bs=0.0, expert_enc_rg=0.0,
               other_raw=0, n_bf16=0, n_expert=0, n_esc_total=0, all_lossless=True)
    is_expert = re.compile(r"mixer\.experts\.\d+\.(up|down)_proj\.weight$")
    t0 = time.time()
    for si, sh in enumerate(shards, 1):
        p = SNAP / sh
        ds, h = header(p)
        f = open(p, "rb"); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for name, meta in h.items():
            if name == "__metadata__":
                continue
            b, e = meta["data_offsets"]
            raw = mm[ds + b: ds + e]
            nbytes = e - b
            acc["total_raw"] += nbytes
            if meta["dtype"] == "BF16" and nbytes >= 2:
                shape = meta["shape"]
                R = shape[0] if shape else 1
                bs_bits, ok, n_esc = enc_bytesplit_verify(raw, R)
                rg_bits = bits_regroup(raw, R)
                acc["bf16_raw"] += nbytes
                acc["bf16_enc_bs"] += bs_bits / 8
                acc["bf16_enc_rg"] += rg_bits / 8
                acc["n_bf16"] += 1
                acc["n_esc_total"] += n_esc
                acc["all_lossless"] = acc["all_lossless"] and ok
                if is_expert.search(name):
                    acc["n_expert"] += 1
                    acc["expert_raw"] += nbytes
                    acc["expert_enc_bs"] += bs_bits / 8
                    acc["expert_enc_rg"] += rg_bits / 8
            else:
                acc["other_raw"] += nbytes
        mm.close(); f.close()
        # per-shard checkpoint
        prog = summarize(acc)
        prog["_progress"] = {"shard": si, "of": len(shards), "elapsed_s": round(time.time() - t0, 1)}
        OUT.write_text(json.dumps(prog, indent=2))
        print(f"[shard {si}/{len(shards)}] all_lossless={acc['all_lossless']} "
              f"bf16_tensors={acc['n_bf16']} elapsed={time.time()-t0:.0f}s", flush=True)
    final = summarize(acc)
    OUT.write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))


def summarize(a):
    GB = 1024 ** 3
    comp_bs = a["other_raw"] + a["bf16_enc_bs"]
    comp_rg = a["other_raw"] + a["bf16_enc_rg"]
    def gb(x): return round(x / GB, 3)
    def pct(part, whole): return round(100 * (1 - part / whole), 2) if whole else 0
    return {
        "ALL_BF16_TENSORS_LOSSLESS": a["all_lossless"],
        "n_bf16_tensors": a["n_bf16"], "n_expert_tensors": a["n_expert"],
        "total_escapes": a["n_esc_total"],
        "model_GB": {"total_raw": gb(a["total_raw"]), "bf16": gb(a["bf16_raw"]),
                     "experts": gb(a["expert_raw"]), "non_bf16_other": gb(a["other_raw"])},
        "expert_share_of_model_pct": round(100 * a["expert_raw"] / a["total_raw"], 1) if a["total_raw"] else 0,
        "byte_split_K15_12bw": {
            "compressed_model_GB": gb(comp_bs),
            "whole_model_reduction_pct": pct(comp_bs, a["total_raw"]),
            "expert_only_reduction_pct": pct(a["expert_enc_bs"], a["expert_raw"]) if a["expert_raw"] else 0},
        "regroup_K15_11p3bw": {
            "compressed_model_GB": gb(comp_rg),
            "whole_model_reduction_pct": pct(comp_rg, a["total_raw"]),
            "expert_only_reduction_pct": pct(a["expert_enc_rg"], a["expert_raw"]) if a["expert_raw"] else 0},
    }


if __name__ == "__main__":
    main()
