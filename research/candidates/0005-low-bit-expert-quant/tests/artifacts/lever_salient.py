"""Lever: salient-channel mixed-precision quantization of routed experts.

Idea: keep the top-k highest-magnitude *input channels* (rows of W, since
X @ W contracts over in_features) at INT8, quantize the rest at INT4 (or INT3).
Saliency proxy = per-channel max-abs of W (max over out_features of |W[i,:]|).

Question: how much does protecting a few salient channels cut matmul output
error, and at what extra bits/weight? Find the knee of error-vs-bits.

All quantization is symmetric per-group RTN along axis=1 (out_features), one
fp16 max-abs scale per group, group_size=64 (divides 2688 and 1856). A salient
channel just uses 8-bit levels for its groups instead of the low-bit levels.

Overhead accounted in bits/weight:
  payload  = sum over channels of (bits_of_channel * out_features)
  scale    = 16 bits * (num groups)            (one fp16 scale per group, all bit widths)
  index    = in_features bits (1-bit mask: salient or not), as codebook_bits

Runs on N>=16 REAL layer-1 experts (up_proj and down_proj), reports measured
numbers averaged over experts x tensors.
"""

from __future__ import annotations

import csv
import os

import torch

import stage1_probe as sp

SHARD = (
    r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    r"\hf_snapshot\model-00001-of-00013.safetensors"
)
GROUP_SIZE = 64           # divides both 2688 (up out) and 1856 (down out)
N_EXPERTS = 16            # layer-1 experts 0..15
TENSORS = ["up_proj.weight", "down_proj.weight"]


def quant_rows_uniform(W: torch.Tensor, nbits: int, group_size: int) -> torch.Tensor:
    """Symmetric per-group RTN along axis=1 at `nbits` for every row. Returns float32 recon."""
    qmax = (1 << (nbits - 1)) - 1            # int8->127, int4->7, int3->3
    in_f, out_f = W.shape
    n_groups = out_f // group_size
    Wg = W.reshape(in_f, n_groups, group_size)
    max_abs = Wg.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(Wg / scale), -qmax, qmax)
    What = (q * scale).reshape(in_f, out_f)
    return What


def salient_mixed(W: torch.Tensor, low_bits: int, frac: float, group_size: int):
    """Top `frac` of input channels (by max-abs) at INT8, rest at `low_bits`.

    Returns (W_prime, eff_bits_per_weight).
    """
    in_f, out_f = W.shape
    k = max(0, round(frac * in_f))

    # low-bit recon for all rows, then overwrite salient rows with 8-bit recon
    W_low = quant_rows_uniform(W, low_bits, group_size)
    if k > 0:
        saliency = W.abs().amax(dim=1)               # per input channel
        top_idx = torch.topk(saliency, k).indices
        W_hi = quant_rows_uniform(W[top_idx], 8, group_size)
        W_prime = W_low.clone()
        W_prime[top_idx] = W_hi
    else:
        W_prime = W_low

    # --- bits/weight accounting ---
    num_weights = in_f * out_f
    n_salient = k
    n_low = in_f - k
    payload_bits = (n_salient * 8 + n_low * low_bits) * out_f
    n_groups_total = in_f * (out_f // group_size)
    scale_bits = 16 * n_groups_total            # one fp16 scale per group
    index_bits = in_f                            # 1-bit salient mask per channel
    bpw = sp.bits_per_weight(
        num_weights,
        payload_bits=payload_bits,
        scale_bits=scale_bits,
        codebook_bits=index_bits,
    )
    return W_prime, bpw


def uniform_codec(W: torch.Tensor, nbits: int, group_size: int):
    """Pure uniform per-group RTN baseline at nbits. Returns (W_prime, bpw)."""
    W_prime = quant_rows_uniform(W, nbits, group_size)
    in_f, out_f = W.shape
    num_weights = in_f * out_f
    n_groups_total = in_f * (out_f // group_size)
    bpw = sp.bits_per_weight(
        num_weights,
        payload_bits=nbits * num_weights,
        scale_bits=16 * n_groups_total,
    )
    return W_prime, bpw


def main():
    fracs = [0.005, 0.01, 0.02, 0.05]
    # (name, kind, low_bits, frac)
    configs = [
        ("INT8 (ref)",        "uniform", 8, None),
        ("INT4 uniform",      "uniform", 4, None),
        ("INT3 uniform",      "uniform", 3, None),
    ]
    for f in fracs:
        configs.append((f"INT4 + {f*100:.1f}% salient INT8", "salient", 4, f))
    for f in fracs:
        configs.append((f"INT3 + {f*100:.1f}% salient INT8", "salient", 3, f))

    # accumulate over experts x tensors
    acc = {c[0]: {"rel_err": [], "cos": [], "bpw": []} for c in configs}

    # cache loaded weights + inputs
    cache = {}
    for e in range(N_EXPERTS):
        for t in TENSORS:
            name = f"backbone.layers.1.mixer.experts.{e}.{t}"
            W = sp.load_expert(SHARD, name)
            X = sp.make_inputs(W.shape[0], batch=256, seed=0)
            cache[(e, t)] = (W, X)

    for cname, kind, bits, frac in configs:
        for (e, t), (W, X) in cache.items():
            if kind == "uniform":
                Wp, bpw = uniform_codec(W, bits, GROUP_SIZE)
            else:
                Wp, bpw = salient_mixed(W, bits, frac, GROUP_SIZE)
            fid = sp.fidelity(W, Wp, X)
            acc[cname]["rel_err"].append(fid["rel_err"])
            acc[cname]["cos"].append(fid["mean_cosine"])
            acc[cname]["bpw"].append(bpw)

    rows = []
    for cname, _, _, _ in configs:
        a = acc[cname]
        bpw = sum(a["bpw"]) / len(a["bpw"])
        rel = sum(a["rel_err"]) / len(a["rel_err"]) * 100
        cos = sum(a["cos"]) / len(a["cos"])
        vram = sp.implied_vram_gb(bpw)
        rows.append((cname, bpw, rel, cos, vram))

    # print + CSV
    out_csv = os.path.join(os.path.dirname(__file__), "lever_salient_results.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "bits_per_weight", "rel_err_pct", "mean_cosine", "implied_vram_gb"])
        for cname, bpw, rel, cos, vram in rows:
            w.writerow([cname, f"{bpw:.4f}", f"{rel:.4f}", f"{cos:.6f}", f"{vram:.2f}"])

    print(f"N={N_EXPERTS} experts x {len(TENSORS)} tensors (layer 1), group_size={GROUP_SIZE}")
    print(f"{'config':<32} {'bits/w':>8} {'rel_err%':>9} {'cos':>10} {'vram_GB':>8}")
    for cname, bpw, rel, cos, vram in rows:
        print(f"{cname:<32} {bpw:>8.4f} {rel:>9.4f} {cos:>10.6f} {vram:>8.2f}")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
