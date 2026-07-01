"""VQ / product-quantization probe for routed experts (candidate 0011).

The ledger closed *scalar* post-hoc quant (RTN/GPTQ/AWQ) at INT8 (safe) / INT4 (+5% ppl).
The under-explored frontier for sub-4-bit is VECTOR quantization with learned codebooks +
incoherence transforms (AQLM / QuIP# / QTIP family). This probe measures product/residual
VQ on real layer-1 experts against the same matmul-fidelity harness the scalar sweep used,
so the numbers are directly comparable.

Levers:
  - product quantization: split the in-axis into length-d subvectors, K=2^nbits codebook.
  - residual VQ: R stacked codebooks, each quantizes the previous residual (bits add).
  - incoherence (QuIP#): random-sign + Walsh-Hadamard rotation of the in-axis, absorbed
    losslessly into the matmul (Y = X W = (X R^T)(R W)); spreads outliers so VQ bites.
  - activation-aware: weight the k-means distance by per-in-channel energy (real X).

bits/weight ~= R*nbits/d (+ tiny codebook/scale overhead). d=8,nbits=8,R=2 -> ~2.0 b/w.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ART = Path(__file__).resolve().parent
S1_DIR = ART.parents[2] / "0005-low-bit-expert-quant" / "tests" / "artifacts"
sys.path.insert(0, str(S1_DIR))
import stage1_probe as s1  # noqa: E402

ACT = S1_DIR / "activations"
SHARD = (r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
         r"\hf_snapshot\model-00001-of-00013.safetensors")


# --------------------------------------------------------------------------- transforms
def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def fwht(x):
    """In-place-safe fast Walsh-Hadamard transform along dim 0 (x: [n, ...]), n a pow2.
    Unnormalized; caller divides by sqrt(n)."""
    x = x.clone()
    n = x.shape[0]
    h = 1
    while h < n:
        x = x.view(n // (2 * h), 2, h, *x.shape[1:])
        a = x[:, 0, ...]
        b = x[:, 1, ...]
        x = torch.stack([a + b, a - b], dim=1).view(n, *x.shape[3:])
        h *= 2
    return x


class Incoherence:
    """R = (1/sqrt(P)) * H @ diag(signs) on the padded in-axis P. Orthonormal, so
    Y = X @ W = (X @ R^T) @ (R @ W). Apply R to W (rows), R^T = signs*H/sqrt(P)... but H is
    symmetric so R^T = diag(signs) H /sqrt(P). We store signs + P and apply via FWHT."""

    def __init__(self, in_dim, seed=0):
        self.in_dim = in_dim
        self.P = _next_pow2(in_dim)
        g = torch.Generator().manual_seed(seed)
        self.signs = (torch.randint(0, 2, (self.P,), generator=g) * 2 - 1).float()

    def rot_W(self, W):  # W: [in_dim, out] -> [P, out]
        Wp = torch.zeros(self.P, W.shape[1])
        Wp[: self.in_dim] = W
        Wp = Wp * self.signs[:, None]          # diag(signs)
        return fwht(Wp) / (self.P ** 0.5)       # H/sqrt(P)

    def rot_X(self, X):  # X: [batch, in_dim] -> [batch, P]; X R^T = (R X^T)^T
        Xp = torch.zeros(X.shape[0], self.P)
        Xp[:, : self.in_dim] = X
        # R X^T = H/sqrt(P) @ diag(signs) @ Xp^T  ; do along axis 0 of Xp^T
        t = (Xp * self.signs[None, :]).t()      # [P, batch]
        t = fwht(t) / (self.P ** 0.5)
        return t.t()                            # [batch, P]


# --------------------------------------------------------------------------- weighted kmeans
def weighted_kmeans(V, K, dim_w=None, n_iter=15, seed=0):
    """V: [N, d]. dim_w: [d] per-dimension weight (activation energy). Returns
    (codebook [K,d], idx [N]). Distances weighted by dim_w."""
    g = torch.Generator().manual_seed(seed)
    N = V.shape[0]
    w = torch.ones(V.shape[1]) if dim_w is None else dim_w.clamp_min(1e-8)
    ws = w.sqrt()
    Vw = V * ws[None, :]
    # kmeans++ lite: random distinct init
    init = torch.randperm(N, generator=g)[:K]
    C = Vw[init].clone()
    idx = torch.zeros(N, dtype=torch.long)
    chunk = 65536
    for _ in range(n_iter):
        for s in range(0, N, chunk):
            d = torch.cdist(Vw[s:s + chunk], C)
            idx[s:s + chunk] = d.argmin(1)
        newC = C.clone()
        for k in range(K):
            m = idx == k
            if m.any():
                newC[k] = Vw[m].mean(0)
        if torch.allclose(newC, C, atol=1e-6):
            C = newC
            break
        C = newC
    return (C / ws[None, :]), idx   # de-weight codebook back to real space


# --------------------------------------------------------------------------- product VQ
def product_vq(W, d, nbits, R=1, dim_w=None, seed=0, kmeans_iter=12):
    """W: [in, out]. PQ along in-axis with subvector length d, K=2^nbits, R residual
    stages. Returns (W_hat [in,out], meta)."""
    in_dim, out = W.shape
    assert in_dim % d == 0, f"in {in_dim} % d {d}"
    nsub = in_dim // d
    K = 1 << nbits
    # per-output-column max-abs scale (fp16), then PQ the normalized weight
    col_scale = W.abs().amax(0, keepdim=True).clamp_min(1e-12)   # [1, out]
    Wn = W / col_scale
    # vectors: [nsub*out, d]  (subvector b of column c)
    V = Wn.reshape(nsub, d, out).permute(0, 2, 1).reshape(nsub * out, d)
    # per-dimension weight repeats over subvector position: dim_w is length in_dim -> [nsub,d]
    dw = None
    if dim_w is not None:
        dw = dim_w.reshape(nsub, d).mean(0)   # average energy per within-subvector position
    resid = V.clone()
    Vhat = torch.zeros_like(V)
    codebooks = []
    for r in range(R):
        C, idx = weighted_kmeans(resid, K, dim_w=dw, n_iter=kmeans_iter, seed=seed + r)
        q = C[idx]
        Vhat += q
        resid = resid - q
        codebooks.append(C)
    Wn_hat = Vhat.reshape(nsub, out, d).permute(0, 2, 1).reshape(in_dim, out)
    W_hat = Wn_hat * col_scale
    numw = in_dim * out
    payload = numw * R * nbits / d
    codebook_bits = R * K * d * 16
    scale_bits = out * 16
    meta = {
        "d": d, "nbits": nbits, "R": R, "K": K,
        "bits_per_weight": s1.bits_per_weight(numw, payload, scale_bits, codebook_bits),
    }
    return W_hat, meta


# --------------------------------------------------------------------------- run
def load_real():
    Xr = torch.load(ACT / "real_X_layer1.pt").float() if (ACT / "real_X_layer1.pt").exists() \
        else torch.from_numpy(np.load(ACT / "real_X_layer1.npy")).float()
    e = torch.from_numpy(np.load(ACT / "channel_energy_layer1.npy")).float()
    return Xr, e  # [187,2688], [2688]


def run(expert_names, configs, use_incoherence_variants=True):
    Xr, energy = load_real()
    in_features = Xr.shape[1]
    assert in_features == 2688
    rows = []
    for name in expert_names:
        W = s1.load_expert(SHARD, name).t().contiguous()   # [in=2688, out=1856]
        assert W.shape[0] == in_features
        # baselines
        Wi8, mi8 = s1.int8_per_group_rtn(W, group_size=128, axis=0)
        f8 = s1.fidelity(W, Wi8, Xr)
        rows.append({"name": name, "codec": "INT8-RTN", "bpw": round(mi8["bits_per_weight"], 3),
                     "rel_err_pct": round(f8["rel_err"] * 100, 3)})
        for cfg in configs:
            for inco in ([False, True] if use_incoherence_variants else [cfg.get("inco", True)]):
                aw = cfg.get("act_weight", False)
                if inco:
                    ic = Incoherence(in_features, seed=0)
                    Wr = ic.rot_W(W)                       # [P, out]
                    Xrot = ic.rot_X(Xr)                    # [batch, P]
                    dw = None  # incoherence scrambles channel identity; energy weighting N/A
                    Wh, meta = product_vq(Wr, cfg["d"], cfg["nbits"], cfg["R"], dim_w=dw,
                                          kmeans_iter=cfg.get("iter", 12))
                    f = s1.fidelity(Wr, Wh, Xrot)
                else:
                    dw = energy if aw else None
                    Wh, meta = product_vq(W, cfg["d"], cfg["nbits"], cfg["R"], dim_w=dw,
                                          kmeans_iter=cfg.get("iter", 12))
                    f = s1.fidelity(W, Wh, Xr)
                rows.append({
                    "name": name,
                    "codec": f"PQ d{cfg['d']} n{cfg['nbits']} R{cfg['R']}"
                             f"{'+H' if inco else ''}{'+AW' if aw and not inco else ''}",
                    "bpw": round(meta["bits_per_weight"], 3),
                    "rel_err_pct": round(f["rel_err"] * 100, 3),
                    "vram_gb": round(s1.implied_vram_gb(meta["bits_per_weight"]), 2),
                })
                print(rows[-1])
    return rows


if __name__ == "__main__":
    names = [f"backbone.layers.1.mixer.experts.{i}.up_proj.weight" for i in range(int(os.environ.get("NEXP", "2")))]
    configs = [
        {"d": 8, "nbits": 8, "R": 1, "iter": 10},   # ~1.0 b/w
        {"d": 8, "nbits": 8, "R": 2, "iter": 10},   # ~2.0 b/w
        {"d": 8, "nbits": 8, "R": 3, "iter": 10},   # ~3.0 b/w
        {"d": 4, "nbits": 8, "R": 1, "iter": 10},   # ~2.0 b/w
    ]
    rows = run(names, configs)
    out = ART / "vq_probe_result.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", out)
