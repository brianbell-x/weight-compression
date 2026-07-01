"""Output-aware low-bit expert quant: randomized-Hadamard incoherence + GPTQ (QuIP#-lite).

The ledger's scalar GPTQ (no incoherence) tied RTN on held-out. QuIP#'s key untested
addition is an incoherence rotation that removes weight/activation outliers so simple
k-bit quant in the rotated space becomes near output-optimal. The rotation is absorbed
losslessly into the matmul: Y = X W = (X R^T)(R W), R orthogonal.

R = (Q_b (x) H_a) diag(signs) on the in-axis (2688 = 128*21), so NO power-of-2 padding
(padding would inflate stored weights 1.52x and eat the bit savings). Scored on real
held-out activations by output error ||X(W-W')|| / ||X W|| — the honest generalization metric.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ART = Path(__file__).resolve().parent
S1_DIR = ART.parents[2] / "0005-low-bit-expert-quant" / "tests" / "artifacts"
sys.path.insert(0, str(S1_DIR))
import stage1_probe as s1  # noqa

CORP = S1_DIR / "gptq_powered" / "activations_corpus"
SHARD1 = (r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
          r"\hf_snapshot\model-00001-of-00013.safetensors")
torch.set_num_threads(torch.get_num_threads())


def hadamard(n):
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)


def rand_orth(n, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    return Q


def build_R(in_dim, a=128, seed=0):
    b = in_dim // a
    assert a * b == in_dim, f"{in_dim} != {a}*{b}"
    R = torch.kron(rand_orth(b, seed).contiguous(), hadamard(a).contiguous())  # [in,in], orthonormal
    g = torch.Generator().manual_seed(seed + 1)
    signs = (torch.randint(0, 2, (in_dim,), generator=g) * 2 - 1).float()
    return R * signs[None, :]                            # R diag(signs)


def out_err(W, Wq, X):
    Y = X @ W
    return ((X @ Wq - Y).norm() / Y.norm().clamp_min(1e-12)).item()


def quant_rtn(W, bits, group=128, axis=0):
    Wp, _ = _pergroup(W, bits, group, axis)
    return Wp


def _pergroup(W, bits, group, axis):
    maxq = 2 ** (bits - 1) - 1
    Wt = W.movedim(axis, -1)
    sh = Wt.shape
    n = sh[-1]
    g = group if n % group == 0 else n
    Wg = Wt.reshape(*sh[:-1], n // g, g)
    s = (Wg.abs().amax(-1, keepdim=True) / maxq).clamp_min(1e-12)
    q = torch.clamp(torch.round(Wg / s), -maxq, maxq) * s
    return q.reshape(sh).movedim(-1, axis), (n // g)


def gptq(W, H, bits, group=128):
    """GPTQ on W [in, out] with Hessian H [in,in]. Per-(in-group,out) scale. Returns Wq [in,out]."""
    W = W.clone().float()
    in_dim, out = W.shape
    # precompute per-group scales along in-axis (group rows share a scale, per out-col)
    maxq = 2 ** (bits - 1) - 1
    Wt = W.t().contiguous()  # [out, in]
    damp = 0.01 * torch.diag(H).mean()
    Hd = H + damp * torch.eye(in_dim)
    L = torch.linalg.cholesky(Hd)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)  # upper tri
    Q = torch.zeros_like(Wt)
    # per-column (in) scale from grouped max-abs of the ORIGINAL rotated weights
    g = group if in_dim % group == 0 else in_dim
    scales = torch.zeros(out, in_dim)
    Wt_orig = Wt.clone()
    for c0 in range(0, in_dim, g):
        blk = Wt_orig[:, c0:c0 + g]
        sc = (blk.abs().amax(1, keepdim=True) / maxq).clamp_min(1e-12)  # [out,1]
        scales[:, c0:c0 + g] = sc
    for i in range(in_dim):
        w = Wt[:, i]
        s = scales[:, i]
        q = torch.clamp(torch.round(w / s), -maxq, maxq) * s
        Q[:, i] = q
        d = Hinv[i, i]
        err = (w - q) / d
        Wt[:, i:] -= err[:, None] * Hinv[i, i:][None, :]
    return Q.t().contiguous()


def load_XY():
    Xc = torch.from_numpy(np.load(CORP / "X_cal.npy")).float()
    Xh = torch.from_numpy(np.load(CORP / "X_heldout.npy")).float()
    return Xc, Xh


def run(names, bits_list=(4, 3, 2), groups=(128,), seed=0):
    Xc, Xh = load_XY()
    in_dim = Xc.shape[1]
    R = build_R(in_dim, seed=seed)
    Xc_r = Xc @ R.t()
    Xh_r = Xh @ R.t()
    Hc = Xc.t() @ Xc
    Hc_r = Xc_r.t() @ Xc_r
    rows = []
    for name in names:
        W = s1.load_expert(SHARD1, name).t().contiguous()   # [in=2688, out]
        Wr = R @ W                                          # rotated weight
        for bits in bits_list:
            for group in groups:
                bpw = s1.bits_per_weight(W.numel(), bits * W.numel(),
                                         scale_bits=16 * (W.numel() // group))
                # 1) RTN, no incoherence
                e_rtn = out_err(W, quant_rtn(W, bits, group, axis=0), Xh)
                # 2) GPTQ, no incoherence
                e_gptq = out_err(W, gptq(W, Hc, bits, group), Xh)
                # 3) GPTQ + incoherence (QuIP#-lite): quantize Wr, eval in rotated space
                e_quip = out_err(Wr, gptq(Wr, Hc_r, bits, group), Xh_r)
                # 4) RTN + incoherence
                e_rtn_h = out_err(Wr, quant_rtn(Wr, bits, group, axis=0), Xh_r)
                rec = {"name": name.split(".mixer.")[-1], "bits": bits, "group": group,
                       "bpw": round(bpw, 3), "vram_gb": round(s1.implied_vram_gb(bpw), 2),
                       "rtn": round(e_rtn * 100, 3), "gptq": round(e_gptq * 100, 3),
                       "rtn+H": round(e_rtn_h * 100, 3), "gptq+H_quip": round(e_quip * 100, 3)}
                # 5) residual product-VQ in the incoherent (RHT) space, honest bits (no padding)
                #    d=8, nbits=8 codebook -> 1 b/stage; R=bits stages ~= `bits` b/w
                import vq_probe as vqp
                Wr_vq, meta_vq = vqp.product_vq(Wr, d=8, nbits=8, R=bits, dim_w=None, kmeans_iter=10)
                e_vq = out_err(Wr, Wr_vq, Xh_r)
                rec["vq+H_bpw"] = round(meta_vq["bits_per_weight"], 3)
                rec["vq+H"] = round(e_vq * 100, 3)
                rows.append(rec)
                print(json.dumps(rec))
    return rows


if __name__ == "__main__":
    import os
    ne = int(os.environ.get("NEXP", "1"))
    names = []
    for i in range(ne):
        names.append(f"backbone.layers.1.mixer.experts.{i}.up_proj.weight")
    t0 = time.time()
    rows = run(names, bits_list=(4, 3, 2))
    print(f"elapsed {time.time()-t0:.1f}s")
    (ART / "gptq_rht_result.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote gptq_rht_result.json")
