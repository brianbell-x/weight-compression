"""Output-aware VQ: does Hessian-diagonal weighting in the incoherent space break the
2-bit floor, or is diag(H_r) flat (=> 17% is the real rate-distortion wall)?

In RHT space, ||X_r (W-W')||^2 = tr((W-W')^T H_r (W-W')) ~= sum_j h_jj ||row_j||^2 when
H_r is ~diagonal. So row-scale W_r by sqrt(diag(H_r)), do plain product VQ, unscale.
"""
from __future__ import annotations
import json, sys, time
import numpy as np, torch
from pathlib import Path
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ART))
import gptq_rht as g
import vq_probe as vqp
import stage1_probe as s1


def oaware_vq(Wr, Hdiag, d, nbits, R, iters=12):
    s = Hdiag.clamp_min(1e-12).sqrt()              # [in]
    Ws = Wr * s[:, None]                           # row-scale
    Wh_s, meta = vqp.product_vq(Ws, d=d, nbits=nbits, R=R, dim_w=None, kmeans_iter=iters)
    Wh = Wh_s / s[:, None]                          # unscale reconstruction
    return Wh, meta


if __name__ == "__main__":
    Xc, Xh = g.load_XY()
    in_dim = Xc.shape[1]
    R = g.build_R(in_dim, seed=0)
    Xc_r = Xc @ R.t(); Xh_r = Xh @ R.t()
    Hc_r = Xc_r.t() @ Xc_r
    Hc = Xc.t() @ Xc
    dr = torch.diag(Hc_r); du = torch.diag(Hc)
    print(f"diag(H) unrotated  max/mean = {(du.max()/du.mean()).item():.2f}")
    print(f"diag(H_r) rotated  max/mean = {(dr.max()/dr.mean()).item():.2f}  (flat => 2-bit wall real)")

    name = "backbone.layers.1.mixer.experts.0.up_proj.weight"
    W = s1.load_expert(g.SHARD1, name).t().contiguous()
    Wr = R @ W
    rows = []
    for bits in (2, 3):
        plain, m1 = vqp.product_vq(Wr, d=8, nbits=8, R=bits, dim_w=None, kmeans_iter=12)
        e_plain = g.out_err(Wr, plain, Xh_r)
        oa, m2 = oaware_vq(Wr, dr, d=8, nbits=8, R=bits)
        e_oa = g.out_err(Wr, oa, Xh_r)
        rec = {"bits": bits, "bpw": round(m2["bits_per_weight"], 3),
               "vq_plain_%": round(e_plain * 100, 3), "vq_oaware_%": round(e_oa * 100, 3)}
        rows.append(rec); print(json.dumps(rec))
    (ART / "oaware_result.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
