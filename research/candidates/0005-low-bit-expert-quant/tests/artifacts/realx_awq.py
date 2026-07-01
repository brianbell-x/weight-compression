"""AWQ-style activation-aware quantization of Nemotron-3 layer-1 routed experts,
measured with REAL cached activations via the Stage-1 matmul-fidelity probe.

Goal (the open lever from test-001): close the sub-4-bit gap that all data-free /
weight-magnitude levers failed to close (best 4-bit ~9.7% output error). AWQ is the
field-standard method for exactly this gap; it needs REAL activations, which test-001
did not have. We now have them (activation_capture.py): X = post-layers.1.norm hidden
state [187, 2688], and per-input-channel activation energy [2688].

AWQ core idea implemented here
------------------------------
For a linear y = X @ W (W oriented [in, out]), pick a per-INPUT-channel scale s[in].
Because (X @ diag(1/s)) @ (diag(s) @ W) == X @ W exactly in full precision, we can
quantize the *scaled* weight diag(s)@W instead of W. Salient channels (large
activation energy) get scaled up so their relative quantization error shrinks by 1/s,
at the cost of enlarging the group max-abs for the rest. AWQ grid-searches the scale
exponent alpha in s = act_energy**alpha to minimize the REAL-activation output error.
A second AWQ component, per-group max-abs CLIPPING, is also searched (shrink the grid
range to trade clipping a few outliers for finer steps on the bulk).

We report, averaged over N>=16 real layer-1 experts:
  - INT8 / INT4 / INT3 plain per-group RTN on real X (baselines)
  - INT4 / INT3 AWQ (scale search)            <- the named method
  - INT4 / INT3 AWQ + clip
  - INT4 AWQ + salient-channel INT8 (mixed precision on top of AWQ)
on up_proj (cached X applies directly) and down_proj (real per-expert intermediate
relu(X@up.T)^2 computed on the fly).

All numbers measured on true BF16 weights, float32 CPU math. No fabrication.
"""
import os, sys, json, time
import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_probe as S1

SNAP = r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot"
SHARD1 = os.path.join(SNAP, "model-00001-of-00013.safetensors")
ACT = os.path.join(HERE, "activations")

# ------------------------------------------------------------------ cached real X
X = torch.load(os.path.join(ACT, "real_X_layer1.pt")).to(torch.float32)        # [187,2688]
chan_energy = torch.load(os.path.join(ACT, "channel_energy_layer1.pt")).float() # [2688]
N_TOK, HID = X.shape
_ratio = (chan_energy.max() / chan_energy.mean()).item()
print(f"real X: shape={tuple(X.shape)}  energy max/mean={_ratio:.2f}")

# ------------------------------------------------------------------ quantizers
def quant_pergroup(W, bits, group_size=128, group_axis=0, clip=1.0):
    """Symmetric signed per-group RTN quant+dequant of W along `group_axis`.

    bits-bit signed levels [-(2^(b-1)-1), +(2^(b-1)-1)]. One fp16 max-abs scale per
    group, optionally shrunk by `clip` (clip<1 clips outliers for finer bulk steps).
    Returns W_hat (float32, same shape as W).
    """
    qmax = 2 ** (bits - 1) - 1
    Wt = W.movedim(group_axis, -1)
    shp = Wt.shape
    n_along = shp[-1]
    assert n_along % group_size == 0, f"axis len {n_along} not divisible by {group_size}"
    ng = n_along // group_size
    Wg = Wt.reshape(*shp[:-1], ng, group_size)
    max_abs = Wg.abs().amax(dim=-1, keepdim=True) * clip
    scale = (max_abs / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(Wg / scale), -qmax, qmax)
    Wg_hat = q * scale
    return Wg_hat.reshape(shp).movedim(-1, group_axis).reshape(W.shape)

def awq_quant(W, Xin, energy, bits, group_size=128, group_axis=0,
              alphas=None, clips=None):
    """AWQ: search per-input-channel scale exponent (and optional clip) to minimize
    real-activation output error, then return W' (the inference-equivalent recon).

    W [in,out], Xin [n,in], energy [in]. The scale s[in] = (energy/mean)^alpha is
    folded into the weight (diag(s)@W) before quant; at inference X absorbs diag(1/s).
    W' = dequant(diag(s)@W) * (1/s) is exactly what X@W' computes, so we pass W' to the
    standard fidelity(W, W', X). Returns (W_prime, best_alpha, best_clip).
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if clips is None:
        clips = [1.0]
    e = (energy / energy.mean()).clamp_min(1e-8)            # [in]
    Y = Xin @ W                                             # reference output
    denom = Y.norm().clamp_min(1e-12)
    best = (None, None, None, float("inf"))
    for a in alphas:
        s = e.pow(a)
        s = s / s.mean().clamp_min(1e-12)                   # keep magnitudes centered
        s = s.clamp_min(1e-4)
        Wsc = W * s[:, None]                                # diag(s) @ W
        for c in clips:
            Wsc_hat = quant_pergroup(Wsc, bits, group_size, group_axis, clip=c)
            Wp = Wsc_hat / s[:, None]
            err = ((Xin @ Wp - Y).norm() / denom).item()
            if err < best[3]:
                best = (Wp, a, c, err)
    return best[0], best[1], best[2]

def salient_mixed(W, Xin, energy, base_bits, group_size, group_axis,
                  topk_frac, alpha, clip):
    """AWQ scale (fixed alpha/clip) + keep top activation-energy input channels at
    INT8, the rest at base_bits. Returns (W_prime, frac_kept)."""
    e = (energy / energy.mean()).clamp_min(1e-8)
    s = e.pow(alpha); s = (s / s.mean()).clamp_min(1e-4)
    Wsc = W * s[:, None]
    lo = quant_pergroup(Wsc, base_bits, group_size, group_axis, clip=clip)
    hi = quant_pergroup(Wsc, 8, group_size, group_axis, clip=1.0)
    k = max(1, int(round(topk_frac * W.shape[0])))
    idx = torch.topk(energy, k).indices
    out = lo.clone()
    out[idx] = hi[idx]
    Wp = out / s[:, None]
    return Wp, k / W.shape[0]

# ------------------------------------------------------------------ bits/weight
def bpw(payload_bits, in_dim, out_dim, group_size, awq_scale=True, salient_frac=0.0):
    n = in_dim * out_dim
    ngroups = n // group_size
    total = payload_bits * n + 16 * ngroups          # payload + fp16 group scales
    if awq_scale:
        total += 16 * in_dim                         # one fp16 per-channel AWQ scale
    # salient: extra (8-base) bits already folded into payload_bits by caller
    return total / n

# ------------------------------------------------------------------ experts
N_EXPERTS = 16
EXPERTS = list(range(N_EXPERTS))
GS = 128
ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CLIPS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

def relu2(z):
    return torch.relu(z) ** 2

results = {}   # config -> list of (rel_err, cosine)
def add(cfg, fid):
    results.setdefault(cfg, []).append((fid["rel_err"], fid["mean_cosine"]))

t0 = time.time()
for ei in EXPERTS:
    up_raw = S1.load_expert(SHARD1, f"backbone.layers.1.mixer.experts.{ei}.up_proj.weight")   # [1856,2688]
    dn_raw = S1.load_expert(SHARD1, f"backbone.layers.1.mixer.experts.{ei}.down_proj.weight")  # [2688,1856]

    # ---- up_proj: oriented [in=2688, out=1856]; cached X feeds it directly. group axis0 (2688)
    Wu = up_raw.t().contiguous()                     # [2688,1856]
    eu = chan_energy                                  # [2688] real per-input-channel energy

    # baselines (plain RTN, real X)
    add("up_INT8_RTN", S1.fidelity(Wu, quant_pergroup(Wu, 8, GS, 0), X))
    add("up_INT4_RTN", S1.fidelity(Wu, quant_pergroup(Wu, 4, GS, 0), X))
    add("up_INT3_RTN", S1.fidelity(Wu, quant_pergroup(Wu, 3, GS, 0), X))
    # AWQ scale only
    Wp,_,_ = awq_quant(Wu, X, eu, 4, GS, 0, ALPHAS, [1.0]); add("up_INT4_AWQ", S1.fidelity(Wu, Wp, X))
    Wp,_,_ = awq_quant(Wu, X, eu, 3, GS, 0, ALPHAS, [1.0]); add("up_INT3_AWQ", S1.fidelity(Wu, Wp, X))
    # AWQ scale + clip
    Wp,a4,c4 = awq_quant(Wu, X, eu, 4, GS, 0, ALPHAS, CLIPS); add("up_INT4_AWQ_clip", S1.fidelity(Wu, Wp, X))
    Wp,a3,c3 = awq_quant(Wu, X, eu, 3, GS, 0, ALPHAS, CLIPS); add("up_INT3_AWQ_clip", S1.fidelity(Wu, Wp, X))
    # AWQ + 1% salient INT8
    Wp,_ = salient_mixed(Wu, X, eu, 4, GS, 0, 0.01, a4, c4); add("up_INT4_AWQclip_salient1pct", S1.fidelity(Wu, Wp, X))

    # ---- down_proj: input is the per-expert intermediate relu2(X@up.T). oriented [in=1856,out=2688]
    #      in=1856 not divisible by 128 -> group along out-axis (2688). AWQ scale still per input chan (1856).
    inter = relu2(X @ Wu)                              # [187,1856] real down_proj input
    Wd = dn_raw.t().contiguous()                       # [1856,2688]
    ed = inter.pow(2).mean(0).sqrt()                   # [1856] real activation energy
    add("dn_INT8_RTN", S1.fidelity(Wd, quant_pergroup(Wd, 8, GS, 1), inter))
    add("dn_INT4_RTN", S1.fidelity(Wd, quant_pergroup(Wd, 4, GS, 1), inter))
    Wp,_,_ = awq_quant(Wd, inter, ed, 4, GS, 1, ALPHAS, CLIPS); add("dn_INT4_AWQ_clip", S1.fidelity(Wd, Wp, inter))
    Wp,_,_ = awq_quant(Wd, inter, ed, 3, GS, 1, ALPHAS, CLIPS); add("dn_INT3_AWQ_clip", S1.fidelity(Wd, Wp, inter))

    print(f"  expert {ei:2d} done  (t={time.time()-t0:.1f}s)")

# ------------------------------------------------------------------ summarize
def bits_for(cfg):
    in_u, out_u = 2688, 1856
    in_d, out_d = 1856, 2688
    if cfg.startswith("up"):
        ind, outd = in_u, out_u
    else:
        ind, outd = in_d, out_d
    if "INT8" in cfg and "salient" not in cfg:
        pay = 8.0
    elif "INT4" in cfg:
        pay = 4.0
    elif "INT3" in cfg:
        pay = 3.0
    else:
        pay = 4.0
    awq = "AWQ" in cfg or "salient" in cfg
    if "salient1pct" in cfg:
        pay = 0.99 * 4.0 + 0.01 * 8.0       # 1% channels to INT8
    return bpw(pay, ind, outd, GS, awq_scale=awq)

print("\n=== AWQ real-activation results (mean over %d experts) ===" % N_EXPERTS)
print(f"{'config':30s} {'b/w':>6s} {'rel_err%':>9s} {'cosine':>9s} {'vram_GB':>8s}")
table = []
for cfg, vals in results.items():
    arr = np.array(vals)
    m = float(arr[:, 0].mean()) * 100
    cos = float(arr[:, 1].mean())
    b = bits_for(cfg)
    v = S1.implied_vram_gb(b)
    table.append((cfg, b, m, cos, v))
    print(f"{cfg:30s} {b:6.3f} {m:9.3f} {cos:9.6f} {v:8.2f}")

# machine-readable
print("\nSUMMARY " + json.dumps({
    "n_experts": N_EXPERTS, "n_tokens": N_TOK,
    "rows": [{"config": c, "bpw": round(b,4), "rel_err_pct": round(m,4),
              "mean_cosine": round(cos,6), "vram_gb": round(v,2)}
             for (c,b,m,cos,v) in table],
    "wall_s": round(time.time()-t0,1),
}))
