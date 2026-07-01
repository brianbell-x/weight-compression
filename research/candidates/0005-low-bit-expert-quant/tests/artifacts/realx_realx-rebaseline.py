"""realx-rebaseline: re-score the Stage-1 codecs on REAL activations vs random-X proxy.

Question: does measuring matmul-output error on the cached REAL layer-1 routed-expert
input X change the numbers vs the random unit-norm proxy used in test-001? Does the
proxy over- or under-state the INT4 gap? This calibrates how much to trust Stage-1.

Three codecs (the relevant ones from test-001), each scored on the SAME experts with
TWO inputs so the only difference is X:
  1. INT8 per-group RTN  (g=128)              -- the safe floor
  2. INT4 per-group RTN  (g=128)              -- the gap baseline
  3. codebook16 per-group (4b non-uniform)    -- test-001's best <=4b point

Orientation (verified in activation_capture.py): up_proj.weight is the nn.Linear weight
[out=1856, in=2688]; the true op is y = X @ W.T. So we orient W' = up_proj.weight.T =
[in=2688, out=1856] and group-quantize along axis=0 (the 2688 in-axis; 1856 is NOT
divisible by 128). Real X = post-layers.1.norm hidden state [187, 2688] feeds up_proj
verbatim, so this is the *true* matmul the expert performs.

Random-X control: make_inputs(2688, batch=187, seed=0) -- same shape, unit-norm Gaussian
rows, NO activation outliers. Comparing the two isolates the effect of real activations.

All numbers measured on real BF16 layer-1 experts. Codebook fit by Lloyd-Max on the
pooled per-group-normalized weight population (shape-shared, amortized over 29.4e9).
"""
from __future__ import annotations

import csv, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_probe as s1

SHARD = (
    r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    r"\hf_snapshot\model-00001-of-00013.safetensors"
)
ACT_DIR = os.path.join(HERE, "activations")
REAL_X = os.path.join(ACT_DIR, "real_X_layer1.pt")
CSV_PATH = os.path.join(HERE, "realx_realx-rebaseline_results.csv")

LAYER = 1
N_EXPERTS = 16
PROJ = "up_proj"          # [out=1856, in=2688]
GROUP_SIZE = 128
AXIS = 0                  # group along the in-axis (2688) after orienting to [in, out]
FIT_SAMPLE = 3_000_000


# --- codebook helpers (same math as lever_codebook.py, axis=0 here) ------------------
def grouped(W, group_size, axis):
    Wt = W.movedim(axis, -1)
    moved_shape = Wt.shape
    n_along = moved_shape[-1]
    if n_along % group_size != 0:
        raise ValueError(f"axis len {n_along} not divisible by group_size {group_size}")
    n_groups = n_along // group_size
    Wg = Wt.reshape(*moved_shape[:-1], n_groups, group_size)
    def restore(Wg_mod):
        return Wg_mod.reshape(moved_shape).movedim(-1, axis).reshape(W.shape)
    return Wg, restore, n_groups


def normalize_per_group(W, group_size, axis):
    Wg, restore, n_groups = grouped(W, group_size, axis)
    scale = Wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return Wg / scale, scale, Wg, restore


def fit_codebook(values, n_levels, iters=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = values.flatten()
    if v.numel() > FIT_SAMPLE:
        v = v[torch.randperm(v.numel(), generator=g)[:FIT_SAMPLE]]
    v = v.to(torch.float32)
    qs = torch.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
    centers = torch.quantile(v, qs)
    for _ in range(iters):
        a = (v.unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)
        new = centers.clone()
        for k in range(n_levels):
            m = a == k
            if m.any():
                new[k] = v[m].mean()
        if torch.allclose(new, centers, atol=1e-7):
            centers = new; break
        centers = new
    return torch.sort(centers).values


def quantize_to_codebook(norm, centers):
    d = (norm.unsqueeze(-1) - centers.view(*([1] * norm.dim()), -1)).abs()
    return centers[d.argmin(dim=-1)]


# --- codecs (all return W_prime oriented [in, out]) ----------------------------------
def codec_int_rtn(W, bits, group_size, axis):
    qmax = (1 << (bits - 1)) - 1
    Wg, restore, _ = grouped(W, group_size, axis)
    scale = (Wg.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(Wg / scale), -qmax, qmax)
    return restore(q * scale)


def codec_codebook_pergroup(W, centers, group_size, axis):
    norm, scale, Wg, restore = normalize_per_group(W, group_size, axis)
    return restore(quantize_to_codebook(norm, centers) * scale)


def main():
    t0 = time.time()
    # --- load real X (post layers.1.norm hidden state) ---
    X_real = torch.load(REAL_X).to(torch.float32)        # [187, 2688]
    n_tokens, in_features = X_real.shape
    assert in_features == 2688
    X_rand = s1.make_inputs(in_features, batch=n_tokens, seed=0)  # control, same shape
    # energy tail to confirm real X has outliers the proxy lacks
    e_real = X_real.pow(2).mean(0).sqrt(); r_real = (e_real.max() / e_real.mean()).item()
    e_rand = X_rand.pow(2).mean(0).sqrt(); r_rand = (e_rand.max() / e_rand.mean()).item()
    print(f"real X {tuple(X_real.shape)}  channel-energy max/mean: real={r_real:.2f} rand={r_rand:.2f}")

    # --- load N real layer-1 experts, orient to [in=2688, out=1856] ---
    names = [f"backbone.layers.{LAYER}.mixer.experts.{i}.{PROJ}.weight" for i in range(N_EXPERTS)]
    print(f"loading {N_EXPERTS} experts {PROJ} layer {LAYER} ...")
    Ws = [s1.load_expert(SHARD, n).t().contiguous() for n in names]  # [2688, 1856]
    numw = Ws[0].numel()
    assert Ws[0].shape[0] == in_features, "oriented W rows must match X in_features"
    print(f"  oriented expert shape={tuple(Ws[0].shape)} (in,out)  numel={numw}")

    # --- fit shared 16-level codebook on pooled per-group-normalized population ---
    print("fitting 16-level Lloyd-Max codebook on pooled normalized weights ...")
    pg_pool = torch.cat([normalize_per_group(W, GROUP_SIZE, AXIS)[0].flatten() for W in Ws])
    cb16 = fit_codebook(pg_pool, 16)

    # bits/weight (axis=0 grouping; one fp16 scale per group; codebook amortized over 29.4e9)
    n_groups = numw // GROUP_SIZE
    bpw_int8 = s1.bits_per_weight(numw, 8 * numw, scale_bits=16 * n_groups)
    bpw_int4 = s1.bits_per_weight(numw, 4 * numw, scale_bits=16 * n_groups)
    bpw_cb16 = s1.bits_per_weight(numw, 4 * numw, scale_bits=16 * n_groups) + 16 * 16 / s1.ROUTED_EXPERT_PARAMS

    configs = [
        ("INT8_pg128_RTN",        lambda W: codec_int_rtn(W, 8, GROUP_SIZE, AXIS),               bpw_int8),
        ("INT4_pg128_RTN",        lambda W: codec_int_rtn(W, 4, GROUP_SIZE, AXIS),               bpw_int4),
        ("codebook16_pg128_4b",   lambda W: codec_codebook_pergroup(W, cb16, GROUP_SIZE, AXIS),  bpw_cb16),
    ]

    rows = []
    print(f"\n{'config':24s} {'bpw':>6s} {'real_err%':>10s} {'rand_err%':>10s} "
          f"{'real_cos':>10s} {'ratio r/r':>10s} {'vram_GB':>8s}")
    for name, codec, bpw in configs:
        rr, rd, cr, cd = [], [], [], []
        for W in Ws:
            Wp = codec(W)
            fr = s1.fidelity(W, Wp, X_real)
            fd = s1.fidelity(W, Wp, X_rand)
            rr.append(fr["rel_err"]); cr.append(fr["mean_cosine"])
            rd.append(fd["rel_err"]); cd.append(fd["mean_cosine"])
        real_err = sum(rr) / len(rr); rand_err = sum(rd) / len(rd)
        real_cos = sum(cr) / len(cr); rand_cos = sum(cd) / len(cd)
        ratio = real_err / rand_err if rand_err else float("nan")
        vram = s1.implied_vram_gb(bpw)
        rows.append({
            "config": name, "bits_per_weight": round(bpw, 4),
            "real_X_rel_err_pct": round(real_err * 100, 4),
            "random_X_rel_err_pct": round(rand_err * 100, 4),
            "real_over_random": round(ratio, 3),
            "real_X_mean_cosine": round(real_cos, 6),
            "random_X_mean_cosine": round(rand_cos, 6),
            "implied_vram_gb": round(vram, 2),
        })
        print(f"{name:24s} {bpw:6.3f} {real_err*100:10.4f} {rand_err*100:10.4f} "
              f"{real_cos:10.6f} {ratio:10.3f} {vram:8.2f}")

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {CSV_PATH}  (wall {time.time()-t0:.1f}s)")
    print("SUMMARY " + json.dumps({
        "n_experts": N_EXPERTS, "proj": PROJ, "n_tokens": n_tokens,
        "energy_ratio_real": round(r_real, 2), "energy_ratio_rand": round(r_rand, 2),
        "rows": rows,
    }))


if __name__ == "__main__":
    main()
