"""GPTQ-style error-feedback quantization of REAL layer-1 routed experts, scored with
the cached REAL activations through stage1_probe.fidelity().

Motivation (candidate 0005, test-001): with RANDOM X, every data-free sub-4-bit codec
hit >= 9.7% matmul output error -- the "4-bit wall". GPTQ could not be tested then
because it NEEDS real activations: it builds the input Hessian H = X^T X and does
Hessian-aware sequential column quantization with error feedback (Frantar et al. 2022,
the OBQ/GPTQ algorithm). This script feeds the cached REAL X into GPTQ and measures
whether it breaks the wall.

What it does
------------
For N>=16 real layer-1 experts (backbone.layers.1.mixer.experts.{e}):
  * up_proj  [out=1856, in=2688]: input = cached REAL X (post layers.1.norm hidden state,
    187 tokens x 2688). This is the matmul the router-selected expert actually performs.
  * down_proj [out=2688, in=1856]: input = relu2(X @ up_proj^T) -- the TRUE second-hop
    activation, computed per expert from the full-precision up_proj. (act_fn=relu2 per
    config.json mlp_hidden_act.)

For each weight it runs, per group:
  * GPTQ  (Hessian-aware, error-feedback) at 4-bit and 3-bit
  * RTN   (data-free per-group round-to-nearest) at 4-bit and 3-bit  -- the baseline GPTQ
    must beat
  * INT8 RTN reference (the known ~0.27% real-X point)

Honesty controls
----------------
  * Token split: the Hessian is built ONLY from calibration tokens; fidelity is reported
    on HELD-OUT eval tokens (so we are not scoring on the data we fit). Full-set numbers
    are also printed.
  * bits/weight is the EFFECTIVE value incl. the fp16 group scale (stage1_probe.bits_per_weight).
  * All numbers are measured on real weights + real activations. No synthetic values.
"""
import os, sys, json, time
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_probe as S1

SNAP = r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot"
SHARD1 = os.path.join(SNAP, "model-00001-of-00013.safetensors")
ACT = os.path.join(HERE, "activations", "real_X_layer1.pt")

torch.manual_seed(0)
N_EXPERTS = 16
CAL_FRAC = 0.75          # fraction of tokens used to build the Hessian
GROUP_UP = 128           # 2688 % 128 == 0  -> 21 groups
GROUP_DN = 116           # 1856 % 116 == 0  -> 16 groups (1856 = 128*14.5 not int)
PERCDAMP = 0.01


# ---------------------------------------------------------------------------------------
# Symmetric per-group quantizer params (one fp16 scale per (row, group))
# ---------------------------------------------------------------------------------------
def find_scale(Wblock: torch.Tensor, bits: int) -> torch.Tensor:
    """Wblock [out, gs] -> scale [out,1]; symmetric, maxq = 2^(b-1)-1."""
    maxq = 2 ** (bits - 1) - 1
    max_abs = Wblock.abs().amax(dim=1, keepdim=True)
    return (max_abs / maxq).clamp_min(1e-12)


def quant(w: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    maxq = 2 ** (bits - 1) - 1
    q = torch.clamp(torch.round(w / scale), -maxq, maxq)
    return q * scale


# ---------------------------------------------------------------------------------------
# RTN baseline (data-free): per-group symmetric round-to-nearest along the in dimension.
# W is [out, in]; group along in.
# ---------------------------------------------------------------------------------------
def rtn_quantize(W: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    out, cols = W.shape
    Q = torch.empty_like(W)
    for g0 in range(0, cols, group_size):
        g1 = min(g0 + group_size, cols)
        blk = W[:, g0:g1]
        s = find_scale(blk, bits)
        Q[:, g0:g1] = quant(blk, s, bits)
    return Q


# ---------------------------------------------------------------------------------------
# GPTQ: Hessian-aware sequential column quantization with error feedback.
# W [out, in]; H [in, in] = X^T X.  Groups along in (cols), one scale per (row, group).
# Follows Frantar et al. 2022 (Cholesky-inverse formulation).
# ---------------------------------------------------------------------------------------
def gptq_quantize(W: torch.Tensor, H: torch.Tensor, group_size: int, bits: int,
                  percdamp: float = PERCDAMP, blocksize: int = 128) -> torch.Tensor:
    W = W.clone().float()
    out, cols = W.shape
    H = H.clone().float()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    damp = percdamp * torch.mean(torch.diag(H))
    idx = torch.arange(cols)
    H[idx, idx] += damp

    # Hinv = upper-triangular Cholesky factor of H^{-1}
    L = torch.linalg.cholesky(H)
    Hinv_full = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv_full, upper=True)

    Q = torch.zeros_like(W)
    scale_cache = {}  # group-start col -> scale[out,1]

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            col = i1 + i
            w = W1[:, i]
            d = Hinv1[i, i]

            if col % group_size == 0:
                # (re)compute the group scale from the CURRENT weights of this group,
                # reading straight from W (already error-updated for cols < col).
                g1 = min(col + group_size, cols)
                scale_cache[col] = find_scale(W[:, col:g1], bits)
            gstart = col - (col % group_size)
            s = scale_cache[gstart]

            q = quant(w.unsqueeze(1), s, bits).squeeze(1)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return Q


# ---------------------------------------------------------------------------------------
def eff_bits(num_weights: int, group_size: int, bits: int) -> float:
    n_groups = num_weights // group_size
    if num_weights % group_size:
        n_groups += 1  # partial last group still needs a scale (won't happen here)
    return S1.bits_per_weight(num_weights, payload_bits=bits * num_weights,
                              scale_bits=16 * n_groups * (num_weights // num_weights and 1))  # placeholder


def eff_bits2(out_rows: int, cols: int, group_size: int, bits: int) -> float:
    # one fp16 scale per (row, group)
    num_weights = out_rows * cols
    groups_per_row = (cols + group_size - 1) // group_size
    n_scales = out_rows * groups_per_row
    return S1.bits_per_weight(num_weights, payload_bits=bits * num_weights,
                              scale_bits=16 * n_scales)


# ---------------------------------------------------------------------------------------
def run_proj(name, W_oriented, X, group_size, n_cal):
    """W_oriented [in,out] (fidelity orientation). GPTQ works on Wq=[out,in]=W_oriented.T.
    X [n,in]. Build H from first n_cal tokens; eval on the rest (+ full)."""
    in_f, out_f = W_oriented.shape
    Wq = W_oriented.t().contiguous()          # [out, in]
    Xc = X[:n_cal]
    Xe = X[n_cal:]
    H = Xc.t() @ Xc                            # [in, in] Hessian from calibration tokens

    results = {}
    # reference: INT8 per-group RTN (axis=0 = in-axis of W_oriented)
    W8, m8 = S1.int8_per_group_rtn(W_oriented, group_size=group_size, axis=0)
    results["int8_rtn"] = {
        "bits": m8["bits_per_weight"],
        "eval": S1.fidelity(W_oriented, W8, Xe),
        "full": S1.fidelity(W_oriented, W8, X),
    }
    for bits in (4, 3):
        b = eff_bits2(out_f, in_f, group_size, bits)
        # RTN
        Qr = rtn_quantize(Wq, group_size, bits)
        Wr = Qr.t().contiguous()
        # GPTQ
        Qg = gptq_quantize(Wq, H, group_size, bits)
        Wg = Qg.t().contiguous()
        results[f"rtn_{bits}b"] = {"bits": b,
                                   "eval": S1.fidelity(W_oriented, Wr, Xe),
                                   "full": S1.fidelity(W_oriented, Wr, X)}
        results[f"gptq_{bits}b"] = {"bits": b,
                                    "eval": S1.fidelity(W_oriented, Wg, Xe),
                                    "full": S1.fidelity(W_oriented, Wg, X)}
    return results


# ---------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    X = torch.load(ACT).float()                # [187, 2688]
    n_tok, H_in = X.shape
    n_cal = int(round(n_tok * CAL_FRAC))
    print(f"X {tuple(X.shape)}  cal={n_cal} eval={n_tok-n_cal}")

    agg = {}  # key -> {proj -> list of dicts}
    for proj in ("up", "down"):
        agg[proj] = {}

    for e in range(N_EXPERTS):
        with safe_open(SHARD1, framework="pt") as f:
            Wup = f.get_tensor(f"backbone.layers.1.mixer.experts.{e}.up_proj.weight").float()    # [1856,2688]
            Wdn = f.get_tensor(f"backbone.layers.1.mixer.experts.{e}.down_proj.weight").float()  # [2688,1856]

        # up_proj: oriented [in=2688, out=1856], input = real X
        Wup_or = Wup.t().contiguous()
        rup = run_proj("up", Wup_or, X, GROUP_UP, n_cal)

        # down_proj: input = relu2(X @ Wup^T) = relu2(up_proj output)  [187,1856]
        A = torch.relu(X @ Wup.t()) ** 2
        Wdn_or = Wdn.t().contiguous()          # [in=1856, out=2688]
        rdn = run_proj("down", Wdn_or, A, GROUP_DN, n_cal)

        for proj, r in (("up", rup), ("down", rdn)):
            for k, v in r.items():
                agg[proj].setdefault(k, []).append(v)
        print(f"expert {e:2d}  up.gptq4b eval={rup['gptq_4b']['eval']['rel_err']*100:5.2f}%  "
              f"up.gptq3b eval={rup['gptq_3b']['eval']['rel_err']*100:5.2f}%  "
              f"up.rtn4b eval={rup['rtn_4b']['eval']['rel_err']*100:5.2f}%  "
              f"dn.gptq4b eval={rdn['gptq_4b']['eval']['rel_err']*100:5.2f}%")

    # ---- aggregate (mean over experts) ----
    def summ(lst, field):
        ev = sum(d["eval"]["rel_err"] for d in lst) / len(lst)
        fu = sum(d["full"]["rel_err"] for d in lst) / len(lst)
        co = sum(d["eval"]["mean_cosine"] for d in lst) / len(lst)
        return ev, fu, co, lst[0]["bits"]

    print("\n=================  AGGREGATE (mean over %d experts)  =================" % N_EXPERTS)
    print(f"{'proj':5s} {'codec':10s} {'bits':>6s} {'eval_rel%':>10s} {'full_rel%':>10s} {'cos':>10s} {'vram_GB':>9s}")
    rows_out = []
    order = ["int8_rtn", "rtn_4b", "gptq_4b", "rtn_3b", "gptq_3b"]
    for proj in ("up", "down"):
        for k in order:
            lst = agg[proj][k]
            ev, fu, co, b = summ(lst, k)
            vram = S1.implied_vram_gb(b)
            print(f"{proj:5s} {k:10s} {b:6.3f} {ev*100:10.3f} {fu*100:10.3f} {co:10.6f} {vram:9.2f}")
            rows_out.append({"proj": proj, "codec": k, "bits": round(b, 4),
                             "eval_rel_err_pct": round(ev*100, 4),
                             "full_rel_err_pct": round(fu*100, 4),
                             "eval_mean_cosine": round(co, 6),
                             "implied_vram_gb": round(vram, 2)})

    print(f"\nwall={time.time()-t0:.1f}s")
    print("SUMMARY " + json.dumps({"n_experts": N_EXPERTS, "n_cal": n_cal,
                                   "n_eval": n_tok - n_cal, "rows": rows_out}))


if __name__ == "__main__":
    main()
