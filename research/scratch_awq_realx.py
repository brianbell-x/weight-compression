import numpy as np
import torch
from safetensors import safe_open

np.random.seed(0)
ACTS = r"C:\dev\compression\research\artifacts\layer1_expert_input_acts.npy"
ST = r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot\model-00001-of-00013.safetensors"
GROUP = 128

X = np.load(ACTS).astype(np.float64)  # [96, 2688]
print("X shape", X.shape, "channel std spread",
      X.std(0).max() / (X.std(0).min() + 1e-12))

def load_expert(n):
    name = f"backbone.layers.1.mixer.experts.{n}.up_proj.weight"
    with safe_open(ST, framework="pt") as f:
        keys = [k for k in f.keys() if name == k]
        if not keys:
            return None
        return f.get_tensor(name).float().numpy().astype(np.float64)  # [1856,2688]

# discover available experts
with safe_open(ST, framework="pt") as f:
    allk = list(f.keys())
import re
exp_ids = sorted({int(re.search(r"experts\.(\d+)\.up_proj", k).group(1))
                  for k in allk if "layers.1.mixer.experts." in k and "up_proj.weight" in k})
print("available experts:", len(exp_ids), exp_ids[:5], "...")
EXPERTS = exp_ids[:16]
print("using", len(EXPERTS), "experts")

def quant_per_group_int(W, bits, scale_vec=None):
    """RTN symmetric per-group (group along input dim / columns). W [out,in].
    Groups of GROUP columns. Returns dequant Wq and bits-overhead count."""
    out, inn = W.shape
    qmax = 2**(bits-1) - 1
    Wq = np.empty_like(W)
    ngroups = 0
    for c0 in range(0, inn, GROUP):
        c1 = min(c0+GROUP, inn)
        block = W[:, c0:c1]
        maxabs = np.max(np.abs(block), axis=1, keepdims=True)  # per-row per-group
        scale = maxabs / qmax
        scale[scale == 0] = 1.0
        q = np.round(block / scale)
        q = np.clip(q, -qmax-1, qmax)
        Wq[:, c0:c1] = q * scale
        ngroups += 1
    # overhead: one fp16 scale per (row, group)
    return Wq

def out_err(W, Wq):
    Y = X @ W.T
    Yq = X @ Wq.T
    return np.linalg.norm(Y - Yq) / np.linalg.norm(Y)

# importance vectors
s_meanabs = np.mean(np.abs(X), axis=0)  # [2688]
s_std = np.std(X, axis=0)
energy = np.sum(X**2, axis=0)  # [2688]

def bits_per_weight(bits, n_int8_cols=0, inn=2688):
    # base bits + per-group fp16 scale overhead = 16/GROUP = 0.125 b/w per group
    # mixed: n_int8_cols at 8 bits, rest at `bits`
    n4 = inn - n_int8_cols
    avg = (n4*bits + n_int8_cols*8) / inn
    return avg + 16.0/GROUP  # scale overhead per group of 128

results = []

# Pre-load experts
Ws = [load_expert(n) for n in EXPERTS]

# 1. INT4 RTN baseline
errs = [out_err(W, quant_per_group_int(W, 4)) for W in Ws]
results.append(("INT4 RTN", 4 + 0.125, np.mean(errs)))
int4_rtn = np.mean(errs)
print("INT4 RTN realX err", int4_rtn)

# 5. INT8 reference
errs8 = [out_err(W, quant_per_group_int(W, 8)) for W in Ws]
int8_ref = np.mean(errs8)
results.append(("INT8 RTN", 8 + 0.125, int8_ref))
print("INT8 ref", int8_ref)

# 2. AWQ-style per-channel scaling
def awq(W, s, alpha, bits=4):
    sc = np.power(s + 1e-12, alpha)  # [in]
    sc = sc / np.exp(np.mean(np.log(sc)))  # normalize geomean to 1 (stability)
    Ws_ = W * sc[None, :]   # scale columns
    Wq = quant_per_group_int(Ws_, bits)
    return Wq / sc[None, :]

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
awq_res = {}
for sname, s in [("mean|X|", s_meanabs), ("std(X)", s_std)]:
    for a in alphas:
        errs = [out_err(W, awq(W, s, a)) for W in Ws]
        awq_res[(sname, a)] = np.mean(errs)
        print(f"AWQ {sname} alpha={a}: {np.mean(errs):.4f}")

best_awq = min(awq_res.items(), key=lambda kv: kv[1])
print("BEST AWQ", best_awq)
best_s_name, best_alpha = best_awq[0]
best_s = s_meanabs if best_s_name == "mean|X|" else s_std
results.append((f"AWQ {best_s_name} a={best_alpha}", 4.125, best_awq[1]))

# 3. Activation-energy salient mixing: top-k cols INT8, rest INT4
def salient_mix(W, sel_cols, bits_lo=4, bits_hi=8):
    out, inn = W.shape
    Wq = quant_per_group_int(W, bits_lo).copy()
    # high-precision for salient columns: quantize those columns at INT8 individually
    Wsel = quant_per_group_int(W[:, sel_cols], bits_hi)
    Wq[:, sel_cols] = Wsel
    return Wq

sal_res = {}
for frac in [0.005, 0.01, 0.02, 0.05]:
    k = max(1, int(round(frac*2688)))
    sel = np.argsort(energy)[::-1][:k]
    errs = [out_err(W, salient_mix(W, sel)) for W in Ws]
    bpw = bits_per_weight(4, n_int8_cols=k)
    sal_res[frac] = (bpw, np.mean(errs))
    print(f"Salient energy k={frac*100}% ({k} cols): bpw={bpw:.3f} err={np.mean(errs):.4f}")
    results.append((f"Salient-energy {frac*100}%", bpw, np.mean(errs)))

best_sal = min(sal_res.items(), key=lambda kv: kv[1][1])
print("BEST salient", best_sal)

# 4. Combine best AWQ + salient mixing
def combo(W, s, alpha, sel_cols, bits_lo=4, bits_hi=8):
    sc = np.power(s + 1e-12, alpha)
    sc = sc / np.exp(np.mean(np.log(sc)))
    Ws_ = W * sc[None, :]
    Wq = quant_per_group_int(Ws_, bits_lo)
    Wsel = quant_per_group_int(Ws_[:, sel_cols], bits_hi)
    Wq[:, sel_cols] = Wsel
    return Wq / sc[None, :]

combo_res = {}
for frac in [0.01, 0.02]:
    k = max(1, int(round(frac*2688)))
    sel = np.argsort(energy)[::-1][:k]
    errs = [out_err(W, combo(W, best_s, best_alpha, sel)) for W in Ws]
    bpw = bits_per_weight(4, n_int8_cols=k)
    combo_res[frac] = (bpw, np.mean(errs))
    print(f"COMBO awq+salient {frac*100}%: bpw={bpw:.3f} err={np.mean(errs):.4f}")
    results.append((f"COMBO awq+sal {frac*100}%", bpw, np.mean(errs)))

best_combo = min(combo_res.items(), key=lambda kv: kv[1][1])

print("\n=== TABLE ===")
print(f"{'config':30s} {'bits/w':>8s} {'realX err':>10s}")
for name, bpw, err in results:
    print(f"{name:30s} {bpw:8.3f} {err*100:9.3f}%")

# verdict checks: anything <=4 bits with <=2% or <=4%
le4 = [(n,b,e) for n,b,e in results if b <= 4.2 and "INT8" not in n]  # ~4bit incl overhead
under2 = [(n,b,e) for n,b,e in le4 if e <= 0.02]
under4 = [(n,b,e) for n,b,e in le4 if e <= 0.04]
print("under2% at <=4bit:", under2)
print("under4% at <=4bit:", under4)

import json
out = {
  "experts": len(EXPERTS),
  "int4_rtn": int4_rtn,
  "int8_ref": int8_ref,
  "best_alpha": best_alpha,
  "best_s_name": best_s_name,
  "best_awq_err": best_awq[1],
  "best_sal": [best_sal[0], best_sal[1][0], best_sal[1][1]],
  "best_combo": [best_combo[0], best_combo[1][0], best_combo[1][1]],
  "salient_best_err": min(v[1] for v in sal_res.values()),
  "results": results,
}
print("JSON", json.dumps(out, default=float))
