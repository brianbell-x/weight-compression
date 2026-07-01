"""Capture REAL input activations feeding the routed experts at the first MoE layer
(layer 1) of Nemotron-3-Nano-30B, over a batch of short real prompts, and cache them.

Method (proven feasible in the scoping step; see capture_real_activations.py):
  Only embeddings + layer 0 (Mamba) + layer 1 pre-MoE norm are built and loaded -- all
  from shard 1 (~2.7 GB peak RSS). The captured tensor X is the post-layers.1.norm hidden
  state [tokens, hidden_size=2688], which the model code feeds verbatim to both the router
  (gate, in=2688) and every routed expert's up_proj (in=2688). The 128 experts (58 GB) are
  never loaded.

This script:
  1. Runs the partial CPU forward for ~8-16 short real prompts.
  2. Concatenates all token positions into one X matrix [n_tokens, 2688] (float32).
  3. Computes per-input-channel activation energy = RMS magnitude per channel (what AWQ
     needs to pick salient channels), shape [2688].
  4. Caches X, the channel energy, and metadata under tests/artifacts/activations/.
  5. Sanity check: loads the real layer-1 expert0 up_proj weight, orients it so X @ W is
     the true up_proj output ([N,2688] @ [2688,1856]), and runs ONE INT8 per-group RTN
     through stage1_probe.fidelity() with the REAL X. Compares rel_err vs the 0.67%
     random-X baseline.
"""
import os, sys, time, json
import torch
import torch.nn.functional as F
import importlib.util, types
import psutil
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "activations")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, HERE)  # so we can import stage1_probe

SNAP = r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot"
SHARD1 = os.path.join(SNAP, "model-00001-of-00013.safetensors")

proc = psutil.Process()
_peak = {"rss": 0.0}
def rss_gb():
    g = proc.memory_info().rss / 1024**3
    if g > _peak["rss"]:
        _peak["rss"] = g
    return g

# --- Shim mamba_ssm (pure-PyTorch gated RMSNorm; modeling module hard-imports it) -----
def _rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=False):
    dtype = x.dtype
    x = x.float()
    if z is not None and not norm_before_gate:
        x = x * F.silu(z.float())
    if group_size is None:
        rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        out = x * rstd * weight.float()
    else:
        shp = x.shape
        xg = x.reshape(*shp[:-1], shp[-1] // group_size, group_size)
        rstd = torch.rsqrt(xg.square().mean(-1, keepdim=True) + eps)
        out = (xg * rstd).reshape(shp) * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None and norm_before_gate:
        out = out * F.silu(z.float())
    return out.to(dtype)

def _mkmod(name, **attrs):
    m = types.ModuleType(name); [setattr(m, k, v) for k, v in attrs.items()]
    sys.modules[name] = m; return m
_mkmod("mamba_ssm"); _mkmod("mamba_ssm.ops"); _mkmod("mamba_ssm.ops.triton")
_mkmod("mamba_ssm.ops.triton.layernorm_gated", rmsnorm_fn=_rmsnorm_fn)
_mkmod("mamba_ssm.ops.triton.selective_state_update", selective_state_update=None)
_mkmod("mamba_ssm.ops.triton.ssd_combined",
       mamba_chunk_scan_combined=None, mamba_split_conv1d_scan_combined=None)

# --- Load the snapshot's custom model code as a synthetic package ---------------------
_pkg = types.ModuleType("nemo_pkg"); _pkg.__path__ = [SNAP]; sys.modules["nemo_pkg"] = _pkg
def _load(name):
    spec = importlib.util.spec_from_file_location(f"nemo_pkg.{name}", os.path.join(SNAP, name + ".py"))
    mod = importlib.util.module_from_spec(spec); sys.modules[f"nemo_pkg.{name}"] = mod
    spec.loader.exec_module(mod); return mod
NemotronHConfig = _load("configuration_nemotron_h").NemotronHConfig
M = _load("modeling_nemotron_h")

import stage1_probe as S1

# =====================================================================================
t0 = time.time()
cfg = NemotronHConfig(**json.load(open(os.path.join(SNAP, "config.json"))))
H = cfg.hidden_size
print(f"hidden_size={H} block_types[0..3]={cfg.layers_block_type[:4]}")

# --- Build only the modules we need (mamba mixer + two RMSNorms) ----------------------
norm0 = M.NemotronHRMSNorm(H, eps=cfg.layer_norm_epsilon)   # pre-mamba (layer 0)
mixer0 = M.NemotronHMamba2Mixer(cfg, layer_idx=0)            # layer 0 mamba mixer
norm1 = M.NemotronHRMSNorm(H, eps=cfg.layer_norm_epsilon)   # pre-MoE norm (layer 1)

# --- Load just the needed weights from shard 1, cast to float32 -----------------------
want = {
    "backbone.layers.0.norm.weight": (norm0, "weight"),
    "backbone.layers.1.norm.weight": (norm1, "weight"),
}
mixer_prefix = "backbone.layers.0.mixer."
loaded_bytes = 0
with safe_open(SHARD1, framework="pt") as f:
    for name, (mod, attr) in want.items():
        t = f.get_tensor(name).to(torch.float32)
        loaded_bytes += t.numel() * 4
        getattr(mod, attr).data = t
    msd = mixer0.state_dict()
    new = {}
    for k in msd:
        t = f.get_tensor(mixer_prefix + k).to(torch.float32)
        loaded_bytes += t.numel() * 4
        new[k] = t
    mixer0.load_state_dict(new)
    emb = f.get_tensor("backbone.embeddings.weight").to(torch.float32)
    loaded_bytes += emb.numel() * 4
print(f"loaded weights ~{loaded_bytes/1024**3:.3f} GB (f32); embedding={tuple(emb.shape)}")
print(f"RSS after load: {rss_gb():.2f} GB  (t={time.time()-t0:.1f}s)")

for m in (norm0, mixer0, norm1):
    m.eval()

# --- Real prompts (short, varied domains so outlier channels are well sampled) --------
PROMPTS = [
    "The theory of general relativity describes gravity as the curvature of spacetime.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "In 1969, Apollo 11 landed the first humans on the surface of the Moon.",
    "A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals.",
    "The stock market fell sharply after the central bank raised interest rates.",
    "She opened the old wooden door and stepped into the quiet, dusty library.",
    "To sort a list efficiently, many programmers reach for the quicksort algorithm.",
    "The recipe calls for two cups of flour, a pinch of salt, and three eggs.",
    "Climate change is driving more frequent and intense heat waves across the globe.",
    "He tuned his guitar carefully before walking out onto the crowded stage.",
    "The human immune system defends the body against bacteria, viruses, and parasites.",
    "Quantum computers use qubits that can exist in superpositions of states.",
]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)

# --- Partial forward per prompt; collect post-layers.1.norm hidden states -------------
def capture_one(ids):
    with torch.no_grad():
        h = F.embedding(ids, emb)                 # [1, seq, H] f32
        res = h
        hn = norm0(h)
        mout = mixer0(hn, cache_params=None, cache_position=None, attention_mask=None)
        h = res + mout
        X = norm1(h)                              # [1, seq, H] -- routed-expert input
    return X.reshape(-1, H)

rows = []
per_prompt = []
for i, p in enumerate(PROMPTS):
    ids = tok(p, return_tensors="pt").input_ids
    Xp = capture_one(ids)
    rows.append(Xp)
    per_prompt.append({"prompt": p, "n_tokens": int(Xp.shape[0])})
    rss_gb()
    print(f"  prompt {i:2d}: {Xp.shape[0]:3d} tokens  | {p[:48]}...")

X = torch.cat(rows, dim=0).contiguous()           # [n_tokens, H] f32
n_tokens = X.shape[0]
print(f"\n=== CAPTURED ROUTED-EXPERT INPUT ===")
print(f"X shape={tuple(X.shape)} dtype={X.dtype}  (in_features=hidden_size={H})")
print(f"X stats: mean={X.mean():.4f} std={X.std():.4f} absmax={X.abs().max():.4f}")

# --- Per-input-channel activation energy (RMS magnitude per channel) -- AWQ input -----
chan_energy = X.pow(2).mean(dim=0).sqrt()         # [H], sqrt(mean_token X[:,c]^2)
ratio = (chan_energy.max() / chan_energy.mean()).item()
print(f"channel_energy shape={tuple(chan_energy.shape)} "
      f"max={chan_energy.max():.4f} mean={chan_energy.mean():.4f} max/mean={ratio:.2f}")

# --- Cache to disk --------------------------------------------------------------------
import numpy as np
X_pt = os.path.join(OUT_DIR, "real_X_layer1.pt")
X_npy = os.path.join(OUT_DIR, "real_X_layer1.npy")
E_pt = os.path.join(OUT_DIR, "channel_energy_layer1.pt")
E_npy = os.path.join(OUT_DIR, "channel_energy_layer1.npy")
META = os.path.join(OUT_DIR, "capture_meta.json")
torch.save(X, X_pt)
np.save(X_npy, X.numpy())
torch.save(chan_energy, E_pt)
np.save(E_npy, chan_energy.numpy())
meta = {
    "layer": 1, "role": "first MoE layer; input to gate + every routed expert up_proj",
    "tensor": "post backbone.layers.1.norm hidden state",
    "X_shape": list(X.shape), "n_tokens": n_tokens, "hidden_size": H,
    "dtype": "float32", "n_prompts": len(PROMPTS),
    "channel_energy_def": "sqrt(mean over tokens of X[:,c]^2) per input channel",
    "channel_energy_max_over_mean": ratio,
    "prompts": per_prompt,
    "files": {"X_pt": X_pt, "X_npy": X_npy,
              "channel_energy_pt": E_pt, "channel_energy_npy": E_npy},
}
json.dump(meta, open(META, "w"), indent=2)
print(f"\nsaved:\n  {X_pt}\n  {X_npy}\n  {E_pt}\n  {E_npy}\n  {META}")

# =====================================================================================
# Sanity: ONE INT8 fidelity() through stage1_probe with the REAL X
# =====================================================================================
# up_proj.weight is nn.Linear weight [out=1856, in=2688]. The true op is y = X @ W.T.
# stage1_probe.fidelity does Y = X @ W, so pass W_oriented = up_proj.weight.T = [2688,1856]
# (= [in, out]); group-quantize along axis=0 (the 2688 in-axis; 1856 is NOT divisible
# by group_size=128, so axis=0 is the only valid grouping).
UP = "backbone.layers.1.mixer.experts.0.up_proj.weight"
W_raw = S1.load_expert(SHARD1, UP)                # [1856, 2688]
W = W_raw.t().contiguous()                        # [2688, 1856] = [in, out]
print(f"\n[sanity] up_proj.weight raw shape={tuple(W_raw.shape)} (out,in)")
print(f"[sanity] oriented W shape={tuple(W.shape)} (in,out)  -> X@W valid: "
      f"{W.shape[0]==X.shape[1]}")
assert W.shape[0] == X.shape[1], "X in_features must match oriented W rows"

W_prime, qmeta = S1.int8_per_group_rtn(W, group_size=128, axis=0)
fid_real = S1.fidelity(W, W_prime, X)

# Same codec on random X for direct comparison (random unit-norm rows, in=2688).
X_rand = S1.make_inputs(W.shape[0], batch=n_tokens, seed=0)
fid_rand = S1.fidelity(W, W_prime, X_rand)

print(f"\n=== INT8 per-group RTN (group_size=128, axis=0) on expert0.up_proj ===")
print(f"  REAL-X   rel_err = {fid_real['rel_err']*100:.3f}%   mean_cosine={fid_real['mean_cosine']:.6f}")
print(f"  RANDOM-X rel_err = {fid_rand['rel_err']*100:.3f}%   mean_cosine={fid_rand['mean_cosine']:.6f}")
print(f"  (test-001 random-X baseline was ~0.67%)")
print(f"  bits/weight={qmeta['bits_per_weight']:.4f}  implied_vram={S1.implied_vram_gb(qmeta['bits_per_weight']):.2f} GB")

rss_gb()
print(f"\npeak RSS: {_peak['rss']:.2f} GB   total wall time={time.time()-t0:.1f}s")

# Machine-readable summary line (last line for easy parsing).
print("SUMMARY " + json.dumps({
    "X_pt": X_pt, "X_npy": X_npy, "channel_energy_pt": E_pt, "channel_energy_npy": E_npy,
    "meta": META, "X_shape": list(X.shape), "n_tokens": n_tokens,
    "n_prompts": len(PROMPTS),
    "int8_real_rel_err_pct": round(fid_real["rel_err"]*100, 3),
    "int8_random_rel_err_pct": round(fid_rand["rel_err"]*100, 3),
    "peak_rss_gb": round(_peak["rss"], 2),
}))
