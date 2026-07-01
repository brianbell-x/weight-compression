"""Prototype: capture REAL input activations to the routed experts at the first
MoE layer (layer 1) of Nemotron-3-Nano-30B, using a PARTIAL CPU forward.

Only embeddings + layer 0 (Mamba) + layer 1 pre-MoE norm are built and loaded.
All from shard 1. The captured tensor X is the post-layers.1.norm hidden state
[tokens, hidden_size=2688] -- the exact input fed to both the router (gate) and
each routed expert's up_proj (in_features=2688). No full model load.
"""
import os, sys, time, json
import torch
import psutil
from safetensors import safe_open

SNAP = r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot"
SHARD1 = os.path.join(SNAP, "model-00001-of-00013.safetensors")

import torch.nn.functional as F
import importlib.util, types

# --- Shim mamba_ssm: the modeling module hard-imports rmsnorm_fn at import time, but
#     the actual CPU torch_forward only needs a pure-PyTorch gated RMSNorm. Provide it. ---
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
    m = types.ModuleType(name); [setattr(m, k, v) for k, v in attrs.items()]; sys.modules[name] = m; return m
_mkmod("mamba_ssm"); _mkmod("mamba_ssm.ops"); _mkmod("mamba_ssm.ops.triton")
_mkmod("mamba_ssm.ops.triton.layernorm_gated", rmsnorm_fn=_rmsnorm_fn)
_mkmod("mamba_ssm.ops.triton.selective_state_update", selective_state_update=None)
_mkmod("mamba_ssm.ops.triton.ssd_combined", mamba_chunk_scan_combined=None, mamba_split_conv1d_scan_combined=None)

# Load the snapshot's custom code as a synthetic package so its relative imports resolve.
_pkg = types.ModuleType("nemo_pkg"); _pkg.__path__ = [SNAP]; sys.modules["nemo_pkg"] = _pkg
def _load(name):
    spec = importlib.util.spec_from_file_location(f"nemo_pkg.{name}", os.path.join(SNAP, name + ".py"))
    mod = importlib.util.module_from_spec(spec); sys.modules[f"nemo_pkg.{name}"] = mod
    spec.loader.exec_module(mod); return mod
NemotronHConfig = _load("configuration_nemotron_h").NemotronHConfig
M = _load("modeling_nemotron_h")

proc = psutil.Process()
def rss_gb():
    return proc.memory_info().rss / 1024**3

t0 = time.time()
cfg = NemotronHConfig(**json.load(open(os.path.join(SNAP, "config.json"))))
print(f"hidden_size={cfg.hidden_size} block_types[0..2]={cfg.layers_block_type[:3]}")

# --- Build only the modules we need (tiny: mamba mixer + two RMSNorms) ----------
norm0 = M.NemotronHRMSNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)   # pre-mamba
mixer0 = M.NemotronHMamba2Mixer(cfg, layer_idx=0)                          # layer 0 mamba
norm1 = M.NemotronHRMSNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)   # pre-MoE (layer 1)

# --- Load just the needed weights from shard 1, cast to float32 -----------------
want = {
    "backbone.layers.0.norm.weight": (norm0, "weight"),
    "backbone.layers.1.norm.weight": (norm1, "weight"),
}
mixer_prefix = "backbone.layers.0.mixer."
loaded_bytes = 0
with safe_open(SHARD1, framework="pt") as f:
    keys = list(f.keys())
    # norms
    for name, (mod, attr) in want.items():
        t = f.get_tensor(name).to(torch.float32)
        loaded_bytes += t.numel() * 4
        getattr(mod, attr).data = t
    # mamba mixer params
    msd = mixer0.state_dict()
    new = {}
    for k in msd:
        full = mixer_prefix + k
        t = f.get_tensor(full).to(torch.float32)
        loaded_bytes += t.numel() * 4
        new[k] = t
    mixer0.load_state_dict(new)
    # embedding weight (largest single tensor)
    emb = f.get_tensor("backbone.embeddings.weight").to(torch.float32)
    loaded_bytes += emb.numel() * 4
print(f"loaded weights ~{loaded_bytes/1024**3:.3f} GB (f32); embedding={tuple(emb.shape)}")
print(f"RSS after load: {rss_gb():.2f} GB  (t={time.time()-t0:.1f}s)")

for m in (norm0, mixer0, norm1):
    m.eval()

# --- One real prompt -------------------------------------------------------------
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
prompt = "The theory of general relativity describes gravity as the curvature of spacetime."
ids = tok(prompt, return_tensors="pt").input_ids
print(f"prompt tokens: {ids.shape[1]}  ids[:8]={ids[0,:8].tolist()}")

# --- Partial forward (manual block math, skipping the cuda.stream wrapper) -------
with torch.no_grad():
    h = F.embedding(ids, emb)                       # [1, seq, 2688] f32
    # layer 0 (mamba): residual + mixer(norm(h))
    res = h
    hn = norm0(h)
    mout = mixer0(hn, cache_params=None, cache_position=None, attention_mask=None)
    h = res + mout                                  # [1, seq, 2688]
    # layer 1 pre-MoE norm -> THIS is the routed-expert input X
    X = norm1(h)                                    # [1, seq, 2688]

X2d = X.reshape(-1, cfg.hidden_size)
print(f"\n=== CAPTURED ROUTED-EXPERT INPUT ===")
print(f"X shape={tuple(X2d.shape)} dtype={X2d.dtype}  (in_features=hidden_size={cfg.hidden_size})")
print(f"X stats: mean={X2d.mean():.4f} std={X2d.std():.4f} absmax={X2d.abs().max():.4f}")
print(f"peak RSS: {rss_gb():.2f} GB  total t={time.time()-t0:.1f}s")

# sanity vs experts' in_features
with safe_open(SHARD1, framework="pt") as f:
    up = f.get_tensor("backbone.layers.1.mixer.experts.0.up_proj.weight")
    gate = f.get_tensor("backbone.layers.1.mixer.gate.weight")
print(f"expert0.up_proj.weight shape={tuple(up.shape)} (out,in)=(1856,2688) -> in matches X: {up.shape[1]==X2d.shape[1]}")
print(f"gate.weight shape={tuple(gate.shape)} -> in matches X: {gate.shape[1]==X2d.shape[1]}")

out_path = os.path.join(os.path.dirname(__file__), "real_X_layer1.pt")
torch.save(X2d, out_path)
print(f"saved {out_path}")
