"""
Capture REAL input activations to the layer-1 MoE expert mixer of
NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, WITHOUT loading the full 63 GB model.

Strategy (stays well under 33 GB RAM):
  - Load ONLY from shard 1 the tensors for:
        backbone.embeddings.weight
        backbone.layers.0.*        (full Mamba2 block: norm + mixer)
        backbone.layers.1.norm.weight
  - Build just NemotronHBlock(layer 0) and a standalone RMSNorm for layer 1's
    pre-mixer norm. We do NOT build the 128 experts at all -- we only want the
    INPUT that would be fed to them.
  - Forward:  ids -> embeddings -> block0 (Mamba2 torch_forward CPU path)
              -> layer1.norm  ==> THIS is the per-token expert-mixer input [T, 2688].

Everything runs in float32 on CPU. The Mamba torch_forward path is the pure
PyTorch fallback (Triton/cuda kernels are gated on CUDA, so CPU uses it).
"""

import os, sys, json, importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

SNAP = r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot"
OUT_DIR = r"C:\dev\compression\research\artifacts"
SHARD1 = os.path.join(SNAP, "model-00001-of-00013.safetensors")

PROMPTS = [
    "The capital of France is",
    "In 1969, humans first walked on the surface of the",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Water boils at a temperature of 100 degrees",
    "The mitochondria is the powerhouse of the",
    "To compress a neural network without losing accuracy, one common method is",
    "Once upon a time, in a small village nestled between two mountains,",
    "The derivative of x squared with respect to x is",
]


def load_modeling():
    """Import the snapshot's custom modeling + config modules as a package
    (the modeling file uses relative imports like `.configuration_nemotron_h`)."""
    import types
    # The modeling file hard-imports `rmsnorm_fn` from mamba_ssm at module load
    # time, but the CPU torch_forward path never calls it (it uses the pure
    # PyTorch MambaRMSNormGated). Inject a stub so the import succeeds. We do NOT
    # stub the triton fast-path fns, so is_fast_path_available stays False and
    # the CPU naive path is used.
    def _rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None,
                    norm_before_gate=False, upcast=True):
        # Faithful pure-PyTorch port of mamba_ssm.ops.triton.layernorm_gated
        # reference (rms_norm_ref). group_size groups the last dim; gate z is
        # applied via SiLU. For this model: norm_before_gate=False, bias=None.
        dtype = x.dtype
        w = weight.float()
        b = bias.float() if bias is not None else None
        if upcast:
            x = x.float()
            z = z.float() if z is not None else z
        if z is not None and not norm_before_gate:
            x = x * F.silu(z)
        if group_size is None:
            rstd = 1.0 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
            out = x * rstd * w
        else:
            xg = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
            rstd = 1.0 / torch.sqrt(xg.square().mean(dim=-1, keepdim=True) + eps)
            out = (xg * rstd).reshape(x.shape) * w
        if b is not None:
            out = out + b
        if z is not None and norm_before_gate:
            out = out * F.silu(z)
        return out.to(dtype)
    ms = types.ModuleType("mamba_ssm")
    ops = types.ModuleType("mamba_ssm.ops")
    tr = types.ModuleType("mamba_ssm.ops.triton")
    lg = types.ModuleType("mamba_ssm.ops.triton.layernorm_gated")
    lg.rmsnorm_fn = _rmsnorm_fn
    for name, m in [("mamba_ssm", ms), ("mamba_ssm.ops", ops),
                    ("mamba_ssm.ops.triton", tr),
                    ("mamba_ssm.ops.triton.layernorm_gated", lg)]:
        sys.modules.setdefault(name, m)

    pkg = types.ModuleType("nemo_snap")
    pkg.__path__ = [SNAP]
    sys.modules["nemo_snap"] = pkg

    def _load(name):
        spec = importlib.util.spec_from_file_location(
            f"nemo_snap.{name}", os.path.join(SNAP, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"nemo_snap.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_mod = _load("configuration_nemotron_h")
    mdl_mod = _load("modeling_nemotron_h")
    return cfg_mod, mdl_mod


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)
    cfg_mod, mdl_mod = load_modeling()

    config = cfg_mod.NemotronHConfig.from_pretrained(SNAP)
    H = config.hidden_size
    print(f"hidden_size={H}  layer0={config.layers_block_type[0]}  layer1={config.layers_block_type[1]}")
    assert config.layers_block_type[1] == "moe", "layer 1 is not MoE?!"

    # ---- selectively pull only the tensors we need from shard 1 (fp32) ----
    need_prefixes = ("backbone.embeddings.weight",
                     "backbone.layers.0.",
                     "backbone.layers.1.norm.weight")
    block0_sd, embed_w, norm1_w = {}, None, None
    with safe_open(SHARD1, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            if not k.startswith(need_prefixes):
                continue
            t = f.get_tensor(k).to(torch.float32)
            if k == "backbone.embeddings.weight":
                embed_w = t
            elif k == "backbone.layers.1.norm.weight":
                norm1_w = t
            elif k.startswith("backbone.layers.0."):
                block0_sd[k[len("backbone.layers.0."):]] = t
    assert embed_w is not None and norm1_w is not None and block0_sd
    print(f"loaded embed {tuple(embed_w.shape)}, "
          f"block0 tensors={len(block0_sd)}, norm1 {tuple(norm1_w.shape)}")

    # ---- build only block 0 (Mamba2) and layer 1's pre-mixer RMSNorm ----
    with torch.no_grad():
        block0 = mdl_mod.NemotronHBlock(config, layer_idx=0).to(torch.float32).eval()
        missing, unexpected = block0.load_state_dict(block0_sd, strict=False)
        assert not unexpected, f"unexpected block0 keys: {unexpected}"
        assert not missing, f"missing block0 keys: {missing}"

        norm1 = mdl_mod.NemotronHRMSNorm(H, eps=config.layer_norm_epsilon).to(torch.float32).eval()
        norm1.weight.copy_(norm1_w)

    # ---- tokenize real prompts ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)

    captured, manifest_prompts = [], []
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids  # [1, T]
            emb = F.embedding(ids, embed_w)               # [1, T, H] fp32
            cache_pos = torch.arange(ids.shape[1])
            # layer 0: replicate NemotronHBlock.forward manually (its real forward
            # wraps in torch.cuda.stream which crashes on CPU-only torch).
            # residual_in_fp32 is False for this config.
            residual = emb
            hn = block0.norm(emb.to(block0.norm.weight.dtype))
            mixed = block0.mixer(hn, cache_params=None, cache_position=cache_pos,
                                 attention_mask=None)   # Mamba2 torch_forward CPU path
            h = residual + mixed                          # [1, T, H]
            # layer 1 pre-mixer RMSNorm  ==> expert-mixer input
            mixer_in = norm1(h)                            # [1, T, H]
            v = mixer_in[0].to(torch.float32).numpy()     # [T, H]
            captured.append(v)
            manifest_prompts.append({"prompt": p, "tokens": int(ids.shape[1])})
            print(f"  '{p[:40]}...'  ->  {v.shape}")

    acts = np.concatenate(captured, axis=0)  # [total_tokens, H]
    np.save(os.path.join(OUT_DIR, "layer1_expert_input_acts.npy"), acts)

    # ---- sanity stats ----
    finite = np.isfinite(acts).all()
    mean, std = float(acts.mean()), float(acts.std())
    per_ch_std = acts.std(axis=0)            # [H]
    ch_energy = (acts.astype(np.float64) ** 2).sum(axis=0)  # [H]
    total_e = ch_energy.sum()
    order = np.argsort(ch_energy)[::-1]
    fracs = {}
    for k in (1, 8, 32, 64, 128):
        fracs[k] = float(ch_energy[order[:k]].sum() / total_e)

    manifest = {
        "source": "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 / shard1 early-layer partial forward",
        "capture_point": "input to backbone.layers[1].mixer (MoE) = layer1 pre-mixer RMSNorm output",
        "path": "embeddings -> layer0 Mamba2 (torch_forward CPU) -> layer1 RMSNorm",
        "dtype_compute": "float32",
        "shape": list(acts.shape),
        "dtype_saved": str(acts.dtype),
        "n_prompts": len(PROMPTS),
        "total_tokens": int(acts.shape[0]),
        "prompts": manifest_prompts,
        "stats": {
            "all_finite": bool(finite),
            "mean": mean,
            "std": std,
            "per_channel_std_min": float(per_ch_std.min()),
            "per_channel_std_max": float(per_ch_std.max()),
            "per_channel_std_median": float(np.median(per_ch_std)),
            "topk_energy_fraction": fracs,
        },
    }
    with open(os.path.join(OUT_DIR, "layer1_expert_input_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== SANITY ===")
    print(json.dumps(manifest["stats"], indent=2))
    print(f"\nsaved: {os.path.join(OUT_DIR, 'layer1_expert_input_acts.npy')}")


if __name__ == "__main__":
    main()
