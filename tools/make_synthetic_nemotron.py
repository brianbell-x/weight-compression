from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/synthetic/nemotron_tiny/hf_snapshot"


def bf16(shape, scale=0.02):
    return (torch.randn(shape) * scale).to(torch.bfloat16)


def f32(shape, start=0.0, end=1.0):
    return torch.linspace(start, end, steps=shape[0], dtype=torch.float32)


def add_mamba(tensors, layer, hidden=64, inner=96, state=16):
    p = f"backbone.layers.{layer}"
    tensors[f"{p}.norm.weight"] = bf16((hidden,))
    tensors[f"{p}.mixer.in_proj.weight"] = bf16((inner, hidden))
    tensors[f"{p}.mixer.conv1d.weight"] = bf16((inner, 1, 4))
    tensors[f"{p}.mixer.A_log"] = f32((inner,), -4.0, 1.0)
    tensors[f"{p}.mixer.D"] = f32((inner,), 0.0, 1.0)
    tensors[f"{p}.mixer.dt_bias"] = f32((inner,), -1.0, 1.0)
    tensors[f"{p}.mixer.out_proj.weight"] = bf16((hidden, inner))
    tensors[f"{p}.mixer.ssm_state.weight"] = bf16((inner, state))


def add_moe(tensors, layer, hidden=64, expert=48, experts=8):
    p = f"backbone.layers.{layer}"
    tensors[f"{p}.norm.weight"] = bf16((hidden,))
    tensors[f"{p}.mixer.gate.weight"] = bf16((experts, hidden))
    tensors[f"{p}.mixer.gate.e_score_correction_bias"] = f32((experts,), -0.25, 0.25)
    tensors[f"{p}.mixer.shared_experts.up_proj.weight"] = bf16((expert * 2, hidden))
    tensors[f"{p}.mixer.shared_experts.down_proj.weight"] = bf16((hidden, expert * 2))
    for i in range(experts):
        tensors[f"{p}.mixer.experts.{i}.up_proj.weight"] = bf16((expert, hidden), 0.02 + i * 0.001)
        tensors[f"{p}.mixer.experts.{i}.down_proj.weight"] = bf16((hidden, expert), 0.018 + i * 0.001)


def add_attention(tensors, layer, hidden=64, heads=4, kv_heads=1, head_dim=16):
    p = f"backbone.layers.{layer}"
    tensors[f"{p}.norm.weight"] = bf16((hidden,))
    tensors[f"{p}.self_attn.q_proj.weight"] = bf16((heads * head_dim, hidden))
    tensors[f"{p}.self_attn.k_proj.weight"] = bf16((kv_heads * head_dim, hidden))
    tensors[f"{p}.self_attn.v_proj.weight"] = bf16((kv_heads * head_dim, hidden))
    tensors[f"{p}.self_attn.o_proj.weight"] = bf16((hidden, heads * head_dim))


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def write_index(shards):
    weight_map = {name: shard for shard, tensors in shards.items() for name in tensors}
    total_size = sum(tensor_bytes(t) for tensors in shards.values() for t in tensors.values())
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n",
        encoding="utf-8",
    )


def write_config():
    config = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "torch_dtype": "bfloat16",
        "hidden_size": 64,
        "vocab_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 48,
        "layer_types": "ME*M",
        "synthetic": True,
        "note": "Small deterministic tensor set for compression experiments; not an inference model.",
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    torch.manual_seed(20260629)
    OUT.mkdir(parents=True, exist_ok=True)
    shard1 = {"backbone.embeddings.weight": bf16((256, 64))}
    add_mamba(shard1, 0)
    add_moe(shard1, 1)
    shard2 = {}
    add_attention(shard2, 2)
    add_mamba(shard2, 3)
    shard2["backbone.norm_f.weight"] = bf16((64,))
    shard2["lm_head.weight"] = bf16((256, 64))
    shards = {"model-00001-of-00002.safetensors": shard1, "model-00002-of-00002.safetensors": shard2}
    for name, tensors in shards.items():
        save_file(tensors, OUT / name)
    write_index(shards)
    write_config()
    print(json.dumps({"out": str(OUT), "shards": len(shards), "tensors": sum(map(len, shards.values()))}, indent=2))


if __name__ == "__main__":
    main()
