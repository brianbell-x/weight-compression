"""
Streamed full forward of NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 on CPU, WITHOUT
loading the 63 GB model at once.

Idea (keystone for Stage-2 capability eval, see
research/candidates/0008-streamed-forward-stage2-eval/brief.md):

  - Keep resident only:
      * the per-prompt running hidden state  [1, T, 2688]  (tiny), and
      * the ONE layer currently being computed.
  - For each of the 52 layers in order: gather that layer's tensors from the
    shard(s) the index points to (some MoE layers straddle two shards), build
    just that block on CPU in float32, push every prompt's hidden state through
    it, then free the block before moving to the next layer.
  - Finish with backbone.norm_f + lm_head to get next-token logits.

Disk is read exactly ONCE total (layers on the outer loop, prompts on the
inner loop), so adding prompts costs compute, not extra disk passes. Peak RAM is
~(largest single layer in fp32 ~= a MoE layer) + embeddings during layer 0, a
few GB, far under 33.7 GB.

Three block types are handled by replicating NemotronHBlock.forward manually
(the real forward wraps in torch.cuda.stream, which crashes CPU-only torch --
same reason capture_activations.py inlines it):
  * mamba      -> Mamba2 mixer torch_forward (pure-PyTorch CPU path)
  * attention  -> SDPA mixer (this model is NoPE: no rotary, plain causal)
  * moe        -> routed top-6 experts + shared expert

Everything runs in float32 (bf16 weights upcast on load). fp32 compute of the
bf16 weights is the highest-fidelity / trusted-reference path and matches
capture_activations.py; it also serves as the BF16-behavior reference the
in-flight quant hook is compared against.

Public API
----------
streamed_forward(prompts, expert_hook=None, ...) -> dict
    Run a full streamed forward over a list of prompt strings. Returns per-prompt
    logits, greedy continuations, perplexity, and RAM/timing stats.

make_int8_expert_hook(group_size=128) / make_int4_expert_hook(group_size=128)
    Build a per-layer hook(layer_idx, mixer) that quant+dequantizes every routed
    expert's up_proj/down_proj weight in-place before that MoE layer runs. Pass as
    streamed_forward(..., expert_hook=hook) to measure quantized behavior end-to-end.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import time
import types
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open

SNAP = r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot"


# =====================================================================================
# Snapshot custom-modeling import (mirrors capture_activations.py)
# =====================================================================================
def load_modeling(snap: str = SNAP):
    """Import the snapshot's custom modeling + config as a package.

    Stubs mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn (imported at module load
    but unused on the CPU torch_forward path) so the import succeeds while leaving
    is_fast_path_available False -> the pure-PyTorch Mamba path is used.
    """
    def _rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None,
                    norm_before_gate=False, upcast=True):
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
    pkg.__path__ = [snap]
    sys.modules["nemo_snap"] = pkg

    def _load(name):
        spec = importlib.util.spec_from_file_location(
            f"nemo_snap.{name}", os.path.join(snap, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"nemo_snap.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_mod = _load("configuration_nemotron_h")
    mdl_mod = _load("modeling_nemotron_h")
    return cfg_mod, mdl_mod


# =====================================================================================
# On-demand layer weight loading (a layer may straddle two shards)
# =====================================================================================
class ShardIndex:
    def __init__(self, snap: str = SNAP):
        self.snap = snap
        with open(os.path.join(snap, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]

    def load_prefixed(self, prefix: str, strip: bool = True) -> dict:
        """Load every tensor whose name starts with `prefix`, as float32.

        Returns {name (optionally prefix-stripped) -> fp32 tensor}. Groups by shard
        so each shard file is opened at most once.
        """
        wanted = [k for k in self.weight_map if k.startswith(prefix)]
        by_shard: dict[str, list[str]] = {}
        for k in wanted:
            by_shard.setdefault(self.weight_map[k], []).append(k)
        out = {}
        for shard, keys in by_shard.items():
            with safe_open(os.path.join(self.snap, shard), framework="pt", device="cpu") as f:
                for k in keys:
                    out[k[len(prefix):] if strip else k] = f.get_tensor(k).to(torch.float32)
        return out

    def load_one(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        with safe_open(os.path.join(self.snap, shard), framework="pt", device="cpu") as f:
            return f.get_tensor(name).to(torch.float32)


# =====================================================================================
# Per-group RTN quant (vendored from 0005 stage1_probe.py to avoid import path games)
# =====================================================================================
def _per_group_rtn(W: torch.Tensor, bits: int, group_size: int = 128, axis: int = 1) -> torch.Tensor:
    """Symmetric per-group round-to-nearest quant+dequant; returns fp32 reconstruction."""
    qmax = (1 << (bits - 1)) - 1  # INT8 -> 127, INT4 -> 7
    W = W.to(torch.float32)
    orig_shape = W.shape
    Wt = W.movedim(axis, -1)
    moved_shape = Wt.shape
    n_along = moved_shape[-1]
    gs = group_size if n_along % group_size == 0 else n_along  # one group if not divisible
    n_groups = n_along // gs
    Wg = Wt.reshape(*moved_shape[:-1], n_groups, gs)
    scale = (Wg.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(Wg / scale), -qmax, qmax)
    Wg_hat = q * scale
    return Wg_hat.reshape(moved_shape).movedim(-1, axis).reshape(orig_shape)


def _make_expert_hook(bits: int, group_size: int = 128) -> Callable:
    """Quant+dequant every routed expert's up_proj/down_proj weight in-place.

    Router gate, shared expert, and norm are left in fp32 (they are tiny / not the
    target of the expert-quant deliverable). axis=1 = along out_features, matching
    the 0005 baseline.
    """
    @torch.no_grad()
    def hook(layer_idx: int, mixer) -> None:
        for expert in mixer.experts:
            expert.up_proj.weight.copy_(_per_group_rtn(expert.up_proj.weight, bits, group_size, axis=1))
            expert.down_proj.weight.copy_(_per_group_rtn(expert.down_proj.weight, bits, group_size, axis=1))
    return hook


def make_int8_expert_hook(group_size: int = 128) -> Callable:
    return _make_expert_hook(8, group_size)


def make_int4_expert_hook(group_size: int = 128) -> Callable:
    return _make_expert_hook(4, group_size)


# =====================================================================================
# Manual single-block apply (replicates NemotronHBlock.forward without cuda.stream)
# =====================================================================================
@torch.no_grad()
def _apply_block(block, block_type: str, h: torch.Tensor, cache_pos: torch.Tensor) -> torch.Tensor:
    residual = h
    hn = block.norm(h.to(block.norm.weight.dtype))
    if block_type == "mamba":
        mixed = block.mixer(hn, cache_params=None, cache_position=cache_pos, attention_mask=None)
    elif block_type == "attention":
        # NoPE model: no rotary. attention_mask=None -> SDPA is_causal path (q_len>1).
        mixed = block.mixer(hn, attention_mask=None, cache_position=cache_pos)[0]
    elif block_type == "moe":
        mixed = block.mixer(hn)
    else:
        raise ValueError(f"unknown block_type {block_type}")
    return residual + mixed


# =====================================================================================
# The streamed forward
# =====================================================================================
@torch.no_grad()
def streamed_forward(
    prompts: list[str],
    snap: str = SNAP,
    expert_hook: Optional[Callable] = None,
    max_layers: Optional[int] = None,
    verbose: bool = True,
    topk_show: int = 5,
    router_record: Optional[dict] = None,
) -> dict:
    """Run a full streamed forward of the model over `prompts`.

    Parameters
    ----------
    prompts      : list of prompt strings (each run as its own batch-1 sequence).
    expert_hook  : optional callable hook(layer_idx, mixer) invoked on each MoE
                   layer's mixer right after its weights are loaded and before it
                   runs, e.g. make_int8_expert_hook(). Use to measure quantized
                   behavior end-to-end. None = pure fp32 (BF16-reference) forward.
    max_layers   : stop after this many layers (debug/timing); None = all 52.
    topk_show    : how many top next-token candidates to decode for the last token.

    Returns dict with per-prompt:
      logits_last  : [vocab] fp32 next-token logits at the final prompt position
      top1_token   : decoded greedy next token
      topk_tokens  : decoded top-k next tokens
      perplexity   : teacher-forced perplexity over the prompt's own tokens
    plus global: peak_rss_gb, seconds_total, seconds_per_prompt, n_layers.
    """
    import psutil
    proc = psutil.Process()

    def rss_gb():
        return proc.memory_info().rss / (1024 ** 3)

    t0 = time.time()
    cfg_mod, mdl_mod = load_modeling(snap)
    config = cfg_mod.NemotronHConfig.from_pretrained(snap)
    # NOTE: use the "eager" attention class, not "sdpa". This model has
    # num_heads*head_dim = 32*128 = 4096 != hidden_size = 2688 (o_proj maps
    # 4096->2688). The snapshot's NemotronHSdpaAttention.forward wrongly reshapes
    # to hidden_size and crashes; the eager NemotronHAttention.forward correctly
    # uses num_heads*head_dim. Both are plain causal SDPA otherwise (NoPE model).
    config._attn_implementation = "eager"
    H = config.hidden_size
    block_types = list(config.layers_block_type)
    n_layers = len(block_types) if max_layers is None else min(max_layers, len(block_types))
    idx = ShardIndex(snap)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)

    # ---- embed every prompt, then free the embedding table -------------------------
    embed_w = idx.load_one("backbone.embeddings.weight")  # [vocab, H] fp32
    ids_list, h_list, cache_pos_list = [], [], []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids  # [1, T]
        ids_list.append(ids)
        h_list.append(F.embedding(ids, embed_w))      # [1, T, H] fp32
        cache_pos_list.append(torch.arange(ids.shape[1]))
    del embed_w
    gc.collect()
    if verbose:
        lens = [int(i.shape[1]) for i in ids_list]
        print(f"[stream] {len(prompts)} prompts, token lens={lens}, "
              f"after-embed RSS={rss_gb():.2f} GB")

    peak_rss = rss_gb()

    # ---- stream the layers; disk read ONCE, all prompts share each resident layer --
    for li in range(n_layers):
        bt = block_types[li]
        t_load = time.time()
        sd = idx.load_prefixed(f"backbone.layers.{li}.")
        block = mdl_mod.NemotronHBlock(config, layer_idx=li).to(torch.float32).eval()
        missing, unexpected = block.load_state_dict(sd, strict=False)
        assert not unexpected, f"layer {li} unexpected keys: {unexpected}"
        # e_score_correction_bias is a buffer that IS in the checkpoint; nothing else
        # should be missing.
        assert not missing, f"layer {li} missing keys: {missing}"
        del sd
        if bt == "moe" and expert_hook is not None:
            expert_hook(li, block.mixer)
        gate_handle = None
        captured = []
        if bt == "moe" and router_record is not None:
            def _grab(mod, inp, out):
                # out = (topk_indices [T, top_k], topk_weights); keep last token's experts
                captured.append(out[0][-1].detach().clone())
            gate_handle = block.mixer.gate.register_forward_hook(_grab)
        t_compute = time.time()
        for i in range(len(prompts)):
            h_list[i] = _apply_block(block, bt, h_list[i], cache_pos_list[i])
        if gate_handle is not None:
            gate_handle.remove()
            router_record[li] = captured  # one [top_k] tensor per prompt, in prompt order
        del block
        gc.collect()
        peak_rss = max(peak_rss, rss_gb())
        if verbose:
            print(f"[stream] layer {li:2d} {bt:9s}  load={t_compute - t_load:5.1f}s  "
                  f"compute={time.time() - t_compute:5.1f}s  RSS={rss_gb():.2f} GB")

    # ---- final norm + lm_head ------------------------------------------------------
    norm_f_w = idx.load_one("backbone.norm_f.weight")
    norm_f = mdl_mod.NemotronHRMSNorm(H, eps=config.layer_norm_epsilon).to(torch.float32).eval()
    norm_f.weight.copy_(norm_f_w)
    lm_head_w = idx.load_one("lm_head.weight")  # [vocab, H] fp32
    peak_rss = max(peak_rss, rss_gb())

    results = []
    for i, p in enumerate(prompts):
        hN = norm_f(h_list[i])[0]                 # [T, H]
        logits = hN @ lm_head_w.T                 # [T, vocab]
        ids = ids_list[i][0]                      # [T]
        last = logits[-1]                         # [vocab]
        topk = torch.topk(last, topk_show)
        top_ids = topk.indices.tolist()
        # teacher-forced perplexity over the prompt's own tokens (predict ids[t+1])
        if logits.shape[0] > 1:
            logp = F.log_softmax(logits[:-1].float(), dim=-1)
            tgt = ids[1:]
            nll = -logp[torch.arange(tgt.shape[0]), tgt]
            ppl = float(torch.exp(nll.mean()))
        else:
            ppl = float("nan")
        finite = bool(torch.isfinite(logits).all())
        results.append({
            "prompt": p,
            "n_tokens": int(ids.shape[0]),
            "top1_token": tok.decode([top_ids[0]]),
            "topk_tokens": [tok.decode([t]) for t in top_ids],
            "topk_logits": [round(float(v), 3) for v in topk.values.tolist()],
            "perplexity": ppl,
            "logits_finite": finite,
            "logits_last": last,  # kept for downstream KL vs quantized run
            "logits_all": logits.detach().clone(),  # [T, vocab] fp32, all positions
        })

    secs = time.time() - t0
    return {
        "results": results,
        "peak_rss_gb": peak_rss,
        "seconds_total": secs,
        "seconds_per_prompt": secs / max(1, len(prompts)),
        "n_layers": n_layers,
        "config_block_types": block_types,
    }


# =====================================================================================
# CLI: BF16 sanity over a handful of short real prompts
# =====================================================================================
DEFAULT_PROMPTS = [
    "The capital of France is",
    "The opposite of hot is",
    "2 + 2 =",
    "Water is made of hydrogen and",
    "The sun rises in the",
    "Roses are red, violets are",
]


def main():
    import argparse
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--quant", choices=["none", "int8", "int4"], default="none")
    args = ap.parse_args()

    hook = None
    if args.quant == "int8":
        hook = make_int8_expert_hook()
    elif args.quant == "int4":
        hook = make_int4_expert_hook()

    out = streamed_forward(DEFAULT_PROMPTS, expert_hook=hook, max_layers=args.max_layers)

    print("\n================ BF16 SANITY ================" if args.quant == "none"
          else f"\n================ {args.quant.upper()} ================")
    for r in out["results"]:
        print(f"\nPROMPT: {r['prompt']!r}  (tokens={r['n_tokens']}, "
              f"finite={r['logits_finite']}, ppl={r['perplexity']:.2f})")
        print(f"  top1 -> {r['top1_token']!r}")
        cont = "  ".join(f"{t!r}({l})" for t, l in zip(r["topk_tokens"], r["topk_logits"]))
        print(f"  topk -> {cont}")
    print(f"\npeak_rss = {out['peak_rss_gb']:.2f} GB")
    print(f"seconds_total = {out['seconds_total']:.1f}  "
          f"seconds_per_prompt = {out['seconds_per_prompt']:.1f}")
    ppls = [r["perplexity"] for r in out["results"]]
    print(f"perplexity: min={min(ppls):.2f} max={max(ppls):.2f}")


if __name__ == "__main__":
    main()
