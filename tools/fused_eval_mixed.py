"""Mixed-precision Stage-2 eval for routed experts.

Streams each Nemotron layer once and carries several quantization policies forward:
- bf16 reference
- int8 all routed experts
- int4 all routed experts
- u4d8: up_proj INT4, down_proj INT8
- u8d4: up_proj INT8, down_proj INT4
- last6_int8: last six MoE layers INT8, other MoE layers INT4
- even6_int8: six evenly-spaced MoE layers INT8, other MoE layers INT4

The goal is to test whether a mixed policy can recover most INT8 behavior while
staying much closer to INT4 resident VRAM than the 32 GB all-INT8 floor.

Run from tools/:
  PYTHONPATH= uv run python fused_eval_mixed.py
Optional:
  MIXED_MAX_ITEMS=25 PYTHONPATH= uv run python fused_eval_mixed.py
"""
import os, sys, gc, time, json, math
import torch
import torch.nn.functional as F
from streamed_forward import (SNAP, load_modeling, ShardIndex, _apply_block,
                              _per_group_rtn)
from prompts_large import PROMPTS, CATS, HELDOUT, DRIFT_CATS

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CKPT = os.environ.get("MIXED_CKPT", "fused_mixed_ckpt.pt")
OUT = os.environ.get("MIXED_OUT", "fused_mixed_summary.json")
MAX_ITEMS = int(os.environ.get("MIXED_MAX_ITEMS", "0") or 0)
CONDS = ["bf16", "int8", "int4", "u4d8", "u8d4", "last6_int8", "even6_int8"]
MOE_LAYERS = [1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51]
LAST6 = set(MOE_LAYERS[-6:])
EVEN6 = set([1, 10, 20, 29, 40, 51])

ITEMS = [(p, c) for p, c in zip(PROMPTS, CATS)]
for hid, text in HELDOUT:
    ITEMS.append((text, f"heldout:{hid}"))
if MAX_ITEMS:
    # Keep deterministic short run: first MAX_ITEMS short prompts plus both heldouts when room.
    short = ITEMS[:min(MAX_ITEMS, len(PROMPTS))]
    if MAX_ITEMS >= len(short) + 2:
        short += ITEMS[-2:]
    ITEMS = short
TEXTS = [t for t, _ in ITEMS]
KINDS = [k for _, k in ITEMS]


def policy_bits(cond: str, layer_idx: int):
    """Return (up_bits, down_bits) for routed experts, or (None, None) for BF16."""
    if cond == "bf16":
        return None, None
    if cond == "int8":
        return 8, 8
    if cond == "int4":
        return 4, 4
    if cond == "u4d8":
        return 4, 8
    if cond == "u8d4":
        return 8, 4
    if cond == "last6_int8":
        b = 8 if layer_idx in LAST6 else 4
        return b, b
    if cond == "even6_int8":
        b = 8 if layer_idx in EVEN6 else 4
        return b, b
    raise KeyError(cond)


def effective_expert_bits(cond: str) -> float:
    """Approx effective bits/weight including ~0.125 b/w fp16 group-scale overhead.

    up/down routed expert matrices have equal weight counts in this model, so per-proj
    and per-MoE-layer averages are simple means. Non-expert 4.4 GB floor is handled in
    implied_vram_gb().
    """
    vals = []
    for li in MOE_LAYERS:
        u, d = policy_bits(cond, li)
        if u is None:
            vals.extend([16.0, 16.0])  # BF16 payload; no quant scales
        else:
            vals.extend([u + 0.125, d + 0.125])
    return sum(vals) / len(vals)


def implied_vram_gb(bits: float) -> float:
    # Matches project convention: 4.4 GB non-expert floor + 29.4B routed expert params.
    return 4.4 + (29.4e9 * bits / 8) / (1024 ** 3)


@torch.no_grad()
def quant_experts_policy_inplace(mixer, cond: str, layer_idx: int):
    ub, db = policy_bits(cond, layer_idx)
    if ub is None:
        return
    for e in mixer.experts:
        e.up_proj.weight.copy_(_per_group_rtn(e.up_proj.weight, ub, 128, axis=1))
        e.down_proj.weight.copy_(_per_group_rtn(e.down_proj.weight, db, 128, axis=1))


def main():
    torch.set_grad_enabled(False)
    proc = __import__("psutil").Process()
    rss = lambda: proc.memory_info().rss / 1024**3
    t0 = time.time()

    cfg_mod, mdl_mod = load_modeling(SNAP)
    config = cfg_mod.NemotronHConfig.from_pretrained(SNAP)
    config._attn_implementation = "eager"
    H = config.hidden_size
    block_types = list(config.layers_block_type)
    n_layers = len(block_types)
    idx = ShardIndex(SNAP)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)

    if os.path.exists(CKPT):
        st = torch.load(CKPT)
        start = st["next_layer"]
        h = st["h"]
        router = st["router"]
        ids_list = st["ids_list"]
        cache_pos_list = st["cache_pos_list"]
        print(f"[resume] from layer {start}/{n_layers}, conds={len(CONDS)}, RSS={rss():.1f}GB", flush=True)
    else:
        embed_w = idx.load_one("backbone.embeddings.weight")
        ids_list, cache_pos_list, h0 = [], [], []
        for txt in TEXTS:
            ids = tok(txt, return_tensors="pt").input_ids
            ids_list.append(ids)
            cache_pos_list.append(torch.arange(ids.shape[1]))
            h0.append(F.embedding(ids, embed_w))
        del embed_w; gc.collect()
        h = {c: [t.clone() for t in h0] for c in CONDS}
        router = {c: {} for c in CONDS}
        start = 0
        toklens = [int(i.shape[1]) for i in ids_list]
        print(f"[start] {len(ITEMS)} items, total_tokens={sum(toklens)}, max_len={max(toklens)}, "
              f"conds={len(CONDS)}, RSS={rss():.1f}GB", flush=True)

    for li in range(start, n_layers):
        bt = block_types[li]
        t_load = time.time()
        sd = idx.load_prefixed(f"backbone.layers.{li}.")
        block = mdl_mod.NemotronHBlock(config, layer_idx=li).to(torch.float32).eval()
        block.load_state_dict(sd, strict=False)
        del sd
        t_comp = time.time()

        if bt == "moe":
            snap_w = [(e.up_proj.weight.clone(), e.down_proj.weight.clone()) for e in block.mixer.experts]
            for c in CONDS:
                quant_experts_policy_inplace(block.mixer, c, li)
                cap = []
                gh = block.mixer.gate.register_forward_hook(
                    lambda m, i, o: cap.append(o[0][-1].detach().clone()))
                for pi in range(len(ITEMS)):
                    h[c][pi] = _apply_block(block, bt, h[c][pi], cache_pos_list[pi])
                gh.remove()
                router[c][li] = cap
                # restore originals for next condition
                for e, (u, d) in zip(block.mixer.experts, snap_w):
                    e.up_proj.weight.copy_(u); e.down_proj.weight.copy_(d)
            del snap_w
        else:
            for c in CONDS:
                for pi in range(len(ITEMS)):
                    h[c][pi] = _apply_block(block, bt, h[c][pi], cache_pos_list[pi])

        del block; gc.collect()
        print(f"[L{li:2d} {bt:9s}] load={t_comp-t_load:5.1f}s comp={time.time()-t_comp:6.1f}s "
              f"RSS={rss():.1f}GB elapsed={time.time()-t0:6.1f}s", flush=True)
        torch.save({"next_layer": li+1, "h": h, "router": router,
                    "ids_list": ids_list, "cache_pos_list": cache_pos_list}, CKPT)

    norm_f_w = idx.load_one("backbone.norm_f.weight")
    norm_f = mdl_mod.NemotronHRMSNorm(H, eps=config.layer_norm_epsilon).to(torch.float32).eval()
    norm_f.weight.copy_(norm_f_w)
    lm_w = idx.load_one("lm_head.weight")

    pos_kl = {c: [] for c in CONDS if c != "bf16"}
    pos_kl_cat = {}
    agree = {c: [0, 0] for c in CONDS if c != "bf16"}
    nll = {c: {"prompt": [0.0, 0], "heldout": [0.0, 0]} for c in CONDS}
    drift = {c: [] for c in CONDS if c not in ("bf16", "int8")}

    def logits_of(cond, pi):
        hN = norm_f(h[cond][pi])[0]
        return hN @ lm_w.T

    for pi in range(len(ITEMS)):
        kind = KINDS[pi]
        is_heldout = kind.startswith("heldout")
        ids = ids_list[pi][0]
        lg_ref = logits_of("bf16", pi)
        lp_ref = F.log_softmax(lg_ref.float(), -1)
        if lg_ref.shape[0] > 1:
            nllv = -lp_ref[:-1][torch.arange(ids.shape[0]-1), ids[1:]]
            bucket = "heldout" if is_heldout else "prompt"
            nll["bf16"][bucket][0] += float(nllv.sum()); nll["bf16"][bucket][1] += nllv.numel()
        ref_argmax = lg_ref.argmax(-1)
        for cond in [c for c in CONDS if c != "bf16"]:
            lg = logits_of(cond, pi)
            lp = F.log_softmax(lg.float(), -1)
            k = (lp_ref.exp() * (lp_ref - lp)).sum(-1)
            if not is_heldout:
                pos_kl[cond].extend(k.tolist())
                pos_kl_cat.setdefault(kind, {}).setdefault(cond, []).extend(k.tolist())
                am = (ref_argmax == lg.argmax(-1))
                agree[cond][0] += int(am.sum()); agree[cond][1] += am.numel()
            if lg.shape[0] > 1:
                nllv = -lp[:-1][torch.arange(ids.shape[0]-1), ids[1:]]
                bucket = "heldout" if is_heldout else "prompt"
                nll[cond][bucket][0] += float(nllv.sum()); nll[cond][bucket][1] += nllv.numel()
            if cond in drift and kind in DRIFT_CATS and lg.shape[0] > 1:
                rec = {"text_head": TEXTS[pi][:60], "n_pred": int(ids.shape[0]-1), "diverge": []}
                ref_next = ref_argmax[:-1]
                cmp_next = lg.argmax(-1)[:-1]
                for t in range(ref_next.shape[0]):
                    if int(ref_next[t]) != int(cmp_next[t]):
                        rec["diverge"].append({
                            "pos": t,
                            "ctx": tok.decode(ids[max(0, t-4):t+1].tolist()),
                            "bf16": tok.decode([int(ref_next[t])]),
                            cond: tok.decode([int(cmp_next[t])]),
                        })
                rec["n_diverge"] = len(rec["diverge"])
                rec["diverge"] = rec["diverge"][:8]
                drift[cond].append(rec)
            del lg, lp
        del lg_ref, lp_ref

    def stats(xs):
        if not xs:
            return {}
        t = torch.tensor(xs)
        q = torch.quantile(t, torch.tensor([0.5, 0.9, 0.99]))
        return {"n": len(xs), "mean": float(t.mean()), "std": float(t.std()),
                "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), "max": float(t.max())}

    def ppl(cond, bucket):
        s, n = nll[cond][bucket]
        return float(math.exp(s / n)) if n else float("nan")

    def router_overlap(rb, rq):
        sh, n = 0, 0
        for li in rb:
            for a, b in zip(rb[li], rq[li]):
                sh += len(set(a.tolist()) & set(b.tolist())); n += 1
        return sh / (n * 6) if n else float("nan")

    summary = {
        "conditions": CONDS,
        "moe_layers": MOE_LAYERS,
        "last6_int8_layers": sorted(LAST6),
        "even6_int8_layers": sorted(EVEN6),
        "n_items_total": len(ITEMS),
        "n_short_prompts": sum(1 for k in KINDS if not k.startswith("heldout")),
        "n_heldout_passages": sum(1 for k in KINDS if k.startswith("heldout")),
        "n_predicted_positions_short": next(iter(agree.values()))[1] if agree else 0,
        "effective_expert_bits": {c: round(effective_expert_bits(c), 4) for c in CONDS},
        "implied_resident_vram_gb": {c: round(implied_vram_gb(effective_expert_bits(c)), 2) for c in CONDS},
        "corpus_ppl_short": {c: round(ppl(c, "prompt"), 4) for c in CONDS},
        "corpus_ppl_heldout": {c: round(ppl(c, "heldout"), 4) for c in CONDS},
        "kl_spread": {cond: stats(pos_kl[cond]) for cond in pos_kl},
        "kl_by_category": {cat: {cond: round(float(torch.tensor(vals).mean()), 6)
                                  for cond, vals in d.items()} for cat, d in pos_kl_cat.items()},
        "top1_agreement_pct": {cond: round(100 * agree[cond][0] / agree[cond][1], 3) for cond in agree},
        "router_overlap_frac": {cond: round(router_overlap(router["bf16"], router[cond]), 4)
                                for cond in agree},
        "generation_drift": drift,
        "seconds_this_run": round(time.time() - t0, 1),
        "peak_rss_gb": round(rss(), 2),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n==== DONE ====")
    print(json.dumps(summary, indent=2))
    if os.path.exists(CKPT):
        os.remove(CKPT)


if __name__ == "__main__":
    main()
