"""Fused Stage-2 eval: stream each layer's weights from disk ONCE and run all
three conditions (BF16 / INT8-experts / INT4-experts) through it, carrying three
tiny hidden states forward. Checkpoints every layer so a kill resumes instead of
restarting (background runs have a ~10-min wall limit; a full cold pass is close).

Run repeatedly until it prints DONE:  uv run python fused_eval.py
"""
import os, sys, gc, time, json
import torch
import torch.nn.functional as F
from streamed_forward import (SNAP, load_modeling, ShardIndex, _apply_block,
                              _per_group_rtn)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CKPT = os.environ.get("FUSED_CKPT", "fused_ckpt.pt")
OUT = os.environ.get("FUSED_OUT", "fused_summary.json")
_PF = os.environ.get("FUSED_PROMPTS")
if _PF and os.path.exists(_PF):
    PROMPTS = [ln.rstrip("\n") for ln in open(_PF, encoding="utf-8") if ln.strip()]
else:
    PROMPTS = [
        "The capital of France is",
        "Water is made of hydrogen and",
        "The sun rises in the",
        "Roses are red, violets are",
        "The opposite of hot is",
        "The largest planet in our solar system is",
        "A group of lions is called a",
        "The chemical symbol for gold is",
    ]
CONDS = ["bf16", "int8", "int4"]
BITS = {"int8": 8, "int4": 4}


@torch.no_grad()
def quant_experts_inplace(mixer, bits):
    for e in mixer.experts:
        e.up_proj.weight.copy_(_per_group_rtn(e.up_proj.weight, bits, 128, axis=1))
        e.down_proj.weight.copy_(_per_group_rtn(e.down_proj.weight, bits, 128, axis=1))


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
        h = st["h"]                       # {cond: [ [1,T,H] per prompt ]}
        router = st["router"]             # {cond: {layer: [ [top_k] per prompt ]}}
        ids_list = st["ids_list"]
        cache_pos_list = st["cache_pos_list"]
        print(f"[resume] from layer {start}, RSS={rss():.1f}GB")
    else:
        embed_w = idx.load_one("backbone.embeddings.weight")
        ids_list, cache_pos_list = [], []
        h0 = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids
            ids_list.append(ids)
            cache_pos_list.append(torch.arange(ids.shape[1]))
            h0.append(F.embedding(ids, embed_w))
        del embed_w; gc.collect()
        h = {c: [t.clone() for t in h0] for c in CONDS}
        router = {c: {} for c in CONDS}
        start = 0
        print(f"[start] {len(PROMPTS)} prompts, RSS={rss():.1f}GB")

    for li in range(start, n_layers):
        bt = block_types[li]
        t_load = time.time()
        sd = idx.load_prefixed(f"backbone.layers.{li}.")
        block = mdl_mod.NemotronHBlock(config, layer_idx=li).to(torch.float32).eval()
        block.load_state_dict(sd, strict=False)
        del sd
        t_comp = time.time()

        if bt == "moe":
            snap_w = [(e.up_proj.weight.clone(), e.down_proj.weight.clone())
                      for e in block.mixer.experts]
            for c in CONDS:
                if c != "bf16":
                    quant_experts_inplace(block.mixer, BITS[c])
                cap = []
                gh = block.mixer.gate.register_forward_hook(
                    lambda m, i, o: cap.append(o[0][-1].detach().clone()))
                for pi in range(len(PROMPTS)):
                    h[c][pi] = _apply_block(block, bt, h[c][pi], cache_pos_list[pi])
                gh.remove()
                router[c][li] = cap
                if c != "bf16":  # restore originals for the next condition
                    for e, (u, d) in zip(block.mixer.experts, snap_w):
                        e.up_proj.weight.copy_(u); e.down_proj.weight.copy_(d)
            del snap_w
        else:
            for c in CONDS:
                for pi in range(len(PROMPTS)):
                    h[c][pi] = _apply_block(block, bt, h[c][pi], cache_pos_list[pi])

        del block; gc.collect()
        print(f"[L{li:2d} {bt:9s}] load={t_comp-t_load:4.1f}s comp={time.time()-t_comp:4.1f}s "
              f"RSS={rss():.1f}GB", flush=True)

        torch.save({"next_layer": li+1, "h": h, "router": router,
                    "ids_list": ids_list, "cache_pos_list": cache_pos_list}, CKPT)

    # ---- finalize: norm_f + lm_head per condition --------------------------------
    norm_f_w = idx.load_one("backbone.norm_f.weight")
    norm_f = mdl_mod.NemotronHRMSNorm(H, eps=config.layer_norm_epsilon).to(torch.float32).eval()
    norm_f.weight.copy_(norm_f_w)
    lm_w = idx.load_one("lm_head.weight")

    logits = {c: [] for c in CONDS}
    ppl = {c: [] for c in CONDS}
    for c in CONDS:
        for pi in range(len(PROMPTS)):
            hN = norm_f(h[c][pi])[0]
            lg = hN @ lm_w.T            # [T, vocab]
            logits[c].append(lg)
            ids = ids_list[pi][0]
            if lg.shape[0] > 1:
                logp = F.log_softmax(lg[:-1].float(), -1)
                nll = -logp[torch.arange(ids.shape[0]-1), ids[1:]]
                ppl[c].append(float(torch.exp(nll.mean())))

    def kl_agree(ref, cmp):
        kls, agree, tot, pk = [], 0, 0, []
        for lr, lc in zip(ref, cmp):
            lp = F.log_softmax(lr.float(), -1); lq = F.log_softmax(lc.float(), -1)
            k = (lp.exp() * (lp - lq)).sum(-1)
            kls.append(k); pk.append(float(k.mean()))
            a = (lr.argmax(-1) == lc.argmax(-1)); agree += int(a.sum()); tot += a.numel()
        return float(torch.cat(kls).mean()), pk, agree/tot

    def router_overlap(rb, rq):
        sh, n = 0, 0
        for li in rb:
            for a, b in zip(rb[li], rq[li]):
                sh += len(set(a.tolist()) & set(b.tolist())); n += 1
        return sh/(n*6) if n else float("nan")

    gm = lambda xs: float(torch.exp(torch.log(torch.tensor(xs)).mean()))
    summary = {"prompts": PROMPTS, "bf16_ppl": [round(x,3) for x in ppl["bf16"]],
               "bf16_ppl_geomean": round(gm(ppl["bf16"]),4)}
    for c in ["int8", "int4"]:
        kl, pk, ag = kl_agree(logits["bf16"], logits[c])
        summary[c] = {"mean_kl": kl, "per_prompt_kl": [round(x,6) for x in pk],
                      "ppl": [round(x,3) for x in ppl[c]], "ppl_geomean": round(gm(ppl[c]),4),
                      "top1_agreement_pct": round(100*ag,2),
                      "router_overlap_frac": round(router_overlap(router["bf16"], router[c]),4)}
    summary["seconds_this_run"] = round(time.time()-t0,1)
    summary["peak_rss_gb"] = round(rss(),2)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n==== DONE ====")
    print(json.dumps(summary, indent=2))
    if os.path.exists(CKPT):
        os.remove(CKPT)


if __name__ == "__main__":
    main()
