"""Large-scale fused Stage-2 INT4 eval. Streams each layer's weights from disk
ONCE and runs BF16 / INT8-experts / INT4-experts through it (routed experts only,
per-group RTN, group 128), carrying three hidden states per item forward.
Checkpoints every layer so a kill resumes. Memory-light finalize: per-item KL is
computed on the fly (no giant logit buffers).

Adds vs fused_eval.py:
  - ~77 diverse short prompts (6 categories) + 2 held-out public-domain passages
  - per-position KL spread (mean/std/median/p90/p99/max), per category
  - corpus perplexity (token-weighted) for short prompts AND held-out passages
  - generation-drift proxy: teacher-forced per-position argmax divergence
    BF16 vs INT4 on the multi-sentence items (true autoregressive gen would cost
    one full ~13-min streamed pass PER token, so we use the multi-position proxy)

Run repeatedly (or in background) until it prints DONE:  uv run python fused_eval_large.py
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

SMOKE = os.environ.get("SMOKE")  # if set: limit layers/items for a fast pipeline test
CKPT = "smoke_ckpt.pt" if SMOKE else "fused_large_ckpt.pt"
OUT = "smoke_summary.json" if SMOKE else "fused_large_summary.json"
CONDS = ["bf16", "int8", "int4"]
BITS = {"int8": 8, "int4": 4}

# unified item list: (text, kind)  kind = category name OR "heldout:<id>"
ITEMS = [(p, c) for p, c in zip(PROMPTS, CATS)]
for hid, text in HELDOUT:
    ITEMS.append((text, f"heldout:{hid}"))
if SMOKE:
    # keep a couple of each-ish + one heldout for a fast end-to-end check
    ITEMS = ITEMS[:6] + ITEMS[-1:]
TEXTS = [t for t, _ in ITEMS]
KINDS = [k for _, k in ITEMS]


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
    if SMOKE:
        n_layers = int(SMOKE)  # SMOKE=N runs only first N layers (must include a moe)
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
        print(f"[resume] from layer {start}/{n_layers}, RSS={rss():.1f}GB", flush=True)
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
        print(f"[start] {len(ITEMS)} items, total_tokens={sum(toklens)}, "
              f"max_len={max(toklens)}, RSS={rss():.1f}GB", flush=True)

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
                for pi in range(len(ITEMS)):
                    h[c][pi] = _apply_block(block, bt, h[c][pi], cache_pos_list[pi])
                gh.remove()
                router[c][li] = cap
                if c != "bf16":
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

    # ---- finalize: norm_f + lm_head, memory-light per-item KL ---------------------
    norm_f_w = idx.load_one("backbone.norm_f.weight")
    norm_f = mdl_mod.NemotronHRMSNorm(H, eps=config.layer_norm_epsilon).to(torch.float32).eval()
    norm_f.weight.copy_(norm_f_w)
    lm_w = idx.load_one("lm_head.weight")

    # accumulators
    pos_kl = {"int8": [], "int4": []}            # per-position KL, all short prompts
    pos_kl_cat = {}                               # cat -> {cond -> [kl...]}
    agree = {"int8": [0, 0], "int4": [0, 0]}      # [matches, total] over short prompts
    nll = {c: {"prompt": [0.0, 0], "heldout": [0.0, 0]} for c in CONDS}  # [sum, count]
    heldout_ppl = {}                              # id -> {cond -> ppl}
    drift = []                                    # per drift-item divergence records

    def logits_of(cond, pi):
        hN = norm_f(h[cond][pi])[0]               # [T, H]
        return hN @ lm_w.T                        # [T, vocab]

    for pi in range(len(ITEMS)):
        kind = KINDS[pi]
        is_heldout = kind.startswith("heldout")
        ids = ids_list[pi][0]
        lg_ref = logits_of("bf16", pi)            # bf16 reference
        lp_ref = F.log_softmax(lg_ref.float(), -1)
        # bf16 ppl
        if lg_ref.shape[0] > 1:
            nllv = -lp_ref[:-1][torch.arange(ids.shape[0]-1), ids[1:]]
            bucket = "heldout" if is_heldout else "prompt"
            nll["bf16"][bucket][0] += float(nllv.sum()); nll["bf16"][bucket][1] += nllv.numel()
            if is_heldout:
                heldout_ppl.setdefault(kind.split(":")[1], {})["bf16"] = float(torch.exp(nllv.mean()))
        ref_argmax = lg_ref.argmax(-1)
        for cond in ["int8", "int4"]:
            lg = logits_of(cond, pi)
            lp = F.log_softmax(lg.float(), -1)
            k = (lp_ref.exp() * (lp_ref - lp)).sum(-1)    # [T] per-position KL
            if not is_heldout:
                pos_kl[cond].extend(k.tolist())
                cat = kind
                pos_kl_cat.setdefault(cat, {}).setdefault(cond, []).extend(k.tolist())
                am = (ref_argmax == lg.argmax(-1))
                agree[cond][0] += int(am.sum()); agree[cond][1] += am.numel()
            # ppl
            if lg.shape[0] > 1:
                nllv = -lp[:-1][torch.arange(ids.shape[0]-1), ids[1:]]
                bucket = "heldout" if is_heldout else "prompt"
                nll[cond][bucket][0] += float(nllv.sum()); nll[cond][bucket][1] += nllv.numel()
                if is_heldout:
                    heldout_ppl.setdefault(kind.split(":")[1], {})[cond] = float(torch.exp(nllv.mean()))
            # generation-drift proxy on int4, multi-sentence items
            if cond == "int4" and kind in DRIFT_CATS and lg.shape[0] > 1:
                rec = {"text_head": TEXTS[pi][:60], "n_pred": int(ids.shape[0]-1), "diverge": []}
                ref_next = ref_argmax[:-1]
                int4_next = lg.argmax(-1)[:-1]
                for t in range(ref_next.shape[0]):
                    if int(ref_next[t]) != int(int4_next[t]):
                        rec["diverge"].append({
                            "pos": t,
                            "ctx": tok.decode(ids[max(0, t-4):t+1].tolist()),
                            "bf16": tok.decode([int(ref_next[t])]),
                            "int4": tok.decode([int(int4_next[t])]),
                        })
                rec["n_diverge"] = len(rec["diverge"])
                rec["diverge"] = rec["diverge"][:8]   # cap examples
                drift.append(rec)
            del lg, lp
        del lg_ref, lp_ref

    def stats(xs):
        if not xs:
            return {}
        t = torch.tensor(xs)
        q = torch.quantile(t, torch.tensor([0.5, 0.9, 0.99]))
        return {"n": len(xs), "mean": float(t.mean()), "std": float(t.std()),
                "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
                "max": float(t.max())}

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
        "n_items_total": len(ITEMS),
        "n_short_prompts": len(PROMPTS),
        "n_heldout_passages": len(HELDOUT),
        "n_predicted_positions_short": agree["int4"][1],
        "categories": {c: CATS.count(c) for c in set(CATS)},
        "corpus_ppl_short": {c: round(ppl(c, "prompt"), 4) for c in CONDS},
        "corpus_ppl_heldout": {c: round(ppl(c, "heldout"), 4) for c in CONDS},
        "heldout_ppl_per_passage": {k: {c: round(v.get(c, float('nan')), 4) for c in CONDS}
                                    for k, v in heldout_ppl.items()},
        "kl_spread": {cond: stats(pos_kl[cond]) for cond in ["int8", "int4"]},
        "kl_by_category": {cat: {cond: round(float(torch.tensor(d[cond]).mean()), 6)
                                 for cond in d} for cat, d in pos_kl_cat.items()},
        "top1_agreement_pct": {cond: round(100 * agree[cond][0] / agree[cond][1], 3)
                               for cond in ["int8", "int4"]},
        "router_overlap_frac": {cond: round(router_overlap(router["bf16"], router[cond]), 4)
                                for cond in ["int8", "int4"]},
        "generation_drift_int4": drift,
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
