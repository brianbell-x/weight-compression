"""Run BF16 / INT8 / INT4 streamed forwards on the SAME prompts and report
KL, perplexity, top-1 agreement, and router top-6 overlap."""
import sys, json
import torch
import torch.nn.functional as F
from streamed_forward import (streamed_forward, make_int8_expert_hook,
                              make_int4_expert_hook)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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

def run(hook):
    rec = {}
    out = streamed_forward(PROMPTS, expert_hook=hook, verbose=True,
                           router_record=rec)
    return out, rec

print("=== BF16 ===");  bf16, rec_bf16 = run(None)
print("=== INT8 ===");  int8, rec_int8 = run(make_int8_expert_hook(group_size=128))
print("=== INT4 ===");  int4, rec_int4 = run(make_int4_expert_hook(group_size=128))

def pooled_ppl(out):
    # pool nll over all predicting positions across all prompts
    nlls = []
    for r in out["results"]:
        lg = r["logits_all"]  # [T, vocab]
        # use the same prompt-token teacher forcing isn't available here without ids;
        # fall back to per-prompt ppl already computed (exp mean nll over own tokens)
        nlls.append(r["perplexity"])
    return nlls

def kl_and_agree(ref, cmp):
    """mean KL(ref||cmp) over all positions, and top-1 agreement fraction."""
    kls = []
    agree = 0
    total = 0
    per_prompt_kl = []
    for rr, rc in zip(ref["results"], cmp["results"]):
        lr = rr["logits_all"].float()  # [T, vocab]
        lc = rc["logits_all"].float()
        logp = F.log_softmax(lr, dim=-1)
        logq = F.log_softmax(lc, dim=-1)
        p = logp.exp()
        kl = (p * (logp - logq)).sum(dim=-1)  # [T]
        kls.append(kl)
        per_prompt_kl.append(float(kl.mean()))
        a = (lr.argmax(-1) == lc.argmax(-1))
        agree += int(a.sum()); total += a.numel()
    allkl = torch.cat(kls)
    return float(allkl.mean()), per_prompt_kl, agree / total

def router_overlap(rb, rq):
    """mean |top6_bf16 ∩ top6_quant| / 6 over all (moe_layer, prompt)."""
    shared, count = 0, 0
    for li in rb:
        for a, b in zip(rb[li], rq[li]):
            sa = set(a.tolist()); sb = set(b.tolist())
            shared += len(sa & sb); count += 1
    return shared / (count * 6), count

bf16_ppl = [r["perplexity"] for r in bf16["results"]]
int8_ppl = [r["perplexity"] for r in int8["results"]]
int4_ppl = [r["perplexity"] for r in int4["results"]]

kl8, pp_kl8, agree8 = kl_and_agree(bf16, int8)
kl4, pp_kl4, agree4 = kl_and_agree(bf16, int4)
ro8, ncells = router_overlap(rec_bf16, rec_int8)
ro4, _ = router_overlap(rec_bf16, rec_int4)

import statistics as st
def gm(xs): return float(torch.exp(torch.log(torch.tensor(xs)).mean()))

summary = {
    "prompts": PROMPTS,
    "n_prompts": len(PROMPTS),
    "bf16_ppl_per_prompt": [round(x,3) for x in bf16_ppl],
    "bf16_ppl_geomean": round(gm(bf16_ppl),4),
    "int8": {
        "mean_kl": kl8,
        "per_prompt_kl": [round(x,6) for x in pp_kl8],
        "ppl_per_prompt": [round(x,3) for x in int8_ppl],
        "ppl_geomean": round(gm(int8_ppl),4),
        "top1_agreement_pct": round(100*agree8,3),
        "router_overlap_frac": round(ro8,5),
        "router_cells": ncells,
    },
    "int4": {
        "mean_kl": kl4,
        "per_prompt_kl": [round(x,6) for x in pp_kl4],
        "ppl_per_prompt": [round(x,3) for x in int4_ppl],
        "ppl_geomean": round(gm(int4_ppl),4),
        "top1_agreement_pct": round(100*agree4,3),
        "router_overlap_frac": round(ro4,5),
        "router_cells": ncells,
    },
    "peak_rss_gb": bf16["peak_rss_gb"],
    "seconds_total_bf16": bf16["seconds_total"],
}
print("\n==== SUMMARY JSON ====")
print(json.dumps(summary, indent=2))
with open("quant_eval_summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
print("\nWROTE quant_eval_summary.json")
