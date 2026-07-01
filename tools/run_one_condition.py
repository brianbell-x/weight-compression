"""Run ONE condition (bf16|int8|int4) of the streamed forward and persist
logits_all + router records to disk, so the 3 passes are independent and
robust to interruption. KL is computed afterward from the saved tensors."""
import sys, json
import torch
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

cond = sys.argv[1]
hook = {"bf16": None,
        "int8": make_int8_expert_hook(group_size=128) if cond == "int8" else None,
        "int4": make_int4_expert_hook(group_size=128) if cond == "int4" else None}[cond]

rec = {}
out = streamed_forward(PROMPTS, expert_hook=hook, verbose=True, router_record=rec)

payload = {
    "cond": cond,
    "prompts": PROMPTS,
    "logits_all": [r["logits_all"].half() for r in out["results"]],  # half to save space
    "perplexity": [r["perplexity"] for r in out["results"]],
    "top1": [r.get("top1_token") for r in out["results"]],
    "router": {int(li): [t.tolist() for t in rec[li]] for li in rec},
    "peak_rss_gb": out.get("peak_rss_gb"),
    "seconds_total": out.get("seconds_total"),
}
torch.save(payload, f"cond_{cond}.pt")
print(f"\nWROTE cond_{cond}.pt  ppl={[round(x,2) for x in payload['perplexity']]}  "
      f"peak_rss={payload['peak_rss_gb']}GB  secs={payload['seconds_total']}")
