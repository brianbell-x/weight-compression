"""Self-contained toy-model-of-superposition probe (CPU, seeded).

Question: when inputs are sparse high-dim feature vectors, can a d-dim
bottleneck faithfully carry many MORE than d features? This measures
OCCUPANCY DENSITY = how many features F pack into d dims before
reconstruction degrades, as a function of sparsity.

Setup (Anthropic toy model):
  x in R^F : each feature active w.p. p, magnitude ~ U(0,1), else 0.
  h = W x            (W is d x F, the down-projection / bottleneck)
  x_hat = ReLU(W^T h + b)   (tied up-projection + bias + nonlinearity)
  loss = mean over batch of importance-weighted ||x - x_hat||^2
         (importance uniform = 1 here)

We sweep overcompleteness F/d at a few d, and sparsity p.
For each setting we report:
  - final reconstruction loss (per-feature-normalized)
  - fraction of features "recovered" (per-feature MSE below threshold)
  - the F/d at which packing breaks for each d (recovery drops below 0.5)
"""

import json
import math
import time

import torch

torch.manual_seed(0)
DEVICE = "cpu"
torch.set_num_threads(max(1, torch.get_num_threads()))


def make_batch(F, p, batch, gen):
    """Sparse inputs: active mask w.p. p, magnitude U(0,1)."""
    mag = torch.rand(batch, F, generator=gen)
    mask = (torch.rand(batch, F, generator=gen) < p).float()
    return mag * mask


def train_one(F, d, p, steps=4000, batch=1024, lr=1e-2, seed=0):
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(d, F, generator=gen) * (1.0 / math.sqrt(F))
    W.requires_grad_(True)
    b = torch.zeros(F, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)

    for _ in range(steps):
        x = make_batch(F, p, batch, gen)
        h = x @ W.t()                 # (batch, d)
        x_hat = torch.relu(h @ W + b) # (batch, F)
        loss = ((x - x_hat) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluation on a fresh large batch
    with torch.no_grad():
        eval_gen = torch.Generator().manual_seed(seed + 12345)
        xe = make_batch(F, p, 8192, eval_gen)
        he = xe @ W.t()
        xh = torch.relu(he @ W + b)
        # per-feature MSE
        per_feat_mse = ((xe - xh) ** 2).mean(dim=0)        # (F,)
        # variance of each feature's input (active prob p, U(0,1)):
        # baseline "predict mean" MSE per feature = Var(x_i)
        per_feat_var = xe.var(dim=0, unbiased=False).clamp_min(1e-9)
        # recovered = feature reconstructed much better than its own variance
        frac_recovered = (per_feat_mse < 0.25 * per_feat_var).float().mean().item()
        # normalized loss: total MSE / total input variance
        norm_loss = (per_feat_mse.sum() / per_feat_var.sum()).item()
        raw_loss = per_feat_mse.mean().item()

    return {
        "F": F, "d": d, "p": p, "ratio": F / d,
        "raw_loss": raw_loss,
        "norm_loss": norm_loss,
        "frac_recovered": frac_recovered,
    }


def main():
    t0 = time.time()
    ds = [8, 16, 32]
    ratios = [1, 2, 4, 8, 16, 32]
    ps = [0.3, 0.1, 0.03, 0.01]

    results = []
    for p in ps:
        for d in ds:
            for r in ratios:
                F = d * r
                res = train_one(F, d, p, steps=3000, seed=hash((F, d)) % 100000)
                results.append(res)
                print(f"p={p:<5} d={d:<3} F={F:<5} F/d={r:<3} "
                      f"norm_loss={res['norm_loss']:.4f} "
                      f"recovered={res['frac_recovered']:.3f}")
        print("-" * 60)

    # breakpoint: smallest F/d where recovery drops below 0.5, per (p,d)
    print("\n=== Overcompleteness F/d where packing breaks (recovery<0.5) ===")
    breaks = {}
    for p in ps:
        for d in ds:
            sub = [r for r in results if r["p"] == p and r["d"] == d]
            sub.sort(key=lambda x: x["ratio"])
            brk = None
            for r in sub:
                if r["frac_recovered"] < 0.5:
                    brk = r["ratio"]
                    break
            # max ratio that still kept >=0.9 recovered
            faithful = [r["ratio"] for r in sub if r["frac_recovered"] >= 0.9]
            max_faithful = max(faithful) if faithful else 0
            breaks[f"p={p},d={d}"] = {"break_ratio": brk, "max_faithful_0.9": max_faithful}
            print(f"p={p:<5} d={d:<3} break@F/d={str(brk):<5} max_faithful(>=0.9)@F/d={max_faithful}")

    out = {"results": results, "breaks": breaks, "seconds": time.time() - t0}
    with open("superposition_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['seconds']:.1f}s -> superposition_results.json")


if __name__ == "__main__":
    main()
