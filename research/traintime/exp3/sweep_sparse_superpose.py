"""sweep_sparse_superpose.py -- decisive test of sparsity-gated superposition.

Holds task + training loop FIXED (imports train.run). Varies only the
sparse_superpose FFN family. For each param budget (matched to the dense curve)
we pick (H, K, W2-rank) configs that land at the SAME total params as dense, then
sweep the SPARSITY fraction k_active/H -- which does NOT change params -- to test
the prediction "more sparsity helps superposition".

Reports (total_params, config, val_loss mean over seeds) vs the dense baseline at
matched params.

    uv run python sweep_sparse_superpose.py
"""
from __future__ import annotations

import json
import sys
import time

import layers
from train import run

D_MODEL = 64
N_BLOCKS = 2

# dense baseline (total params, val_loss) at budgets 90k/180k/350k/580k
DENSE = {
    90_000:  {"params": 134016, "val_loss": 1.6832},
    180_000: {"params": 223872, "val_loss": 1.6179},
    350_000: {"params": 393856, "val_loss": 1.5839},
    580_000: {"params": 624000, "val_loss": 1.5576},
}


def solve_H(target_per_module, K, w2_rank):
    """H so one SparseSuperposeFFN module ~= target_per_module params."""
    d = D_MODEL
    if w2_rank is None:
        # params = H*(K+1+d) + d*K + d
        H = (target_per_module - d * K - d) / (K + 1 + d)
    else:
        # params = H*(K+1+w2_rank) + d*(K + w2_rank + 1)
        H = (target_per_module - d * (K + w2_rank + 1)) / (K + 1 + w2_rank)
    return max(8, int(round(H)))


def run_cfg(budget, K, w2_rank, sparsity, seeds):
    target_per_module = budget / N_BLOCKS   # dense total budget split over blocks
    H = solve_H(target_per_module, K, w2_rank)
    layers.SP_H = H
    layers.SP_K = K
    layers.SP_W2_RANK = w2_rank
    layers.SP_SPARSITY = sparsity
    vals, params = [], None
    for s in seeds:
        r = run("sparse_superpose", budget, seed=s)
        vals.append(r["val_loss"])
        params = r["params"]
    k_active = max(1, min(H, int(round(sparsity * H))))
    return {"budget": budget, "H": H, "K": K, "w2_rank": w2_rank,
            "sparsity": sparsity, "k_active": k_active,
            "params": params, "vals": vals,
            "val_loss": sum(vals) / len(vals)}


def main():
    # configs: (budget, K, w2_rank).  Sparsity swept inside.
    # dense-W2 (caps H ~2x d_ff) AND low-rank-W2 (pushes H to 4-16x).
    budgets = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [90_000, 180_000]
    sparsities = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1.0, 0.5, 0.12, 0.03]
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1]

    configs = []
    for b in budgets:
        # dense W2, modest dictionary
        configs.append((b, 16, None))      # H ~ 2x dense d_ff
        # low-rank W2 -> wide H (superposition stress)
        configs.append((b, 16, 8))         # H ~ 4-6x dense d_ff
        configs.append((b, 8, 4))          # H ~ 8-12x dense d_ff (very wide)

    rows = []
    for (b, K, w2r) in configs:
        for sp in sparsities:
            t0 = time.time()
            row = run_cfg(b, K, w2r, sp, seeds)
            wall = time.time() - t0
            d_ff_dense = round(b / (2 * D_MODEL * N_BLOCKS))
            row["H_over_dff"] = round(row["H"] / d_ff_dense, 2)
            row["dense_val"] = DENSE.get(b, {}).get("val_loss")
            row["dense_params"] = DENSE.get(b, {}).get("params")
            row["gap"] = (row["val_loss"] - row["dense_val"]) if row["dense_val"] else None
            rows.append(row)
            seedstr = ",".join(f"{v:.4f}" for v in row["vals"])
            print(f"b={b:>7d} K={K:>2d} w2r={str(w2r):>4s} H={row['H']:>5d}"
                  f"({row['H_over_dff']}x) kact={row['k_active']:>4d} sp={sp:.2f} "
                  f"params={row['params']:>7d} val={row['val_loss']:.4f} "
                  f"({seedstr}) "
                  f"dense={row['dense_val']} gap={row['gap']:+.4f} {wall:.0f}s",
                  flush=True)
    print("\nJSON_ROWS=" + json.dumps(rows))


if __name__ == "__main__":
    main()
