"""sweep_shared_dict.py -- port of exp1's shared_dict contender to exp2.

Holds the task + training loop fixed (imports train.run). Varies only K (atom
count) of the shared_dict family across the SAME param budgets as the dense
curve. Reports (total_params, val_loss) for every (budget, K) point so we can
compare shared_dict to dense at MATCHED params in the capacity-bound regime.
"""
from __future__ import annotations

import json
import time

import layers
from train import run

# same budgets that produced the dense curve params [134016,223872,393856,624000]
BUDGETS = [90_000, 180_000, 350_000, 580_000]
KS = [1, 2, 3, 4, 6, 8]
SEEDS = [0, 1]

DENSE = [
    {"params": 134016, "val_loss": 1.6832},
    {"params": 223872, "val_loss": 1.6179},
    {"params": 393856, "val_loss": 1.5839},
    {"params": 624000, "val_loss": 1.5576},
]


def run_sd(budget, K, seed):
    layers.SHARED_DICT_K = K
    layers.reset_shared_dict()
    r = run("shared_dict", budget, seed=seed)
    return r


def main():
    rows = []
    for budget in BUDGETS:
        for K in KS:
            vals = []
            params = None
            t0 = time.time()
            for seed in SEEDS:
                r = run_sd(budget, K, seed)
                vals.append(r["val_loss"])
                params = r["params"]
            wall = time.time() - t0
            mean_val = sum(vals) / len(vals)
            row = {"budget": budget, "K": K, "params": params,
                   "val_loss_seed0": vals[0], "val_loss_seed1": vals[1],
                   "val_loss_mean": mean_val}
            rows.append(row)
            print(f"budget={budget:>7d} K={K} params={params:>7d} "
                  f"val[s0]={vals[0]:.4f} val[s1]={vals[1]:.4f} "
                  f"mean={mean_val:.4f} wall={wall:.1f}s", flush=True)
    print("\nJSON_ROWS=" + json.dumps(rows))


if __name__ == "__main__":
    main()
