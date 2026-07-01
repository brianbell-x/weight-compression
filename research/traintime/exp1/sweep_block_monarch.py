import json
from train import run

# Dense reference: 290304 total params -> final loss 1.1162367725372315
DENSE_REF_PARAMS = 290304

# Sweep param_budget so the model's TOTAL params span ~0.25x..1.5x the dense ref.
# Because block_monarch targets ~`budget` params per swappable matrix (same
# interface as dense), the model total lands near dense total at the same budget.
budgets = [18000, 36000, 72576, 108000, 145000]

print("family seed budget params swappable d_ff final_loss val_loss")
results = []
for bud in budgets:
    r = run("block_monarch", param_budget=bud, seed=0, verbose=False)
    results.append(r)
    print(f"block_monarch 0 {bud} {r['params']} {r['swappable_params']} "
          f"{r['d_ff']} {r['final_loss']:.5f} {r['val_loss']:.5f}")

# Also run dense at the matching budgets to draw the reference curve.
print("\n--- dense reference curve at same budgets ---")
for bud in budgets:
    r = run("dense", param_budget=bud, seed=0, verbose=False)
    print(f"dense 0 {bud} {r['params']} {r['swappable_params']} "
          f"{r['d_ff']} {r['final_loss']:.5f} {r['val_loss']:.5f}")

print("\nJSON:")
print(json.dumps([{"params": r["params"], "swappable": r["swappable_params"],
                   "final_loss": r["final_loss"], "val_loss": r["val_loss"]}
                  for r in results]))
