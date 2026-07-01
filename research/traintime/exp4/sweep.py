"""exp4 sweep: dense baseline + sparse_superpose_v2 at MATCHED total params.

Fair test fixing exp3's input bottleneck: full-rank W1, shared output dictionary
(W2=D@S, M<H atoms), top-k sparse activation. d_model=128. Sweep sparsity k/H.
"""
import json, sys, time
import layers
from train import run, D_MODEL, N_BLOCKS

BUDGETS = [90_000, 180_000, 350_000]

def solve_H(per_module, M):
    # params = H*(d_model+1+M) + d_model*M + d_model  -> solve H
    d = D_MODEL
    return max(8, int(round((per_module - d*M - d) / (d + 1 + M))))

def v2_run(budget, M, sparsity, seed):
    per_module = budget / N_BLOCKS
    H = solve_H(per_module, M)
    layers.SPV2_H = H; layers.SPV2_M = M; layers.SPV2_SPARSITY = sparsity
    r = run("sparse_superpose_v2", budget, seed=seed)
    k_active = max(1, min(H, int(round(sparsity*H))))
    return {"budget": budget, "M": M, "H": H, "sparsity": sparsity,
            "k_active": k_active, "params": r["params"], "val_loss": r["val_loss"]}

def main():
    seed = 0
    # ---- dense baseline ----
    dense = {}
    print("=== DENSE baseline (d_model=128) ===", flush=True)
    for b in BUDGETS:
        t0=time.time(); r=run("dense", b, seed=seed); dt=time.time()-t0
        dense[b]={"params":r["params"],"val_loss":r["val_loss"],"d_ff":r["d_ff"]}
        print(f"dense b={b:>7d} d_ff={r['d_ff']:>4d} params={r['params']:>7d} "
              f"val={r['val_loss']:.4f} {dt:.0f}s", flush=True)

    # ---- v2 sweep ----
    sparsities = [1.0, 0.5, 0.25, 0.12, 0.06]
    # (budget, M) configs
    configs = [(90_000,32),
               (180_000,16),(180_000,32),(180_000,64),
               (350_000,32)]
    rows=[]
    print("\n=== sparse_superpose_v2 (matched params) ===", flush=True)
    for (b,M) in configs:
        d_ff = round(b/(2*D_MODEL*N_BLOCKS))
        for sp in sparsities:
            t0=time.time(); row=v2_run(b,M,sp,seed); dt=time.time()-t0
            row["d_ff"]=d_ff; row["H_over_dff"]=round(row["H"]/d_ff,2)
            dv=dense[b]["val_loss"]; row["dense_val"]=dv
            row["dense_params"]=dense[b]["params"]
            row["gap"]=row["val_loss"]-dv
            rows.append(row)
            print(f"b={b:>7d} M={M:>3d} H={row['H']:>5d}({row['H_over_dff']}x) "
                  f"kact={row['k_active']:>4d} sp={sp:.2f} params={row['params']:>7d} "
                  f"val={row['val_loss']:.4f} dense={dv:.4f} gap={row['gap']:+.4f} {dt:.0f}s",
                  flush=True)
    out={"dense":{str(k):v for k,v in dense.items()}, "v2":rows}
    print("\nJSON="+json.dumps(out))
    with open("results.json","w") as f: json.dump(out,f,indent=2)

if __name__=="__main__":
    main()
