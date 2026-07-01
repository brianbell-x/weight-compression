import layers
from train import run, D_MODEL, N_BLOCKS

def solve_H(per_module, M):
    d=D_MODEL; return max(8,int(round((per_module-d*M-d)/(d+1+M))))

def v2(budget,M,sp,seed):
    H=solve_H(budget/N_BLOCKS,M)
    layers.SPV2_H=H; layers.SPV2_M=M; layers.SPV2_SPARSITY=sp
    return run("sparse_superpose_v2",budget,seed=seed)["val_loss"]

for seed in (1,):
    d90=run("dense",90_000,seed=seed)["val_loss"]
    d180=run("dense",180_000,seed=seed)["val_loss"]
    print(f"seed{seed} dense90={d90:.4f} dense180={d180:.4f}",flush=True)
    for (b,M,sp) in [(90_000,32,1.0),(90_000,32,0.25),(180_000,64,0.25),(180_000,64,1.0)]:
        v=v2(b,M,sp,seed); base=d90 if b==90_000 else d180
        print(f"seed{seed} v2 b={b} M={M} sp={sp} val={v:.4f} gap={v-base:+.4f}",flush=True)
