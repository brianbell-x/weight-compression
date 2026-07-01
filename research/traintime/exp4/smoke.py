import time, torch, layers
from train import run, D_MODEL, N_BLOCKS
from layers import sparse_superpose_v2_params, SparseSuperposeV2FFN

# vocab size
from task import build_data
t = build_data()
print("d_model", D_MODEL, "vocab", t.vocab_size)

# param accounting check for v2 module
g = torch.Generator().manual_seed(0)
H, M = 400, 32
mod = SparseSuperposeV2FFN(D_MODEL, H, M, 0.12, g)
actual = mod.param_count()
formula = sparse_superpose_v2_params(D_MODEL, H, M)
print("v2 module params actual", actual, "formula", formula, "match", actual==formula)

# native compute: confirm no [d_out,H] dense W2 tensor exists as a param/buffer
names = [n for n,_ in mod.named_parameters()]
print("v2 params:", names)
assert "W1" in names and "D" in names and "S" in names
# W2 dense never stored: largest output-side tensors are D[d_out,M],S[M,H]
print("D shape", tuple(mod.D.shape), "S shape", tuple(mod.S.shape), "(no [d_out,H] W2)")
# forward sanity
x = torch.randn(2, 5, D_MODEL)
y = mod(x); print("fwd out", tuple(y.shape))

# time a dense run at the new d_model, and confirm capacity-bound direction quickly
t0=time.time(); r = run("dense", 180_000, seed=0); dt=time.time()-t0
print(f"dense 180k: total_params={r['params']} d_ff={r['d_ff']} val={r['val_loss']:.4f} wall={dt:.0f}s")
