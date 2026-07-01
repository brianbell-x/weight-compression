import time, torch
import layers
from train import run
from layers import sparse_superpose_params
from sweep_sparse_superpose import solve_H

# 1) native-compute check: forward must never allocate an (H, d_in) tensor.
m = layers.SparseSuperposeFFN(64, 1000, 16, 0.12, None, torch.Generator().manual_seed(0))
H, d_in = m.H, m.d_in
import torch.utils.hooks  # noqa
_seen = []
_orig_mm = torch.matmul
# Confirm equivalence to explicit (materialized) W1 path, while forward itself
# only ever multiplies x@A^T (->K) then z@C^T (->H): largest weight tensor is C(H,K).
W1_explicit = (m.C @ m.A)  # (H,d_in) -- built ONLY here for the equivalence check
x = torch.randn(2, 5, 64)
pre_native = (x @ m.A.t()) @ m.C.t()
pre_mat = x @ W1_explicit.t()
assert torch.allclose(pre_native, pre_mat, atol=1e-4), "native != materialized!"
largest = max(p.numel() for p in m.parameters())
assert largest < H * d_in or m.w2_rank is None, "stored a full (H,d_in) param!"
print("native==materialized OK; no stored (H,d_in) input weight; largest param",
      largest, "vs H*d_in", H * d_in)
x = torch.randn(2, 5, 64)
y = m(x)
print("native-compute OK  out", tuple(y.shape), "k_active", m.k_active, "params", m.param_count())

pre = (x @ m.A.t()) @ m.C.t() + m.b1
h = torch.nn.functional.gelu(pre)
thresh = torch.topk(h, m.k_active, dim=-1).values[..., -1:]
hs = torch.where(h >= thresh, h, torch.zeros_like(h))
nz = (hs != 0).sum(-1).float().mean().item()
print("avg nonzero hidden per token:", nz, "of", m.H)

# 2) param-budget solver hits target
for b in [90000, 180000]:
    for K, w2r in [(16, None), (16, 8), (8, 4)]:
        H = solve_H(b / 2, K, w2r)
        p = sparse_superpose_params(64, H, K, w2r)
        print(f"b={b} K={K} w2r={w2r} H={H} per_module={p} total~{p*2} target={b}")

# 3) dense reproduces + time a sparse run
t0 = time.time(); rd = run("dense", 90000, seed=0)
print("dense 90k val", round(rd['val_loss'], 4), "params", rd['params'], f"{time.time()-t0:.0f}s")
layers.SP_H = 1000; layers.SP_K = 16; layers.SP_W2_RANK = 8; layers.SP_SPARSITY = 0.12
t0 = time.time(); rs = run("sparse_superpose", 90000, seed=0)
print("sparse 90k val", round(rs['val_loss'], 4), "params", rs['params'],
      "swap", rs['swappable_params'], f"{time.time()-t0:.0f}s")
