import numpy as np, torch
from safetensors import safe_open

PATH = r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot\model-00001-of-00013.safetensors"
G = 128
rng = np.random.default_rng(0)
N_EXP = 24
N_IN = 8

def quant_group(W, group=G, bits=4):
    # W: [out, in], per-group along input dim, symmetric
    out, inn = W.shape
    qmax = 2**(bits-1)-1
    Wd = np.empty_like(W)
    ng = inn // group
    for g in range(ng):
        sl = slice(g*group,(g+1)*group)
        blk = W[:, sl]
        scale = np.abs(blk).max(axis=1, keepdims=True) / qmax
        scale[scale==0] = 1e-12
        q = np.round(blk/scale).clip(-qmax,qmax)
        Wd[:, sl] = q*scale
    return Wd

def qcodes(W, group=G, bits=4):
    out, inn = W.shape
    qmax = 2**(bits-1)-1
    ng = inn//group
    codes = np.empty_like(W)
    for g in range(ng):
        sl=slice(g*group,(g+1)*group)
        blk=W[:,sl]
        scale=np.abs(blk).max(axis=1,keepdims=True)/qmax
        scale[scale==0]=1e-12
        codes[:,sl]=np.round(blk/scale).clip(-qmax,qmax)
    return codes

def outerr(W, Wp, X):
    A = X@W; B = X@Wp
    rel = np.linalg.norm(A-B)/np.linalg.norm(A)
    cos = (A*B).sum()/(np.linalg.norm(A)*np.linalg.norm(B))
    return rel, cos

f = safe_open(PATH, framework="pt")
res_b4 = []; res_b4r = []; res_i8 = []; res_i8r = []; res_direct4=[]
ent_list=[]; sparse_list=[]; effbits_list=[]
for n in range(N_EXP):
    W = f.get_tensor(f"backbone.layers.1.mixer.experts.{n}.up_proj.weight").float().numpy()
    # W shape [1856,2688]; matmul X[in=2688] @ W^T? up_proj: out=1856,in=2688
    # XW: treat as X[8,2688] @ W^T[2688,1856]
    Wt = W.T  # [2688,1856] so X@Wt valid
    X = rng.standard_normal((N_IN, 2688)).astype(np.float32)
    X /= np.linalg.norm(X,axis=1,keepdims=True)

    # quantize along input dim of the matmul = rows of Wt = columns of W...
    # per-group group=128 along input dim (2688). Work on W with input dim = axis1 (2688). Good.
    W4 = quant_group(W,bits=4)
    r1,_ = outerr(Wt, W4.T, X)
    res_direct4.append(r1)

    # base+residual int4
    R = W - W4
    R4 = quant_group(R, bits=4)
    Wbr = W4 + R4
    r2,_ = outerr(Wt, Wbr.T, X)
    res_b4r.append(r2)

    # direct int8
    W8 = quant_group(W, bits=8)
    r3,_ = outerr(Wt, W8.T, X)
    res_i8.append(r3)

    # int8 + residual (above-int8)
    R8 = W - W8
    R8q = quant_group(R8, bits=8)
    W8r = W8 + R8q
    r4,_ = outerr(Wt, W8r.T, X)
    res_i8r.append(r4)

    # residual compressibility: residual codes from base+int4 residual (R4 codes)
    codes = qcodes(R, bits=4).astype(np.int64).ravel()
    vals, cnts = np.unique(codes, return_counts=True)
    p = cnts/cnts.sum()
    ent = -(p*np.log2(p)).sum()
    ent_list.append(ent)
    sparse = (codes==0).mean()
    sparse_list.append(sparse)

print("N experts:", N_EXP)
print("INT4 base only err  mean %.4f%%" % (100*np.mean(res_direct4)))
print("INT4 base+INT4 resid err mean %.4f%%" % (100*np.mean(res_b4r)))
print("direct INT8 err mean %.4f%%" % (100*np.mean(res_i8)))
print("INT8+INT8 resid err mean %.6f%%" % (100*np.mean(res_i8r)))
print("residual(R4) entropy bits/code mean %.4f" % np.mean(ent_list))
print("residual zero-fraction mean %.4f" % np.mean(sparse_list))
