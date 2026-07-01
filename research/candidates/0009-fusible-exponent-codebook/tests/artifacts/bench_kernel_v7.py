"""H100 diagnostic v7: WHERE is the ~2.2 TB/s fused ceiling? Isolate each cost.
 - k_bf16        : bf16 GEMV (single stream + x)                 -> peak ref
 - k_lowonly     : read low(uint32) only, sum (no idx, no x)     -> single narrow plane ceiling
 - k_dq_nolut    : read idx16+low32, assemble, sum (NO x, NO LUT)-> two-stream+unpack ceiling
 - k_dq_lut      : + 16-entry LUT gather (NO x)                  -> LUT cost
 - k_full        : + x multiply (strided)                       -> x cost
If k_dq_nolut >> 2.2, x is the killer (fix layout). If it caps ~2.2, restructure/CUDA needed."""
import json, numpy as np, torch, triton
import triton.language as tl
DEV = "cuda"; SAMPLE = "/workspace/gpu_sample.npz"


@triton.jit
def k_bf16(y, w, x, R, C, BLOCK: tl.constexpr):
    r = tl.program_id(0); base = r.to(tl.int64)*C
    av = tl.zeros((BLOCK,), tl.float32)
    for t in range(tl.cdiv(C, BLOCK)):
        c = t*BLOCK + tl.arange(0, BLOCK); m = c < C
        wv = tl.load(w+base+c, mask=m, other=0.0).to(tl.float32)
        av += tl.where(m, wv*tl.load(x+c, mask=m, other=0.0), 0.0)
    tl.store(y+r, tl.sum(av))


@triton.jit
def k_lowonly(y, low32, R, C, BLOCK: tl.constexpr):
    r = tl.program_id(0); Q = C // 4; b = r.to(tl.int64)*Q
    av = tl.zeros((BLOCK,), tl.float32)
    for t in range(tl.cdiv(Q, BLOCK)):
        j = t*BLOCK + tl.arange(0, BLOCK); mj = j < Q
        lw = tl.load(low32+b+j, mask=mj, other=0)
        for s in tl.static_range(4):
            av += tl.where(mj, ((lw >> (8*s)) & 0xFF).to(tl.float32), 0.0)
    tl.store(y+r, tl.sum(av))


@triton.jit
def k_dq(y, idx16, low32, cb, x, R, C, BLOCK: tl.constexpr, LUT: tl.constexpr, USEX: tl.constexpr):
    r = tl.program_id(0); Q = C // 4; b = r.to(tl.int64)*Q
    av = tl.zeros((BLOCK,), tl.float32)
    for t in range(tl.cdiv(Q, BLOCK)):
        j = t*BLOCK + tl.arange(0, BLOCK); mj = j < Q
        pw = tl.load(idx16+b+j, mask=mj, other=0).to(tl.int32)
        lw = tl.load(low32+b+j, mask=mj, other=0)
        for s in tl.static_range(4):
            nib = (pw >> (4*s)) & 0xF
            lo = ((lw >> (8*s)) & 0xFF).to(tl.int32)
            hi = tl.load(cb+nib, mask=mj, other=0).to(tl.int32) if LUT else nib
            w = ((hi << 8) | lo).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            if USEX:
                cc = 4*j + s
                av += tl.where(cc < C, w*tl.load(x+cc, mask=cc < C, other=0.0), 0.0)
            else:
                av += tl.where(mj, w, 0.0)
    tl.store(y+r, tl.sum(av))


def cuda_time(fn, iters=80, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/iters


def best(mk, cfgs):
    b = 1e9
    for B, w in cfgs:
        try: b = min(b, cuda_time(mk(B, w)))
        except Exception: pass
    return b


def main(K=370):
    npz = np.load(SAMPLE); g = lambda k: npz["up__"+k]
    R0, C = [int(v) for v in g("shape")]; R = R0*K
    cb16 = np.zeros(16, np.uint8); cbk = g("codebook").astype(np.uint8); cb16[:15] = cbk; cb16[15] = cbk[0]
    idx16 = torch.from_numpy(np.tile(g("idx_packed").astype(np.uint8), K).view(np.uint16)).to(DEV)
    low32 = torch.from_numpy(np.tile(g("low").astype(np.uint8), K).view(np.uint32)).to(DEV)
    cb = torch.from_numpy(cb16).to(DEV)
    W = torch.from_numpy(g("raw_u16").view(np.int16)).to(DEV).view(torch.bfloat16).reshape(R0, C).repeat(K, 1)
    x = torch.randn(C, device=DEV, dtype=torch.float32)
    y = torch.empty(R, device=DEV, dtype=torch.float32); grid = (R,)
    cfgs = [(128, 2), (256, 4), (512, 8), (256, 2), (512, 4), (1024, 8)]
    wb = R*C
    res = {
        "bf16_2.0Bw": (best(lambda B, w: (lambda: k_bf16[grid](y, W, x, R, C, BLOCK=B, num_warps=w)), cfgs), 2.0),
        "lowonly_1.0Bw": (best(lambda B, w: (lambda: k_lowonly[grid](y, low32, R, C, BLOCK=B, num_warps=w)), cfgs), 1.0),
        "dq_nolut_nox_1.5Bw": (best(lambda B, w: (lambda: k_dq[grid](y, idx16, low32, cb, x, R, C, BLOCK=B, num_warps=w, LUT=False, USEX=False)), cfgs), 1.5),
        "dq_lut_nox_1.5Bw": (best(lambda B, w: (lambda: k_dq[grid](y, idx16, low32, cb, x, R, C, BLOCK=B, num_warps=w, LUT=True, USEX=False)), cfgs), 1.5),
        "dq_lut_x_1.5Bw": (best(lambda B, w: (lambda: k_dq[grid](y, idx16, low32, cb, x, R, C, BLOCK=B, num_warps=w, LUT=True, USEX=True)), cfgs), 1.5),
    }
    out = {"gpu": torch.cuda.get_device_name(0), "K": K,
           "us": {k: round(v[0]*1000, 1) for k, v in res.items()},
           "GBps": {k: round(v[1]*wb/(v[0]*1e9), 2) for k, v in res.items()}}
    print(json.dumps(out, indent=2)); open("/workspace/kernel_v7_result.json", "w").write(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys; main(int(sys.argv[1]) if len(sys.argv) > 1 else 370)
