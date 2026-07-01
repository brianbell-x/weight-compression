"""H100 kernel fix v6: wide coalesced loads (uint16 idx + uint32 low, 4 cols/lane),
vector accumulate + SINGLE reduce per tile (v5's bug: reduced inside the unroll)."""
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
def k_v6(y, idx16, low32, cb, x, R, C, BLOCK: tl.constexpr, LUT: tl.constexpr):
    r = tl.program_id(0); Q = C // 4; b = r.to(tl.int64)*Q
    av = tl.zeros((BLOCK,), tl.float32)
    for t in range(tl.cdiv(Q, BLOCK)):
        j = t*BLOCK + tl.arange(0, BLOCK); mj = j < Q
        pw = tl.load(idx16+b+j, mask=mj, other=0).to(tl.int32)
        lw = tl.load(low32+b+j, mask=mj, other=0)
        for s in tl.static_range(4):
            nib = (pw >> (4*s)) & 0xF
            lo = ((lw >> (8*s)) & 0xFF).to(tl.int32)
            if LUT:
                hi = tl.load(cb+nib, mask=mj, other=0).to(tl.int32)
            else:
                hi = nib
            w = ((hi << 8) | lo).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            cc = 4*j + s
            av += tl.where(cc < C, w*tl.load(x+cc, mask=cc < C, other=0.0), 0.0)
    tl.store(y+r, tl.sum(av))


def cuda_time(fn, iters=80, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/iters


def best(mk, cfgs):
    b = 1e9; bc = None
    for B, w in cfgs:
        try:
            t = cuda_time(mk(B, w))
            if t < b: b, bc = t, (B, w)
        except Exception:
            pass
    return b, bc


def main(K=370):
    npz = np.load(SAMPLE); g = lambda k: npz["up__"+k]
    R0, C = [int(v) for v in g("shape")]; R = R0*K
    cb16 = np.zeros(16, np.uint8); cbk = g("codebook").astype(np.uint8); cb16[:15] = cbk; cb16[15] = cbk[0]
    idxp_np = np.tile(g("idx_packed").astype(np.uint8), K)
    low_np = np.tile(g("low").astype(np.uint8), K)
    idx16 = torch.from_numpy(idxp_np.view(np.uint16)).to(DEV)
    low32 = torch.from_numpy(low_np.view(np.uint32)).to(DEV)
    cb = torch.from_numpy(cb16).to(DEV)
    W = torch.from_numpy(g("raw_u16").view(np.int16)).to(DEV).view(torch.bfloat16).reshape(R0, C).repeat(K, 1)
    x = torch.randn(C, device=DEV, dtype=torch.float32)
    y = torch.empty(R, device=DEV, dtype=torch.float32); grid = (R,)
    cfgs = [(128, 2), (256, 4), (512, 8), (256, 2), (512, 4), (1024, 8), (128, 4)]
    wbytes = R*C
    tb, cbb = best(lambda B, w: (lambda: k_bf16[grid](y, W, x, R, C, BLOCK=B, num_warps=w)), cfgs)
    tnl, cnl = best(lambda B, w: (lambda: k_v6[grid](y, idx16, low32, cb, x, R, C, BLOCK=B, num_warps=w, LUT=False)), cfgs)
    tv, cv = best(lambda B, w: (lambda: k_v6[grid](y, idx16, low32, cb, x, R, C, BLOCK=B, num_warps=w, LUT=True)), cfgs)
    yo = torch.empty(R, device=DEV, dtype=torch.float32); yb = torch.empty(R, device=DEV, dtype=torch.float32)
    k_v6[grid](yo, idx16, low32, cb, x, R, C, BLOCK=256, num_warps=4, LUT=True)
    k_bf16[grid](yb, W, x, R, C, BLOCK=256, num_warps=4)
    rel = ((yo-yb).abs().max()/(yb.abs().max()+1e-9)).item()
    out = {"gpu": torch.cuda.get_device_name(0), "K": K,
           "us": {"bf16": round(tb*1000, 1), "v6_noLUT": round(tnl*1000, 1), "v6_LUT": round(tv*1000, 1)},
           "GBps": {"bf16": round(2*wbytes/(tb*1e9), 2), "v6_noLUT": round(1.5*wbytes/(tnl*1e9), 2), "v6_LUT": round(1.5*wbytes/(tv*1e9), 2)},
           "ratio_vs_bf16": {"v6_noLUT": round(tnl/tb, 3), "v6_LUT": round(tv/tb, 3), "ideal": 0.75},
           "best_cfg": {"bf16": cbb, "v6_noLUT": cnl, "v6_LUT": cv}, "v6_rel_err": float(f"{rel:.2e}")}
    print(json.dumps(out, indent=2)); open("/workspace/kernel_v6_result.json", "w").write(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys; main(int(sys.argv[1]) if len(sys.argv) > 1 else 370)
