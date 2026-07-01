"""H100 kernel fix v4: coalesced WIDE loads for both planes.
Root cause (v3): idxp c//2 broadcast + two narrow streams cap at 2.13 TB/s.
Fix: 2 (or 4) cols per lane; load idxp as contiguous bytes and low as contiguous
uint16 (or uint32) so both weight streams are fully coalesced. x is tiny/L2-cached."""
import json, numpy as np, torch, triton
import triton.language as tl
DEV = "cuda"; SAMPLE = "/workspace/gpu_sample.npz"


@triton.jit
def k_bf16(y, w, x, R, C, BLOCK: tl.constexpr):
    r = tl.program_id(0); base = r.to(tl.int64)*C; acc = 0.0
    for t in range(tl.cdiv(C, BLOCK)):
        c = t*BLOCK + tl.arange(0, BLOCK); m = c < C
        wv = tl.load(w+base+c, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(m, wv*tl.load(x+c, mask=m, other=0.0), 0.0))
    tl.store(y+r, acc)


@triton.jit
def k_pair(y, idxp, low16, cb, x, R, C, BLOCK: tl.constexpr):
    # 2 cols/lane: idxp uint8 coalesced + low uint16 coalesced
    r = tl.program_id(0); b = r.to(tl.int64)*(C//2); acc = 0.0; HALF = C//2
    for t in range(tl.cdiv(HALF, BLOCK)):
        j = t*BLOCK + tl.arange(0, BLOCK); mj = j < HALF
        pb = tl.load(idxp+b+j, mask=mj, other=0)
        lo16 = tl.load(low16+b+j, mask=mj, other=0)
        loe = (lo16 & 0xFF).to(tl.int32); loo = ((lo16 >> 8) & 0xFF).to(tl.int32)
        hlo = tl.load(cb+(pb & 0xF), mask=mj, other=0).to(tl.int32)
        hhi = tl.load(cb+(pb >> 4), mask=mj, other=0).to(tl.int32)
        we = ((hlo << 8) | loe).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        wo = ((hhi << 8) | loo).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        ce = 2*j; co = 2*j+1
        xe = tl.load(x+ce, mask=ce < C, other=0.0); xo = tl.load(x+co, mask=co < C, other=0.0)
        acc += tl.sum(tl.where(mj, we*xe, 0.0)) + tl.sum(tl.where(mj, wo*xo, 0.0))
    tl.store(y+r, acc)


@triton.jit
def k_quad(y, idxp16, low32, cb, x, R, C, BLOCK: tl.constexpr):
    # 4 cols/lane: idxp uint16 (4 nibbles) + low uint32 (4 low bytes)
    r = tl.program_id(0); b = r.to(tl.int64)*(C//4); acc = 0.0; Q = C//4
    for t in range(tl.cdiv(Q, BLOCK)):
        j = t*BLOCK + tl.arange(0, BLOCK); mj = j < Q
        pw = tl.load(idxp16+b+j, mask=mj, other=0).to(tl.int32)
        lw = tl.load(low32+b+j, mask=mj, other=0).to(tl.int64)
        for s in tl.static_range(4):
            nib = (pw >> (4*s)) & 0xF
            lo = ((lw >> (8*s)) & 0xFF).to(tl.int32)
            hi = tl.load(cb+nib, mask=mj, other=0).to(tl.int32)
            w = ((hi << 8) | lo).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            cc = 4*j + s
            acc += tl.sum(tl.where(cc < C, w*tl.load(x+cc, mask=cc < C, other=0.0), 0.0))
    tl.store(y+r, acc)


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
    idxp = torch.from_numpy(idxp_np).to(DEV)
    idxp16 = torch.from_numpy(idxp_np.view(np.uint16)).to(DEV)
    low16 = torch.from_numpy(low_np.view(np.uint16)).to(DEV)
    low32 = torch.from_numpy(low_np.view(np.uint32)).to(DEV)
    cb = torch.from_numpy(cb16).to(DEV)
    W = torch.from_numpy(g("raw_u16").view(np.int16)).to(DEV).view(torch.bfloat16).reshape(R0, C).repeat(K, 1)
    x = torch.randn(C, device=DEV, dtype=torch.float32)
    y = torch.empty(R, device=DEV, dtype=torch.float32); grid = (R,)
    cfgs = [(256, 2), (512, 4), (1024, 8), (2048, 8), (512, 2), (1024, 4)]
    wbytes = R*C
    tb, cb_ = best(lambda B, w: (lambda: k_bf16[grid](y, W, x, R, C, BLOCK=B, num_warps=w)), cfgs)
    tp, cp_ = best(lambda B, w: (lambda: k_pair[grid](y, idxp, low16, cb, x, R, C, BLOCK=B, num_warps=w)), cfgs)
    tq, cq_ = best(lambda B, w: (lambda: k_quad[grid](y, idxp16, low32, cb, x, R, C, BLOCK=B, num_warps=w)), cfgs)
    # correctness check for k_pair vs bf16 (single tile, small)
    yp = torch.empty(R, device=DEV, dtype=torch.float32); yb = torch.empty(R, device=DEV, dtype=torch.float32)
    k_pair[grid](yp, idxp, low16, cb, x, R, C, BLOCK=512, num_warps=4)
    k_bf16[grid](yb, W, x, R, C, BLOCK=512, num_warps=4)
    rel = ((yp-yb).abs().max()/(yb.abs().max()+1e-9)).item()
    out = {"gpu": torch.cuda.get_device_name(0), "K": K,
           "us": {"bf16": round(tb*1000, 1), "pair": round(tp*1000, 1), "quad": round(tq*1000, 1)},
           "GBps": {"bf16": round(2*wbytes/(tb*1e9), 2), "pair": round(1.5*wbytes/(tp*1e9), 2), "quad": round(1.5*wbytes/(tq*1e9), 2)},
           "ratio_vs_bf16": {"pair": round(tp/tb, 3), "quad": round(tq/tb, 3), "ideal": 0.75},
           "best_cfg": {"bf16": cb_, "pair": cp_, "quad": cq_},
           "pair_vs_bf16_rel_err": float(f"{rel:.2e}")}
    print(json.dumps(out, indent=2)); open("/workspace/kernel_v4_result.json", "w").write(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys; main(int(sys.argv[1]) if len(sys.argv) > 1 else 370)
