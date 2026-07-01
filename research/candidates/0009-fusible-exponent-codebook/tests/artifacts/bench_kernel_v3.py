"""H100 kernel fix v3: isolate load vs assembly, test coalesced-idx layouts + 16-entry LUT.
Goal: a fused 12-b/w matvec that hits ~HBM bandwidth (ratio ~0.75 vs bf16)."""
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
def k_loadonly(y, idxp, low, x, R, C, BLOCK: tl.constexpr):
    # loads idxp(c//2)+low, value = low only (no assembly) -> pure load throughput probe
    r = tl.program_id(0); bC = r.to(tl.int64)*C; bH = r.to(tl.int64)*(C//2); acc = 0.0
    for t in range(tl.cdiv(C, BLOCK)):
        c = t*BLOCK + tl.arange(0, BLOCK); m = c < C
        pb = tl.load(idxp+bH+(c//2), mask=m, other=0).to(tl.float32)
        lo = tl.load(low+bC+c, mask=m, other=0).to(tl.float32)
        acc += tl.sum(tl.where(m, (lo+pb)*tl.load(x+c, mask=m, other=0.0), 0.0))
    tl.store(y+r, acc)


@triton.jit
def k_g16(y, idxp, low, cb, x, R, C, BLOCK: tl.constexpr):
    # 16-entry cb gather (cheap: 16 bytes) + assemble, idxp via c//2
    r = tl.program_id(0); bC = r.to(tl.int64)*C; bH = r.to(tl.int64)*(C//2); acc = 0.0
    for t in range(tl.cdiv(C, BLOCK)):
        c = t*BLOCK + tl.arange(0, BLOCK); m = c < C
        pb = tl.load(idxp+bH+(c//2), mask=m, other=0)
        nib = tl.where((c & 1) == 1, pb >> 4, pb & 0xF).to(tl.int32)
        hi = tl.load(cb+nib, mask=m, other=0).to(tl.int32)
        lo = tl.load(low+bC+c, mask=m, other=0).to(tl.int32)
        w = ((hi << 8) | lo).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        acc += tl.sum(tl.where(m, w*tl.load(x+c, mask=m, other=0.0), 0.0))
    tl.store(y+r, acc)


@triton.jit
def k_2col(y, idxp, low, cb, x, R, C, BLOCK: tl.constexpr):
    # coalesced idxp (lane j -> byte j, 2 cols/lane); low stride-2
    r = tl.program_id(0); bC = r.to(tl.int64)*C; bH = r.to(tl.int64)*(C//2); acc = 0.0
    HALF = C // 2
    for t in range(tl.cdiv(HALF, BLOCK)):
        j = t*BLOCK + tl.arange(0, BLOCK); mj = j < HALF
        pb = tl.load(idxp+bH+j, mask=mj, other=0)
        hlo = tl.load(cb+(pb & 0xF), mask=mj, other=0).to(tl.int32)
        hhi = tl.load(cb+(pb >> 4), mask=mj, other=0).to(tl.int32)
        ce = 2*j; co = 2*j+1; me = ce < C; mo = co < C
        loe = tl.load(low+bC+ce, mask=me, other=0).to(tl.int32)
        loo = tl.load(low+bC+co, mask=mo, other=0).to(tl.int32)
        we = ((hlo << 8) | loe).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        wo = ((hhi << 8) | loo).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        acc += tl.sum(tl.where(me, we*tl.load(x+ce, mask=me, other=0.0), 0.0))
        acc += tl.sum(tl.where(mo, wo*tl.load(x+co, mask=mo, other=0.0), 0.0))
    tl.store(y+r, acc)


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
    idxp = torch.from_numpy(np.tile(g("idx_packed").astype(np.uint8), K)).to(DEV)
    low = torch.from_numpy(np.tile(g("low").astype(np.uint8), K)).to(DEV)
    cb = torch.from_numpy(cb16).to(DEV)
    W = torch.from_numpy(g("raw_u16").view(np.int16)).to(DEV).view(torch.bfloat16).reshape(R0, C).repeat(K, 1)
    x = torch.randn(C, device=DEV, dtype=torch.float32)
    y = torch.empty(R, device=DEV, dtype=torch.float32); grid = (R,)
    cfgs = [(256, 1), (512, 2), (1024, 4), (2048, 8), (1024, 8), (2048, 16)]
    wbytes = R*C
    ts = {
        "bf16_16b": best(lambda B, w: (lambda: k_bf16[grid](y, W, x, R, C, BLOCK=B, num_warps=w)), cfgs),
        "loadonly_12b": best(lambda B, w: (lambda: k_loadonly[grid](y, idxp, low, x, R, C, BLOCK=B, num_warps=w)), cfgs),
        "g16_12b": best(lambda B, w: (lambda: k_g16[grid](y, idxp, low, cb, x, R, C, BLOCK=B, num_warps=w)), cfgs),
        "2col_12b": best(lambda B, w: (lambda: k_2col[grid](y, idxp, low, cb, x, R, C, BLOCK=B, num_warps=w)), cfgs),
    }
    bpw = {"bf16_16b": 2.0, "loadonly_12b": 1.5, "g16_12b": 1.5, "2col_12b": 1.5}
    out = {"gpu": torch.cuda.get_device_name(0), "K": K,
           "us": {k: round(v*1000, 1) for k, v in ts.items()},
           "GBps": {k: round(bpw[k]*wbytes/(v*1e9), 2) for k, v in ts.items()},
           "ratio_vs_bf16": {k: round(v/ts["bf16_16b"], 3) for k, v in ts.items()},
           "ideal_fused_ratio": 0.75}
    print(json.dumps(out, indent=2)); open("/workspace/kernel_v3_result.json", "w").write(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys; main(int(sys.argv[1]) if len(sys.argv) > 1 else 370)
