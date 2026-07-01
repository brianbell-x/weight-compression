"""H100 kernel fix v8: KILL the LUT gather with arithmetic decode.
Codebook high bytes are ~contiguous exp ranges (sign0:56-61, sign1:183-189), so a
sign+offset code decodes by ARITHMETIC, no gather. Two structures:
 - k_arith1: 1 col/lane, CONTIGUOUS x + low, broadcast idx, arithmetic decode (best hope)
 - k_arith4: 4 cols/lane, wide loads, strided x, arithmetic decode
BASE is a perf stand-in (exact value irrelevant to throughput; correctness verified separately)."""
import json, numpy as np, torch, triton
import triton.language as tl
DEV = "cuda"; SAMPLE = "/workspace/gpu_sample.npz"; BASE = 55


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
def k_arith1(y, idxp, low, x, R, C, BLOCK: tl.constexpr):
    # 1 col/lane: contiguous x + low, broadcast idx (c//2), arithmetic decode (no gather)
    r = tl.program_id(0); bC = r.to(tl.int64)*C; bH = r.to(tl.int64)*(C//2)
    av = tl.zeros((BLOCK,), tl.float32)
    for t in range(tl.cdiv(C, BLOCK)):
        c = t*BLOCK + tl.arange(0, BLOCK); m = c < C
        pb = tl.load(idxp+bH+(c//2), mask=m, other=0)
        code = tl.where((c & 1) == 1, pb >> 4, pb & 0xF).to(tl.int32)
        hi = ((code >> 3) << 7) | (55 + (code & 7))
        lo = tl.load(low+bC+c, mask=m, other=0).to(tl.int32)
        w = ((hi << 8) | lo).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        av += tl.where(m, w*tl.load(x+c, mask=m, other=0.0), 0.0)
    tl.store(y+r, tl.sum(av))


@triton.jit
def k_arith4(y, idx16, low32, x, R, C, BLOCK: tl.constexpr):
    r = tl.program_id(0); Q = C // 4; b = r.to(tl.int64)*Q
    av = tl.zeros((BLOCK,), tl.float32)
    for t in range(tl.cdiv(Q, BLOCK)):
        j = t*BLOCK + tl.arange(0, BLOCK); mj = j < Q
        pw = tl.load(idx16+b+j, mask=mj, other=0).to(tl.int32)
        lw = tl.load(low32+b+j, mask=mj, other=0)
        for s in tl.static_range(4):
            code = (pw >> (4*s)) & 0xF
            hi = ((code >> 3) << 7) | (55 + (code & 7))
            lo = ((lw >> (8*s)) & 0xFF).to(tl.int32)
            w = ((hi << 8) | lo).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            cc = 4*j + s
            av += tl.where(cc < C, w*tl.load(x+cc, mask=cc < C, other=0.0), 0.0)
    tl.store(y+r, tl.sum(av))


def cuda_time(fn, iters=100, warm=25):
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
        except Exception: pass
    return b, bc


def main(K=370):
    npz = np.load(SAMPLE); g = lambda k: npz["up__"+k]
    R0, C = [int(v) for v in g("shape")]; R = R0*K
    idxp = torch.from_numpy(np.tile(g("idx_packed").astype(np.uint8), K)).to(DEV)
    idx16 = torch.from_numpy(np.tile(g("idx_packed").astype(np.uint8), K).view(np.uint16)).to(DEV)
    low = torch.from_numpy(np.tile(g("low").astype(np.uint8), K)).to(DEV)
    low32 = torch.from_numpy(np.tile(g("low").astype(np.uint8), K).view(np.uint32)).to(DEV)
    W = torch.from_numpy(g("raw_u16").view(np.int16)).to(DEV).view(torch.bfloat16).reshape(R0, C).repeat(K, 1)
    x = torch.randn(C, device=DEV, dtype=torch.float32)
    y = torch.empty(R, device=DEV, dtype=torch.float32); grid = (R,)
    cfgs = [(256, 2), (512, 4), (1024, 8), (2048, 8), (128, 2), (512, 8)]
    wb = R*C
    tb, cb = best(lambda B, w: (lambda: k_bf16[grid](y, W, x, R, C, BLOCK=B, num_warps=w)), cfgs)
    t1, c1 = best(lambda B, w: (lambda: k_arith1[grid](y, idxp, low, x, R, C, BLOCK=B, num_warps=w)), cfgs)
    t4, c4 = best(lambda B, w: (lambda: k_arith4[grid](y, idx16, low32, x, R, C, BLOCK=B, num_warps=w)), cfgs)
    out = {"gpu": torch.cuda.get_device_name(0), "K": K,
           "us": {"bf16": round(tb*1000, 1), "arith1_1col": round(t1*1000, 1), "arith4_wide": round(t4*1000, 1)},
           "GBps": {"bf16": round(2*wb/(tb*1e9), 2), "arith1": round(1.5*wb/(t1*1e9), 2), "arith4": round(1.5*wb/(t4*1e9), 2)},
           "ratio_vs_bf16": {"arith1": round(t1/tb, 3), "arith4": round(t4/tb, 3), "ideal": 0.75},
           "best_cfg": {"bf16": cb, "arith1": c1, "arith4": c4}}
    print(json.dumps(out, indent=2)); open("/workspace/kernel_v8_result.json", "w").write(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys; main(int(sys.argv[1]) if len(sys.argv) > 1 else 370)
