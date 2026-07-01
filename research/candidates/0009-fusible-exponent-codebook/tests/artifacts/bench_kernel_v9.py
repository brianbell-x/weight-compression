"""H100 kernel fix v9: WIDE loads + INTERLEAVED layout for contiguous x + arithmetic decode.
Column permutation: lane j owns cols {j, Q+j, 2Q+j, 3Q+j} (Q=C/4). Weights pre-permuted so
one uint16(idx)+one uint32(low) load per lane covers 4 cols (few load instrs), and x[s*Q+j]
is contiguous across lanes. Arithmetic decode (no gather). This is the Marlin-style combo."""
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
def k_il(y, il_idx, il_low, x, R, C, Q, BLOCK: tl.constexpr):
    # lane j owns cols {s*Q+j}; wide coalesced loads; contiguous x per s; arithmetic decode
    r = tl.program_id(0); b = r.to(tl.int64)*Q
    av = tl.zeros((BLOCK,), tl.float32)
    for t in range(tl.cdiv(Q, BLOCK)):
        j = t*BLOCK + tl.arange(0, BLOCK); mj = j < Q
        pw = tl.load(il_idx+b+j, mask=mj, other=0).to(tl.int32)
        lw = tl.load(il_low+b+j, mask=mj, other=0)
        for s in tl.static_range(4):
            code = (pw >> (4*s)) & 0xF
            hi = ((code >> 3) << 7) | (55 + (code & 7))
            lo = ((lw >> (8*s)) & 0xFF).to(tl.int32)
            w = ((hi << 8) | lo).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            xs = tl.load(x + s*Q + j, mask=mj, other=0.0)   # contiguous across lanes
            av += tl.where(mj, w*xs, 0.0)
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
    R0, C = [int(v) for v in g("shape")]; R = R0*K; Q = C // 4
    # per-row planes for the real up-expert (R0 rows), then tile K
    low0 = g("low").astype(np.uint8).reshape(R0, C)
    idxp0 = g("idx_packed").astype(np.uint8).reshape(R0, C // 2)
    codes0 = np.empty((R0, C), np.uint8)
    codes0[:, 0::2] = idxp0 & 0x0F
    codes0[:, 1::2] = idxp0 >> 4
    # interleave: il[r, 4j+s] = plane[r, s*Q+j]  ==  plane[r].reshape(4,Q).T.reshape(C)
    il_low0 = low0.reshape(R0, 4, Q).transpose(0, 2, 1).reshape(R0, C)
    il_codes0 = codes0.reshape(R0, 4, Q).transpose(0, 2, 1).reshape(R0, C)  # [r, 4j+s]
    il_idx0 = (il_codes0[:, 0::4].astype(np.uint16)
               | (il_codes0[:, 1::4].astype(np.uint16) << 4)
               | (il_codes0[:, 2::4].astype(np.uint16) << 8)
               | (il_codes0[:, 3::4].astype(np.uint16) << 12))   # [R0, Q] uint16
    il_low = torch.from_numpy(np.tile(il_low0.reshape(R0, C).view(np.uint32), (K, 1)).reshape(-1)).to(DEV)  # [R*Q] u32
    il_idx = torch.from_numpy(np.tile(il_idx0, (K, 1)).reshape(-1)).to(DEV)                                 # [R*Q] u16
    W = torch.from_numpy(g("raw_u16").view(np.int16)).to(DEV).view(torch.bfloat16).reshape(R0, C).repeat(K, 1)
    x = torch.randn(C, device=DEV, dtype=torch.float32)
    y = torch.empty(R, device=DEV, dtype=torch.float32); grid = (R,)
    cfgs = [(128, 2), (256, 4), (512, 8), (256, 2), (512, 4), (1024, 8)]
    wb = R*C
    tb, cb = best(lambda B, w: (lambda: k_bf16[grid](y, W, x, R, C, BLOCK=B, num_warps=w)), cfgs)
    ti, ci = best(lambda B, w: (lambda: k_il[grid](y, il_idx, il_low, x, R, C, Q, BLOCK=B, num_warps=w)), cfgs)
    out = {"gpu": torch.cuda.get_device_name(0), "K": K,
           "us": {"bf16": round(tb*1000, 1), "il_4col": round(ti*1000, 1)},
           "GBps": {"bf16": round(2*wb/(tb*1e9), 2), "il": round(1.5*wb/(ti*1e9), 2)},
           "ratio_vs_bf16": {"il": round(ti/tb, 3), "ideal": 0.75},
           "best_cfg": {"bf16": cb, "il": ci}}
    print(json.dumps(out, indent=2)); open("/workspace/kernel_v9_result.json", "w").write(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys; main(int(sys.argv[1]) if len(sys.argv) > 1 else 370)
