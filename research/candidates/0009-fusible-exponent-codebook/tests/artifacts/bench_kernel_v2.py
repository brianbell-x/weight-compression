"""H100 kernel fix: make fused narrow-dequant matvec saturate memory bandwidth.

The v1 kernel capped at ~1.2 TB/s because of a per-element GATHER (wlut[nib*256+lo]) —
scattered, uncoalesced. Here we compare dequant strategies to find one that hits HBM BW:
  - k_bf16     : reads full BF16 (16 b/w)                        [bandwidth ceiling ref]
  - k_gather   : v1 approach, wlut gather (12 b/w)               [the slow one]
  - k_alu      : 16-entry codebook via ALU select, NO gather (12 b/w)
  - k_noLUT    : reads idxp+low but skips lookup (diagnostic: load-pattern ceiling)
All use int64 offsets (v1 overflowed int32 at large tiles). Tiled to a bandwidth-bound
working set. Reports effective GB/s and fused/bf16 ratio (ideal 0.75).
"""
import json, numpy as np, torch, triton
import triton.language as tl

DEV = "cuda"
SAMPLE = "/workspace/gpu_sample.npz"


@triton.jit
def k_bf16(y, w, x, R, C, BLOCK: tl.constexpr):
    r = tl.program_id(0); base = r.to(tl.int64) * C; acc = 0.0
    for t in range(tl.cdiv(C, BLOCK)):
        c = t * BLOCK + tl.arange(0, BLOCK); m = c < C
        wv = tl.load(w + base + c, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(m, wv * tl.load(x + c, mask=m, other=0.0), 0.0))
    tl.store(y + r, acc)


@triton.jit
def k_gather(y, idxp, low, wlut, x, R, C, BLOCK: tl.constexpr):
    r = tl.program_id(0); bC = r.to(tl.int64) * C; bH = r.to(tl.int64) * (C // 2); acc = 0.0
    for t in range(tl.cdiv(C, BLOCK)):
        c = t * BLOCK + tl.arange(0, BLOCK); m = c < C
        pb = tl.load(idxp + bH + (c // 2), mask=m, other=0)
        nib = tl.where((c & 1) == 1, pb >> 4, pb & 0xF).to(tl.int32)
        lo = tl.load(low + bC + c, mask=m, other=0).to(tl.int32)
        w = tl.load(wlut + nib * 256 + lo, mask=m, other=0.0)
        acc += tl.sum(tl.where(m, w * tl.load(x + c, mask=m, other=0.0), 0.0))
    tl.store(y + r, acc)


@triton.jit
def k_alu(y, idxp, low, cb, x, R, C, BLOCK: tl.constexpr):
    r = tl.program_id(0); bC = r.to(tl.int64) * C; bH = r.to(tl.int64) * (C // 2); acc = 0.0
    for t in range(tl.cdiv(C, BLOCK)):
        c = t * BLOCK + tl.arange(0, BLOCK); m = c < C
        pb = tl.load(idxp + bH + (c // 2), mask=m, other=0)
        nib = tl.where((c & 1) == 1, pb >> 4, pb & 0xF).to(tl.int32)
        hi = tl.zeros((BLOCK,), tl.int32)
        for k in tl.static_range(16):
            hi += tl.load(cb + k).to(tl.int32) * (nib == k).to(tl.int32)
        lo = tl.load(low + bC + c, mask=m, other=0).to(tl.int32)
        w = ((hi << 8) | lo).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        acc += tl.sum(tl.where(m, w * tl.load(x + c, mask=m, other=0.0), 0.0))
    tl.store(y + r, acc)


@triton.jit
def k_noLUT(y, idxp, low, x, R, C, BLOCK: tl.constexpr):
    r = tl.program_id(0); bC = r.to(tl.int64) * C; bH = r.to(tl.int64) * (C // 2); acc = 0.0
    for t in range(tl.cdiv(C, BLOCK)):
        c = t * BLOCK + tl.arange(0, BLOCK); m = c < C
        pb = tl.load(idxp + bH + (c // 2), mask=m, other=0).to(tl.int32)
        lo = tl.load(low + bC + c, mask=m, other=0).to(tl.int32)
        w = (((pb & 0xF) << 8) | lo).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        acc += tl.sum(tl.where(m, w * tl.load(x + c, mask=m, other=0.0), 0.0))
    tl.store(y + r, acc)


def cuda_time(fn, iters=80, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters


def best(fn_factory, cfgs):
    b = 1e9
    for BLK, nw in cfgs:
        try: b = min(b, cuda_time(fn_factory(BLK, nw)))
        except Exception: pass
    return b


def main(K=370):
    npz = np.load(SAMPLE); g = lambda k: npz["up__" + k]
    R0, C = [int(v) for v in g("shape")]; R = R0 * K
    codebook = g("codebook").astype(np.uint8); cb16 = np.zeros(16, np.uint8)
    cb16[:15] = codebook; cb16[15] = codebook[0]
    nn, ll = np.meshgrid(np.arange(16), np.arange(256), indexing="ij")
    # wlut[nib*256+low] = f32 value of bf16 bits (cb[nib]<<8 | low)
    ub = ((cb16[nn].astype(np.uint16) << 8) | ll.astype(np.uint16)).reshape(-1)
    wlut = torch.from_numpy((ub.astype(np.uint32) << 16).view(np.float32).astype(np.float32)).to(DEV)
    idxp = torch.from_numpy(np.tile(g("idx_packed").astype(np.uint8), K)).to(DEV)
    low = torch.from_numpy(np.tile(g("low").astype(np.uint8), K)).to(DEV)
    cb = torch.from_numpy(cb16).to(DEV)
    W = torch.from_numpy(g("raw_u16").view(np.int16)).to(DEV).view(torch.bfloat16).reshape(R0, C).repeat(K, 1)
    x = torch.randn(C, device=DEV, dtype=torch.float32)
    y = torch.empty(R, device=DEV, dtype=torch.float32); grid = (R,)
    cfgs = [(256, 1), (512, 2), (1024, 4), (2048, 8), (512, 4), (1024, 8)]

    t_bf16 = best(lambda B, w: (lambda: k_bf16[grid](y, W, x, R, C, BLOCK=B, num_warps=w)), cfgs)
    t_gather = best(lambda B, w: (lambda: k_gather[grid](y, idxp, low, wlut, x, R, C, BLOCK=B, num_warps=w)), cfgs)
    t_alu = best(lambda B, w: (lambda: k_alu[grid](y, idxp, low, cb, x, R, C, BLOCK=B, num_warps=w)), cfgs)
    t_noL = best(lambda B, w: (lambda: k_noLUT[grid](y, idxp, low, x, R, C, BLOCK=B, num_warps=w)), cfgs)

    wbytes = R * C
    def gbps(bytes_per_w, ms): return round(bytes_per_w * wbytes / (ms * 1e9), 2)
    out = {"gpu": torch.cuda.get_device_name(0), "K": K, "read_GB_bf16": round(2 * wbytes / 1e9, 2),
           "us": {"bf16_16b": round(t_bf16*1000,1), "gather_12b": round(t_gather*1000,1),
                  "alu_12b": round(t_alu*1000,1), "noLUT_12b": round(t_noL*1000,1)},
           "GBps": {"bf16": gbps(2, t_bf16), "gather": gbps(1.5, t_gather),
                    "alu": gbps(1.5, t_alu), "noLUT": gbps(1.5, t_noL)},
           "ratio_vs_bf16": {"gather": round(t_gather/t_bf16,3), "alu": round(t_alu/t_bf16,3),
                             "noLUT": round(t_noL/t_bf16,3), "ideal": 0.75}}
    print(json.dumps(out, indent=2))
    open("/workspace/kernel_v2_result.json", "w").write(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 370)
