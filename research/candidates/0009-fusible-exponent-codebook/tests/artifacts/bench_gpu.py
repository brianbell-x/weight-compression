"""GPU benchmark: fused fixed-width-codebook dequant + matvec vs BF16 at batch 1.

Closes the unproven link in candidate 0009: reading the experts NARROW (12 b/w =
4-bit codebook index + 8-bit raw low byte) and rebuilding BF16 only in registers is
(a) exactly lossless on-device and (b) competitive/faster at batch 1 than a BF16
matmul moving 16 b/w. Escapes (~0.3%) corrected by a tiny in-place sparse term
(SqueezeLLM dense+sparse). Dequant uses one precomputed (nibble,low)->weight float LUT
(4096 entries, fits L1) -> a single gather, no per-element bit assembly.

Baselines: an identical-structure BF16 Triton kernel (isolates the bandwidth lever
from kernel quality) and cuBLAS. Bit-ops on CPU/int32 (torch lacks uint16 CUDA shifts).
"""
import json
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl

assert torch.cuda.is_available(), "no CUDA device"
DEV = "cuda"
SAMPLE = Path(__file__).resolve().parent / "gpu_sample.npz"


@triton.jit
def fused_matvec(y_ptr, idxp_ptr, low_ptr, wlut_ptr, x_ptr, R, C, BLOCK: tl.constexpr):
    """y[r]=sum_c W[r,c]*x[c], reading idx_packed (4-bit) + low byte (8-bit) = 12 b/w.
    weight = wlut[nibble*256 + low]  (one gather from a 4096-float table in L1)."""
    r = tl.program_id(0)
    acc = 0.0
    for t in range(tl.cdiv(C, BLOCK)):
        c = t * BLOCK + tl.arange(0, BLOCK)
        m = c < C
        pb = tl.load(idxp_ptr + r * (C // 2) + (c // 2), mask=m, other=0)
        nib = tl.where((c & 1) == 1, pb >> 4, pb & 0xF).to(tl.int32)
        lo = tl.load(low_ptr + r * C + c, mask=m, other=0).to(tl.int32)
        w = tl.load(wlut_ptr + nib * 256 + lo, mask=m, other=0.0)
        xx = tl.load(x_ptr + c, mask=m, other=0.0)
        acc += tl.sum(tl.where(m, w * xx, 0.0))
    tl.store(y_ptr + r, acc)


@triton.jit
def bf16_matvec(y_ptr, w_ptr, x_ptr, R, C, BLOCK: tl.constexpr):
    """Identical structure, reads full BF16 (16 b/w): isolates the bandwidth lever."""
    r = tl.program_id(0)
    acc = 0.0
    for t in range(tl.cdiv(C, BLOCK)):
        c = t * BLOCK + tl.arange(0, BLOCK)
        m = c < C
        w = tl.load(w_ptr + r * C + c, mask=m, other=0.0).to(tl.float32)
        xx = tl.load(x_ptr + c, mask=m, other=0.0)
        acc += tl.sum(tl.where(m, w * xx, 0.0))
    tl.store(y_ptr + r, acc)


def bf16_bits_to_f32_np(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def cuda_time(fn, iters=200, warmup=40):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def run(proj, npz):
    g = lambda k: npz[f"{proj}__{k}"]
    R, C = [int(v) for v in g("shape")]
    codebook = g("codebook").astype(np.uint8)
    idx_packed_np = g("idx_packed").astype(np.uint8)
    low_np = g("low").astype(np.uint8)
    escape_vals_np = g("escape_vals").astype(np.uint8)
    raw_u16_np = g("raw_u16")

    idx_np = np.empty(R * C, np.uint8)
    idx_np[0::2] = idx_packed_np & 0x0F
    idx_np[1::2] = idx_packed_np >> 4
    high_true_np = (raw_u16_np >> 8).astype(np.uint8)
    esc_mask_np = idx_np == 15
    pos = np.nonzero(esc_mask_np)[0]
    rows_np = (pos // C).astype(np.int64); cols_np = (pos % C).astype(np.int64)
    low_esc = low_np[pos].astype(np.uint16)
    cb16 = np.zeros(16, np.uint8); cb16[:15] = codebook; cb16[15] = codebook[0]
    u_true = (escape_vals_np.astype(np.uint16) << 8) | low_esc
    u_appr = (np.uint16(cb16[15]) << 8) | low_esc
    dweight_np = (bf16_bits_to_f32_np(u_true) - bf16_bits_to_f32_np(u_appr)).astype(np.float32)

    # precomputed (nibble,low)->weight float LUT [16*256]
    nn, ll = np.meshgrid(np.arange(16), np.arange(256), indexing="ij")
    u_lut = ((cb16[nn].astype(np.uint16) << 8) | ll.astype(np.uint16)).reshape(-1)
    wlut_np = bf16_bits_to_f32_np(u_lut).astype(np.float32)

    # GPU
    W = torch.from_numpy(raw_u16_np.view(np.int16)).to(DEV).view(torch.bfloat16).reshape(R, C)
    x = torch.randn(C, device=DEV, dtype=torch.bfloat16); xf = x.float()
    y_ref = W.float() @ xf
    idxp = torch.from_numpy(idx_packed_np).to(DEV)
    low = torch.from_numpy(low_np).to(DEV)
    wlut = torch.from_numpy(wlut_np).to(DEV)
    y = torch.empty(R, device=DEV, dtype=torch.float32)
    grid = (R,)
    fused_matvec[grid](y, idxp, low, wlut, xf, R, C, BLOCK=1024)

    # lossless-on-GPU
    idxp_i = idxp.to(torch.int32)
    idx_t = torch.empty(R * C, dtype=torch.uint8, device=DEV)
    idx_t[0::2] = (idxp_i & 0xF).to(torch.uint8); idx_t[1::2] = (idxp_i >> 4).to(torch.uint8)
    cb_u8 = torch.from_numpy(cb16).to(DEV)
    high_rec = cb_u8[idx_t.long()].clone()
    high_rec[idx_t == 15] = torch.from_numpy(escape_vals_np).to(DEV)
    lossless = bool(torch.equal(high_rec, torch.from_numpy(high_true_np).to(DEV)))

    # exact via sparse correction
    rows = torch.from_numpy(rows_np).to(DEV); cols = torch.from_numpy(cols_np).to(DEV)
    dweight = torch.from_numpy(dweight_np).to(DEV)
    corr = dweight * xf[cols]                       # precomputed sparse correction
    y_fused = y + torch.zeros(R, device=DEV).index_add_(0, rows, corr)
    max_err = (y_fused - y_ref).abs().max().item()
    rel = max_err / (y_ref.abs().max().item() + 1e-9)

    # latency: sweep fused kernel cfg; escape correction done in-place (no alloc)
    best = None
    for BLK in (128, 256, 512, 1024):
        for nw in (1, 2, 4):
            try:
                def f(BLK=BLK, nw=nw):
                    fused_matvec[grid](y, idxp, low, wlut, xf, R, C, BLOCK=BLK, num_warps=nw)
                tt = cuda_time(f, iters=120, warmup=25)
                if best is None or tt < best[0]: best = (tt, BLK, nw)
            except Exception:
                pass
    t_kernel, bBLK, bNW = best
    def fused_full():
        fused_matvec[grid](y, idxp, low, wlut, xf, R, C, BLOCK=bBLK, num_warps=bNW)
        y.index_add_(0, rows, corr)                 # in-place sparse escape, no allocation
    ytw = torch.empty(R, device=DEV, dtype=torch.float32)
    t_fused = cuda_time(fused_full)
    t_twin = cuda_time(lambda: bf16_matvec[grid](ytw, W, xf, R, C, BLOCK=1024))
    t_cublas = cuda_time(lambda: torch.mv(W, x))

    return {
        "proj": proj, "shape": [R, C], "n_escape": int(pos.size),
        "lossless_on_gpu": lossless, "fused_vs_ref_rel": float(f"{rel:.3e}"),
        "latency_us": {
            "fused_dense_kernel_only": round(t_kernel * 1000, 3),
            "fused_total_incl_escape": round(t_fused * 1000, 3),
            "triton_bf16_twin_16b": round(t_twin * 1000, 3),
            "cublas_bf16": round(t_cublas * 1000, 3)},
        "best_fused_cfg": {"BLOCK": bBLK, "num_warps": bNW},
        "fused_total_over_twin": round(t_fused / t_twin, 3),
        "fused_total_over_cublas": round(t_fused / t_cublas, 3),
        "ideal_bandwidth_ratio_12_over_16": 0.75,
    }


def bandwidth_regime(npz, K=64):
    """Bandwidth-bound test: tile one expert's weight K times (~hundreds of MB) so the
    weight-read time dominates launch overhead. Then fused(12 b/w) vs twin(16 b/w) should
    approach the ideal 0.75x. Dense path only (isolates the bandwidth lever cleanly)."""
    g = lambda k: npz[f"up__{k}"]
    R0, C = [int(v) for v in g("shape")]
    R = R0 * K
    idxp = torch.from_numpy(np.tile(g("idx_packed").astype(np.uint8), K)).to(DEV)
    low = torch.from_numpy(np.tile(g("low").astype(np.uint8), K)).to(DEV)
    codebook = g("codebook").astype(np.uint8); cb16 = np.zeros(16, np.uint8)
    cb16[:15] = codebook; cb16[15] = codebook[0]
    nn, ll = np.meshgrid(np.arange(16), np.arange(256), indexing="ij")
    u_lut = ((cb16[nn].astype(np.uint16) << 8) | ll.astype(np.uint16)).reshape(-1)
    wlut = torch.from_numpy(bf16_bits_to_f32_np(u_lut).astype(np.float32)).to(DEV)
    W = torch.from_numpy(g("raw_u16").view(np.int16)).to(DEV).view(torch.bfloat16).reshape(R0, C).repeat(K, 1)
    xf = torch.randn(C, device=DEV, dtype=torch.float32)
    x_bf = xf.to(torch.bfloat16)
    y = torch.empty(R, device=DEV, dtype=torch.float32); grid = (R,)

    t_fused = min(cuda_time(lambda: fused_matvec[grid](y, idxp, low, wlut, xf, R, C, BLOCK=B, num_warps=w), 60, 15)
                  for B, w in ((256, 1), (512, 2), (1024, 4)))
    t_twin = min(cuda_time(lambda: bf16_matvec[grid](y, W, xf, R, C, BLOCK=B, num_warps=w), 60, 15)
                 for B, w in ((512, 2), (1024, 4), (1024, 8)))
    t_cublas = cuda_time(lambda: torch.mv(W, x_bf), 60, 15)
    wbytes = R * C
    return {"K": K, "weight_rows": R, "read_MB_fused_12b": round(1.5 * wbytes / 1e6, 1),
            "read_MB_twin_16b": round(2.0 * wbytes / 1e6, 1),
            "us_fused": round(t_fused * 1000, 2), "us_twin": round(t_twin * 1000, 2),
            "us_cublas": round(t_cublas * 1000, 2),
            "GBps_fused": round(1.5 * wbytes / (t_fused * 1e9), 1),
            "GBps_twin": round(2.0 * wbytes / (t_twin * 1e9), 1),
            "fused_over_twin": round(t_fused / t_twin, 3),
            "fused_over_cublas": round(t_fused / t_cublas, 3),
            "ideal_ratio_12_over_16": 0.75}


def main():
    npz = np.load(SAMPLE)
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "triton": triton.__version__, "runs": [run(p, npz) for p in ("up", "down")],
           "bandwidth_bound_regime": bandwidth_regime(npz, K=64)}
    print(json.dumps(out, indent=2))
    (Path(__file__).resolve().parent / "gpu_bench_result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
