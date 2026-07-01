"""CPU validation of the GPU benchmark's ALGEBRA (no Triton/CUDA needed).

Mirrors bench_gpu.py's correctness path in numpy:
  - reconstruct high plane from (codebook, idx, escape) -> must equal true high (lossless)
  - dense-approx (codebook[0] at escape positions) matvec + sparse escape correction
    -> must equal the exact BF16 matvec
bf16 bit-pattern u16 -> float32 value: (u16 << 16) viewed as float32 (exact).
If both pass, only the Triton kernel remains to validate on-device.
"""
import json
from pathlib import Path
import numpy as np

SAMPLE = Path("C:/dev/compression/research/candidates/0009-fusible-exponent-codebook/tests/artifacts/gpu_sample.npz")


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main():
    npz = np.load(SAMPLE)
    rng = np.random.default_rng(0)
    for proj in ("up", "down"):
        g = lambda k: npz[f"{proj}__{k}"]
        R, C = [int(v) for v in g("shape")]
        codebook = g("codebook").astype(np.int64)         # [15]
        idx_packed = g("idx_packed")
        low = g("low").astype(np.int64)
        escape_vals = g("escape_vals").astype(np.int64)
        raw_u16 = g("raw_u16")

        idx = np.empty(R * C, dtype=np.uint8)
        idx[0::2] = idx_packed & 0x0F
        idx[1::2] = idx_packed >> 4
        idx = idx.astype(np.int64)

        # ---- lossless: reconstruct high plane ----
        high_true = (raw_u16 >> 8).astype(np.int64)
        esc_mask = idx == 15
        high_rec = codebook[np.minimum(idx, 14)].copy()
        high_rec[esc_mask] = escape_vals             # row-major order matches
        lossless = bool(np.array_equal(high_rec, high_true))

        # ---- exact reference matvec from true bf16 weights ----
        W = bf16_to_f32(raw_u16).reshape(R, C)
        x = rng.standard_normal(C).astype(np.float32)
        y_ref = W @ x

        # ---- dense-approx + sparse correction (mirror of GPU path) ----
        high_approx = codebook[np.minimum(idx, 14)].copy()   # escape -> codebook[0]? no: min(15,14)=14
        # NB: GPU maps escape code 15 -> codebook[0]; replicate that exactly:
        high_approx[esc_mask] = codebook[0]
        u_approx = ((high_approx << 8) | low).astype(np.uint16)
        W_approx = bf16_to_f32(u_approx).reshape(R, C)
        y_dense = W_approx @ x

        pos = np.nonzero(esc_mask)[0]
        rows = pos // C; cols = pos % C
        u_true = ((escape_vals << 8) | low[pos]).astype(np.uint16)
        u_appr = ((codebook[0] << 8) | low[pos]).astype(np.uint16)
        delta = (bf16_to_f32(u_true) - bf16_to_f32(u_appr)) * x[cols]
        y_corr = np.zeros(R, dtype=np.float32)
        np.add.at(y_corr, rows, delta)
        y_fused = y_dense + y_corr

        max_err = float(np.abs(y_fused - y_ref).max())
        rel = max_err / (float(np.abs(y_ref).max()) + 1e-9)
        print(json.dumps({
            "proj": proj, "shape": [R, C],
            "lossless_high_plane": lossless,
            "n_escape": int(pos.size),
            "dense+sparse_vs_exact_max_abs_err": round(max_err, 6),
            "rel": float(f"{rel:.3e}"),
            "PASS": bool(lossless and rel < 1e-6),
        }, indent=2))


if __name__ == "__main__":
    main()
