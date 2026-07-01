"""Powered Stage-1 GPTQ eval for candidate 0005's live INT4 thread.

This consumes the document-disjoint activation caches produced by
`capture_corpus_activations.py` and asks the narrow question that remained open after
0008's powered end-to-end eval:

    Does INT4 GPTQ, with enough real calibration tokens to form a useful Hessian,
    beat plain INT4 RTN on held-out routed-expert inputs by enough to plausibly close
    the gap to INT8-class behavior?

Scope and honesty notes
-----------------------
* This is still a Stage-1 matmul-fidelity probe, not a capability verdict. Survivors
  must still go through the Stage-2 streamed full-forward harness.
* Calibration and held-out X come from disjoint documents when the cache was produced.
* Defaults run layer-1 up_proj only, because layer-1 X is exactly the gate/up_proj
  input cached by the cheap partial forward. `--proj down` is available, but each
  expert's down_proj Hessian is built from that expert's dense full-precision
  intermediate `relu2(X @ up_proj.T)`, not from routed-token-only activations.

Run examples
------------
Smoke on the current cache:
  uv run python powered_gptq_eval.py --smoke

Larger up-proj run after capturing more tokens:
  uv run python powered_gptq_eval.py --max-cal 30000 --max-heldout 2500 \
      --n-experts 8 --damp 0.01,0.03,0.1,0.3,1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import psutil
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
sys.path.insert(0, ART)
import stage1_probe as S1  # noqa: E402

SNAP = r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot"
SHARD1 = os.path.join(SNAP, "model-00001-of-00013.safetensors")
DEFAULT_CACHE = os.path.join(HERE, "activations_corpus")

GROUP_UP = 128   # up_proj in-axis 2688 is divisible by 128
GROUP_DN = 116   # down_proj in-axis 1856 is divisible by 116

PROC = psutil.Process()


def rss_gb() -> float:
    return PROC.memory_info().rss / 1024**3


@dataclass(frozen=True)
class QuantSpec:
    bits: int
    group_size: int


# =====================================================================================
# Cache loading
# =====================================================================================
def _load_npy(path: str, max_rows: int | None) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Activation cache file is missing: {path}. Run "
            "`uv run python capture_corpus_activations.py --smoke` first for a small "
            "cache, or without --smoke for the powered cache."
        )
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D activation array at {path}, got shape {arr.shape}.")
    n = arr.shape[0] if max_rows is None or max_rows <= 0 else min(max_rows, arr.shape[0])
    # Copy into a real contiguous tensor: the GPTQ path repeatedly slices and matmuls it.
    return torch.from_numpy(np.array(arr[:n], dtype=np.float32, copy=True))


def load_activation_cache(cache_dir: str, max_cal: int | None, max_heldout: int | None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    X_cal = _load_npy(os.path.join(cache_dir, "X_cal.npy"), max_cal)
    X_held = _load_npy(os.path.join(cache_dir, "X_heldout.npy"), max_heldout)
    meta_path = os.path.join(cache_dir, "capture_meta.json")
    meta = json.load(open(meta_path, "r", encoding="utf-8")) if os.path.exists(meta_path) else {}
    if X_cal.shape[1] != X_held.shape[1]:
        raise ValueError(
            f"Calibration and held-out hidden sizes differ: cal={tuple(X_cal.shape)} "
            f"heldout={tuple(X_held.shape)}. Regenerate the cache."
        )
    print(
        f"[cache] cal={tuple(X_cal.shape)} heldout={tuple(X_held.shape)} "
        f"cache_dir={cache_dir} RSS={rss_gb():.2f}GB",
        flush=True,
    )
    if meta:
        print(
            "[cache-meta] original_tokens="
            f"cal:{meta.get('n_cal_tokens')} heldout:{meta.get('n_heldout_tokens')} "
            f"energy_max_over_mean:{meta.get('channel_energy_max_over_mean_cal')} "
            f"source_peak_rss:{meta.get('peak_rss_gb')}GB",
            flush=True,
        )
    return X_cal, X_held, meta


# =====================================================================================
# Quantization helpers
# =====================================================================================
def find_scale(Wblock: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-row max-abs scale for Wblock [out, group_size]."""
    qmax = (1 << (bits - 1)) - 1
    max_abs = Wblock.abs().amax(dim=1, keepdim=True)
    return (max_abs / qmax).clamp_min(1e-12)


def quant(w: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


def rtn_quantize(W: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Data-free symmetric RTN. W is [out, in], grouped along the in dimension."""
    out, cols = W.shape
    if cols % spec.group_size != 0:
        raise ValueError(
            f"RTN group_size={spec.group_size} must divide in_features={cols}; "
            "choose --group-up 128 for up_proj or --group-down 116 for down_proj."
        )
    Q = torch.empty_like(W)
    for g0 in range(0, cols, spec.group_size):
        g1 = g0 + spec.group_size
        blk = W[:, g0:g1]
        s = find_scale(blk, spec.bits)
        Q[:, g0:g1] = quant(blk, s, spec.bits)
    return Q


def effective_bits(out_rows: int, cols: int, spec: QuantSpec) -> float:
    groups_per_row = (cols + spec.group_size - 1) // spec.group_size
    num_weights = out_rows * cols
    n_scales = out_rows * groups_per_row
    return S1.bits_per_weight(
        num_weights,
        payload_bits=spec.bits * num_weights,
        scale_bits=16 * n_scales,
    )


# =====================================================================================
# Hessian / GPTQ
# =====================================================================================
def compute_hessian(X: torch.Tensor, chunk_rows: int = 8192) -> torch.Tensor:
    """Return H = X^T X in float32, chunked to keep memory predictable."""
    if X.ndim != 2:
        raise ValueError(f"Expected X [tokens, hidden], got {tuple(X.shape)}")
    d = X.shape[1]
    H = torch.zeros((d, d), dtype=torch.float32)
    t0 = time.time()
    for s in range(0, X.shape[0], chunk_rows):
        xb = X[s : s + chunk_rows].float()
        H.addmm_(xb.t(), xb)
    print(
        f"[hessian] X={tuple(X.shape)} H={tuple(H.shape)} "
        f"diag_mean={torch.diag(H).mean().item():.6g} "
        f"seconds={time.time()-t0:.1f} RSS={rss_gb():.2f}GB",
        flush=True,
    )
    return H


def prepare_hinv(H: torch.Tensor, percdamp: float) -> tuple[torch.Tensor, float]:
    """Return upper Cholesky factor of H^{-1}, adding damp until Cholesky is stable."""
    if percdamp < 0:
        raise ValueError(f"percdamp must be non-negative, got {percdamp}")
    diag_mean = torch.diag(H).mean().clamp_min(1e-12)
    base = float(percdamp * diag_mean)
    idx = torch.arange(H.shape[0])
    last_error = None
    multipliers = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    for mult in multipliers:
        Hd = H.clone()
        actual = base * mult
        if actual == 0.0:
            actual = float(1e-6 * diag_mean)
        Hd[idx, idx] += actual
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv_full = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv_full, upper=True)
            return Hinv, actual
        except RuntimeError as exc:
            last_error = str(exc).splitlines()[0]
    raise RuntimeError(
        f"Could not factor damped Hessian. percdamp={percdamp}, base_damp={base}, "
        f"last_error={last_error}. Try a larger --damp value or more calibration tokens."
    )


def gptq_quantize_with_hinv(
    W: torch.Tensor,
    Hinv: torch.Tensor,
    spec: QuantSpec,
    blocksize: int = 128,
) -> torch.Tensor:
    """GPTQ/OBQ sequential error-feedback quantization.

    W is [out, in]. Hinv is the upper Cholesky factor of the damped inverse Hessian.
    Groups are along the input-column dimension, with one fp16 scale per (row, group).
    """
    W = W.clone().float()
    out, cols = W.shape
    if Hinv.shape != (cols, cols):
        raise ValueError(f"Hinv shape {tuple(Hinv.shape)} does not match in_features={cols}.")
    if cols % spec.group_size != 0:
        raise ValueError(f"group_size={spec.group_size} must divide in_features={cols}.")
    if blocksize < spec.group_size or blocksize % spec.group_size != 0:
        raise ValueError(
            f"blocksize={blocksize} must be a positive multiple of group_size={spec.group_size}. "
            "This keeps GPTQ group-scale boundaries aligned with block error updates."
        )

    Q = torch.zeros_like(W)
    scale_cache: dict[int, torch.Tensor] = {}

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            col = i1 + i
            w = W1[:, i]
            d = Hinv1[i, i].clamp_min(1e-12)

            if col % spec.group_size == 0:
                g1 = min(col + spec.group_size, cols)
                # Dynamic group scale from the current error-updated group.
                scale_cache[col] = find_scale(W[:, col:g1], spec.bits)
            gstart = col - (col % spec.group_size)
            s = scale_cache[gstart]

            q = quant(w.unsqueeze(1), s, spec.bits).squeeze(1)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        Q[:, i1:i2] = Q1
        if i2 < cols:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return Q


# =====================================================================================
# Metrics / runners
# =====================================================================================
def fidelity_chunked(W_oriented: torch.Tensor, Wp_oriented: torch.Tensor, X: torch.Tensor, chunk_rows: int = 2048) -> dict:
    """S1.fidelity-compatible metric without materializing all outputs at once."""
    diff_sq = 0.0
    ref_sq = 0.0
    cos_sum = 0.0
    n_rows = 0
    for s in range(0, X.shape[0], chunk_rows):
        xb = X[s : s + chunk_rows].float()
        Y = xb @ W_oriented.float()
        Yp = xb @ Wp_oriented.float()
        diff_sq += float((Y - Yp).square().sum())
        ref_sq += float(Y.square().sum())
        cos = torch.nn.functional.cosine_similarity(Y, Yp, dim=1, eps=1e-12)
        cos_sum += float(cos.sum())
        n_rows += int(cos.numel())
    rel = (diff_sq ** 0.5) / max(ref_sq ** 0.5, 1e-12)
    return {"rel_err": rel, "mean_cosine": cos_sum / max(n_rows, 1)}


def load_layer1_expert(expert_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(SHARD1, framework="pt", device="cpu") as f:
        up = f.get_tensor(f"backbone.layers.1.mixer.experts.{expert_idx}.up_proj.weight").to(torch.float32)
        down = f.get_tensor(f"backbone.layers.1.mixer.experts.{expert_idx}.down_proj.weight").to(torch.float32)
    return up, down


def add_rtn_rows(rows: list[dict], proj: str, expert: int, Wq: torch.Tensor, X_eval: torch.Tensor, specs: Iterable[QuantSpec]) -> None:
    # Wq [out,in], W_oriented [in,out] for fidelity.
    W_or = Wq.t().contiguous()
    for spec in specs:
        t0 = time.time()
        Q = rtn_quantize(Wq, spec)
        Wp_or = Q.t().contiguous()
        f = fidelity_chunked(W_or, Wp_or, X_eval)
        bits = effective_bits(Wq.shape[0], Wq.shape[1], spec)
        row = {
            "proj": proj,
            "expert": expert,
            "codec": f"rtn_{spec.bits}b",
            "bits": bits,
            "damp": None,
            "actual_damp": None,
            "rel_err": f["rel_err"],
            "mean_cosine": f["mean_cosine"],
            "implied_vram_gb": S1.implied_vram_gb(bits),
            "seconds": time.time() - t0,
        }
        rows.append(row)
        print(
            f"[row] proj={proj} expert={expert:03d} codec={row['codec']} "
            f"bits={bits:.3f} heldout_rel={100*f['rel_err']:.3f}% "
            f"cos={f['mean_cosine']:.6f} seconds={row['seconds']:.1f}",
            flush=True,
        )


def add_gptq_rows(
    rows: list[dict],
    proj: str,
    expert: int,
    Wq: torch.Tensor,
    X_eval: torch.Tensor,
    specs: Iterable[QuantSpec],
    hinv_by_damp: dict[float, tuple[torch.Tensor, float]],
    blocksize: int,
) -> None:
    W_or = Wq.t().contiguous()
    for percdamp, (Hinv, actual_damp) in hinv_by_damp.items():
        for spec in specs:
            t0 = time.time()
            Q = gptq_quantize_with_hinv(Wq, Hinv, spec, blocksize=blocksize)
            Wp_or = Q.t().contiguous()
            f = fidelity_chunked(W_or, Wp_or, X_eval)
            bits = effective_bits(Wq.shape[0], Wq.shape[1], spec)
            row = {
                "proj": proj,
                "expert": expert,
                "codec": f"gptq_{spec.bits}b",
                "bits": bits,
                "damp": percdamp,
                "actual_damp": actual_damp,
                "rel_err": f["rel_err"],
                "mean_cosine": f["mean_cosine"],
                "implied_vram_gb": S1.implied_vram_gb(bits),
                "seconds": time.time() - t0,
            }
            rows.append(row)
            print(
                f"[row] proj={proj} expert={expert:03d} codec={row['codec']} "
                f"damp={percdamp:g} bits={bits:.3f} "
                f"heldout_rel={100*f['rel_err']:.3f}% cos={f['mean_cosine']:.6f} "
                f"seconds={row['seconds']:.1f}",
                flush=True,
            )


def aggregate_rows(rows: list[dict]) -> list[dict]:
    keys: dict[tuple, list[dict]] = {}
    for r in rows:
        keys.setdefault((r["proj"], r["codec"], r["bits"], r["damp"]), []).append(r)
    out = []
    for (proj, codec, bits, damp), grp in sorted(keys.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][3]))):
        rels = [g["rel_err"] for g in grp]
        coss = [g["mean_cosine"] for g in grp]
        out.append(
            {
                "proj": proj,
                "codec": codec,
                "bits": round(bits, 6),
                "damp": damp,
                "n_experts": len(grp),
                "mean_rel_err_pct": round(100 * float(np.mean(rels)), 6),
                "max_rel_err_pct": round(100 * float(np.max(rels)), 6),
                "mean_cosine": round(float(np.mean(coss)), 8),
                "implied_vram_gb": round(S1.implied_vram_gb(bits), 4),
            }
        )
    return out


def write_outputs(prefix: str, rows: list[dict], summary: dict) -> tuple[str, str]:
    csv_path = prefix + "_rows.csv"
    json_path = prefix + "_summary.json"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return csv_path, json_path


# =====================================================================================
# CLI
# =====================================================================================
def parse_csv_floats(s: str) -> list[float]:
    try:
        return [float(x) for x in s.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated floats like 0.01,0.1; got {s!r}") from exc


def parse_csv_ints(s: str) -> list[int]:
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated ints like 4 or 4,3; got {s!r}") from exc


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--max-cal", type=int, default=0, help="0 means use all cached calibration rows")
    ap.add_argument("--max-heldout", type=int, default=0, help="0 means use all cached held-out rows")
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--experts", default="", help="optional comma-separated expert ids; overrides --n-experts")
    ap.add_argument("--proj", choices=["up", "down", "both"], default="up")
    ap.add_argument("--bits", type=parse_csv_ints, default=[4], help="comma-separated GPTQ/RTN payload bits, e.g. 4 or 4,3")
    ap.add_argument("--damp", type=parse_csv_floats, default=[0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--group-up", type=int, default=GROUP_UP)
    ap.add_argument("--group-down", type=int, default=GROUP_DN)
    ap.add_argument("--hessian-chunk", type=int, default=8192)
    ap.add_argument("--out-prefix", default=os.path.join(HERE, "powered_gptq_eval"))
    ap.add_argument("--smoke", action="store_true", help="small quick run: one expert, 1024 cal, 512 heldout, two damp values")
    args = ap.parse_args()

    if args.smoke:
        args.n_experts = 1
        args.max_cal = 1024 if args.max_cal == 0 else min(args.max_cal, 1024)
        args.max_heldout = 512 if args.max_heldout == 0 else min(args.max_heldout, 512)
        args.damp = [0.01, 0.1]
        args.bits = [4]

    t_start = time.time()
    torch.set_grad_enabled(False)

    X_cal, X_held, cache_meta = load_activation_cache(args.cache_dir, args.max_cal, args.max_heldout)
    if X_cal.shape[1] != 2688:
        raise ValueError(f"Layer-1 up_proj cache should have hidden_size=2688, got {X_cal.shape[1]}.")

    experts = parse_csv_ints(args.experts) if args.experts.strip() else list(range(args.n_experts))
    if not experts:
        raise ValueError("No experts selected. Use --n-experts N or --experts 0,7,13.")
    for e in experts:
        if e < 0 or e >= 128:
            raise ValueError(f"Expert id {e} is out of range; expected 0..127.")

    bit_specs_up = [QuantSpec(bits=b, group_size=args.group_up) for b in args.bits]
    bit_specs_dn = [QuantSpec(bits=b, group_size=args.group_down) for b in args.bits]
    int8_up = QuantSpec(bits=8, group_size=args.group_up)
    int8_dn = QuantSpec(bits=8, group_size=args.group_down)

    rows: list[dict] = []
    proj_set = {"up", "down"} if args.proj == "both" else {args.proj}

    def aligned_blocksize(requested: int, group_size: int, proj: str) -> int:
        if requested >= group_size and requested % group_size == 0:
            return requested
        chosen = group_size
        print(
            f"[blocksize] proj={proj} requested={requested} is not aligned with "
            f"group_size={group_size}; using blocksize={chosen} so group scales align "
            "with GPTQ error-update blocks.",
            flush=True,
        )
        return chosen

    blocksize_up = aligned_blocksize(args.blocksize, args.group_up, "up") if "up" in proj_set else args.blocksize
    blocksize_dn = aligned_blocksize(args.blocksize, args.group_down, "down") if "down" in proj_set else args.blocksize

    up_hinv_by_damp: dict[float, tuple[torch.Tensor, float]] = {}
    if "up" in proj_set:
        H_up = compute_hessian(X_cal, chunk_rows=args.hessian_chunk)
        for d in args.damp:
            t0 = time.time()
            Hinv, actual = prepare_hinv(H_up, d)
            up_hinv_by_damp[d] = (Hinv, actual)
            print(
                f"[hinv] proj=up damp={d:g} actual_damp={actual:.6g} "
                f"seconds={time.time()-t0:.1f} RSS={rss_gb():.2f}GB",
                flush=True,
            )
        del H_up

    for e in experts:
        t0 = time.time()
        Wup, Wdn = load_layer1_expert(e)
        print(f"[expert] id={e:03d} loaded up={tuple(Wup.shape)} down={tuple(Wdn.shape)} seconds={time.time()-t0:.1f} RSS={rss_gb():.2f}GB", flush=True)

        if "up" in proj_set:
            Wq_up = Wup.contiguous()  # checkpoint orientation [out=1856, in=2688]
            add_rtn_rows(rows, "up", e, Wq_up, X_held, [int8_up, *bit_specs_up])
            add_gptq_rows(rows, "up", e, Wq_up, X_held, bit_specs_up, up_hinv_by_damp, blocksize_up)

        if "down" in proj_set:
            # Dense full-precision second-hop activations for this expert.
            A_cal = torch.relu(X_cal @ Wup.t()) ** 2
            A_held = torch.relu(X_held @ Wup.t()) ** 2
            H_dn = compute_hessian(A_cal, chunk_rows=args.hessian_chunk)
            dn_hinv_by_damp = {}
            for d in args.damp:
                t_h = time.time()
                Hinv, actual = prepare_hinv(H_dn, d)
                dn_hinv_by_damp[d] = (Hinv, actual)
                print(
                    f"[hinv] proj=down expert={e:03d} damp={d:g} actual_damp={actual:.6g} "
                    f"seconds={time.time()-t_h:.1f} RSS={rss_gb():.2f}GB",
                    flush=True,
                )
            Wq_dn = Wdn.contiguous()  # checkpoint orientation [out=2688, in=1856]
            add_rtn_rows(rows, "down", e, Wq_dn, A_held, [int8_dn, *bit_specs_dn])
            add_gptq_rows(rows, "down", e, Wq_dn, A_held, bit_specs_dn, dn_hinv_by_damp, blocksize_dn)
            del A_cal, A_held, H_dn, dn_hinv_by_damp

        del Wup, Wdn

    aggregates = aggregate_rows(rows)
    summary = {
        "stage": "Stage-1 matmul fidelity, powered GPTQ cache",
        "cost_axes": ["Resident VRAM", "Per-token bandwidth"],
        "litmus": "Weights would stay INT4/INT8 into fused kernels; this eval dequantizes only to measure fidelity.",
        "cache_dir": os.path.abspath(args.cache_dir),
        "cache_meta": cache_meta,
        "n_cal_used": int(X_cal.shape[0]),
        "n_heldout_used": int(X_held.shape[0]),
        "experts": experts,
        "proj": args.proj,
        "bits": args.bits,
        "damp": args.damp,
        "blocksize_requested": args.blocksize,
        "blocksize_up": blocksize_up,
        "blocksize_down": blocksize_dn,
        "rows_csv": os.path.abspath(args.out_prefix + "_rows.csv"),
        "aggregates": aggregates,
        "seconds_total": round(time.time() - t_start, 1),
        "peak_rss_gb_at_end": round(rss_gb(), 2),
        "bar": "INT4 must approach INT8-class behavior; Stage-2 bar is KL≈1e-3 / flat perplexity.",
    }
    csv_path, json_path = write_outputs(args.out_prefix, rows, summary)

    print("\n==== AGGREGATE ====")
    for a in aggregates:
        d = "" if a["damp"] is None else f" damp={a['damp']:g}"
        print(
            f"{a['proj']:4s} {a['codec']:8s}{d:12s} n={a['n_experts']:2d} "
            f"bits={a['bits']:.3f} rel={a['mean_rel_err_pct']:.3f}% "
            f"max={a['max_rel_err_pct']:.3f}% cos={a['mean_cosine']:.6f} "
            f"vram={a['implied_vram_gb']:.2f}GB"
        )
    print(f"[write] rows={csv_path}")
    print(f"[write] summary={json_path}")
    print("SUMMARY " + json.dumps({
        "n_cal_used": summary["n_cal_used"],
        "n_heldout_used": summary["n_heldout_used"],
        "experts": experts,
        "aggregates": aggregates,
        "seconds_total": summary["seconds_total"],
    }))


if __name__ == "__main__":
    main()
