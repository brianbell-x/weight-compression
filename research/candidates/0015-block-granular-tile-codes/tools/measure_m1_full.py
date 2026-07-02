"""measure_m1_full.py -- candidate 0015: convert the M1 projection into a fully
MEASURED whole-model number, plus the two named follow-up measurements.

Thin driver over verified modules (no new coder, no new accounting):
probe_block_codes (v1: loaders, quantizer, packing, stz parity),
probe_block_codes_v2 (v2: coder, DP tiers, tables), probe_mantissa_phase
(mp: M1 sym10 coder path, generalized tables, k-bit L3 seeds, realized cells,
round-trip gate), probe_emission_peel (ep: emitting coder, randomness
batteries, plane certificates, sampling).

T1 (--tasks t1, per layer; --aggregate across all 23):
  Score EVERY expert tensor of the layer (256 real: 128 experts x {up,down})
  under the frozen v2 cell (W=128, T=4 DP tiers, P100, L1+L3, k=7 verbatim
  mantissa) and under frozen+M1 (sym10 = u>>6 folds the mantissa MSB into the
  coded symbol; 1024-entry 12-bit rANS table charged pad8(1024 + nnz*12);
  6-bit verbatim plane pad8(6n - 12nb)). All side costs charged; per-tensor
  stz parity vs the 0009 reference (must stay exact); per-tensor cross-check
  vs the stored frozen artifacts (bit-identical bpw); round-trip gate per
  tensor per mechanism ({first, last, argmin, argmax} blocks: emitted bits ==
  accounted bits, symbols exact, L3 payload from the decoder's final state,
  SHA-256-exact BF16 reconstruction with flush-borne bits destroyed first).
  --aggregate combines the 23 per-layer summaries into THE whole-model number
  under the EXACT 10.7311 convention (all 23 expert layers have identical
  numel, so the expert plane is the plain mean of per-layer deltas; the
  non-expert ~7% stays at stz):
      wm = stz_wm - expert_share * mean_23(stz_ref_L - format_bpw_L)
  and gates itself by reproducing 10.7311 from the frozen recompute.

T2 (--tasks t2): MI decomposition of the mantissa MSB (and 2nd bit) --
  H(msb), H(msb|sign), H(msb|exp8), H(msb|sym9) (= H(sym10) - H(sym9)), the
  incremental sign term given exp, per-exponent-value p(msb=1|e) profile, and
  the same for bit2 vs sym10 (what sym11 would capture). Entropy-level
  diagnostics (a 256x2x12-bit table model would cost ~0.0012 b/w/tensor --
  noted, not charged; this locates structure, it is not a code).

T3 (--tasks t3): re-peel of the M1 emission (the recursive peel-until-random
  loop applied to what M1 emits). Per tensor: the ACTUAL sym10 coded payload
  plane (12-bit flush + renorm bits, serializer-gated against the reference
  byte packer) through the peel's battery (order-0/1/2 bit entropy, bit-pair
  MI vs circular-shift null, lzma -9e), and the residual 6-bit mantissa plane
  through H(bit | position mod 6) + native-lag MI (lags 6/12/18/6C/756).
  Deliverable: converged certificate (both ceilings < 0.01 b/w) or a new
  quantified ceiling.

Usage:
  uv run python measure_m1_full.py --synthetic                 # smoke, all T's
  uv run python measure_m1_full.py --layer 13                  # real, all T's
  uv run python measure_m1_full.py --layer 13 --tasks t1       # one deliverable
  uv run python measure_m1_full.py --layer 13 --summary        # T1 tables only
  uv run python measure_m1_full.py --aggregate                 # whole-model
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import probe_block_codes as v1        # noqa: E402  -- verified infrastructure
import probe_block_codes_v2 as v2     # noqa: E402  -- coder, DP, tables
import probe_emission_peel as ep      # noqa: E402  -- batteries, sampling
import probe_mantissa_phase as mp     # noqa: E402  -- M1 coder + accounting

stz = v1.stz
M, FLUSH_BITS = v1.M, v1.FLUSH_BITS
pad8, ceil_div, die = v1.pad8, v1.ceil_div, v1.die
ART = v1.ART
BORROW_BITS = v2.BORROW_BITS
W = mp.W                      # 128, frozen
T_MAX = mp.T_MAX              # 4, frozen
FROZEN_KEY = v2.FROZEN_KEY    # W128_T4_P100_L11L31L40

LAYERS_ALL = (1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36,
              38, 40, 43, 45, 47, 49, 51)
LAYER_NUMEL = 1_277_165_568           # identical on all 23 expert layers
STZ_WM_BPW = 10.897505                # stz realized whole-model (0009 ref)
EXPERT_SHARE = 0.930232               # expert numel share of the whole model
FROZEN_WM_REF = 10.7311               # fully measured frozen whole-model
WM_REPRO_TOL = 2e-4                   # aggregate must reproduce it this close
T2_MODEL_NOTE = ("entropy-level; a transmitted 256x2 12-bit model would cost "
                 "pad8(256*2*12)=6144 bits/tensor ~ 0.0012 b/w -- noted, not "
                 "charged (diagnostic, not a code)")
T3_PHASE = 6                          # residual plane period under M1
T3_EXPERTS_PER_PROJ = 4               # >= 8 tensors per layer (x2 projections)
STRUCT_EPS_BPW = ep.STRUCT_EPS_BPW    # 0.01 b/w convergence bar

ACCT = {"schema": 1, "probe": "measure_m1_full", "W": W, "T_MAX": T_MAX,
        "P": 100, "L1": 1, "L3": 1, "L4": 0,
        "MP_ACCT": mp.ACCT_STAMP,     # inherits the M1 accounting verbatim
        "EP_ACCT": ep.ACCT_STAMP,     # inherits the battery constants
        "LAYERS_ALL": LAYERS_ALL, "LAYER_NUMEL": LAYER_NUMEL,
        "STZ_WM_BPW": STZ_WM_BPW, "EXPERT_SHARE": EXPERT_SHARE,
        "FROZEN_WM_REF": FROZEN_WM_REF, "WM_REPRO_TOL": WM_REPRO_TOL,
        "T3_PHASE": T3_PHASE, "T3_EXPERTS_PER_PROJ": T3_EXPERTS_PER_PROJ,
        "CODER": v2.CODER_SPEC}
ACCT_STAMP = hashlib.sha256(json.dumps(ACCT, sort_keys=True).encode()).hexdigest()[:12]

hbin, h0_bits = mp.hbin, mp.h0_bits


def check_stamp(rows: list[dict], jsonl: Path):
    bad = [r for r in rows if r.get("acct") != ACCT_STAMP]
    if bad:
        die(f"{len(bad)}/{len(rows)} rows in {jsonl} carry accounting stamp "
            f"{bad[0].get('acct')!r} != current {ACCT_STAMP!r} -- move that "
            f"file aside and re-run")


# ------------------------------------------------------------------ targets ---
def layer_targets(snap: Path, synthetic: bool, layer: int | None) -> list[dict]:
    """T1: EVERY expert tensor of the layer (256 real). Synthetic: all."""
    if synthetic:
        return v1.enum_targets(snap, True)
    v1.TARGET_LAYER = layer
    return v1.enum_targets(snap, False)


def t3_targets(snap: Path, synthetic: bool, layer: int | None) -> list[dict]:
    """T3: deterministic subsample of the ep sample -- T3_EXPERTS_PER_PROJ
    experts per projection (>= 8 tensors per real layer)."""
    tg = ep.sample_targets(snap, synthetic, layer)
    out = []
    for L in sorted({t["layer"] for t in tg}):
        for proj in ("up", "down"):
            cand = sorted((t for t in tg
                           if t["layer"] == L and t["proj"] == proj),
                          key=lambda t: t["expert"])
            k = min(T3_EXPERTS_PER_PROJ, len(cand))
            sel = sorted({int(i) for i in
                          np.linspace(0, len(cand) - 1, k).round()})
            out.extend(cand[i] for i in sel)
    return out


def load_stz_ref(names: list[str]) -> dict:
    if not v1.STATS_JSONL.exists():
        die(f"missing stz parity reference {v1.STATS_JSONL}")
    want = set(names)
    ref = {}
    for line in v1.STATS_JSONL.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["name"] in want:
                ref[r["name"]] = r["bpw"]
    missing = [nm for nm in names if nm not in ref]
    if missing:
        die(f"{len(missing)} targets absent from stz stats "
            f"(first: {missing[0]})")
    return ref


def load_stored_frozen(synthetic: bool, layer: int | None) -> dict:
    """name -> stored frozen-cell bpw from the prior frozen artifacts (layer
    27 lives in the full-grid jsonl). Missing file/rows -> partial/empty map;
    every PRESENT row is an exact-equality gate."""
    if synthetic:
        paths = [ART / "blockcodes_v2_frozen_results_synthetic.jsonl"]
    elif layer == 27:
        paths = [ART / "blockcodes_v2_results.jsonl"]
    else:
        paths = [ART / f"blockcodes_v2_frozen_results_layer{layer}.jsonl"]
    out = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            c = r.get("cells", {}).get(FROZEN_KEY)
            if c is not None:
                out.setdefault(r["name"], c["bpw"])
    return out


# ----------------------------------------------------------------- T1 core ---
def t1_tensor(raw: bytes, t: dict, stats_ref: dict, stored_frozen: dict,
              synthetic: bool) -> dict:
    u = np.frombuffer(raw, "<u2")
    n = u.size
    R, Ccols = t["shape"]
    assert n == R * Ccols, (t["name"], n, R, Ccols)
    if n % W:
        die(f"n={n} not divisible by W={W} on {t['name']}")
    nb = n // W
    starts = np.arange(nb, dtype=np.int64) * W

    sym9 = (u >> 7).astype(np.int64)
    mant = (u & 0x7F).astype(np.int64)
    sym10 = (u >> 6).astype(np.int64)
    m6 = (u & 0x3F).astype(np.int64)
    hist9 = np.bincount(sym9, minlength=512).astype(np.int64)
    hist10 = np.bincount(sym10, minlength=1024).astype(np.int64)
    H9, H10 = h0_bits(hist9), h0_bits(hist10)

    # ---- stz parity (must stay exact; v2's gate verbatim)
    plan = stz.plan_regroup(hist9, n, R)
    base_bpw = plan["bits"] / n
    if synthetic:
        codec, chunks, st = stz.enc_tensor(raw, t["shape"])
        ref = st["bpw"]
        if codec == 1:
            realized = sum(len(c) for c in chunks) * 8
            if realized != plan["bits"]:
                die(f"PARITY: {t['name']} plan {plan['bits']} != "
                    f"realized {realized}")
    else:
        ref = stats_ref[t["name"]]
    pdiff = abs(round(base_bpw, 4) - ref)
    if pdiff > v2.PARITY_TOL:
        die(f"PARITY FAILURE on {t['name']}: {base_bpw:.4f} vs {ref:.4f}")

    # ---- quantizer parity gate (generalized == v1 at A=512)
    if not np.array_equal(mp.quantize_hist_a(hist9, n, 512),
                          v1.quantize_hist(hist9, n)):
        die(f"QUANTIZER PARITY on {t['name']}: generalized != v1 at A=512")

    sha_o, sha_r = hashlib.sha256(), hashlib.sha256()
    cells, rt = {}, {}

    # ---- frozen (k=7), asserted equal to ep.realized_cell + stored artifact
    q0, cum0, _clq0, tab0_bits, nnz9 = v2.build_table(hist9, n)
    ds7 = mp.d_seed_k(mant, starts, 7)
    x0_7 = (M + ds7).astype(np.int64)
    rb7 = v2.rans_sim_blocks(q0[sym9].reshape(nb, W),
                             cum0[sym9].reshape(nb, W), x0_7)
    cells["frozen"] = mp.realized_cell_k(rb7, nb, n, tab0_bits, 7)
    epc = ep.realized_cell(rb7, nb, n, tab0_bits)
    epc.pop("_flags")
    if epc != cells["frozen"]:
        die(f"FROZEN PARITY on {t['name']}: realized_cell_k(7) != "
            f"ep.realized_cell")
    stored = stored_frozen.get(t["name"])
    if stored is not None and stored != cells["frozen"]["bpw"]:
        die(f"STORED-FROZEN MISMATCH on {t['name']}: recomputed "
            f"{cells['frozen']['bpw']} != stored artifact {stored}")
    rt["frozen"] = mp.rt_extended(raw, t["name"], "frozen", sym9, mant,
                                  mp.MECH_SPEC["frozen"], q0, cum0, x0_7,
                                  ds7, rb7, starts, sha_o, sha_r)

    # ---- M1 (sym10, k=6)
    q10, cum10, tab10_bits, nnz10 = mp.build_table_a(hist10, n, 1024)
    ds6 = mp.d_seed_k(m6, starts, 6)
    x0_6 = (M + ds6).astype(np.int64)
    rb10 = v2.rans_sim_blocks(q10[sym10].reshape(nb, W),
                              cum10[sym10].reshape(nb, W), x0_6)
    cells["M1"] = mp.realized_cell_k(rb10, nb, n, tab10_bits, 6)
    rt["M1"] = mp.rt_extended(raw, t["name"], "M1", sym10, m6,
                              mp.MECH_SPEC["M1"], q10, cum10, x0_6, ds6,
                              rb10, starts, sha_o, sha_r)

    if sha_o.digest() != sha_r.digest():
        die(f"ROUND-TRIP ({t['name']}): SHA-256 mismatch over sampled spans")
    rt["sha256_ok"] = True

    return {"name": t["name"], "layer": t["layer"], "expert": t["expert"],
            "proj": t["proj"], "n": int(n), "nb": int(nb),
            "acct": ACCT_STAMP,
            "H_sym": round(H9, 6), "H_sym10": round(H10, 6),
            "nnz": {"sym9": nnz9, "sym10": nnz10},
            "floors": {"floor7": round(H9 + 7.0, 6),
                       "bound_m1": round(H10 + 6.0, 6)},
            "stz": {"ref_bpw": ref, "parity_abs_diff": round(pdiff, 6)},
            "stored_frozen_checked": stored is not None,
            "cells": cells,
            "delta_bpw": round(cells["frozen"]["bpw"] - cells["M1"]["bpw"], 6),
            "roundtrip": rt}


def t1_summarize(tg: list[dict], jsonl: Path, summaryp: Path,
                 synthetic: bool, layer: int | None) -> dict:
    rows = v1.load_rows(jsonl)
    check_stamp(rows, jsonl)
    rec = {}
    for r in rows:
        rec.setdefault(r["name"], r)
    names = [t["name"] for t in tg]
    miss = [nm for nm in names if nm not in rec]
    if miss:
        die(f"T1 summary requires all tensors done; {len(miss)} missing "
            f"(first: {miss[0]}) -- re-invoke to resume")
    recs = [rec[nm] for nm in names]
    n_tot = sum(r["n"] for r in recs)
    wsum = lambda f: sum(f(r) for r in recs)
    bits = lambda r, m: r["cells"][m]["sym_bits"] + r["cells"][m]["mant_bits"]

    frozen_bpw = wsum(lambda r: bits(r, "frozen")) / n_tot
    m1_bpw = wsum(lambda r: bits(r, "M1")) / n_tot
    stz_ref = wsum(lambda r: r["stz"]["ref_bpw"] * r["n"]) / n_tot
    floor7 = wsum(lambda r: r["floors"]["floor7"] * r["n"]) / n_tot
    bound_m1 = wsum(lambda r: r["floors"]["bound_m1"] * r["n"]) / n_tot
    deltas = [r["delta_bpw"] for r in recs]
    rt_ok = all(r["roundtrip"]["sha256_ok"] for r in recs)
    per_proj = {}
    for proj in sorted({r["proj"] for r in recs}):
        rs = [r for r in recs if r["proj"] == proj]
        nn = sum(r["n"] for r in rs)
        per_proj[proj] = {
            "tensors": len(rs),
            "frozen_bpw": round(sum(bits(r, "frozen") for r in rs) / nn, 6),
            "m1_bpw": round(sum(bits(r, "M1") for r in rs) / nn, 6),
        }
        per_proj[proj]["delta_bpw"] = round(
            per_proj[proj]["frozen_bpw"] - per_proj[proj]["m1_bpw"], 6)

    summary = {
        "mode": "synthetic" if synthetic else "real",
        "layer": layer, "acct_stamp": ACCT_STAMP,
        "targets": len(recs), "total_params": int(n_tot),
        "stz_ref_weighted_bpw": round(stz_ref, 6),
        "parity_max_abs_diff": max(r["stz"]["parity_abs_diff"] for r in recs),
        "stored_frozen_checked": sum(r["stored_frozen_checked"] for r in recs),
        "frozen_bpw": round(frozen_bpw, 6),
        "m1_bpw": round(m1_bpw, 6),
        "delta_bpw": round(frozen_bpw - m1_bpw, 6),
        "delta_vs_stz_m1_bpw": round(stz_ref - m1_bpw, 6),
        "floor7_bpw": round(floor7, 6),
        "bound_m1_bpw": round(bound_m1, 6),
        "per_proj": per_proj,
        "per_tensor_delta": {"min": min(deltas), "max": max(deltas),
                             "n_nonpositive": sum(d <= 0 for d in deltas)},
        "roundtrip": {"frozen_blocks": wsum(lambda r: r["roundtrip"]["frozen"]),
                      "m1_blocks": wsum(lambda r: r["roundtrip"]["M1"]),
                      "all_ok": bool(rt_ok)},
        "accounting_note": ("identical mechanics/charges to "
                            "probe_mantissa_phase (frozen k=7 vs M1 sym10 "
                            "k=6); stz parity + stored-frozen cross-check + "
                            "round-trip gates all fatal on mismatch"),
    }
    summaryp.write_text(json.dumps(summary, indent=2))
    tagl = "synthetic" if synthetic else f"layer {layer}"
    print(f"\n[T1 {tagl}] {len(recs)} tensors, {n_tot:,} params | "
          f"stz {stz_ref:.4f} | frozen {frozen_bpw:.4f} | M1 {m1_bpw:.4f} | "
          f"delta +{frozen_bpw - m1_bpw:.4f} b/w | floor7 {floor7:.4f} -> "
          f"bound_m1 {bound_m1:.4f} | RT "
          f"{summary['roundtrip']['frozen_blocks']}+"
          f"{summary['roundtrip']['m1_blocks']} blocks "
          f"{'PASS' if rt_ok else 'FAIL'} | parity max "
          f"{summary['parity_max_abs_diff']} | stored-frozen checked "
          f"{summary['stored_frozen_checked']}/{len(recs)}")
    print(f"[T1 {tagl}] summary written: {summaryp}")
    return summary


def t1_run(a, snap: Path, jsonl: Path, summaryp: Path, t0: float):
    tg = layer_targets(snap, a.synthetic, a.layer)
    names = [t["name"] for t in tg]
    stats_ref = {} if a.synthetic else load_stz_ref(names)
    stored_frozen = load_stored_frozen(a.synthetic, a.layer)
    prior = v1.load_rows(jsonl)
    check_stamp(prior, jsonl)
    done = {r["name"] for r in prior}
    processed = 0
    for i, t in enumerate(tg):
        if t["name"] in done:
            continue
        if a.limit and processed >= a.limit:
            print(f"\n[T1 limit] {a.limit} tensors this invocation -- "
                  f"re-invoke to resume.")
            sys.exit(0)
        if time.time() - t0 > a.budget_s:
            print(f"\n[T1 budget] {a.budget_s:.0f}s reached after {processed} "
                  f"tensors -- progress saved, re-invoke to resume.",
                  flush=True)
            sys.exit(0)
        raw = v1.read_raw(snap, t)
        rec = t1_tensor(raw, t, stats_ref, stored_frozen, a.synthetic)
        with jsonl.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        done.add(t["name"])
        processed += 1
        if processed % 16 == 0 or len(done) == len(tg):
            print(f"[T1 {i + 1}/{len(tg)}] {processed} tensors this run, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return t1_summarize(tg, jsonl, summaryp, a.synthetic, a.layer)


# ------------------------------------------------------------ T1 aggregate ---
def aggregate():
    per = {}
    for L in LAYERS_ALL:
        p = ART / f"m1full_summary_layer{L}.json"
        if not p.exists():
            die(f"aggregate requires all 23 layer summaries; missing {p}")
        s = json.loads(p.read_text())
        if s["targets"] != 256 or s["total_params"] != LAYER_NUMEL:
            die(f"layer {L}: {s['targets']} tensors / {s['total_params']} "
                f"params != the full 256 / {LAYER_NUMEL}")
        if not s["roundtrip"]["all_ok"]:
            die(f"layer {L}: round-trip not OK")
        per[L] = s
    d_frozen = [per[L]["stz_ref_weighted_bpw"] - per[L]["frozen_bpw"]
                for L in LAYERS_ALL]
    d_m1 = [per[L]["stz_ref_weighted_bpw"] - per[L]["m1_bpw"]
            for L in LAYERS_ALL]
    frozen_wm = STZ_WM_BPW - EXPERT_SHARE * (sum(d_frozen) / len(d_frozen))
    m1_wm = STZ_WM_BPW - EXPERT_SHARE * (sum(d_m1) / len(d_m1))
    if abs(frozen_wm - FROZEN_WM_REF) > WM_REPRO_TOL:
        die(f"AGGREGATE GATE: frozen whole-model recompute {frozen_wm:.6f} "
            f"does not reproduce the reference {FROZEN_WM_REF} "
            f"(tol {WM_REPRO_TOL})")
    deltas = {L: round(per[L]["frozen_bpw"] - per[L]["m1_bpw"], 6)
              for L in LAYERS_ALL}
    out = {
        "acct_stamp": ACCT_STAMP,
        "convention": ("wm = stz_wm - expert_share * mean_23(stz_ref_L - "
                       "format_bpw_L); all 23 expert layers have identical "
                       "numel so the expert plane is the plain mean of "
                       "per-layer deltas; non-expert ~7% held at stz -- "
                       "IDENTICAL to the 10.7311 computation"),
        "stz_whole_model_bpw": STZ_WM_BPW,
        "expert_share": EXPERT_SHARE,
        "layers": {str(L): {"stz_ref": per[L]["stz_ref_weighted_bpw"],
                            "frozen": per[L]["frozen_bpw"],
                            "m1": per[L]["m1_bpw"],
                            "delta_m1_vs_frozen": deltas[L]}
                   for L in LAYERS_ALL},
        "frozen_whole_model_recomputed_bpw": round(frozen_wm, 6),
        "frozen_whole_model_ref_bpw": FROZEN_WM_REF,
        "m1_whole_model_bpw": round(m1_wm, 6),
        "m1_delta_vs_frozen_wm_bpw": round(frozen_wm - m1_wm, 6),
        "m1_delta_vs_stz_wm_bpw": round(STZ_WM_BPW - m1_wm, 6),
        "expert_delta_m1_vs_frozen": {
            "mean": round(sum(deltas.values()) / len(deltas), 6),
            "min": min(deltas.values()), "max": max(deltas.values()),
            "min_layer": min(deltas, key=deltas.get),
            "max_layer": max(deltas, key=deltas.get)},
        "roundtrip_blocks_total": sum(
            per[L]["roundtrip"]["frozen_blocks"]
            + per[L]["roundtrip"]["m1_blocks"] for L in LAYERS_ALL),
        "note": ("MEASURED on every expert tensor of every expert layer "
                 "(23 x 256); replaces the 10.6923 projection"),
    }
    p = ART / "m1full_whole_model.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n=== whole-model (fully measured) ===")
    print(f"{'layer':>6}{'stz':>10}{'frozen':>10}{'M1':>10}{'delta':>9}")
    for L in LAYERS_ALL:
        s = per[L]
        print(f"{L:>6}{s['stz_ref_weighted_bpw']:>10.4f}"
              f"{s['frozen_bpw']:>10.4f}{s['m1_bpw']:>10.4f}"
              f"{deltas[L]:>+9.4f}")
    print(f"frozen whole-model recomputed: {frozen_wm:.4f} "
          f"(reference {FROZEN_WM_REF}: reproduced)")
    print(f"M1 whole-model MEASURED: {m1_wm:.4f} b/w "
          f"(delta vs frozen {frozen_wm - m1_wm:+.4f}, vs stz "
          f"{STZ_WM_BPW - m1_wm:+.4f})")
    print(f"written: {p}")
    return out


# ----------------------------------------------------------------- T2 core ---
def cond_h_bits(joint2: np.ndarray) -> float:
    """H(bit | ctx) from an (A, 2) count matrix."""
    return v2.cond_entropy_bits(joint2, int(joint2.sum()))


def t2_tensor(raw: bytes, t: dict) -> dict:
    u = np.frombuffer(raw, "<u2")
    n = u.size
    sgn = (u >> 15).astype(np.int64)
    e8 = ((u >> 7) & 0xFF).astype(np.int64)
    msb = ((u >> 6) & 1).astype(np.int64)
    b2 = ((u >> 5) & 1).astype(np.int64)
    sym9 = (u >> 7).astype(np.int64)
    H9 = h0_bits(np.bincount(sym9, minlength=512))
    H10 = h0_bits(np.bincount((u >> 6).astype(np.int64), minlength=1024))
    H11 = h0_bits(np.bincount((u >> 5).astype(np.int64), minlength=2048))

    H_msb = h0_bits(np.bincount(msb, minlength=2))
    H_msb_sign = cond_h_bits(np.bincount(sgn * 2 + msb,
                                         minlength=4).reshape(2, 2))
    H_msb_exp = cond_h_bits(np.bincount(e8 * 2 + msb,
                                        minlength=512).reshape(256, 2))
    H_msb_sym = H10 - H9                       # exact: sym10 = (sym9, msb)

    H_b2 = h0_bits(np.bincount(b2, minlength=2))
    H_b2_exp = cond_h_bits(np.bincount(e8 * 2 + b2,
                                       minlength=512).reshape(256, 2))
    H_b2_sym9 = cond_h_bits(np.bincount(sym9 * 2 + b2,
                                        minlength=1024).reshape(512, 2))
    H_b2_sym10 = H11 - H10                     # what sym11 folding would see

    # per-exponent profile (top by mass): p(msb=1 | e), and split by sign
    cnt_e = np.bincount(e8, minlength=256)
    ones_e = np.bincount(e8, weights=msb.astype(np.float64), minlength=256)
    order = np.argsort(-cnt_e)
    prof = []
    for e in order[:12]:
        if cnt_e[e] == 0:
            break
        sel_p = (e8 == e) & (sgn == 0)
        sel_n = (e8 == e) & (sgn == 1)
        prof.append({"exp": int(e), "mass": round(cnt_e[e] / n, 6),
                     "p1_msb": round(float(ones_e[e] / cnt_e[e]), 6),
                     "p1_msb_pos": (round(float(msb[sel_p].mean()), 6)
                                    if sel_p.any() else None),
                     "p1_msb_neg": (round(float(msb[sel_n].mean()), 6)
                                    if sel_n.any() else None)})

    return {"name": t["name"], "layer": t["layer"], "proj": t["proj"],
            "n": int(n),
            "H_msb": round(H_msb, 6),
            "H_msb_given_sign": round(H_msb_sign, 6),
            "H_msb_given_exp": round(H_msb_exp, 6),
            "H_msb_given_sym": round(H_msb_sym, 6),
            "H_b2": round(H_b2, 6),
            "H_b2_given_exp": round(H_b2_exp, 6),
            "H_b2_given_sym9": round(H_b2_sym9, 6),
            "H_b2_given_sym10": round(H_b2_sym10, 6),
            "exp_profile": prof}


def t2_run(a, snap: Path, outp: Path):
    tg = ep.sample_targets(snap, a.synthetic, a.layer)
    recs = []
    for i, t in enumerate(tg):
        recs.append(t2_tensor(v1.read_raw(snap, t), t))
        print(f"[T2 {i + 1}/{len(tg)}] {t['name']}", flush=True)
    n_tot = sum(r["n"] for r in recs)
    w = lambda k: sum(r[k] * r["n"] for r in recs) / n_tot
    keys = ("H_msb", "H_msb_given_sign", "H_msb_given_exp", "H_msb_given_sym",
            "H_b2", "H_b2_given_exp", "H_b2_given_sym9", "H_b2_given_sym10")
    agg = {k: round(w(k), 6) for k in keys}
    mi = {
        "MI_msb_sign": round(agg["H_msb"] - agg["H_msb_given_sign"], 6),
        "MI_msb_exp": round(agg["H_msb"] - agg["H_msb_given_exp"], 6),
        "MI_msb_sym": round(agg["H_msb"] - agg["H_msb_given_sym"], 6),
        "MI_msb_sign_given_exp": round(agg["H_msb_given_exp"]
                                       - agg["H_msb_given_sym"], 6),
        "MI_b2_exp": round(agg["H_b2"] - agg["H_b2_given_exp"], 6),
        "MI_b2_sym9": round(agg["H_b2"] - agg["H_b2_given_sym9"], 6),
        "MI_b2_sym10": round(agg["H_b2"] - agg["H_b2_given_sym10"], 6),
    }
    headroom = {
        "msb_marginal_bias": round(1.0 - agg["H_msb"], 6),
        "msb_conditioning": mi["MI_msb_sym"],
        "msb_total_vs_verbatim": round(1.0 - agg["H_msb_given_sym"], 6),
        "b2_marginal_bias": round(1.0 - agg["H_b2"], 6),
        "b2_conditioning_at_sym10": mi["MI_b2_sym10"],
        "b2_total_vs_verbatim": round(1.0 - agg["H_b2_given_sym10"], 6),
    }
    exp_share = (mi["MI_msb_exp"] / mi["MI_msb_sym"]
                 if mi["MI_msb_sym"] > 0 else None)
    per_layer = {}
    for L in sorted({r["layer"] for r in recs}):
        rs = [r for r in recs if r["layer"] == L]
        nn = sum(r["n"] for r in rs)
        per_layer[f"L{L}"] = {
            k: round(sum(r[k] * r["n"] for r in rs) / nn, 6) for k in keys}
    # pooled exponent profile over the sample (mass-weighted union)
    pool = {}
    for r in recs:
        for p in r["exp_profile"]:
            d = pool.setdefault(p["exp"], [0.0, 0.0])
            d[0] += p["mass"] * r["n"]
            d[1] += p["p1_msb"] * p["mass"] * r["n"]
    prof = [{"exp": e, "mass": round(m / n_tot, 6),
             "p1_msb": round(s / m, 6)}
            for e, (m, s) in sorted(pool.items(), key=lambda kv: -kv[1][0])]
    verdict = (
        f"sym-conditioned MSB structure is EXPONENT-MAGNITUDE structure: "
        f"conditioning on the 8-bit exponent captures "
        f"{100 * exp_share:.1f}% of MI(msb; sym) "
        f"({mi['MI_msb_exp']:.4f} of {mi['MI_msb_sym']:.4f} b/w); sign adds "
        f"{mi['MI_msb_sign_given_exp']:.4f} b/w given exp (sign alone "
        f"{mi['MI_msb_sign']:.4f})" if exp_share is not None else
        "no sym-conditioned MSB structure in this sample")
    out = {"mode": "synthetic" if a.synthetic else "real",
           "layer": a.layer, "acct_stamp": ACCT_STAMP,
           "targets": len(recs), "total_params": int(n_tot),
           "layers": sorted({r["layer"] for r in recs}),
           "weighted": agg, "mi_bpw": mi, "headroom_bpw": headroom,
           "exp_share_of_sym_mi": (round(exp_share, 6)
                                   if exp_share is not None else None),
           "per_layer": per_layer, "exp_profile_pooled": prof[:12],
           "model_cost_note": T2_MODEL_NOTE,
           "verdict": verdict,
           "per_tensor": recs}
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n[T2] H(msb) {agg['H_msb']:.4f} | given sign "
          f"{agg['H_msb_given_sign']:.4f} | given exp "
          f"{agg['H_msb_given_exp']:.4f} | given sym "
          f"{agg['H_msb_given_sym']:.4f}")
    print(f"[T2] MI(msb;sym) {mi['MI_msb_sym']:.4f} = exp {mi['MI_msb_exp']:.4f}"
          f" + sign|exp {mi['MI_msb_sign_given_exp']:.4f} "
          f"(+cross terms); bit2 total headroom vs verbatim "
          f"{headroom['b2_total_vs_verbatim']:.4f} "
          f"(marginal {headroom['b2_marginal_bias']:.4f} + sym10-cond "
          f"{headroom['b2_conditioning_at_sym10']:.4f})")
    print(f"[T2] {verdict}")
    print(f"[T2] written: {outp}")
    return out


# ----------------------------------------------------------------- T3 core ---
def t3_tensor(raw: bytes, t: dict) -> dict:
    u = np.frombuffer(raw, "<u2")
    n = u.size
    R, Ccols = t["shape"]
    if n % W:
        die(f"n={n} not divisible by W={W} on {t['name']}")
    nb = n // W
    starts = np.arange(nb, dtype=np.int64) * W
    sym10 = (u >> 6).astype(np.int64)
    m6 = (u & 0x3F).astype(np.int64)
    hist10 = np.bincount(sym10, minlength=1024).astype(np.int64)
    q10, cum10, tab10_bits, _nnz = mp.build_table_a(hist10, n, 1024)
    ds6 = mp.d_seed_k(m6, starts, 6)
    x0 = (M + ds6).astype(np.int64)

    # ---- emit the ACTUAL M1 payload plane (bit-identical coder)
    rb, K, V, xf = ep.rans_sim_emit(q10[sym10].reshape(nb, W),
                                    cum10[sym10].reshape(nb, W), x0)
    payload = ep.emit_payload_plane(K, V, xf)
    if payload.size != int(rb.sum()):
        die(f"payload plane size {payload.size} != accounted {int(rb.sum())} "
            f"on {t['name']}")

    # serializer gate: reference byte packer must reproduce the plane slices
    off = np.concatenate([[0], np.cumsum(rb)])
    ql, cl = q10.tolist(), cum10.tolist()
    for i in sorted({0, nb - 1, int(np.argmin(rb)), int(np.argmax(rb))}):
        s0 = int(starts[i])
        fl, bits = v2.rans_enc_block(sym10[s0:s0 + W].tolist(), ql, cl,
                                     int(x0[i]))
        data, nbits = v1.pack_block(fl, bits)
        if nbits != int(rb[i]):
            die(f"SERIALIZER ({t['name']} block {i}): emitted {nbits} != "
                f"accounted {int(rb[i])}")
        refb = np.unpackbits(np.frombuffer(data, np.uint8))[:nbits]
        if not np.array_equal(refb, payload[off[i]:off[i] + nbits]):
            die(f"SERIALIZER ({t['name']} block {i}): plane bits != "
                f"reference bytes")

    pay_cert = ep.plane_cert(payload.size, n, ep.bit_orders(payload),
                             ep.lzma_bits_of(np.packbits(payload).tobytes()),
                             ep.bit_mi_battery(payload), None, payload.size)

    # ---- residual 6-bit mantissa plane (k=6 layout, first 12 bits ride flush)
    b = ((m6[:, None] >> np.arange(5, -1, -1)) & 1).astype(np.uint8)
    mplane = np.ascontiguousarray(
        b.reshape(nb, 6 * W)[:, BORROW_BITS:]).ravel()
    # phase of coded position j is (j + BORROW_BITS) % 6 == j % 6 (12 % 6 == 0)
    phase_pat = np.arange(BORROW_BITS, 6 * W) % T3_PHASE
    m2 = mplane.reshape(nb, 6 * W - BORROW_BITS)
    ph_h, ph_p1 = 0.0, []
    for ph in range(T3_PHASE):
        colsel = phase_pat == ph
        tot = int(nb) * int(colsel.sum())
        ones = int(m2[:, colsel].sum())
        ph_p1.append(round(ones / tot, 6))
        ph_h += (tot / mplane.size) * h0_bits(
            np.array([tot - ones, ones], np.int64))
    mant_extra = [{"name": "phase6", "h": ph_h,
                   "model_bits": T3_PHASE * 2 * ep.MODEL_BITS_PER_CELL,
                   "p1": ph_p1}]
    stride = 6 * W - BORROW_BITS                       # 756
    mant_lags = [T3_PHASE, 2 * T3_PHASE, 3 * T3_PHASE, 6 * Ccols, stride]
    row_lag = Ccols // W if (Ccols >= W and Ccols % W == 0) else None
    if row_lag:
        mant_lags.append(stride * row_lag)
    mant_cert = ep.plane_cert(
        mplane.size, n, ep.bit_orders(mplane),
        ep.lzma_bits_of(np.packbits(mplane).tobytes()),
        ep.bit_mi_battery(mplane, mant_lags, (T3_PHASE, stride)),
        None, mplane.size, extra_ent=mant_extra)

    return {"name": t["name"], "layer": t["layer"], "proj": t["proj"],
            "n": int(n), "nb": int(nb), "acct": ACCT_STAMP,
            "coded_bpw": round(float(rb.sum()) / n, 6),
            "payload": pay_cert,
            "mant6": mant_cert,
            "phase6_p1": ph_p1,
            "phase6_h": round(ph_h, 6)}


def t3_run(a, snap: Path, jsonl: Path, outp: Path, t0: float):
    tg = t3_targets(snap, a.synthetic, a.layer)
    prior = v1.load_rows(jsonl)
    check_stamp(prior, jsonl)
    done = {r["name"] for r in prior}
    for i, t in enumerate(tg):
        if t["name"] in done:
            continue
        if time.time() - t0 > a.budget_s:
            print(f"\n[T3 budget] {a.budget_s:.0f}s reached -- progress "
                  f"saved, re-invoke to resume.", flush=True)
            sys.exit(0)
        rec = t3_tensor(v1.read_raw(snap, t), t)
        with jsonl.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        done.add(t["name"])
        print(f"[T3 {i + 1}/{len(tg)}] {t['name']} "
              f"(payload {rec['payload']['ceiling_bpw']:.4f}, mant6 "
              f"{rec['mant6']['ceiling_bpw']:.4f} b/w ceilings)", flush=True)

    rows = v1.load_rows(jsonl)
    check_stamp(rows, jsonl)
    rec = {}
    for r in rows:
        rec.setdefault(r["name"], r)
    recs = [rec[t["name"]] for t in tg if t["name"] in rec]
    if len(recs) != len(tg):
        die("T3 summary requires all tensors done -- re-invoke to resume")
    n_tot = sum(r["n"] for r in recs)
    w = lambda f: sum(f(r) for r in recs) / n_tot
    pay_ceil = w(lambda r: r["payload"]["ceiling_bpw"] * r["n"])
    man_ceil = w(lambda r: r["mant6"]["ceiling_bpw"] * r["n"])
    pay_hits = sum(bool(r["payload"]["mi_hit"]) for r in recs)
    man_hits = sum(bool(r["mant6"]["mi_hit"]) for r in recs)
    ph_pool = np.zeros(T3_PHASE)
    for r in recs:
        ph_pool += np.array(r["phase6_p1"]) * r["n"]
    ph_pool = [round(float(x / n_tot), 6) for x in ph_pool]
    converged = pay_ceil < STRUCT_EPS_BPW and man_ceil < STRUCT_EPS_BPW
    verdict = (
        f"CONVERGED at the {STRUCT_EPS_BPW} b/w bar: M1 payload ceiling "
        f"{pay_ceil:.4f}, residual 6-bit plane ceiling {man_ceil:.4f}"
        if converged else
        f"NEW QUANTIFIED CEILING: payload {pay_ceil:.4f}, residual 6-bit "
        f"plane {man_ceil:.4f} b/w (bar {STRUCT_EPS_BPW})")
    out = {"mode": "synthetic" if a.synthetic else "real",
           "layer": a.layer, "acct_stamp": ACCT_STAMP,
           "targets": len(recs), "total_params": int(n_tot),
           "payload_ceiling_bpw_weighted": round(pay_ceil, 6),
           "mant6_ceiling_bpw_weighted": round(man_ceil, 6),
           "payload_mi_hit_tensors": int(pay_hits),
           "mant6_mi_hit_tensors": int(man_hits),
           "phase6_p1_pooled": ph_pool,
           "phase6_note": ("residual plane phases 0..5 are the OLD mantissa "
                           "phases 1..6 (the MSB left the plane with M1); a "
                           "non-flat profile here is the second-bit signal "
                           "T2 quantifies as H(b2|sym10)"),
           "converged": bool(converged),
           "verdict": verdict}
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n[T3] {len(recs)} tensors | payload ceiling {pay_ceil:.4f} b/w "
          f"(MI hits {pay_hits}/{len(recs)}) | mant6 ceiling {man_ceil:.4f} "
          f"b/w (MI hits {man_hits}/{len(recs)})")
    print(f"[T3] residual phase p(1): "
          + " ".join(f"{x:.4f}" for x in ph_pool))
    print(f"[T3] {verdict}")
    print(f"[T3] written: {outp}")
    return out


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--layer", type=int, default=None,
                    help="expert layer to score (required for real T1)")
    ap.add_argument("--tasks", default="t1,t2,t3",
                    help="comma list of t1,t2,t3 (default all)")
    ap.add_argument("--summary", action="store_true",
                    help="T1 summary only (requires all tensors done)")
    ap.add_argument("--aggregate", action="store_true",
                    help="combine all 23 T1 layer summaries -> whole-model")
    ap.add_argument("--limit", type=int, default=0,
                    help="max T1 tensors this invocation (0 = no cap)")
    ap.add_argument("--budget-s", type=float, default=3300.0)
    a = ap.parse_args()

    if a.aggregate:
        aggregate()
        return

    tasks = [s.strip() for s in a.tasks.split(",") if s.strip()]
    bad = [s for s in tasks if s not in ("t1", "t2", "t3")]
    if bad:
        die(f"unknown tasks {bad}")
    if not a.synthetic and a.layer is None:
        die("--layer is required for real runs (or use --aggregate)")

    snap = v1.SYN_SNAP if a.synthetic else v1.REAL_SNAP
    tag = (("_synthetic" if a.synthetic else "")
           + (f"_layer{a.layer}" if a.layer is not None else ""))
    ART.mkdir(parents=True, exist_ok=True)
    t1_jsonl = ART / f"m1full_results{tag}.jsonl"
    t1_sum = ART / f"m1full_summary{tag}.json"
    t2_out = ART / f"m1full_mi{tag}.json"
    t3_jsonl = ART / f"m1full_repeel_results{tag}.jsonl"
    t3_out = ART / f"m1full_repeel{tag}.json"

    lockp = ART / f"m1full{tag}.lock"
    try:
        fd = os.open(lockp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = lockp.read_text().strip() or "?"
        except OSError:
            holder = "?"
        die(f"lock file {lockp} exists (pid {holder}); if no run is live, "
            f"delete it and retry")
    with os.fdopen(fd, "w") as lf:
        lf.write(str(os.getpid()))
    try:
        t0 = time.time()
        if a.summary:
            tg = layer_targets(snap, a.synthetic, a.layer)
            t1_summarize(tg, t1_jsonl, t1_sum, a.synthetic, a.layer)
            return
        if "t1" in tasks:
            t1_run(a, snap, t1_jsonl, t1_sum, t0)
        if "t2" in tasks:
            t2_run(a, snap, t2_out)
        if "t3" in tasks:
            t3_run(a, snap, t3_jsonl, t3_out, t0)
        print(f"\ndone in {time.time() - t0:.0f}s")
    finally:
        try:
            lockp.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
