"""Contrarian probes on the two real expert tensors in gpu_sample.npz.

A) True joint (left,up) exponent context vs the hashed 8-bit context.
B) Sub-tensor nonstationarity: per-row/col/tile conditional entropy; per-row
   codebooks vs per-tensor codebook (escape rates at K=15 and K=7).
C) Sign dependence: MI(sign; exp), MI(sign; neighbor signs), MI(sign; column).
D) Recomputed joint floor: H(exp|ctx) + H(sign|ctx) + H(mant|exp).

All conditional entropies are reported two ways:
  plugin  = plug-in on all data (optimistic; overfits with many contexts)
  holdout = train on even rows, cross-entropy on odd rows with KT smoothing
            (honest: achievable by a two-pass or adaptive coder)
"""
import numpy as np, json, sys

NPZ = r"C:\dev\compression\research\candidates\0009-fusible-exponent-codebook\tests\artifacts\gpu_sample.npz"


def H0(x):
    c = np.bincount(x.reshape(-1))
    p = c[c > 0] / x.size
    return float(-(p * np.log2(p)).sum())


def cond_plugin(sym, ctx):
    """H(sym|ctx), plug-in. sym,ctx int64 flat."""
    ns = int(sym.max()) + 1
    j = ctx * ns + sym
    cj = np.bincount(j)
    cc = np.bincount(ctx)
    n = sym.size
    pj = cj[cj > 0] / n
    Hj = -(pj * np.log2(pj)).sum()
    pc = cc[cc > 0] / n
    Hc = -(pc * np.log2(pc)).sum()
    return float(Hj - Hc)


def cond_holdout(sym, ctx, mask_train):
    """Cross-entropy (bits/sym) of KT-smoothed per-context model fit on train,
    scored on eval."""
    ns = int(sym.max()) + 1
    nc = int(ctx.max()) + 1
    j = ctx * ns + sym
    cj = np.bincount(j[mask_train], minlength=nc * ns).astype(np.float64)
    cj += 0.5  # KT smoothing over full alphabet per context
    cc = cj.reshape(nc, ns).sum(1)
    logp = np.log2(cj.reshape(nc, ns)) - np.log2(cc)[:, None]
    ev = ~mask_train
    return float(-logp[ctx[ev], sym[ev]].mean())


def remap(x):
    """Map values to dense 0..k-1."""
    u, inv = np.unique(x, return_inverse=True)
    return inv.astype(np.int64), len(u)


def probe(name, u16, shape):
    u16 = u16.reshape(shape)
    R, C = shape
    exp = ((u16 >> 7) & 0xFF).astype(np.int64)
    sign = (u16 >> 15).astype(np.int64)
    mant = (u16 & 0x7F).astype(np.int64)
    out = {"tensor": name, "shape": list(map(int, shape))}

    # sign+exp joint symbol (what 0009 codebooks)
    se = (sign << 8) | exp

    # ---------- A: true joint context on exponent ----------
    e = exp[1:, 1:]
    left = exp[1:, :-1]
    up = exp[:-1, 1:]
    ed, ne = remap(e.ravel())
    ld, nl = remap(left.ravel())
    ud, nu = remap(up.ravel())
    hashed = ((left ^ (up * 131)) & 0xFF).ravel()
    hd, nh = remap(hashed)
    joint_ctx = ld * nu + ud
    rows = np.repeat(np.arange(1, R), C - 1)
    mtr = (rows % 2) == 0
    out["A"] = {
        "H0_exp": H0(exp.ravel().astype(np.int64)),
        "H_left_plugin": cond_plugin(ed, ld),
        "H_hashed_plugin": cond_plugin(ed, hd),
        "H_joint_plugin": cond_plugin(ed, joint_ctx),
        "H_left_holdout": cond_holdout(ed, ld, mtr),
        "H_hashed_holdout": cond_holdout(ed, hd, mtr),
        "H_joint_holdout": cond_holdout(ed, joint_ctx, mtr),
        "n_ctx_joint": int(nl * nu),
        "distinct_exp": ne,
    }

    # ---------- B: nonstationarity ----------
    ef = exp.ravel().astype(np.int64)
    ed2, _ = remap(ef)
    row_id = np.repeat(np.arange(R), C)
    col_id = np.tile(np.arange(C), R)
    T = 64
    tile_id = (row_id // T) * ((C + T - 1) // T) + (col_id // T)
    mtr2 = (col_id % 2) == 0  # split by column parity for row/tile conditioning
    out["B"] = {
        "H_row_plugin": cond_plugin(ed2, row_id),
        "H_col_plugin": cond_plugin(ed2, col_id),
        "H_tile64_plugin": cond_plugin(ed2, tile_id),
        "H_row_holdout": cond_holdout(ed2, row_id, mtr2),
        "H_tile64_holdout": cond_holdout(ed2, tile_id, mtr2),
    }

    # per-row vs per-tensor codebook escape (on sign+exp symbol, like 0009)
    sef = se.ravel()
    def esc_global(K):
        c = np.bincount(sef)
        top = np.argsort(c)[::-1][:K]
        cov = c[top].sum() / sef.size
        return float(1.0 - cov)
    def esc_perrow(K):
        misses = 0
        se2 = se.reshape(R, C)
        for r in range(R):
            c = np.bincount(se2[r])
            k = min(K, (c > 0).sum())
            top = np.argsort(c)[::-1][:k]
            misses += se2[r].size - c[top].sum()
        return float(misses / sef.size)
    def esc_perblock(K, B):
        misses = 0
        se2 = se.reshape(R, C)
        for r0 in range(0, R, B):
            blk = se2[r0:r0 + B].ravel()
            c = np.bincount(blk)
            k = min(K, (c > 0).sum())
            top = np.argsort(c)[::-1][:k]
            misses += blk.size - c[top].sum()
        return float(misses / sef.size)
    out["B"]["esc_K15_global"] = esc_global(15)
    out["B"]["esc_K15_perrow"] = esc_perrow(15)
    out["B"]["esc_K15_per64rows"] = esc_perblock(15, 64)
    out["B"]["esc_K7_global"] = esc_global(7)
    out["B"]["esc_K7_perrow"] = esc_perrow(7)
    out["B"]["esc_K7_per64rows"] = esc_perblock(7, 64)
    # side cost of per-64-row codebooks: 15 x 9 bits per block
    nblk = (R + 63) // 64
    out["B"]["side_bits_per_w_64row_K15"] = nblk * 15 * 9 / sef.size

    # ---------- C: sign ----------
    s = sign[1:, 1:].ravel()
    sl = sign[1:, :-1].ravel()
    su = sign[:-1, 1:].ravel()
    e_here = ed  # dense exp at same positions
    out["C"] = {
        "H_sign": H0(sign.ravel()),
        "H_sign_given_exp": cond_plugin(s, e_here),
        "H_sign_given_leftsign": cond_plugin(s, sl),
        "H_sign_given_upsign": cond_plugin(s, su),
        "H_sign_given_bothsigns": cond_plugin(s, sl * 2 + su),
        "H_sign_given_col": cond_plugin(sign.ravel(), col_id),
        "H_sign_given_exp_and_leftsign": cond_plugin(s, e_here * 2 + sl),
    }

    # ---------- D: recomputed joint floor ----------
    md, nm = remap(mant.ravel())
    out["D"] = {
        "H_mant_given_exp_plugin": cond_plugin(md, ed2),
        "H_mant_given_exp_holdout": cond_holdout(md, ed2, mtr2),
        "H_value16_order0": H0(u16.ravel().astype(np.int64)),
    }
    fl_old = out["D"]["H_value16_order0"]
    fl_new = (out["A"]["H_joint_holdout"] + out["C"]["H_sign_given_exp"]
              + out["D"]["H_mant_given_exp_holdout"])
    out["D"]["floor_order0_value"] = fl_old
    out["D"]["floor_ctx_joint"] = fl_new
    out["D"]["floor_ctx_reduction_pct_of_16"] = 100 * (1 - fl_new / 16)
    return out


def main():
    d = np.load(NPZ)
    res = []
    for t in ("up", "down"):
        shape = d[f"{t}__shape"]
        res.append(probe(t, d[f"{t}__raw_u16"], tuple(int(x) for x in shape)))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
