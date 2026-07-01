"""layers.py -- the swappable layer-family registry.

The whole point of this harness: hold the task, the model skeleton, and the
training loop FIXED, and vary only the *family* of weight matrix used for the
model's main maps. A "family" is a recipe for turning an (in, out, budget)
request into an nn.Module that maps (..., in) -> (..., out) using approximately
`budget` trainable parameters.

A later agent adds a new family by writing one function and decorating it with
@register("my_family"). Nothing else in the harness changes.

Contract for a family builder `f(in_features, out_features, budget, gen)`:
  * returns an nn.Module mapping (..., in_features) -> (..., out_features)
  * should aim to use ~`budget` trainable parameters (it may use fewer if its
    structure can't spend that many; the harness records the ACTUAL count)
  * must initialize its parameters using the provided torch.Generator `gen`
    so runs are reproducible
  * has NO nonlinearity inside -- it is a pure linear map. Nonlinearities live
    in the model skeleton, so every family is compared on equal footing.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

LAYER_FAMILIES: dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable):
        LAYER_FAMILIES[name] = fn
        return fn
    return deco


def make_linear(family: str, in_features: int, out_features: int,
                budget: int, gen: torch.Generator) -> nn.Module:
    """Factory: build a map in->out from the named family at ~`budget` params."""
    if family not in LAYER_FAMILIES:
        raise KeyError(
            f"unknown layer family {family!r}; "
            f"registered: {sorted(LAYER_FAMILIES)}"
        )
    return LAYER_FAMILIES[family](in_features, out_features, budget, gen)


def _kaiming_(w: torch.Tensor, fan_in: int, gen: torch.Generator):
    std = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        w.uniform_(-std, std, generator=gen)


# --------------------------------------------------------------------------- #
# DENSE baseline. A plain full-rank matrix. Params = in*out (bias-free).
# `budget` is advisory here: a dense map of a fixed interface has a fixed cost,
# and the run() harness sizes the architecture so that the dense family lands at
# the requested budget. Cheaper families reuse the same interface for less.
# --------------------------------------------------------------------------- #
class Dense(nn.Module):
    def __init__(self, in_features: int, out_features: int, gen: torch.Generator):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        _kaiming_(self.weight, in_features, gen)

    def forward(self, x):
        return x @ self.weight.t()


@register("dense")
def _dense(in_features, out_features, budget, gen):
    return Dense(in_features, out_features, gen)


# --------------------------------------------------------------------------- #
# LOW-RANK family (included to prove the swap + budget mechanism works end to
# end; it is NOT part of the verified dense baseline). Factorizes the map as
# U @ V with inner rank r chosen so (in+out)*r ~ budget. At budget >= in*out it
# is effectively full rank. This is the natural "compressed" comparison point.
# --------------------------------------------------------------------------- #
class LowRank(nn.Module):
    def __init__(self, in_features, out_features, rank, gen):
        super().__init__()
        self.U = nn.Parameter(torch.empty(in_features, rank))
        self.V = nn.Parameter(torch.empty(rank, out_features))
        _kaiming_(self.U, in_features, gen)
        _kaiming_(self.V, rank, gen)

    def forward(self, x):
        return (x @ self.U) @ self.V


@register("lowrank")
def _lowrank(in_features, out_features, budget, gen):
    max_rank = min(in_features, out_features)
    rank = max(1, min(max_rank, round(budget / (in_features + out_features))))
    return LowRank(in_features, out_features, rank, gen)


# --------------------------------------------------------------------------- #
# BLOCK_MONARCH family. Monarch-style structured linear map:
#
#     y = (P2  Bdiag(W2)  P1  Bdiag(W1)) x
#
# i.e. a product of two BLOCK-DIAGONAL factors with a fixed permutation between
# them. Each factor only mixes coordinates *within* its blocks; the permutation
# in the middle reshuffles coordinates so that the second factor's blocks pool
# information from *different* first-factor blocks. The composition is therefore
# dense in effect (every output can depend on every input) while storing only
# O((in+out)*m / b) parameters instead of in*out.
#
# This is strictly more expressive than a low-rank map of equal parameter count:
# low-rank forces ALL information through a single shared `rank`-dim bottleneck,
# whereas Monarch routes it through `b` independent block-bottlenecks that the
# permutation cross-wires -- a block-structured (not rank-1-sum) factorization.
#
# Budget control: with `b` blocks and internal width `m`, the parameter cost is
#     params = m*(in_p + out_p) / b
# (in_p/out_p are in/out padded up to a multiple of b). We fix a small `b` and
# solve for the internal width `m` that lands nearest the requested budget.
#
# NATIVE COMPUTE: forward() runs two batched per-block matmuls plus a
# reshape/transpose permutation. The full dense (out x in) matrix is NEVER
# materialized -- compute and memory stay on the two small factor tensors.
# --------------------------------------------------------------------------- #
class BlockMonarch(nn.Module):
    def __init__(self, in_features, out_features, nblocks, inner, gen):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        b = nblocks
        self.b = b
        # pad each dimension up to a multiple of b so blocks divide evenly
        self.in_p = ((in_features + b - 1) // b) * b
        self.out_p = ((out_features + b - 1) // b) * b
        self.m_p = max(b, ((inner + b - 1) // b) * b)

        self.in_blk = self.in_p // b      # cols per first-factor block
        self.mid_blk = self.m_p // b      # rows per first-factor block
        self.out_blk = self.out_p // b    # rows per second-factor block

        # Factor 1: b independent blocks, each (mid_blk x in_blk).
        self.W1 = nn.Parameter(torch.empty(b, self.mid_blk, self.in_blk))
        # Factor 2: b independent blocks, each (out_blk x mid_blk), applied
        # after the middle permutation regroups the m_p coordinates.
        self.W2 = nn.Parameter(torch.empty(b, self.out_blk, self.mid_blk))
        _kaiming_(self.W1, self.in_blk, gen)
        _kaiming_(self.W2, self.mid_blk, gen)

    def forward(self, x):
        lead = x.shape[:-1]
        f = x.shape[-1]
        if f < self.in_p:                       # pad input to in_p
            x = torch.nn.functional.pad(x, (0, self.in_p - f))
        # --- Factor 1: block-diagonal, per-block matmul --------------------
        x = x.reshape(*lead, self.b, self.in_blk)
        x = torch.einsum("...bi,boi->...bo", x, self.W1)   # (..., b, mid_blk)
        # --- Monarch permutation: transpose the (block, within-block) grid -
        x = x.transpose(-1, -2).reshape(*lead, self.m_p)   # cross-wire blocks
        # --- Factor 2: block-diagonal over the regrouped coordinates --------
        x = x.reshape(*lead, self.b, self.mid_blk)
        x = torch.einsum("...bi,boi->...bo", x, self.W2)   # (..., b, out_blk)
        x = x.reshape(*lead, self.out_p)
        if self.out_p > self.out_features:                 # drop padding rows
            x = x[..., :self.out_features]
        return x


@register("block_monarch")
def _block_monarch(in_features, out_features, budget, gen):
    # Pick a small block count so each block stays non-degenerate, then solve
    # for the internal width m that lands the parameter cost near `budget`.
    b = max(2, min(8, int(round(math.sqrt(min(in_features, out_features) / 2)))))
    in_p = ((in_features + b - 1) // b) * b
    out_p = ((out_features + b - 1) // b) * b
    # params = m*(in_p+out_p)/b  ->  m = budget*b/(in_p+out_p)
    inner = max(b, int(round(budget * b / (in_p + out_p))))
    return BlockMonarch(in_features, out_features, b, inner, gen)


# --------------------------------------------------------------------------- #
# SHARED-DICTIONARY family ("shared_dict").
#
# Idea (train-time version of the post-hoc "shared basis" compression): instead
# of every FFN matrix carrying its own out*in weights, ALL the swappable matrix
# sites in the model are reconstructed from ONE shared trained dictionary of K
# atoms. Each atom is a full flattened matrix (length M = out*in). Every site i
# owns only a tiny coefficient vector c_i in R^K and rebuilds its matrix as
#
#       W_i = (c_i @ D).reshape(out, in)            D: (K, M) shared
#
# So the K atoms are shared across all sites; the per-site cost is just K
# coefficients. Param budget is controlled by K:
#       params = K*M (dictionary, counted ONCE)  +  K * n_sites (coefficients).
#
# In this harness there are 4 swappable sites (fc1+fc2 in each of 2 blocks).
# fc1 is (d_ff, d_model) and fc2 is (d_model, d_ff): different shapes but the
# SAME element count M = d_model*d_ff, so a single flat dictionary serves all 4.
# With n_sites=4, K=4 already spans any 4 matrices (the sweep's K<4 points are
# the genuine-compression regime; K>4 is pure over-parameterization).
#
# K is read from the module-level SHARED_DICT_K (set by the sweep driver). The
# dictionary must be SHARED across the 4 independent make_linear() calls of one
# model build, so it is cached keyed by the init generator identity (+ M, K) and
# reset between runs via reset_shared_dict(). The shared Parameter is registered
# on every site that uses it; PyTorch's parameters()/named_parameters() dedupe
# shared Parameters, so model.total_params() and the optimizer count it once.
# (Note: model.swappable_params() iterates each block.ffn separately and so
# DOUBLE-COUNTS the cross-block shared dictionary; total_params() is correct and
# is what we report.)
#
# NATIVE COMPUTE: forward() reconstructs (materializes) this site's dense matrix
# W = (c_i @ D).reshape(out,in) once, then does a standard matmul. A
# non-materializing form exists -- y = sum_k c_k (x @ A_k^T) over atom matrices
# A_k -- but it costs K matmuls instead of 1, so materialization is the sensible
# path. shared_dict is therefore a STORAGE/parameter-sharing compression, not a
# native-compute (matrix-free) family.
# --------------------------------------------------------------------------- #
SHARED_DICT_K: int | None = None       # set by the sweep driver; None -> auto
_SHARED_DICT_CACHE: dict = {}


def reset_shared_dict():
    """Clear the per-build shared-dictionary cache. Call before each run()."""
    _SHARED_DICT_CACHE.clear()


class SharedDictLinear(nn.Module):
    def __init__(self, in_features, out_features, D, gen):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.D = D                          # shared (K, M) Parameter
        K = D.shape[0]
        coeff = torch.empty(K)
        with torch.no_grad():
            coeff.normal_(0.0, 1.0 / math.sqrt(K), generator=gen)
        self.coeff = nn.Parameter(coeff)    # per-site, NOT shared

    def forward(self, x):
        W = (self.coeff @ self.D).view(self.out_features, self.in_features)
        return x @ W.t()


@register("shared_dict")
def _shared_dict(in_features, out_features, budget, gen):
    M = in_features * out_features
    K = SHARED_DICT_K
    if K is None:
        K = max(1, round(budget / M))       # default ~1 atom at the dense budget
    K = max(1, int(K))
    key = (id(gen), M, K)
    D = _SHARED_DICT_CACHE.get(key)
    if D is None:
        d = torch.empty(K, M)
        with torch.no_grad():
            # init so a reconstructed W has ~kaiming scale: var(W)=K*var(c)*var(D)
            # with var(c)=1/K and var(D)=1/in_features -> var(W)=1/in_features.
            d.normal_(0.0, 1.0 / math.sqrt(in_features), generator=gen)
        D = nn.Parameter(d)
        _SHARED_DICT_CACHE[key] = D
    return SharedDictLinear(in_features, out_features, D, gen)


# --------------------------------------------------------------------------- #
# SPARSE_SUPERPOSE family (exp3) -- a FULL FFN, not a pure linear map.
#
# Hypothesis under test (exp1+exp2 synthesis): structural weight-sharing fails
# per-param because DENSE FFN activations interfere; but the superposition toy
# showed SPARSE activations let d dims carry ~10-16x more features without
# interference. So a WIDE hidden FFN that is (a) read out with TOP-K SPARSE
# activation and (b) built from a SHARED ATOM POOL (superposed input weights)
# might beat dense capability-per-param, because sparsity makes the superposed
# atoms non-interfering. Train-time mirror of "finer sparse experts sharing a
# parameter pool" (the real MoE).
#
# Structure (maps d_model -> d_model; owns its nonlinearity, so it is a full FFN
# and sets is_full_ffn=True so model.FeedForward uses it alone):
#
#     z   = x @ A^T                      A : (K, d_in)   shared atom pool
#     pre = z @ C^T  (+ b1)              C : (H, K)      per-hidden-unit mixes
#     h   = gelu(pre)                    H hidden units (H >> dense d_ff)
#     s   = top_k(h) over H per token    keep k_active, zero the rest  (SPARSE)
#     y   = s @ W2^T (+ b2)              W2: (d_out, H)  output projection
#
# The H input weights are W1[H,d_in] = C @ A but W1 is NEVER materialized: the
# forward computes (x @ A^T) @ C^T natively. Storage of the expansion is
# K*d_in + H*K instead of H*d_in -- the superposition saving (H>>K).
#
# W2 may be DENSE (params d_out*H) or LOW-RANK / atom-shared
# (W2 = G @ E, G:(d_out,Kw) E:(Kw,H), params d_out*Kw + Kw*H). Dense W2 is the
# simple/correct default but caps H near ~2x the dense d_ff at matched params
# (W2 dominates the budget); a low-rank W2 (set SP_W2_RANK) lets H reach 4-16x to
# genuinely test wide superposition.
#
# SPARSITY is a FORWARD-TIME mask: k_active does NOT change the parameter count,
# so we can hold (H,K,W2) -- hence params -- FIXED and sweep the sparsity
# fraction k_active/H to directly test "more sparsity helps superposition".
#
# Config is read from module-level globals (set by the sweep driver), mirroring
# the SHARED_DICT_K pattern, because the builder signature is fixed:
#   SP_H        hidden width H
#   SP_K        atom-pool size K (input dictionary)
#   SP_SPARSITY k_active / H   (fraction of H kept per token; 1.0 = dense act)
#   SP_W2_RANK  None -> dense W2; int -> low-rank/atom-shared W2 of that rank
# --------------------------------------------------------------------------- #
SP_H: int | None = None
SP_K: int | None = None
SP_SPARSITY: float = 1.0
SP_W2_RANK: int | None = None


class SparseSuperposeFFN(nn.Module):
    is_full_ffn = True

    def __init__(self, d_model, H, K, sparsity, w2_rank, gen):
        super().__init__()
        self.d_in = d_model
        self.d_out = d_model
        self.H = H
        self.K = K
        self.k_active = max(1, min(H, int(round(sparsity * H))))

        # --- shared atom pool A (K, d_in) and per-unit mix C (H, K) ---------- #
        A = torch.empty(K, d_model)
        C = torch.empty(H, K)
        with torch.no_grad():
            # init so pre = (x@A^T)@C^T has ~unit-fan-in scale on each hidden unit.
            # var(pre_h) = d_in * var(x) * [ var over K of (sum_k C_hk A_k.) ]
            # Choose var(A)=1/d_in and var(C)=1/K -> effective fan-in ~1. This
            # matches a kaiming W1[H,d_in] with fan_in=d_in in expectation.
            A.normal_(0.0, 1.0 / math.sqrt(d_model), generator=gen)
            C.normal_(0.0, 1.0 / math.sqrt(K), generator=gen)
        self.A = nn.Parameter(A)
        self.C = nn.Parameter(C)
        self.b1 = nn.Parameter(torch.zeros(H))

        # --- output projection W2: dense or low-rank ------------------------- #
        self.w2_rank = w2_rank
        if w2_rank is None:
            W2 = torch.empty(d_model, H)
            _kaiming_(W2, H, gen)            # fan_in = H (k_active active in use)
            self.W2 = nn.Parameter(W2)
        else:
            # W2 = G @ E,  G:(d_out,Kw)  E:(Kw,H).  y = (s @ E^T) @ G^T
            E = torch.empty(w2_rank, H)
            G = torch.empty(d_model, w2_rank)
            _kaiming_(E, H, gen)
            _kaiming_(G, w2_rank, gen)
            self.E = nn.Parameter(E)
            self.G = nn.Parameter(G)
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # native expansion: NEVER materialize W1[H,d_in] = C @ A
        z = x @ self.A.t()                       # (..., K)
        pre = z @ self.C.t() + self.b1           # (..., H)
        h = torch.nn.functional.gelu(pre)        # nonlinearity
        # top-k SPARSE activation over the H hidden units, per token
        if self.k_active < self.H:
            thresh = torch.topk(h, self.k_active, dim=-1).values[..., -1:]
            h = torch.where(h >= thresh, h, torch.zeros_like(h))
        # output projection
        if self.w2_rank is None:
            y = h @ self.W2.t() + self.b2        # (..., d_out)
        else:
            y = (h @ self.E.t()) @ self.G.t() + self.b2
        return y

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def sparse_superpose_params(d_model, H, K, w2_rank):
    """Honest total param count for one SparseSuperposeFFN module."""
    p = K * d_model + H * K + H            # A + C + b1
    if w2_rank is None:
        p += d_model * H                   # dense W2
    else:
        p += w2_rank * H + d_model * w2_rank  # E + G
    p += d_model                          # b2
    return p


@register("sparse_superpose")
def _sparse_superpose(in_features, out_features, budget, gen):
    # in_features == d_model; out_features (the dense d_ff) is IGNORED -- this is
    # a full FFN mapping d_model -> d_model with its own wide hidden layer.
    d_model = in_features
    H = SP_H
    K = SP_K
    if H is None:
        # default: spend the whole per-module budget on a dense-W2 wide layer.
        # per-module budget = 2*budget (the FFN replaces fc1+fc2; budget here is
        # the per-MATRIX dense cost = d_model*d_ff = half a block's FFN).
        H = max(8, int(round(2 * budget / (2 * d_model))))
    if K is None:
        K = max(1, min(d_model, H // 8))
    return SparseSuperposeFFN(d_model, int(H), int(K), SP_SPARSITY, SP_W2_RANK, gen)


# --------------------------------------------------------------------------- #
# SPARSE_SUPERPOSE_V2 family (exp4) -- the FAIR test, fixing exp3's confound.
#
# exp3 confound: input-side atom sharing made W1 = C@A rank-K (only K input
# directions), so widening the hidden layer just stacked units inside a K-dim
# subspace -- a bottleneck, not superposition. exp4 fixes BOTH sides:
#
#   1. W1 is a FULL-RANK dense up-projection  W1[H, d_in]  (a genuine parameter,
#      stored directly, NOT reconstructed from factors). Every one of the H hidden
#      units reads ALL d_in input directions -- no input bottleneck.
#
#   2. The SHARING/superposition lives on the HIDDEN->OUTPUT side. The H hidden
#      units' output vectors (columns of W2[d_out, H]) are drawn from a SHARED
#      dictionary of M < H atoms:   W2 = D[d_out, M] @ S[M, H].
#      So a WIDE hidden layer is cheap on the output side: M*H + d_out*M instead
#      of d_out*H. The H units superpose onto M shared output directions.
#
#   3. TOP-K SPARSE activation over the H hidden units (k_active = sparsity*H),
#      the lever that (per the exp2 toy) makes superposed directions
#      non-interfering. Sparsity is a forward-time mask -> does NOT change params,
#      so we hold (H, M) -- hence params -- fixed and sweep k/H.
#
# NATIVE COMPUTE:
#   * up-projection: pre = x @ W1.t()  -- W1[H,d_in] is the stored full-rank
#     parameter (this IS the honest representation of a full-rank up-proj; there
#     is no factorization to avoid here).
#   * output: y = (h @ S.t()) @ D.t()  -- the dense W2 = D@S [d_out,H] is NEVER
#     materialized; compute stays on the two small factors.
#
# Config from module-level globals (set by the sweep driver), like exp3:
#   SPV2_H        hidden width H (full-rank up-proj rows)
#   SPV2_M        output-dictionary size M (< H; shared output atoms)
#   SPV2_SPARSITY k_active / H  (fraction of H kept per token; 1.0 = dense act)
# --------------------------------------------------------------------------- #
SPV2_H: int | None = None
SPV2_M: int | None = None
SPV2_SPARSITY: float = 1.0


class SparseSuperposeV2FFN(nn.Module):
    is_full_ffn = True

    def __init__(self, d_model, H, M, sparsity, gen):
        super().__init__()
        self.d_in = d_model
        self.d_out = d_model
        self.H = H
        self.M = M
        self.k_active = max(1, min(H, int(round(sparsity * H))))

        # --- FULL-RANK dense up-projection W1[H, d_in] (genuine parameter) ----- #
        W1 = torch.empty(H, d_model)
        _kaiming_(W1, d_model, gen)          # fan_in = d_in
        self.W1 = nn.Parameter(W1)
        self.b1 = nn.Parameter(torch.zeros(H))

        # --- shared output dictionary  W2 = D[d_out,M] @ S[M,H]  --------------- #
        # init so reconstructed W2 has ~kaiming scale fan_in=H:
        #   var(W2_oh) = M * var(D) * var(S);  choose var(D)=1/M, var(S)=1/H
        #   -> var(W2) = 1/H, matching _kaiming_(W2, fan_in=H).
        D = torch.empty(d_model, M)
        S = torch.empty(M, H)
        with torch.no_grad():
            D.normal_(0.0, 1.0 / math.sqrt(M), generator=gen)
            S.normal_(0.0, 1.0 / math.sqrt(H), generator=gen)
        self.D = nn.Parameter(D)
        self.S = nn.Parameter(S)
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        pre = x @ self.W1.t() + self.b1          # (..., H)  full-rank up-proj
        h = torch.nn.functional.gelu(pre)
        # top-k SPARSE activation over the H hidden units, per token
        if self.k_active < self.H:
            thresh = torch.topk(h, self.k_active, dim=-1).values[..., -1:]
            h = torch.where(h >= thresh, h, torch.zeros_like(h))
        # native factored output: NEVER materialize dense W2 = D@S [d_out,H]
        y = (h @ self.S.t()) @ self.D.t() + self.b2
        return y

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def sparse_superpose_v2_params(d_model, H, M):
    """Honest total param count for one SparseSuperposeV2FFN module."""
    return (H * d_model + H              # W1 + b1
            + d_model * M + M * H        # D + S
            + d_model)                   # b2


@register("sparse_superpose_v2")
def _sparse_superpose_v2(in_features, out_features, budget, gen):
    # in_features == d_model; out_features (dense d_ff) IGNORED -- full FFN.
    d_model = in_features
    H = SPV2_H
    M = SPV2_M
    if H is None:
        H = max(8, int(round(2 * budget / (2 * d_model))))
    if M is None:
        M = max(1, min(H - 1, d_model // 4))
    return SparseSuperposeV2FFN(d_model, int(H), int(M), SPV2_SPARSITY, gen)
