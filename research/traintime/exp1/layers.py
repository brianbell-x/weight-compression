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
