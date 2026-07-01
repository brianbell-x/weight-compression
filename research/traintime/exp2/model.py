"""model.py -- the FIXED model skeleton.

A small causal Transformer for char-level LM. Everything is held constant across
experiments EXCEPT the two big feed-forward (FFN) projection matrices in each
block, which are produced by the swappable layers.make_linear factory. The FFN
is where most transformer parameters live and where a weight matrix's expressive
capacity is most directly exercised, so it is the natural knob for a
capability-per-parameter study.

Held fixed across all families:
  * token + positional embeddings
  * causal multi-head self-attention (standard nn.Linear projections)
  * layernorms, residual connections, nonlinearity (GELU)
  * the LM head
Swapped per family / budget:
  * FFN matrix 1:  d_model -> d_ff
  * FFN matrix 2:  d_ff    -> d_model
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import make_linear


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int = 64
    d_model: int = 96
    n_blocks: int = 2
    n_heads: int = 3
    d_ff: int = 512
    family: str = "dense"
    matrix_budget: int = 0   # target params PER swappable matrix


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(
            1, 1, cfg.seq_len, cfg.seq_len)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class FeedForward(nn.Module):
    """The swappable part: two make_linear maps with a GELU between them."""

    def __init__(self, cfg: ModelConfig, gen: torch.Generator):
        super().__init__()
        self.fc1 = make_linear(cfg.family, cfg.d_model, cfg.d_ff,
                               cfg.matrix_budget, gen)
        self.fc2 = make_linear(cfg.family, cfg.d_ff, cfg.d_model,
                               cfg.matrix_budget, gen)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, gen: torch.Generator):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg, gen)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, gen: torch.Generator):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, gen) for _ in range(cfg.n_blocks)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.register_buffer("pos_ids", torch.arange(cfg.seq_len).unsqueeze(0))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(self.pos_ids[:, :T])
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # --- parameter accounting -------------------------------------------- #
    def swappable_params(self) -> int:
        return sum(p.numel() for blk in self.blocks
                   for p in blk.ffn.parameters())

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def derive_d_ff(param_budget: int, d_model: int, n_blocks: int) -> int:
    """Choose the FFN width so the DENSE swappable cost ~ param_budget.

    Dense FFN cost per block = d_model*d_ff (fc1) + d_ff*d_model (fc2)
                             = 2*d_model*d_ff.
    Summed over n_blocks and set equal to param_budget -> solve for d_ff.
    """
    d_ff = round(param_budget / (2 * d_model * n_blocks))
    return max(8, d_ff)
