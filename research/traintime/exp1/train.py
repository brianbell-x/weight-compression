"""train.py -- the FIXED training loop and the single run() entry point.

    run(layer_family, param_budget, seed) -> {params, final_loss, val_loss, ...}

Everything that is not the swapped layer family is held constant: optimizer,
learning-rate schedule, batch size, sequence length, number of steps, data, and
seeds. This makes loss differences attributable to the layer family at the given
parameter budget -- i.e. a capability-per-parameter measurement.
"""

from __future__ import annotations

import math
import time

import torch

from task import build_data
from model import CharTransformer, ModelConfig, derive_d_ff

# ---- fixed training hyperparameters (identical for every family / budget) --- #
SEQ_LEN = 64
D_MODEL = 96
N_BLOCKS = 2
N_HEADS = 3
BATCH_SIZE = 32
TRAIN_STEPS = 800
LEARNING_RATE = 3e-3
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
EVAL_BATCHES = 20
LOSS_SMOOTH = 50          # final_loss = mean train loss over last N steps


def _set_all_seeds(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)


def _lr_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    prog = (step - WARMUP_STEPS) / max(1, TRAIN_STEPS - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * prog))


@torch.no_grad()
def _eval_val(model, task, gen) -> float:
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = task.get_batch("val", BATCH_SIZE, gen)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def run(layer_family: str = "dense", param_budget: int = 200_000,
        seed: int = 0, verbose: bool = False) -> dict:
    """Train one configuration and return its metrics.

    Args:
        layer_family: name registered in layers.LAYER_FAMILIES (e.g. "dense").
        param_budget: target parameter count for the swappable FFN matrices
                      (the architecture's FFN width is sized so the DENSE family
                      lands at this budget; other families reuse the interface).
        seed: full determinism seed.

    Returns dict with:
        params         total model parameters
        swappable_params  params in the swapped FFN matrices (the studied part)
        final_loss     mean train cross-entropy (nats) over last LOSS_SMOOTH steps
        val_loss       held-out cross-entropy (nats)
        final_bpc / val_bpc   the same in bits/char
        d_ff, steps, wall_s, uniform_bpc   bookkeeping
    """
    _set_all_seeds(seed)
    task = build_data(seq_len=SEQ_LEN, seed=1234)  # corpus seed fixed, not run seed

    d_ff = derive_d_ff(param_budget, D_MODEL, N_BLOCKS)
    cfg = ModelConfig(
        vocab_size=task.vocab_size, seq_len=SEQ_LEN, d_model=D_MODEL,
        n_blocks=N_BLOCKS, n_heads=N_HEADS, d_ff=d_ff,
        family=layer_family, matrix_budget=D_MODEL * d_ff,
    )

    init_gen = torch.Generator().manual_seed(seed + 1)
    model = CharTransformer(cfg, init_gen)

    batch_gen = torch.Generator().manual_seed(seed + 2)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99))

    model.train()
    t0 = time.time()
    recent = []
    for step in range(TRAIN_STEPS):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step)
        x, y = task.get_batch("train", BATCH_SIZE, batch_gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        recent.append(loss.item())
        if len(recent) > LOSS_SMOOTH:
            recent.pop(0)
        if verbose and (step % 100 == 0 or step == TRAIN_STEPS - 1):
            print(f"  step {step:4d}  lr {_lr_at(step):.4f}  loss {loss.item():.4f}")
    wall = time.time() - t0

    final_loss = sum(recent) / len(recent)
    val_loss = _eval_val(model, task, batch_gen)
    ln2 = math.log(2)
    return {
        "layer_family": layer_family,
        "param_budget": param_budget,
        "seed": seed,
        "params": model.total_params(),
        "swappable_params": model.swappable_params(),
        "d_ff": d_ff,
        "steps": TRAIN_STEPS,
        "final_loss": final_loss,
        "val_loss": val_loss,
        "final_bpc": final_loss / ln2,
        "val_bpc": val_loss / ln2,
        "uniform_bpc": math.log2(task.vocab_size),
        "wall_s": wall,
    }


if __name__ == "__main__":
    import json
    res = run("dense", 200_000, seed=0, verbose=True)
    print(json.dumps(res, indent=2))
