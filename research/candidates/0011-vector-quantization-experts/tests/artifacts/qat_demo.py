"""Does QAT break the post-hoc rate-distortion wall? A cheap, local, decisive test.

Trains the project's own convergence-tested CharTransformer (real text) to FP, then
compares — at 2/3/4-bit on the transformer's Linear weights (the FFN+attn bulk, analog of
the 30B experts; embeddings/head/norms kept FP = the non-expert floor):
  * POST-HOC : quantize the trained FP weights, eval. (the wall we measured on the 30B)
  * QAT      : train from scratch with straight-through fake-quant so downstream layers
               co-adapt and the optimizer finds a low-bit-representable point on the
               function manifold.
If QAT(2-bit) << POST-HOC(2-bit) and approaches FP, the training lever breaks the wall,
justifying the GPU spend to apply it to the 30B. If not, 90% is unreachable by any means.
"""
from __future__ import annotations
import sys, json, time, copy
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.parametrize as P

EXP4 = Path(__file__).resolve().parents[4] / "traintime" / "exp4"
sys.path.insert(0, str(EXP4))
from model import CharTransformer, ModelConfig  # noqa
from task import build_data                      # noqa

SEQ_LEN, D_MODEL, N_BLOCKS, N_HEADS, D_FF = 96, 128, 2, 4, 512
BATCH, STEPS, LR = 32, 800, 3e-3


def per_channel_quant(W, bits):
    maxq = 2 ** (bits - 1) - 1
    s = (W.abs().amax(dim=1, keepdim=True) / maxq).clamp_min(1e-12)   # per output row
    return torch.clamp(torch.round(W / s), -maxq, maxq) * s


class FakeQuant(nn.Module):
    """STE fake-quant: forward returns exactly Wq; gradient flows to W (identity)."""
    def __init__(self, bits):
        super().__init__()
        self.bits = bits

    def forward(self, W):
        Wq = per_channel_quant(W, self.bits)
        return W + (Wq - W).detach()


def target_linears(model):
    # quantize the transformer-block Linear weights (attn qkv/proj + FFN fc1/fc2);
    # leave token/pos embeddings and the LM head full precision (= the non-expert floor).
    outs = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and "head" not in name:
            outs.append((name, m))
    return outs


def quantize_bits(model, bits):
    for _, m in target_linears(model):
        P.register_parametrization(m, "weight", FakeQuant(bits))


def eval_loss(model, task, gen, batches=30):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = task.get_batch("val", BATCH, gen)
            _, loss = model(x, y)
            tot += loss.item()
    return tot / batches


def train(model, task, steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()
    for step in range(steps):
        lr = LR * min(1.0, (step + 1) / 50)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = task.get_batch("train", BATCH, g)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def build(task, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    cfg = ModelConfig(vocab_size=task.vocab_size, seq_len=SEQ_LEN, d_model=D_MODEL,
                      n_blocks=N_BLOCKS, n_heads=N_HEADS, d_ff=D_FF, family="dense")
    return CharTransformer(cfg, gen)


if __name__ == "__main__":
    t0 = time.time()
    task = build_data(seq_len=SEQ_LEN, seed=1234)
    evg = torch.Generator().manual_seed(999)

    # 1) FP reference
    fp = build(task, seed=0)
    train(fp, task, STEPS, seed=0)
    fp_loss = eval_loss(fp, task, evg)
    print(json.dumps({"cond": "FP", "bits": 16, "val_loss": round(fp_loss, 4)}))

    rows = [{"cond": "FP", "bits": 16, "val_loss": round(fp_loss, 4)}]
    for bits in (4, 3, 2):
        # POST-HOC: quantize the trained FP model
        ph = copy.deepcopy(fp)
        quantize_bits(ph, bits)
        ph_loss = eval_loss(ph, task, evg)
        # QAT: fresh model, train with fake-quant in the loop
        q = build(task, seed=0)
        quantize_bits(q, bits)
        train(q, task, STEPS, seed=0)
        qat_loss = eval_loss(q, task, evg)
        rec = {"cond": "quant", "bits": bits,
               "post_hoc_val_loss": round(ph_loss, 4),
               "qat_val_loss": round(qat_loss, 4),
               "fp_val_loss": round(fp_loss, 4),
               "qat_recovers_%": round(100 * (ph_loss - qat_loss) / max(1e-9, ph_loss - fp_loss), 1)}
        rows.append(rec)
        print(json.dumps(rec))
    print(f"elapsed {time.time()-t0:.1f}s")
    (Path(__file__).parent / "qat_demo_result.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
