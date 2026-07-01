"""Push QAT to the 90% frontier on a real trained transformer: how low can bits go
(down to ternary 1.58b / binary 1b), over how much of the model, before quality breaks?

Reports the COMBINED compression % (actual param-weighted bits vs 16) at each rung and the
val loss vs FP, so we can read the max combined reduction QAT sustains at good quality.
One config per invocation: --bits {4,3,2,1.58,1} --scope {bulk,all}. bulk = transformer-block
Linears (attn+FFN, ~93% of params, the analog of the 30B experts); all = also embeddings+head.
"""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.utils.parametrize as P

EXP4 = Path(__file__).resolve().parents[4] / "traintime" / "exp4"
sys.path.insert(0, str(EXP4))
from model import CharTransformer, ModelConfig  # noqa
from task import build_data                      # noqa

SEQ_LEN, D_MODEL, N_BLOCKS, N_HEADS, D_FF = 96, 128, 2, 4, 512
BATCH, STEPS, LR = 32, 800, 3e-3


def fake_quant(W, bits):
    if bits >= 2:                     # per-output-channel symmetric integer
        maxq = 2 ** (int(bits) - 1) - 1
        s = (W.abs().amax(dim=1, keepdim=True) / maxq).clamp_min(1e-12)
        return torch.clamp(torch.round(W / s), -maxq, maxq) * s
    if abs(bits - 1.58) < 0.05:       # BitNet b1.58 ternary {-1,0,1}, per-tensor absmean
        s = W.abs().mean().clamp_min(1e-8)
        return torch.clamp(torch.round(W / s), -1, 1) * s
    # binary {-1,+1}, per-tensor absmean
    s = W.abs().mean().clamp_min(1e-8)
    return torch.where(W >= 0, s, -s)


class FakeQuant(nn.Module):
    def __init__(self, bits):
        super().__init__(); self.bits = bits

    def forward(self, W):
        Wq = fake_quant(W, self.bits)
        return W + (Wq - W).detach()          # STE


def targets(model, scope):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if scope == "all" or "head" not in name:
                yield name, m


def bits_of(name, scope, bits):
    if isinstance(bits, float) and bits == int(bits):
        bits = int(bits)
    return bits


def combined_pct(model, scope, bits):
    q_params, fp_params = 0, 0
    quant_names = {id(m) for _, m in targets(model, scope)}
    # embeddings only quantized if scope == 'all'
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and id(mod) in quant_names:
            q_params += mod.weight.numel()
        elif isinstance(mod, nn.Embedding) and scope == "all":
            q_params += mod.weight.numel()
        elif isinstance(mod, (nn.Linear, nn.Embedding)):
            fp_params += sum(p.numel() for p in mod.parameters())
    total = q_params + fp_params
    remaining = (q_params * bits + fp_params * 16) / (total * 16)
    return round(100 * (1 - remaining), 2), q_params, fp_params


def quantize_embeddings(model, bits):
    for _, m in model.named_modules():
        if isinstance(m, nn.Embedding):
            P.register_parametrization(m, "weight", FakeQuant(bits))


def build(task, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    cfg = ModelConfig(vocab_size=task.vocab_size, seq_len=SEQ_LEN, d_model=D_MODEL,
                      n_blocks=N_BLOCKS, n_heads=N_HEADS, d_ff=D_FF, family="dense")
    return CharTransformer(cfg, gen)


def eval_loss(model, task, gen, batches=30):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = task.get_batch("val", BATCH, gen)
            _, loss = model(x, y); tot += loss.item()
    return tot / batches


def train(model, task, steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, (step + 1) / 50)
        x, y = task.get_batch("train", BATCH, g)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=float, required=True)
    ap.add_argument("--scope", choices=["bulk", "all"], default="bulk")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--fp", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    task = build_data(seq_len=SEQ_LEN, seed=1234)
    evg = torch.Generator().manual_seed(999)

    if a.fp:
        m = build(task); train(m, task, a.steps);
        print(json.dumps({"cond": "FP", "val_loss": round(eval_loss(m, task, evg), 4)})); sys.exit()

    m = build(task, seed=0)
    for _, mod in targets(m, a.scope):
        P.register_parametrization(mod, "weight", FakeQuant(a.bits))
    if a.scope == "all":
        quantize_embeddings(m, a.bits)
    train(m, task, a.steps, seed=0)
    vl = eval_loss(m, task, evg)
    pct, qp, fpp = combined_pct(m, a.scope, a.bits)
    print(json.dumps({"bits": a.bits, "scope": a.scope, "qat_val_loss": round(vl, 4),
                      "combined_pct": pct, "q_params": qp, "fp_params": fpp,
                      "secs": round(time.time() - t0, 1)}))
