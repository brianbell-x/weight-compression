"""dense_sweep.py -- verify exp2 is CAPACITY-BOUND.

Runs the DENSE family at four param budgets spanning ~0.3x..2x of the ~290k
reference and reports the val-loss curve. Capacity-bound iff val loss decreases
clearly and monotonically across the sweep (target spread >= ~0.1 nats).

    uv run python dense_sweep.py
"""

from __future__ import annotations

import time

from train import run

BUDGETS = [90_000, 180_000, 350_000, 580_000]   # ~0.3x .. 2x of 290k


def main() -> None:
    rows = []
    for b in BUDGETS:
        t0 = time.time()
        r = run("dense", b, seed=0)
        wall = time.time() - t0
        rows.append(r)
        print(f"budget={b:>7d}  params={r['params']:>7d}  d_ff={r['d_ff']:>4d}  "
              f"val_loss={r['val_loss']:.4f}  val_bpc={r['val_bpc']:.3f}  "
              f"wall={wall:.1f}s", flush=True)
    vals = [r["val_loss"] for r in rows]
    spread = max(vals) - min(vals)
    mono = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    print(f"\nval-loss spread : {spread:.4f} nats")
    print(f"monotone down   : {mono}")
    print(f"capacity-bound  : {mono and spread >= 0.1}")


if __name__ == "__main__":
    main()
