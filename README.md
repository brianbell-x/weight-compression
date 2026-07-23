# weight-compression

A research ledger for exact, lossless compression of LLM weights. Each proven
method gets its own folder with code, results, and references; this README is
the running record of what has been tried and what it achieved.

Everything here is bit-for-bit lossless — verified by exact round-trip on real
weights, never estimated.

## Ledger

### Split12 — current best (proven)

Byte-split format for BF16 weights. On the full `zai-org/GLM-5.2` scan
(59,509 tensors, all round-tripped bit-for-bit):

- **24.967%** reduction (12.005 bits/weight) — shipped and verified
- **30.168%** K15 charged-format estimate (11.173 bits/weight) — estimate only

Code, verifier, and references: [`Split12/`](Split12/)
Serving and kernel record (B300 inference tests): [`Split12/inference/`](Split12/inference/)

## Layout

Each method folder is self-contained — code, environment (`pyproject.toml`,
`uv.lock`, `.python-version`), writeup, and references all live inside it.

- `Split12/` — the current proven method: verifier, writeup (`Split12/site/`,
  deployed to GitHub Pages at `/Split12/`), inference record, and references
