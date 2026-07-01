"""Analyze the full-model constant-bit scan: WHICH bits are constant, whether the
yield is subsumed by 0009's exponent codebook (constant bits all in exponent) or
ADDITIVE (any constant MANTISSA bits, which 0009 keeps verbatim), and how the
yield distributes across tensor categories (expert matrices vs small tensors).

BF16 bit layout (MSB->LSB): bit15 = sign, bits14..7 = 8-bit exponent, bits6..0 = 7-bit mantissa.
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "full_model_constbits" / "constant_bits.json").read_text())

SIGN = 1 << 15
EXP = 0xFF << 7          # bits 14..7
MANT = 0x7F              # bits 6..0

def cat(name):
    if ".mlp.experts" in name or ".experts." in name or "expert" in name:
        return "expert"
    if ".mixer." in name and ("in_proj" in name or "out_proj" in name):
        return "mamba_proj"
    if "embed" in name or "lm_head" in name:
        return "embed/head"
    return "other_small"

by = defaultdict(lambda: {"n":0,"bytes":0,"free_bytes":0.0,
                          "exp_bits":0,"mant_bits":0,"sign_bits":0,
                          "mant_const_tensors":0})
mant_examples = []
freebits_hist = defaultdict(int)

for t in data["tensors"]:
    if t["dtype"] != "BF16":
        continue
    m = int(t["const_mask_hex"], 16)
    c = cat(t["name"])
    b = by[c]
    b["n"] += 1
    b["bytes"] += t["bytes"]
    b["free_bytes"] += t["free_bytes"]
    exp_const = bin(m & EXP).count("1")
    mant_const = bin(m & MANT).count("1")
    sign_const = 1 if (m & SIGN) else 0
    b["exp_bits"] += exp_const * t["numel"]
    b["mant_bits"] += mant_const * t["numel"]
    b["sign_bits"] += sign_const * t["numel"]
    if mant_const:
        b["mant_const_tensors"] += 1
        if len(mant_examples) < 12:
            mant_examples.append((t["name"], t["numel"], t["const_mask_hex"], mant_const))
    freebits_hist[t["free_bits_per_elem"]] += t["bytes"]

print("=== per-category BF16 ===")
for c, b in sorted(by.items(), key=lambda kv:-kv[1]["bytes"]):
    gb = b["bytes"]/1e9
    fgb = b["free_bytes"]/1e9
    frac = b["free_bytes"]/b["bytes"] if b["bytes"] else 0
    # bits attributable to exponent vs mantissa (in GB of freed bytes)
    exp_gb = b["exp_bits"]/8/1e9
    mant_gb = b["mant_bits"]/8/1e9
    sign_gb = b["sign_bits"]/8/1e9
    print(f"{c:12s} n={b['n']:5d} bytes={gb:7.3f}GB free={fgb:7.3f}GB ({frac*100:5.2f}%) "
          f"| exp={exp_gb:6.3f} mant={mant_gb:6.4f} sign={sign_gb:6.4f} GB "
          f"| mant-const tensors={b['mant_const_tensors']}")

print("\n=== free_bits_per_elem distribution (weighted by tensor bytes) ===")
for fb in sorted(freebits_hist):
    print(f"  {fb:2d} constant bits/elem : {freebits_hist[fb]/1e9:7.3f} GB of tensors")

print("\n=== any constant MANTISSA bits? (would be ADDITIVE to 0009) ===")
if mant_examples:
    for name, numel, mask, mc in mant_examples:
        print(f"  {mask}  mant_const_bits={mc}  numel={numel}  {name}")
else:
    print("  NONE. Every constant bit across the model is sign or exponent.")
    print("  => constant-bit dropping is a strict subset of 0009's exponent codebook.")
