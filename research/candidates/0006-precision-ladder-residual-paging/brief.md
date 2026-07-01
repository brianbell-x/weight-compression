# Candidate: Precision-Ladder / On-Demand Residual Paging (split quantization)

## Claim
Keep a cheap low-bit BASE resident for all 128 experts, and recover higher
precision only for the ~6 experts that fire per token by fetching+adding a
higher-precision RESIDUAL on demand — "loaded low, used high when needed",
exploiting MoE top-6 sparsity to halve resident VRAM without INT4's quality loss.

## Why It Might Work (user idea, worth testing)
Only 6/128 experts fire per token, so paying for precision on just the active
ones could, in principle, give INT8 quality at INT4 resident cost. The literal
"store INT8, use INT16" gives nothing (you can't recover discarded bits), but the
residual form (base + on-demand correction) is the correct generalization.

## Tensor Group
Routed experts, layer-1 up_proj sampled (generalizes to all 5,888 expert tensors).

## Measurement (run — Stage-1 matmul-fidelity proxy, 24 experts)
INT4 per-group base; residual = INT4 quant of (W − dequant(base)); output error
‖XW−XW′‖/‖XW‖; plus residual entropy and zero-fraction.

## Findings — REJECTED (mechanism works, economics don't)
- **Fidelity recovery works**: INT4 base 12.24% → INT4 base+residual **0.885%**
  (INT8-class; direct INT8 = 0.690%). INT8+residual → 0.0027% (near-BF16) but
  pointless since INT8 is already 0.69%.
- **Residual is incompressible — the dealbreaker**: INT4 residual entropy = 3.87
  bits (of 4 max), only 9.9% zeros. The residual is NOT sparse/cheap; it is
  essentially a second full INT4 tensor. So base(4b)+residual(~4b) = 8 bits total
  = the same as just storing INT8. The ladder relocates bits (4b VRAM + 4b host),
  it does not create a smaller accuracy/size point.
- **Per-token paging cost is large**: residual for active experts ≈ 6 × 23 MoE
  layers × ~5 MB ≈ **~690 MB/token**, ≈ **28–43 ms/token** over PCIe (25→16 GB/s) —
  would dominate decode latency.
- **Net**: only a fits-vs-doesn't-fit play (run on ~15 GB VRAM instead of ~30 GB,
  paying the paging tax). If ~30 GB is available, resident INT8
  ([[0005-low-bit-expert-quant]]) is strictly better — same fidelity, no paging.

## Constructive redirect
The experts have no cheap/sparse residual — their fine bits are high-entropy
(consistent with the near-random BF16 mantissa, [[0001-bf16-exponent-plane]]). So
sub-4-bit cannot come from a uniform top-up; it must come from being SELECTIVE —
spend extra bits only on the few salient (high-magnitude / high-activation-energy)
channels, the rest at INT4. That salient-channel mixed-precision lever is the open
item in 0005 and is where the next test should go.

## Status
Rejected
