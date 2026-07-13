import json
import glob
import os

d = os.path.dirname(os.path.abspath(__file__))
for rel in ("glm_ckpts", os.path.join("..", "tests", "artifacts", "ckpts")):
    files = sorted(glob.glob(os.path.join(d, rel, "*.json")))
    if files:
        break
assert len(files) == 16, files

tot = dict(
    total_raw=0,
    bf16_raw=0,
    bf16_enc_bs=0.0,
    bf16_enc_rg=0.0,
    expert_raw=0,
    expert_enc_bs=0.0,
    expert_enc_rg=0.0,
    other_raw=0,
    n_bf16=0,
    n_expert=0,
    n_esc_total=0,
)
all_lossless = True
dtype_raw = {}
done = []
for f in files:
    j = json.load(open(f))
    assert j["repo"] == "zai-org/GLM-5.2"
    a = j["acc"]
    for k in tot:
        tot[k] += a[k]
    all_lossless &= bool(a["all_lossless"])
    for k, v in a["dtype_raw"].items():
        dtype_raw[k] = dtype_raw.get(k, 0) + v
    done.extend(j["done_shards"])

assert len(done) == 282, len(done)
assert len(set(done)) == 282, "overlap between ranges!"

GB = 1024**3
comp_bs = tot["other_raw"] + tot["bf16_enc_bs"]
comp_rg = tot["other_raw"] + tot["bf16_enc_rg"]
n_weights = tot["bf16_raw"] / 2
res = {
    "repo": "zai-org/GLM-5.2",
    "shards": 282,
    "coverage": "all shards, no overlap (asserted)",
    "ALL_BF16_TENSORS_LOSSLESS": all_lossless,
    "n_bf16_tensors": tot["n_bf16"],
    "n_expert_tensors": tot["n_expert"],
    "n_bf16_weights": int(n_weights),
    "total_raw_GB": round(tot["total_raw"] / GB, 2),
    "bf16_GB": round(tot["bf16_raw"] / GB, 2),
    "non_bf16_GB_by_dtype": {
        k: round(v / GB, 3) for k, v in sorted(dtype_raw.items()) if k != "BF16"
    },
    "bf16_share_pct": round(100 * tot["bf16_raw"] / tot["total_raw"], 2),
    "whole_model_reduction_pct": {
        "byte_split": round(100 * (1 - comp_bs / tot["total_raw"]), 3),
        "regroup_K15": round(100 * (1 - comp_rg / tot["total_raw"]), 3),
    },
    "bf16_only_bits_per_weight": {
        "byte_split": round(8 * tot["bf16_enc_bs"] / n_weights, 4),
        "regroup_K15": round(8 * tot["bf16_enc_rg"] / n_weights, 4),
    },
    "expert_only_reduction_pct": {
        "byte_split": round(100 * (1 - tot["expert_enc_bs"] / tot["expert_raw"]), 3)
        if tot["expert_raw"]
        else None,
        "regroup_K15": round(100 * (1 - tot["expert_enc_rg"] / tot["expert_raw"]), 3)
        if tot["expert_raw"]
        else None,
    },
    "expert_share_of_bf16_pct": round(100 * tot["expert_raw"] / tot["bf16_raw"], 2),
    "escape_rate_pct_of_weights": round(100 * tot["n_esc_total"] / n_weights, 4),
    "compressed_GB": {
        "byte_split": round(comp_bs / GB, 2),
        "regroup_K15": round(comp_rg / GB, 2),
    },
}
out = os.path.join(d, "..", "tests", "artifacts", "glm52_standalone_result.json")
json.dump(res, open(out, "w"), indent=2)
print(json.dumps(res, indent=2))
