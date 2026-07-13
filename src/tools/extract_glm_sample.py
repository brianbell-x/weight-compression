"""Extract one real GLM-5.2 expert tensor into the npz layout bench_kernel_v10 expects."""

import os
import json
import struct
import mmap
import re
import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import hf_hub_download

SHARD = "model-00002-of-00282.safetensors"  # layer-10 routed experts
p = hf_hub_download("zai-org/GLM-5.2", SHARD, local_dir="/root/spd")
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
ds = 8 + n
f = open(p, "rb")
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
bundle = {}
meta = {}
for proj in ("up", "down"):
    ks = sorted(k for k in h if re.search(rf"experts\.\d+\.{proj}_proj\.weight$", k))
    k = ks[0]
    b, e = h[k]["data_offsets"]
    raw = bytes(mm[ds + b : ds + e])
    shape = h[k]["shape"]
    assert h[k]["dtype"] == "BF16" and shape[1] % 4 == 0
    bundle[f"{proj}__raw_u16"] = np.frombuffer(raw, np.uint16).copy()
    bundle[f"{proj}__shape"] = np.array(shape, np.int64)
    meta[proj] = {"tensor": k, "shape": shape}
mm.close()
f.close()
np.savez("/root/gpu_sample.npz", **bundle)
print(json.dumps(meta))
