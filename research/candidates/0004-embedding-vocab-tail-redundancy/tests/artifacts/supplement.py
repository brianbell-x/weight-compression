from __future__ import annotations
import json, struct, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

SNAP = Path(r"C:\dev\compression\models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot")
ROWS, COLS = 131072, 2688
RB = COLS * 2


def tslice(shard, t):
    with open(SNAP / shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n)); ds = 8 + n
        b, e = hdr[t]["data_offsets"]; f.seek(ds + b); return f.read(e - b)


def groups(raw):
    g = defaultdict(list); mv = memoryview(raw)
    for i in range(ROWS):
        g[hashlib.blake2b(mv[i*RB:(i+1)*RB], digest_size=16).digest()].append(i)
    return g


for tag, shard, t in [("embeddings", "model-00001-of-00013.safetensors", "backbone.embeddings.weight"),
                      ("lm_head", "model-00013-of-00013.safetensors", "lm_head.weight")]:
    raw = tslice(shard, t)
    arr = np.frombuffer(raw, np.uint16).reshape(ROWS, COLS)
    f32 = (arr.astype(np.uint32) << 16).view(np.float32)
    l2 = np.sqrt((f32.astype(np.float64)**2).sum(1))
    g = groups(raw)
    dg = {h: v for h, v in g.items() if len(v) > 1}
    big = sorted(dg.values(), key=len, reverse=True)[:5]
    print(f"=== {tag} ===")
    print("dup groups:", len(dg), "redundant removable:", sum(len(v)-1 for v in dg.values()))
    for grp in big:
        ids = sorted(grp)
        frac_special = sum(1 for x in ids if x < 1000) / len(ids)
        print(f"  size={len(grp)} min_id={ids[0]} max_id={ids[-1]} frac_id<1000={frac_special:.2f} l2={l2[ids[0]]:.5f} sample={ids[:8]}")
    # all duplicate-involved ids: how many < 1000
    dup_ids = [i for v in dg.values() for i in v]
    print("  dup-involved ids <1000:", sum(1 for i in dup_ids if i < 1000), "of", len(dup_ids))
    # tail check: L2 of last 5000 vs overall, and min-L2 row id
    print("  L2 argmin id:", int(l2.argmin()), "min:", float(l2.min()))
    print("  L2 mean first 1000:", float(l2[:1000].mean()), " mean 1000..131072:", float(l2[1000:].mean()))
    print("  L2 mean last 5000:", float(l2[-5000:].mean()))
    # special-token rows (0..999) stats
    sp = l2[:1000]
    print("  special(0..999) L2 min/mean/max:", float(sp.min()), float(sp.mean()), float(sp.max()))
