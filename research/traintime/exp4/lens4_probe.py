from __future__ import annotations
"""Lens 4 — structural / exact redundancy probe.
(a) intra-tensor exact row & column repeats (hash rows/cols of 2D BF16 tensors)
(b) exact zeros + run-length (consecutive equal u16) model-wide
(c) most-common repeated value per tensor + global repeat stats
Pure numpy, exact on raw BF16 u16 patterns."""
import json, struct, hashlib, sys
from pathlib import Path
import numpy as np

MODEL = Path("C:/dev/compression/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/hf_snapshot")

def read_header(path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))

def prod(xs):
    o=1
    for x in xs: o*=x
    return o

def scan_shard(shard, agg):
    data_start, header = read_header(shard)
    mm = np.memmap(shard, dtype=np.uint8, mode="r")
    for name, meta in header.items():
        if name == "__metadata__": continue
        if meta["dtype"] != "BF16": continue
        b,e = meta["data_offsets"]
        shape = meta["shape"]; numel = prod(shape)
        if numel == 0: continue
        u16 = mm[data_start+b:data_start+e].view(np.uint16)
        # --- zeros & run-length (consecutive equal) ---
        nz = int((u16==0).sum())
        # run-length: positions equal to previous element (flat order)
        if u16.size>1:
            eqprev = int((u16[1:]==u16[:-1]).sum())
        else:
            eqprev = 0
        agg["total"] += u16.size
        agg["zeros"] += nz
        agg["eqprev"] += eqprev
        # most frequent value coverage (already in survey, skip global here)
        rec = {"name":name,"shape":shape,"numel":numel,
               "frac_zero":nz/u16.size,"frac_eqprev":eqprev/max(u16.size-1,1)}
        # --- row/col dedup for 2D tensors ---
        if len(shape)==2 and numel>=1:
            r,c = shape
            M = u16.reshape(r,c)
            # row dup: hash each row
            # use bytes; cap cost — for very tall tensors (vocab) this is fine
            rowh = {}
            dup_rows=0
            # vectorized-ish: view rows as bytes
            rb = M.view(np.uint16)
            # hashing many rows: use numpy unique on a contiguous byte view
            rows_bytes = np.ascontiguousarray(M).view(np.dtype((np.void, c*2)))
            uniq_rows = np.unique(rows_bytes).size
            dup_rows = r - uniq_rows
            cols_bytes = np.ascontiguousarray(M.T).view(np.dtype((np.void, r*2)))
            uniq_cols = np.unique(cols_bytes).size
            dup_cols = c - uniq_cols
            rec.update(dict(rows=r,cols=c,dup_rows=int(dup_rows),
                            uniq_rows=int(uniq_rows),dup_cols=int(dup_cols),
                            uniq_cols=int(uniq_cols)))
            agg["dup_rows_total"] += int(dup_rows)
            agg["dup_cols_total"] += int(dup_cols)
        agg["records"].append(rec)

def main():
    shards = sorted(MODEL.glob("model-*.safetensors"))
    agg = {"total":0,"zeros":0,"eqprev":0,"dup_rows_total":0,"dup_cols_total":0,"records":[]}
    for i,sh in enumerate(shards):
        scan_shard(sh, agg)
        print(f"shard {i+1}/{len(shards)} done, tensors={len(agg['records'])}", flush=True)
    out = Path("C:/dev/compression/research/traintime/exp4/lens4_result.json")
    # top dup-row tensors
    recs = agg["records"]
    dupr = sorted([r for r in recs if r.get("dup_rows",0)>0], key=lambda x:-x["dup_rows"])[:20]
    dupc = sorted([r for r in recs if r.get("dup_cols",0)>0], key=lambda x:-x["dup_cols"])[:20]
    hi_eqprev = sorted(recs, key=lambda x:-x["frac_eqprev"])[:20]
    hi_zero = sorted(recs, key=lambda x:-x["frac_zero"])[:20]
    summary = {
        "total_weights":agg["total"],
        "total_zeros":agg["zeros"],
        "frac_zero_global":agg["zeros"]/agg["total"],
        "total_eqprev":agg["eqprev"],
        "frac_eqprev_global":agg["eqprev"]/agg["total"],
        "dup_rows_total":agg["dup_rows_total"],
        "dup_cols_total":agg["dup_cols_total"],
        "n_tensors":len(recs),
        "top_dup_rows":dupr,
        "top_dup_cols":dupc,
        "top_eqprev":hi_eqprev,
        "top_zero":hi_zero,
    }
    out.write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)},indent=2))

if __name__=="__main__":
    main()
