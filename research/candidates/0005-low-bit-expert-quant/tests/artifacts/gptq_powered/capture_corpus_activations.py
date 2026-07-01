"""Capture MANY real layer-1 routed-expert input activations from a large, diverse
LOCAL text corpus, split into DOCUMENT-DISJOINT calibration vs held-out sets.

Why (candidate 0005, the live GPTQ thread):
  test-002 ran GPTQ on only 187 tokens -> the input Hessian XtX was rank 1939/2688
  (data-starved) and GPTQ OVERFIT: held-out 5.45% vs RTN 5.12% (worse). Its Next
  Action explicitly deferred GPTQ "until far more tokens are cached (1e4-1e5)". This
  script produces that larger cache cheaply, reusing the proven 2-layer partial
  forward (embeddings + layer-0 Mamba2 + layer-1 pre-MoE norm; ~3.5 GB peak RSS, the
  128 experts / 58 GB are NEVER loaded).

Design (the honest part):
  * The calibration tokens and the held-out tokens come from DISJOINT DOCUMENTS, not
    from a within-document token split (test-002 split tokens inside the same 12
    prompts, which leaks within-prompt correlation into "held-out"). Holding out whole
    documents is the correct generalization test.
  * Both cal and held-out span mixed domains (technical prose, code, natural prose) so
    neither is a single-distribution artifact. Held-out additionally reserves the
    natural-English Austen passages (out-of-distribution vs the technical cal text) as
    a stress slice.
  * Corpus = local repo text only (no network): research notes (.md), the model
    README/breakdown, the snapshot's own modeling source (.py), plus the prompts_large
    diverse prompt set. Each document is tokenized and chunked into <=512-token windows
    (one partial forward per window; Mamba is linear so long windows are cheap).

Output (under this dir, ./activations_corpus/):
  X_cal.npy      [Ncal, 2688] float32   -- calibration routed-expert inputs
  X_heldout.npy  [Nhel, 2688] float32   -- held-out routed-expert inputs (disjoint docs)
  channel_energy_cal.npy [2688]         -- per-channel RMS energy (AWQ signal), cal set
  capture_meta.json                     -- token counts, doc lists, shapes, timing

Run:  uv run python capture_corpus_activations.py [--target-cal 30000] [--smoke]
"""
import os, sys, time, json, glob, argparse
import importlib.util, types
import numpy as np
import torch
import torch.nn.functional as F
import psutil
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)                       # .../tests/artifacts
OUT_DIR = os.path.join(HERE, "activations_corpus")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, ART)                            # import stage1_probe

REPO = r"C:\dev\compression"
SNAP = os.path.join(REPO, r"models\nvidia\NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\hf_snapshot")
SHARD1 = os.path.join(SNAP, "model-00001-of-00013.safetensors")

proc = psutil.Process()
_peak = {"rss": 0.0}
def rss_gb():
    g = proc.memory_info().rss / 1024**3
    _peak["rss"] = max(_peak["rss"], g)
    return g

# --- Shim mamba_ssm (pure-PyTorch gated RMSNorm; modeling hard-imports it) -----------
def _rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=False):
    dtype = x.dtype
    x = x.float()
    if z is not None and not norm_before_gate:
        x = x * F.silu(z.float())
    if group_size is None:
        rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        out = x * rstd * weight.float()
    else:
        shp = x.shape
        xg = x.reshape(*shp[:-1], shp[-1] // group_size, group_size)
        rstd = torch.rsqrt(xg.square().mean(-1, keepdim=True) + eps)
        out = (xg * rstd).reshape(shp) * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None and norm_before_gate:
        out = out * F.silu(z.float())
    return out.to(dtype)

def _mkmod(name, **attrs):
    m = types.ModuleType(name); [setattr(m, k, v) for k, v in attrs.items()]
    sys.modules[name] = m; return m
_mkmod("mamba_ssm"); _mkmod("mamba_ssm.ops"); _mkmod("mamba_ssm.ops.triton")
_mkmod("mamba_ssm.ops.triton.layernorm_gated", rmsnorm_fn=_rmsnorm_fn)
_mkmod("mamba_ssm.ops.triton.selective_state_update", selective_state_update=None)
_mkmod("mamba_ssm.ops.triton.ssd_combined",
       mamba_chunk_scan_combined=None, mamba_split_conv1d_scan_combined=None)

_pkg = types.ModuleType("nemo_pkg"); _pkg.__path__ = [SNAP]; sys.modules["nemo_pkg"] = _pkg
def _load(name):
    spec = importlib.util.spec_from_file_location(f"nemo_pkg.{name}", os.path.join(SNAP, name + ".py"))
    mod = importlib.util.module_from_spec(spec); sys.modules[f"nemo_pkg.{name}"] = mod
    spec.loader.exec_module(mod); return mod
NemotronHConfig = _load("configuration_nemotron_h").NemotronHConfig
M = _load("modeling_nemotron_h")


# =====================================================================================
# Corpus assembly (local files only)
# =====================================================================================
# Held-out documents are reserved up front and NEVER enter calibration. They mix
# natural prose (Austen, OOD vs technical cal text) with one technical doc (in-dist).
HELDOUT_FILES = [
    os.path.join(REPO, "research", "notes", "capability-eval-path.md"),
    os.path.join(REPO, "research", "candidates", "0001-bf16-exponent-plane", "tests", "test-001.md"),
]

AUSTEN = [
    "It is a truth universally acknowledged, that a single man in possession of a good "
    "fortune, must be in want of a wife. However little known the feelings or views of "
    "such a man may be on his first entering a neighbourhood, this truth is so well fixed "
    "in the minds of the surrounding families, that he is considered the rightful property "
    "of some one or other of their daughters. My dear Mr. Bennet, said his lady to him one "
    "day, have you heard that Netherfield Park is let at last? Mr. Bennet replied that he had "
    "not. But it is, returned she; for Mrs. Long has just been here, and she told me all "
    "about it. Mr. Bennet made no answer. Do you not want to know who has taken it? cried his "
    "wife impatiently. You want to tell me, and I have no objection to hearing it.",
    "Elizabeth, having rather expected to affront him, was amazed at his gallantry; but there "
    "was a mixture of sweetness and archness in her manner which made it difficult for her to "
    "affront anybody, and Darcy had never been so bewitched by any woman as he was by her. He "
    "really believed that were it not for the inferiority of her connections, he should be in "
    "some danger. Miss Bingley saw, or suspected, enough to be jealous; and her great anxiety "
    "for the recovery of her dear friend Jane received some assistance from her desire of "
    "getting rid of Elizabeth.",
]

def collect_cal_files():
    files = []
    files += sorted(glob.glob(os.path.join(REPO, "research", "**", "*.md"), recursive=True))
    files += sorted(glob.glob(os.path.join(REPO, "research", "**", "*.py"), recursive=True))
    files += sorted(glob.glob(os.path.join(REPO, "tools", "*.py")))
    files += [os.path.join(SNAP, "modeling_nemotron_h.py"),
              os.path.join(SNAP, "configuration_nemotron_h.py")]
    files += sorted(glob.glob(os.path.join(REPO, "research", "nvidia", "**", "*.md"), recursive=True))
    held = set(os.path.abspath(p) for p in HELDOUT_FILES)
    out, seen = [], set()
    for p in files:
        ap = os.path.abspath(p)
        if ap in held or ap in seen or not os.path.exists(ap):
            continue
        seen.add(ap)
        out.append(ap)
    return out

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-cal", type=int, default=30000, help="stop after this many cal tokens")
    ap.add_argument("--target-heldout", type=int, default=2500)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.target_cal, args.target_heldout = 1500, 600
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    t0 = time.time()
    cfg = NemotronHConfig(**json.load(open(os.path.join(SNAP, "config.json"))))
    H = cfg.hidden_size

    norm0 = M.NemotronHRMSNorm(H, eps=cfg.layer_norm_epsilon)
    mixer0 = M.NemotronHMamba2Mixer(cfg, layer_idx=0)
    norm1 = M.NemotronHRMSNorm(H, eps=cfg.layer_norm_epsilon)

    with safe_open(SHARD1, framework="pt") as f:
        norm0.weight.data = f.get_tensor("backbone.layers.0.norm.weight").to(torch.float32)
        norm1.weight.data = f.get_tensor("backbone.layers.1.norm.weight").to(torch.float32)
        msd = mixer0.state_dict()
        mixer0.load_state_dict({k: f.get_tensor("backbone.layers.0.mixer." + k).to(torch.float32)
                                for k in msd})
        emb = f.get_tensor("backbone.embeddings.weight").to(torch.float32)
    for m in (norm0, mixer0, norm1):
        m.eval()
    print(f"[load] H={H} RSS={rss_gb():.2f}GB t={time.time()-t0:.1f}s", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)

    @torch.no_grad()
    def forward_chunk(ids):
        h = F.embedding(ids, emb)
        h = h + mixer0(norm0(h), cache_params=None, cache_position=None, attention_mask=None)
        return norm1(h).reshape(-1, H)            # [T, H] routed-expert input

    @torch.no_grad()
    def capture_docs(docs, target, label):
        """docs: list of (doc_id, text). Returns X [n,H], list of per-doc token counts."""
        rows, n_tok, used = [], 0, []
        for did, text in docs:
            if n_tok >= target:
                break
            ids_full = tok(text, return_tensors="pt").input_ids[0]   # [L]
            L = ids_full.shape[0]
            dtok = 0
            for s in range(0, L, args.chunk):
                if n_tok >= target:
                    break
                ids = ids_full[s:s + args.chunk].unsqueeze(0)
                if ids.shape[1] < 2:
                    continue
                Xp = forward_chunk(ids)
                rows.append(Xp)
                n_tok += Xp.shape[0]; dtok += Xp.shape[0]
            used.append({"doc": did, "tokens": dtok})
            rss_gb()
            print(f"  [{label}] {did[:54]:54s} +{dtok:5d} -> {n_tok:6d} tok "
                  f"RSS={rss_gb():.2f}GB t={time.time()-t0:.0f}s", flush=True)
        X = torch.cat(rows, dim=0).contiguous() if rows else torch.zeros(0, H)
        return X, used

    # ---- held-out first (reserved docs: Austen prose + 2 reserved technical files) ----
    held_docs = [(f"austen_{i}", t) for i, t in enumerate(AUSTEN)]
    held_docs += [(os.path.relpath(p, REPO), read_text(p)) for p in HELDOUT_FILES]
    Xh, held_used = capture_docs(held_docs, args.target_heldout, "held")

    # ---- calibration (all other local text, shuffled deterministically) ---------------
    cal_paths = collect_cal_files()
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(cal_paths), generator=g).tolist()
    cal_docs = [(os.path.relpath(cal_paths[i], REPO), read_text(cal_paths[i])) for i in perm]
    # append the prompts_large diverse prompts as extra short-form diversity
    try:
        sys.path.insert(0, os.path.join(REPO, "tools"))
        from prompts_large import PROMPTS as PL
        cal_docs.append(("prompts_large_concat", "\n".join(PL)))
    except Exception as e:
        print(f"[warn] prompts_large not loaded: {e}")
    Xc, cal_used = capture_docs(cal_docs, args.target_cal, "cal")

    # ---- channel energy (AWQ signal) on cal set ---------------------------------------
    chan_energy = Xc.pow(2).mean(dim=0).sqrt() if Xc.shape[0] else torch.zeros(H)
    ratio = float(chan_energy.max() / chan_energy.mean()) if Xc.shape[0] else float("nan")

    np.save(os.path.join(OUT_DIR, "X_cal.npy"), Xc.numpy())
    np.save(os.path.join(OUT_DIR, "X_heldout.npy"), Xh.numpy())
    np.save(os.path.join(OUT_DIR, "channel_energy_cal.npy"), chan_energy.numpy())
    meta = {
        "hidden_size": H, "chunk": args.chunk,
        "n_cal_tokens": int(Xc.shape[0]), "n_heldout_tokens": int(Xh.shape[0]),
        "cal_docs": cal_used, "heldout_docs": held_used,
        "channel_energy_max_over_mean_cal": ratio,
        "peak_rss_gb": round(_peak["rss"], 2),
        "wall_s": round(time.time() - t0, 1),
        "note": "X = post backbone.layers.1.norm hidden state = verbatim input to gate "
                "and every routed expert up_proj. cal/held-out are DOCUMENT-disjoint.",
    }
    json.dump(meta, open(os.path.join(OUT_DIR, "capture_meta.json"), "w"), indent=2)
    print(f"\n[done] cal={Xc.shape[0]} heldout={Xh.shape[0]} "
          f"chan_energy max/mean={ratio:.2f} peakRSS={_peak['rss']:.2f}GB "
          f"wall={time.time()-t0:.0f}s", flush=True)
    print("SUMMARY " + json.dumps({k: meta[k] for k in
          ("n_cal_tokens", "n_heldout_tokens", "channel_energy_max_over_mean_cal",
           "peak_rss_gb", "wall_s")}))


if __name__ == "__main__":
    main()
