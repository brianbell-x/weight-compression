"""task.py -- the learning task for the capability-per-parameter study (exp2).

CHARACTER-LEVEL LANGUAGE MODELING, same as exp1, but on a CAPACITY-BOUND corpus.

Why exp1 failed to discriminate: its corpus was a trigram-Markov walk over a
~250-word lexicon. That has a LOW entropy ceiling -- a tiny model memorizes the
word shapes and the weak n-gram table and saturates, so dense val loss was nearly
flat (~1.17-1.19) across the whole FFN-width sweep. The task could not stress
capacity.

exp2 fix: use REAL public-domain English prose (Jane Austen, *Pride and
Prejudice*, Project Gutenberg #1342 -- public domain). Natural language has
genuine multi-scale structure (spelling, morphology, word frequencies, syntax,
long-range agreement) with a high, smooth information floor. A small char-LM
captures more of that structure as you give it more FFN width, so cross-entropy
keeps DROPPING monotonically with parameters -- a capacity-BOUND regime.

The cleaned text is cached to data.txt next to this file on first build, so every
run thereafter sees identical bytes (deterministic, network-free). If the cache
is absent and the download fails, we fall back to a deterministic high-order
synthetic chain so the harness still runs.

Loss is interpretable: cross-entropy in nats -> bits/char via bpc = loss / ln 2.
A uniform predictor over a V-symbol alphabet costs log2(V) bits/char.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data.txt")

# Public-domain source: Pride and Prejudice (Gutenberg #1342).
_BOOK_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"
_START_MARK = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END_MARK = "*** END OF THE PROJECT GUTENBERG EBOOK"

_TARGET_CHARS = 300_000        # cleaned corpus size (train+val together)
_VAL_FRACTION = 0.15

# Restricted, fixed character set: lowercase letters, space, and a few marks.
# Everything else is mapped to space. This keeps the vocab small and stable while
# preserving the real spelling/word/syntax structure that makes the task hard.
_ALLOWED = set("abcdefghijklmnopqrstuvwxyz .,;:'\"!?-")


def _clean(raw: str) -> str:
    """Strip Gutenberg boilerplate, lowercase, restrict charset, collapse ws."""
    s = raw
    i = s.find(_START_MARK)
    if i != -1:
        i = s.find("\n", i)
        s = s[i + 1:]
    j = s.find(_END_MARK)
    if j != -1:
        s = s[:j]
    s = s.lower()
    out = []
    prev_space = False
    for ch in s:
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                prev_space = True
            continue
        if ch in _ALLOWED:
            out.append(ch)
            prev_space = False
        # else: drop the odd character (keeps adjacent letters in one word)
    return "".join(out).strip()


def _download_corpus() -> str | None:
    try:
        import urllib.request
        with urllib.request.urlopen(_BOOK_URL, timeout=60) as r:
            raw = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    text = _clean(raw)
    if len(text) < _TARGET_CHARS + 2_000:
        return None
    # Skip the front-matter (title page / contents) and take a contiguous block
    # of running prose so train/val are the same register.
    start = 2_000
    return text[start:start + _TARGET_CHARS]


# --- deterministic fallback (only if the download is unavailable) ----------- #
_FALLBACK_LEXICON = (
    "the of and to in a is that it was he for on are as with his they at be this "
    "from have or by one had not but what all were when we there can an your which "
    "their said if do will each about how up out them then she many some so these"
).split()


def _fallback_corpus(seed: int = 1234) -> str:
    """High-order (6-gram char) synthetic chain -- used only if offline."""
    rng = random.Random(seed)
    base = []
    walk = list(_FALLBACK_LEXICON)
    for _ in range(60):
        rng.shuffle(walk)
        base.extend(walk)
    stream = " ".join(base)
    order = 6
    table: dict[str, list[str]] = {}
    for k in range(len(stream) - order):
        table.setdefault(stream[k:k + order], []).append(stream[k + order])
    out = list(stream[:order])
    while len(out) < _TARGET_CHARS:
        key = "".join(out[-order:])
        nxt = table.get(key)
        out.append(rng.choice(nxt) if nxt else rng.choice(list(stream)))
    return "".join(out)[:_TARGET_CHARS]


def get_corpus(seed: int = 1234) -> str:
    """Return the cached corpus, building + caching it on first call."""
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return f.read()
    text = _download_corpus()
    if text is None:
        text = _fallback_corpus(seed)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return text


@dataclass
class Task:
    train_ids: torch.Tensor   # 1-D long tensor of char ids
    val_ids: torch.Tensor
    vocab: list[str]
    stoi: dict
    seq_len: int

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def get_batch(self, split: str, batch_size: int, generator: torch.Generator):
        data = self.train_ids if split == "train" else self.val_ids
        n = data.numel() - self.seq_len - 1
        ix = torch.randint(0, n, (batch_size,), generator=generator)
        x = torch.stack([data[i : i + self.seq_len] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.seq_len] for i in ix])
        return x, y


def build_data(seq_len: int = 128, seed: int = 1234) -> Task:
    """Deterministically build the char-LM task (cached corpus + fixed vocab)."""
    text = get_corpus(seed)
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = int(len(ids) * _VAL_FRACTION)
    train_ids = ids[:-n_val]
    val_ids = ids[-n_val:]
    return Task(train_ids, val_ids, vocab, stoi, seq_len)


if __name__ == "__main__":
    import math

    t = build_data()
    print(f"corpus chars : {t.train_ids.numel() + t.val_ids.numel()}")
    print(f"vocab size   : {t.vocab_size}")
    print(f"uniform bpc  : {math.log2(t.vocab_size):.3f} bits/char (loss ceiling)")
    print(f"train tokens : {t.train_ids.numel()}  val tokens: {t.val_ids.numel()}")
    print("sample:", repr(get_corpus()[:160]))
