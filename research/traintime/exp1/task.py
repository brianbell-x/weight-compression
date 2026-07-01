"""task.py -- the learning task for the capability-per-parameter study.

We use CHARACTER-LEVEL LANGUAGE MODELING on a small, fixed, deterministically
generated text. Why this task:

  * It directly stresses a weight matrix's expressivity: the model must store
    and recombine many local char->char and context->char regularities inside a
    small number of weights. More usable capacity -> lower cross-entropy.
  * The loss is interpretable. Cross-entropy in nats maps to bits-per-character
    (bpc = loss / ln 2). A uniform predictor over a V-symbol alphabet costs
    log2(V) bits/char; any learning shows up as bpc well below that ceiling.
  * It is deterministic and self-contained. The text is produced by a seeded
    2nd-order (trigram) Markov generator over a fixed English-like lexicon, so
    the corpus has REAL, LEARNABLE structure (word shapes, spacing, n-gram
    regularities) but needs no external/copyrighted file. The generator's order
    bounds the achievable entropy, giving the task a known information ceiling.

The corpus is cached to data.txt next to this file so every run sees identical
bytes. Everything here is seeded; calling build_data(seed) twice yields the same
tensors.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data.txt")

# Fixed English-like lexicon (public-domain common words; no copyrighted text).
_LEXICON = (
    "the of and to in a is that it was he for on are as with his they at be this "
    "from i have or by one had not but what all were when we there can an your "
    "which their said if do will each about how up out them then she many some so "
    "these would other into has more her two like him see time could no make than "
    "first water been call who oil now find long down day did get come made may "
    "part over new sound take only little work know place year live me back give "
    "most very after thing our just name good sentence man think say great where "
    "help through much before line right too mean old any same tell boy follow "
    "came want show also around form three small set put end does another well "
    "large must big even such because turn here why ask went men read need land "
    "different home us move try kind hand picture again change off play spell air "
    "away animal house point page letter mother answer found study still learn "
    "should world high every near add food between own below country plant last "
    "school father keep tree never start city earth eye light thought head under "
    "story saw left dont few while along might close something seem next hard open"
).split()

_TARGET_CHARS = 36_000        # corpus size (train+val together), a few tens of KB
_VAL_FRACTION = 0.15


def _generate_corpus(seed: int = 1234) -> str:
    """Seeded trigram Markov text over the fixed lexicon -> a string of words."""
    rng = random.Random(seed)

    # Build a trigram transition table from a deterministic "training" stream so
    # the generated corpus has genuine higher-order word structure rather than
    # i.i.d. word draws. The base stream is a long shuffled-but-seeded walk.
    base = []
    walk = list(_LEXICON)
    for _ in range(40):
        rng.shuffle(walk)
        base.extend(walk)

    trigram: dict[tuple[str, str], list[str]] = {}
    for a, b, c in zip(base, base[1:], base[2:]):
        trigram.setdefault((a, b), []).append(c)

    out_words: list[str] = [base[0], base[1]]
    sentence_len = rng.randint(6, 14)
    cur = 2
    while sum(len(w) + 1 for w in out_words) < _TARGET_CHARS:
        key = (out_words[-2], out_words[-1])
        choices = trigram.get(key) or _LEXICON
        nxt = rng.choice(choices)
        out_words.append(nxt)
        cur += 1
        if cur >= sentence_len:
            # Punctuate to add structure the model can learn (caps + period).
            out_words[-1] = out_words[-1] + "."
            cur = 0
            sentence_len = rng.randint(6, 14)

    # Capitalize the first word of each sentence for a touch more structure.
    text_parts: list[str] = []
    cap_next = True
    for w in out_words:
        word = w
        if cap_next and word[:1].isalpha():
            word = word[0].upper() + word[1:]
        text_parts.append(word)
        cap_next = word.endswith(".")
    return " ".join(text_parts)[:_TARGET_CHARS]


def get_corpus(seed: int = 1234) -> str:
    """Return the cached corpus, generating + caching it on first call."""
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return f.read()
    text = _generate_corpus(seed)
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


def build_data(seq_len: int = 64, seed: int = 1234) -> Task:
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
