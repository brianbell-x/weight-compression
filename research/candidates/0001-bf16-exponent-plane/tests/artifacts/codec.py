"""BF16 exponent-plane codec primitives + static rANS, plus measurement helpers.

BF16 in-memory (little-endian) layout per element = 2 bytes:
  byte0 (low)  = bits  7..0  = exp_lsb(bit7) + mantissa(bits 6..0)
  byte1 (high) = bits 15..8  = sign(bit15)   + exp_top7(bits 14..8)

De-interleave: even bytes -> low plane, odd bytes -> high plane.
Sign-fold: mag7 = high & 0x7F ; sign = high >> 7  (exact, reversible).
"""
from __future__ import annotations
import numpy as np


# ---------- de-interleave / re-interleave ----------
def deinterleave(raw: bytes):
    a = np.frombuffer(raw, dtype=np.uint8)
    assert a.size % 2 == 0, "BF16 byte stream must be even length"
    low = a[0::2].copy()   # byte0
    high = a[1::2].copy()  # byte1
    return low, high


def reinterleave(low: np.ndarray, high: np.ndarray) -> bytes:
    out = np.empty(low.size * 2, dtype=np.uint8)
    out[0::2] = low
    out[1::2] = high
    return out.tobytes()


def fold_sign(high: np.ndarray):
    mag7 = (high & 0x7F).astype(np.uint8)
    sign = (high >> 7).astype(np.uint8)
    return mag7, sign


def unfold_sign(mag7: np.ndarray, sign: np.ndarray) -> np.ndarray:
    return ((sign << 7) | mag7).astype(np.uint8)


def pack_bits(sign: np.ndarray) -> bytes:
    return np.packbits(sign).tobytes()


def unpack_bits(packed: bytes, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=n).astype(np.uint8)


# ---------- entropy ----------
def order0_entropy_bits(arr: np.ndarray, alphabet: int = 256) -> float:
    """Shannon order-0 entropy in bits/symbol."""
    counts = np.bincount(arr.astype(np.int64), minlength=alphabet).astype(np.float64)
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def histogram(arr: np.ndarray, alphabet: int = 256) -> np.ndarray:
    return np.bincount(arr.astype(np.int64), minlength=alphabet).astype(np.float64)


def kl_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """KL(P||Q) in bits, with Laplace smoothing so it is always finite."""
    p = p_counts + 1.0
    q = q_counts + 1.0
    p /= p.sum()
    q /= q.sum()
    return float((p * np.log2(p / q)).sum())


# ---------- static rANS (rygorous rans_byte, 32-bit state, 8-bit renorm) ----------
RANS_L = 1 << 23
SCALE_BITS = 14
SCALE = 1 << SCALE_BITS


def build_freqs(symbols: np.ndarray, alphabet: int):
    counts = np.bincount(symbols.astype(np.int64), minlength=alphabet).astype(np.int64)
    used = counts > 0
    # normalize to SCALE, guaranteeing every used symbol gets freq >= 1
    freqs = np.zeros(alphabet, dtype=np.int64)
    total = counts.sum()
    # proportional, floored
    scaled = (counts * SCALE) // total
    scaled[used & (scaled == 0)] = 1
    diff = SCALE - scaled.sum()
    # adjust on the largest-count symbols
    order = np.argsort(-counts)
    i = 0
    while diff != 0:
        s = order[i % alphabet]
        if counts[s] > 0:
            if diff > 0:
                scaled[s] += 1
                diff -= 1
            elif scaled[s] > 1:
                scaled[s] -= 1
                diff += 1
        i += 1
    freqs = scaled
    assert freqs.sum() == SCALE
    assert np.all(freqs[used] >= 1)
    return freqs.astype(np.uint32)


def rans_encode(symbols: np.ndarray, freqs: np.ndarray):
    cum = np.zeros(len(freqs) + 1, dtype=np.uint64)
    cum[1:] = np.cumsum(freqs.astype(np.uint64))
    f = freqs.astype(np.uint64)
    out = bytearray()
    x = RANS_L
    syms = symbols.astype(np.int64).tolist()
    fl = f.tolist()
    cuml = cum.tolist()
    L = RANS_L
    SB = SCALE_BITS
    for s in reversed(syms):
        fs = fl[s]
        x_max = ((L >> SB) << 8) * fs
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // fs) << SB) + (x % fs) + cuml[s]
    for _ in range(4):
        out.append(x & 0xFF)
        x >>= 8
    out.reverse()
    return bytes(out)


def rans_decode(data: bytes, freqs: np.ndarray, n: int):
    cum = np.zeros(len(freqs) + 1, dtype=np.int64)
    cum[1:] = np.cumsum(freqs.astype(np.int64))
    # slot -> symbol table
    slot2sym = np.zeros(SCALE, dtype=np.int32)
    for s in range(len(freqs)):
        if freqs[s] > 0:
            slot2sym[cum[s]:cum[s + 1]] = s
    fl = freqs.astype(np.int64).tolist()
    cuml = cum.tolist()
    s2s = slot2sym.tolist()
    mask = SCALE - 1
    pos = 0
    x = 0
    for _ in range(4):
        x = (x << 8) | data[pos]
        pos += 1
    L = RANS_L
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        slot = x & mask
        s = s2s[slot]
        x = fl[s] * (x >> SCALE_BITS) + slot - cuml[s]
        while x < L:
            x = (x << 8) | data[pos]
            pos += 1
        out[i] = s
    return out
