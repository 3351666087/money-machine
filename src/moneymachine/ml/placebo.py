"""Placebo signal for the ML portfolio: random, but with the same persistence.

Why not plain random numbers: the live signal changes slowly from day to day
(per-coin autocorrelation 0.536 at lag 1, 0.324 at lag 5 — measured on the
LightGBM+Kronos signal, 2024-07 → 2026-03). An i.i.d. random signal would churn
far more and pay more costs, making the real model look better than it is.

Model: z = √w · slow + √(1−w) · fast, where ``slow`` is an AR(1) with ρ = 0.8816
and ``fast`` is white noise; w = 0.608 reproduces both autocorrelations exactly
(w·ρ = 0.536, w·ρ⁵ = 0.324).

Determinism: every shock is a hash of (seed, coin, day, stream), and the AR(1)
is a truncated sum over the last ``K`` days — so the value for a given
(seed, coin, day) is identical whether computed live for one day or in a
backtest over years, and does not depend on process restarts.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.special import ndtri

DAY = 86_400_000
RHO_SLOW = 0.8816
W_SLOW = 0.608
K = 80  # 0.8816**81 ≈ 4e-5: truncation error negligible

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLD = np.uint64(0x9E3779B97F4A7C15)


def _key(seed: int, coin: str, stream: str) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.sha256(f"{seed}|{coin}|{stream}".encode()).digest()[:8], "little"))


def _normals(seed: int, coin: str, day_idx: np.ndarray, stream: str) -> np.ndarray:
    """Standard normals indexed by integer day — splitmix64 hash → uniform → inverse CDF."""
    with np.errstate(over="ignore"):
        x = day_idx.astype(np.uint64) * _GOLD + _key(seed, coin, stream)
        x = (x ^ (x >> np.uint64(30))) * _M1
        x = (x ^ (x >> np.uint64(27))) * _M2
        x = x ^ (x >> np.uint64(31))
    u = ((x >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53
    return ndtri(u)


def placebo_signal(coins: list[str], days, seed: int, *, rho: float = RHO_SLOW, w: float = W_SLOW,
                   k: int = K) -> pd.DataFrame:
    """days × coins placebo z-values for decision times ``days`` (UTC ms)."""
    days = np.asarray(sorted(int(d) for d in days), dtype=np.int64)
    idx = days // DAY
    lo, hi = int(idx.min()) - k, int(idx.max())
    span = np.arange(lo, hi + 1, dtype=np.int64)
    kernel = np.sqrt(1 - rho ** 2) * rho ** np.arange(k + 1)
    out = {}
    for c in coins:
        e_slow = _normals(seed, c, span, "slow")
        slow = np.convolve(e_slow, kernel)[: len(span)]  # slow[i] = Σ_j kernel[j]·e[i−j]
        fast = _normals(seed, c, span, "fast")
        z = np.sqrt(w) * slow + np.sqrt(1 - w) * fast
        out[c] = z[idx - lo]
    return pd.DataFrame(out, index=pd.Index(days, name="t"))
