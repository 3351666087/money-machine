"""Small numpy indicator set. Every function returns an array aligned with input."""
from __future__ import annotations

import numpy as np


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty(len(x), dtype=float)
    if len(x) == 0:
        return out
    a = 2.0 / (n + 1)
    v = float(x[0])
    for i, xi in enumerate(x.tolist()):
        v = v + a * (xi - v) if i else v
        out[i] = v
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev = np.concatenate(([close[0]], close[:-1]))
    return np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder ATR."""
    tr = true_range(high, low, close)
    out = np.empty(len(tr), dtype=float)
    v = float(tr[: n].mean()) if len(tr) >= n else float(tr.mean())
    for i, t in enumerate(tr.tolist()):
        if i < n:
            out[i] = float(tr[: i + 1].mean())
            continue
        v = (v * (n - 1) + t) / n
        out[i] = v
    return out


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    out = np.full(len(close), 50.0)
    au = ad = 0.0
    for i in range(1, len(close)):
        if i <= n:
            au = up[1 : i + 1].mean()
            ad = dn[1 : i + 1].mean()
        else:
            au = (au * (n - 1) + up[i]) / n
            ad = (ad * (n - 1) + dn[i]) / n
        out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out
