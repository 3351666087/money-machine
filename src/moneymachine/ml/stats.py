"""Performance statistics with multiple-testing corrections.

- Probabilistic / Deflated Sharpe Ratio: Bailey & López de Prado (2014),
  "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
  Overfitting and Non-Normality".
- Probability of Backtest Overfitting via CSCV: Bailey, Borwein, López de Prado
  & Zhu (2017), "The Probability of Backtest Overfitting".
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


def sharpe(r: np.ndarray, periods: int = 365) -> float:
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def sortino(r: np.ndarray, periods: int = 365) -> float:
    r = np.asarray(r, dtype=float)
    dn = r[r < 0]
    if len(dn) < 3:
        return float("nan")
    return float(r.mean() / math.sqrt((dn ** 2).mean()) * math.sqrt(periods))


def max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1 + np.nan_to_num(np.asarray(r, dtype=float)))
    peak = np.maximum.accumulate(eq)
    return float(np.max(1 - eq / peak)) if len(eq) else 0.0


def psr(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P[true per-period Sharpe > sr_benchmark] (both per-period, not annualized)."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 10 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3 = st.skew(r)
    g4 = st.kurtosis(r, fisher=False)
    den = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(st.norm.cdf((sr - sr_benchmark) * math.sqrt(T - 1) / den))


def expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """Expected maximum per-period Sharpe among n_trials skill-less strategies."""
    if n_trials <= 1:
        return 0.0
    z1 = st.norm.ppf(1 - 1.0 / n_trials)
    z2 = st.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(max(var_trial_sr, 0.0)) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(r: np.ndarray, trial_sharpes_per_period: np.ndarray) -> dict:
    """DSR = PSR against the Sharpe you'd expect from the best of N lucky trials."""
    srs = np.asarray(trial_sharpes_per_period, dtype=float)
    srs = srs[~np.isnan(srs)]
    n = max(len(srs), 1)
    sr0 = expected_max_sharpe(n, float(np.var(srs, ddof=1)) if n > 1 else 0.0)
    return {"n_trials": n, "sr0_per_period": sr0, "sr0_annual": sr0 * math.sqrt(365), "dsr": psr(r, sr0)}


def pbo_cscv(returns: np.ndarray, n_blocks: int = 16) -> dict:
    """Probability of Backtest Overfitting.

    returns: T × N matrix (rows = periods, columns = candidate configurations,
    all evaluated out-of-sample). Splits time into ``n_blocks`` blocks; for every
    way of choosing half the blocks as "in-sample", picks the best config IS and
    records its relative rank OOS. PBO = share of splits where it lands in the
    bottom half.
    """
    M = np.asarray(returns, dtype=float)
    T, N = M.shape
    if N < 2:
        return {"pbo": float("nan"), "n_splits": 0}
    n_blocks -= n_blocks % 2
    T_use = T - T % n_blocks
    blocks = np.array_split(np.arange(T_use), n_blocks)
    logits = []
    for is_idx in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[i] for i in is_idx])
        oos_rows = np.concatenate([blocks[i] for i in range(n_blocks) if i not in is_idx])
        sr_is = _col_sharpe(M[is_rows])
        sr_oos = _col_sharpe(M[oos_rows])
        best = int(np.nanargmax(sr_is))
        rank = st.rankdata(sr_oos)[best] / (N + 1)  # relative rank in (0, 1)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_splits": len(logits), "logit_median": float(np.median(logits))}


def _col_sharpe(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, m.mean(axis=0) / sd, -np.inf)


def summary(r: np.ndarray, periods: int = 365) -> dict:
    r = np.asarray(r, dtype=float)
    return {
        "sharpe": sharpe(r, periods),
        "sortino": sortino(r, periods),
        "ann_return": float(np.nanmean(r) * periods),
        "ann_vol": float(np.nanstd(r, ddof=1) * math.sqrt(periods)),
        "max_dd": max_drawdown(r),
        "psr0": psr(r, 0.0),
        "days": int(np.sum(~np.isnan(r))),
    }
