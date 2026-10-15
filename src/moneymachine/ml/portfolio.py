"""Cost-aware portfolio construction + daily research backtest.

Each day t we solve (cvxpy / Clarabel):

    maximize   αᵀw − κ·[cᵀ|w − w₀| + dᵀ|w − w₀|^1.5] − λ·wᵀΣw
    subject to βᵀw = 0              (BTC-beta neutral)
               ‖w‖₁ ≤ gross_max
               |wᵢ| ≤ capᵢ          (also ≤ a fraction of the coin's daily volume)
               wᵀΣw ≤ σ²_target     (ex-ante volatility target)
               wᵢ = 0 for coins outside today's universe

- α: Grinold's rule α = IC · σ · z (z = cross-sectional z-score of the signal).
- costs: c = taker fee + half-spread (calibrated vs liquidity); d = square-root
  market impact Y·σ·√(NAV/ADV). Trading only partially toward the frictionless
  optimum falls out of this objective (Gârleanu & Pedersen 2013); κ > 1
  amortises costs over the signal's holding horizon (Jensen–Kelly–Malamud–
  Pedersen's "implementable efficient frontier" idea: optimise *net* of costs).
- Σ: one-factor model β βᵀ σ²_BTC + diag(idiosyncratic variance).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict

import cvxpy as cp
import numpy as np
import pandas as pd

from moneymachine.ml.features import BTC, Panel

log = logging.getLogger(__name__)


@dataclass
class CostModel:
    fee: float = 0.0005                # taker
    # half-spread = hs_a * (1e8 / ADV) ** hs_b, floored at hs_min. Calibrated on 185 live
    # OKX order books (2026-10-12) as the 75th-percentile curve, i.e. conservative:
    # 79% of coins' actual half-spread is at or below it (store/ml/spread_calibration.json).
    hs_a: float = 0.66e-4
    hs_b: float = 0.35
    hs_min: float = 1.0e-4
    hs_max: float = 25e-4
    impact_y: float = 0.5              # square-root law coefficient
    multiplier: float = 1.0            # cost stress (2.0 = everything twice as expensive)

    def linear(self, adv: np.ndarray) -> np.ndarray:
        hs = np.clip(self.hs_a * (1e8 / np.maximum(adv, 1.0)) ** self.hs_b, self.hs_min, self.hs_max)
        return (self.fee + hs) * self.multiplier

    def impact(self, vol_daily: np.ndarray, adv: np.ndarray, nav: float) -> np.ndarray:
        return self.impact_y * vol_daily * np.sqrt(nav / np.maximum(adv, 1.0)) * self.multiplier


@dataclass
class PortfolioConfig:
    ic: float = 0.03                   # assumed information coefficient of the signal
    alpha_horizon: int = 1             # days the signal predicts; α is per-day
    kappa: float = 1.0                 # cost aversion (amortisation)
    risk_aversion: float = 1.0
    vol_target: float = 0.15           # annualised ex-ante
    gross_max: float = 2.0
    cap: float = 0.10                  # max |weight| per coin
    adv_frac: float = 0.01             # max position as fraction of ADV
    beta_neutral: bool = True
    nav: float = 100_000.0             # used for impact + ADV cap (capacity analysis)
    # "grinold": α = IC·σᵢ·z (classic); "rank": α = IC·median(σ)·z — expected return scales
    # with rank only. Diagnosis 2026-10-12: model quintile returns are ~linear in rank and
    # do not scale with volatility, so Grinold over-weights shorting high-vol names.
    alpha_mode: str = "grinold"
    # v3 trial A2: subtract the expected daily funding a long pays (trailing 3-day mean,
    # known at t) from α, so the optimizer prefers positions that collect funding.
    funding_in_alpha: bool = False


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m, s = np.nanmean(x), np.nanstd(x)
    z = (x - m) / s if s > 0 else np.zeros_like(x)
    return np.nan_to_num(np.clip(z, -3, 3))


def optimize(signal: np.ndarray, w0: np.ndarray, active: np.ndarray, beta: np.ndarray, idio_var: np.ndarray,
             btc_var: float, vol_daily: np.ndarray, adv: np.ndarray, cfg: PortfolioConfig,
             costs: CostModel, exp_funding: np.ndarray | None = None) -> np.ndarray:
    n = len(w0)
    z = np.zeros(n)
    z[active] = _zscore(signal[active])
    if cfg.alpha_mode == "rank":
        scale = float(np.median(vol_daily[active])) if active.any() else 0.04
        alpha = cfg.ic * scale * z / max(cfg.alpha_horizon, 1)
    else:
        alpha = cfg.ic * vol_daily * z / max(cfg.alpha_horizon, 1)
    if exp_funding is not None:
        alpha = alpha - np.where(active, exp_funding, 0.0)
    c = costs.linear(adv)
    d = costs.impact(vol_daily, adv, cfg.nav)
    cap = np.minimum(cfg.cap, cfg.adv_frac * adv / cfg.nav)
    cap = np.where(active, cap, 0.0)
    B = beta.reshape(-1, 1)
    w = cp.Variable(n)
    dw = w - w0
    risk = btc_var * cp.square(B.T @ w)[0] + cp.sum(cp.multiply(idio_var, cp.square(w)))
    obj = alpha @ w - cfg.kappa * (c @ cp.abs(dw) + d @ cp.power(cp.abs(dw), 1.5)) - cfg.risk_aversion * risk
    cons = [cp.norm1(w) <= cfg.gross_max, cp.abs(w) <= cap, risk <= (cfg.vol_target ** 2) / 365]
    if cfg.beta_neutral:
        cons.append(beta @ w == 0)
    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        prob.solve(solver=cp.SCS)
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return np.where(active, w0, 0.0)
    out = np.asarray(w.value).ravel()
    out[np.abs(out) < 1e-6] = 0.0
    return out


@dataclass
class BacktestResult:
    daily: pd.DataFrame                # per day: gross, cost, funding, net, turnover, gross_exp, beta_exp, n_long, n_short
    weights: pd.DataFrame              # days × insts
    pnl_by_inst: pd.Series
    config: dict


def run(panel: Panel, signal: pd.DataFrame, cfg: PortfolioConfig, costs: CostModel,
        start: int | None = None, end: int | None = None, executor=None) -> BacktestResult:
    """Daily rebalance at t, hold over (t, t+1]. ``signal``: days × insts (NaN = no view).
    ``executor(t, insts, w_prev, w_target, adv, vol, nav) → (w_realised, cost per coin)``
    replaces the cost model with an execution simulation (v4, ml/execution.py)."""
    days = panel.days
    if start is not None:
        days = days[days >= start]
    if end is not None:
        days = days[days < end]
    insts = panel.insts
    col = {s: i for i, s in enumerate(insts)}
    n = len(insts)
    sig = signal.reindex(index=days, columns=insts)
    U = panel.universe.reindex(index=days, columns=insts).fillna(False).to_numpy()
    beta = panel.beta.reindex(index=days, columns=insts).to_numpy()
    idio = panel.feats["idio_vol60"].reindex(index=days, columns=insts).to_numpy() / math.sqrt(365)
    vold = panel.vol_daily.reindex(index=days, columns=insts).to_numpy()
    adv = panel.adv.reindex(index=days, columns=insts).to_numpy()
    r_fwd = np.expm1(panel.fwd_lr[1].reindex(index=days, columns=insts).to_numpy())
    fund = panel.fund_daily.reindex(index=days, columns=insts).to_numpy()
    btc_vol = panel.vol_daily[BTC].reindex(days).to_numpy()
    ef = (np.nan_to_num(panel.feats["fund3"].reindex(index=days, columns=insts).to_numpy() / 3)
          if cfg.funding_in_alpha else None)

    w_prev = np.zeros(n)
    W = np.zeros((len(days), n))
    rows = []
    pnl_inst = np.zeros(n)
    for k, t in enumerate(days):
        s = sig.iloc[k].to_numpy()
        active = U[k] & ~np.isnan(s) & ~np.isnan(beta[k]) & ~np.isnan(vold[k]) & ~np.isnan(adv[k])
        held = np.abs(w_prev) > 0
        consider = active | held
        b = np.nan_to_num(beta[k], nan=1.0)
        iv = np.nan_to_num(idio[k], nan=np.nanmedian(idio[k]) if np.isfinite(np.nanmedian(idio[k])) else 0.03) ** 2
        vd = np.nan_to_num(vold[k], nan=0.05)
        a = np.nan_to_num(adv[k], nan=1e6)
        bv = (btc_vol[k] if np.isfinite(btc_vol[k]) else 0.03) ** 2
        sub = np.where(consider)[0]
        w_new = np.zeros(n)
        if len(sub):
            w_new[sub] = optimize(np.nan_to_num(s[sub]), w_prev[sub], active[sub], b[sub], iv[sub], bv,
                                  vd[sub], a[sub], cfg, costs, None if ef is None else ef[k][sub])
        if executor is None:
            trade = w_new - w_prev
            cvec = costs.linear(a) * np.abs(trade) + costs.impact(vd, a, cfg.nav) * np.abs(trade) ** 1.5
        else:
            w_new, cvec = executor(int(t), insts, w_prev, w_new, a, vd, cfg.nav)
            trade = w_new - w_prev
        cost = float(np.sum(cvec))
        r = np.nan_to_num(r_fwd[k])
        f = np.nan_to_num(fund[k])
        gross = float(w_new @ r)
        funding = float(-(w_new @ f))
        net = gross + funding - cost
        pnl_inst += w_new * r - w_new * f - cvec
        rows.append((t, gross, funding, cost, net, float(np.abs(trade).sum()), float(np.abs(w_new).sum()),
                     float(w_new @ b), int((w_new > 0).sum()), int((w_new < 0).sum()), int(active.sum())))
        W[k] = w_new
        growth = 1 + net
        w_prev = w_new * (1 + r) / growth if growth > 0 else np.zeros(n)
    daily = pd.DataFrame(rows, columns=["t", "gross", "funding", "cost", "net", "turnover", "gross_exp",
                                        "beta_exp", "n_long", "n_short", "n_active"]).set_index("t")
    return BacktestResult(daily=daily, weights=pd.DataFrame(W, index=days, columns=insts),
                          pnl_by_inst=pd.Series(pnl_inst, index=insts), config={**asdict(cfg), **asdict(costs)})


def run_quantile(panel: Panel, signal: pd.DataFrame, costs: CostModel, *, frac: float = 0.2, gross: float = 1.0,
                 band: float = 0.0, nav: float = 100_000.0, start: int | None = None,
                 end: int | None = None) -> BacktestResult:
    """Simple construction: equal-weight long the top ``frac`` / short the bottom ``frac``
    of the day's universe by signal, beta-neutralised with BTC, daily rebalance.
    ``band``: skip trades smaller than this many units of target weight (no-trade zone)."""
    days = panel.days
    if start is not None:
        days = days[days >= start]
    if end is not None:
        days = days[days < end]
    insts = panel.insts
    n = len(insts)
    btc = insts.index(BTC)
    sig = signal.reindex(index=days, columns=insts).to_numpy()
    U = panel.universe.reindex(index=days, columns=insts).fillna(False).to_numpy()
    beta = panel.beta.reindex(index=days, columns=insts).to_numpy()
    vold = np.nan_to_num(panel.vol_daily.reindex(index=days, columns=insts).to_numpy(), nan=0.05)
    adv = np.nan_to_num(panel.adv.reindex(index=days, columns=insts).to_numpy(), nan=1e6)
    r_fwd = np.nan_to_num(np.expm1(panel.fwd_lr[1].reindex(index=days, columns=insts).to_numpy()))
    fund = np.nan_to_num(panel.fund_daily.reindex(index=days, columns=insts).to_numpy())
    w_prev = np.zeros(n)
    rows, W = [], np.zeros((len(days), n))
    pnl_inst = np.zeros(n)
    for k, t in enumerate(days):
        ok = U[k] & ~np.isnan(sig[k]) & ~np.isnan(beta[k])
        w = np.zeros(n)
        idx = np.where(ok)[0]
        if len(idx) >= 10:
            m = max(1, int(len(idx) * frac))
            order = idx[np.argsort(sig[k][idx])]
            w[order[-m:]] = gross / 2 / m
            w[order[:m]] = -gross / 2 / m
            w[btc] -= float(np.nansum(w * np.nan_to_num(beta[k])))  # BTC hedge → beta neutral
        if band > 0:
            keep = np.abs(w - w_prev) < band * (gross / 2 / max(1, int(len(idx) * frac)))
            w[keep] = w_prev[keep]
        trade = w - w_prev
        lin = costs.linear(adv[k])
        cost = float(np.sum(lin * np.abs(trade) + costs.impact(vold[k], adv[k], nav) * np.abs(trade) ** 1.5))
        gross_ret = float(w @ r_fwd[k])
        funding = float(-(w @ fund[k]))
        net = gross_ret + funding - cost
        pnl_inst += w * r_fwd[k] - w * fund[k] - lin * np.abs(trade)
        rows.append((t, gross_ret, funding, cost, net, float(np.abs(trade).sum()), float(np.abs(w).sum()),
                     float(w @ np.nan_to_num(beta[k], nan=1.0)), int((w > 0).sum()), int((w < 0).sum()), int(ok.sum())))
        W[k] = w
        growth = 1 + net
        w_prev = w * (1 + r_fwd[k]) / growth if growth > 0 else np.zeros(n)
    daily = pd.DataFrame(rows, columns=["t", "gross", "funding", "cost", "net", "turnover", "gross_exp",
                                        "beta_exp", "n_long", "n_short", "n_active"]).set_index("t")
    return BacktestResult(daily=daily, weights=pd.DataFrame(W, index=days, columns=insts),
                          pnl_by_inst=pd.Series(pnl_inst, index=insts),
                          config={"construction": "quantile", "frac": frac, "gross": gross, "band": band, **asdict(costs)})


@dataclass
class Lots:
    """OKX order granularity in coins: orders are multiples of ``lot`` and at least ``min``."""
    lot: np.ndarray
    min: np.ndarray

    @classmethod
    def load(cls, insts: list[str], path: str = "store/ml/instruments.parquet") -> "Lots":
        d = pd.read_parquet(path).set_index("inst").reindex(insts)
        return cls(np.nan_to_num((d["ct_val"] * d["lot_sz"]).to_numpy(float)),
                   np.nan_to_num((d["ct_val"] * d["min_sz"]).to_numpy(float)))

    def round(self, i: int, q: float) -> float:
        if self.lot[i] > 0:
            q = round(q / self.lot[i]) * self.lot[i]
        return 0.0 if abs(q) + 1e-12 < self.min[i] else q


def run_concentrated(panel: Panel, signal: pd.DataFrame, cfg: PortfolioConfig, costs: CostModel,
                     start: int | None = None, end: int | None = None, *, n: int, buffer: int | None = None,
                     nav: float = 1000.0, lots: Lots | None = None, executor=None) -> BacktestResult:
    """Small-account construction (v3 trial B1): hold the ``n`` strongest longs and shorts.

    - enter only from the top / bottom ``n`` of the day's universe; exit once a coin drops
      out of the top / bottom ``buffer`` (default 2n) or leaves the universe
    - each new position gets 0.5·equity/n notional, rounded to the contract lot; coins whose
      lot rounds to zero are skipped (counted). Held positions keep their quantity (no drift
      rebalancing), so trades happen only on entries, exits and the hedge
    - BTC is the hedge, not a candidate: its position offsets the legs' beta exposure each
      day, rounded to its lot
    - equity compounds from ``nav``; costs use the same model as ``run`` with impact at the
      current equity. ``cfg`` is unused (kept for the ``evaluate`` runner signature).
    """
    days = panel.days
    if start is not None:
        days = days[days >= start]
    if end is not None:
        days = days[days < end]
    insts = panel.insts
    N = len(insts)
    btc = insts.index(BTC)
    lots = lots or Lots.load(insts)
    buffer = buffer or 2 * n
    sig = signal.reindex(index=days, columns=insts).to_numpy()
    U = panel.universe.reindex(index=days, columns=insts).fillna(False).to_numpy()
    beta = panel.beta.reindex(index=days, columns=insts).to_numpy()
    px = panel.px.reindex(index=days, columns=insts).ffill(limit=3).to_numpy()
    vold = np.nan_to_num(panel.vol_daily.reindex(index=days, columns=insts).to_numpy(), nan=0.05)
    adv = np.nan_to_num(panel.adv.reindex(index=days, columns=insts).to_numpy(), nan=1e6)
    r_fwd = np.nan_to_num(np.expm1(panel.fwd_lr[1].reindex(index=days, columns=insts).to_numpy()))
    fund = np.nan_to_num(panel.fund_daily.reindex(index=days, columns=insts).to_numpy())

    q = np.zeros(N)                    # signed coins held
    equity = nav
    rows, W = [], np.zeros((len(days), N))
    pnl_inst = np.zeros(N)
    skipped = 0
    for k, t in enumerate(days):
        p = px[k]
        priced = np.isfinite(p) & (p > 0)
        ok = U[k] & np.isfinite(sig[k]) & np.isfinite(beta[k]) & priced
        ok[btc] = False
        idx = np.where(ok)[0]
        target = q.copy()
        if equity <= 0:
            target[:] = 0.0
        elif len(idx) >= 2 * n:
            order = idx[np.argsort(-sig[k][idx], kind="stable")]  # strongest long first
            rank = np.full(N, N)
            rank[order] = np.arange(len(order))
            rank_short = np.where(rank < N, len(order) - 1 - rank, N)
            for i in np.nonzero(q)[0]:
                if i == btc:
                    continue
                if not ok[i] or (q[i] > 0 and rank[i] >= buffer) or (q[i] < 0 and rank_short[i] >= buffer):
                    target[i] = 0.0
            per = 0.5 * equity / n
            for side, ordered in ((1.0, order), (-1.0, order[::-1])):
                held = int(sum(1 for i in np.nonzero(target * side > 0)[0] if i != btc))
                for i in ordered[:n]:
                    if held >= n:
                        break
                    if target[i] != 0:
                        continue
                    qi = lots.round(i, side * per / p[i])
                    if qi == 0:
                        skipped += 1
                        continue
                    target[i] = qi
                    held += 1
        legs = np.arange(N) != btc
        bb = np.nan_to_num(beta[k], nan=1.0)
        expo = float(np.nansum((target * np.nan_to_num(p) * bb)[legs]))
        target[btc] = lots.round(btc, -expo / p[btc]) if priced[btc] and equity > 0 else 0.0

        lin = costs.linear(adv[k])
        if executor is not None and equity > 0:
            w0, w1 = np.nan_to_num(q * p) / equity, np.nan_to_num(target * p) / equity
            w_real, cvec = executor(int(t), insts, w0, w1, adv[k], vold[k], equity)
            target = np.where(np.abs(w_real - w0) < 1e-15, q, target)     # cancelled orders keep the old qty
            trade_w = np.abs(np.nan_to_num((target - q) * p)) / equity
            cost = float(np.sum(cvec))
        else:
            trade_w = np.abs(np.nan_to_num((target - q) * p)) / equity if equity > 0 else np.zeros(N)
            cvec = lin * trade_w
            cost = float(np.sum(lin * trade_w + costs.impact(vold[k], adv[k], max(equity, 1.0)) * trade_w ** 1.5))
        val = np.nan_to_num(target * p)
        q = target
        w = val / equity if equity > 0 else np.zeros(N)
        gross = float(w @ r_fwd[k])
        funding = float(-(w @ fund[k]))
        net = gross + funding - cost
        pnl_inst += w * r_fwd[k] - w * fund[k] - cvec
        rows.append((t, gross, funding, cost, net, float(trade_w.sum()), float(np.abs(w).sum()), float(w @ bb),
                     int((w[legs] > 0).sum()), int((w[legs] < 0).sum()), int(ok.sum())))
        W[k] = w
        equity *= 1 + net
    daily = pd.DataFrame(rows, columns=["t", "gross", "funding", "cost", "net", "turnover", "gross_exp",
                                        "beta_exp", "n_long", "n_short", "n_active"]).set_index("t")
    return BacktestResult(daily=daily, weights=pd.DataFrame(W, index=days, columns=insts),
                          pnl_by_inst=pd.Series(pnl_inst, index=insts),
                          config={"construction": "concentrated", "n": n, "buffer": buffer, "nav": nav,
                                  "skipped_min_order": skipped, "final_equity": equity, **asdict(costs)})
