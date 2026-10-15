"""v4 direction F: funding carry — long spot + short the same quantity of the USDT perpetual,
pre-registered in research/2026-10-15_刷钱机v4_研究计划.md (section 2).

Daily at 00:00 UTC, all prices are daily closes (spot_1D / bars_1D, 1Dutc):
  universe   coins with OKX USDT spot + USDT perp, 30d median quote volume ≥ $20M (perp) and
             ≥ $5M (spot), ≥ 30 days of funding history (point in time)
  signal     funding settled over the last 7 days, annualised
  entry      top K by signal and signal ≥ θ, if the +50%-spike margin test passes
  exit       signal < θ/2, rank > 2K, or out of the universe
  size       spot value 0.9·equity/K per coin at entry; perp short of the same quantity; held unchanged
  costs      spot 0.10% + 1.5× half-spread curve; perp 0.05% + half-spread curve (taker, both legs)
  margin     multi-currency margin: spot counts at OKX's discount rate; maintenance = tier-1 MMR ×
             perp notional; a day where adjusted equity < maintenance is a liquidation (reported)
P&L per held coin over (t, t+1]: qty·[(S₁−S₀) − (P₁−P₀)] + qty·P₀·(funding settled in (t, t+1]).
Discount rates / MMR are today's values (store/ml/carry), used for the whole history.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from moneymachine.ml import data as D
from moneymachine.ml.features import _cum_at, _wide
from moneymachine.ml.portfolio import CostModel
from moneymachine.ml.stats import summary
from moneymachine.ml.validation import HOLDOUT_START

log = logging.getLogger(__name__)
DAY = 86_400_000
ROOT = D.ROOT / "carry"
START = 1_630_454_400_000          # 2021-09-01
END = HOLDOUT_START - 3 * DAY      # 2026-03-29 (exclusive): same end as v3
SPOT_FEE, PERP_FEE = 0.0010, 0.0005


@dataclass
class CarryData:
    days: pd.Index
    coins: list[str]                   # perp inst ids
    spot: pd.DataFrame                 # price at t (close of the day ending at t)
    perp: pd.DataFrame
    fund_daily: pd.DataFrame           # funding rate sum settled in (t, t+1]
    signal: pd.DataFrame               # annualised 7-day funding known at t
    universe: pd.DataFrame
    adv_perp: pd.DataFrame
    adv_spot: pd.DataFrame
    discount: dict = field(default_factory=dict)
    mmr: dict = field(default_factory=dict)


def load(root: Path = D.ROOT) -> CarryData:
    spots = {p.stem for p in (root / "spot_1D").glob("*.parquet")}
    coins = sorted(f"{s}-SWAP" for s in spots if (root / "funding" / f"{s}-SWAP.parquet").exists()
                   and (root / "bars_1D" / f"{s}-SWAP.parquet").exists())
    perp_c = _wide("bars_1D", coins, "close", root, DAY)
    perp_v = _wide("bars_1D", coins, "vol_quote", root, DAY)
    sp = {c: c.replace("-SWAP", "") for c in coins}
    spot_c = _wide("spot_1D", list(sp.values()), "close", root, DAY).rename(columns={v: k for k, v in sp.items()})
    spot_v = _wide("spot_1D", list(sp.values()), "vol_quote", root, DAY).rename(columns={v: k for k, v in sp.items()})
    days = pd.Index(np.arange(START - 40 * DAY, END + DAY, DAY), dtype="int64")
    perp_c, perp_v, spot_c, spot_v = (x.reindex(index=days, columns=coins) for x in (perp_c, perp_v, spot_c, spot_v))
    fr = _wide("funding", coins, "rate", root).reindex(columns=coins)
    F = _cum_at(fr, days)
    first = fr.apply(lambda s: s.first_valid_index())
    has = pd.DataFrame({c: days >= (first[c] if pd.notna(first[c]) else np.inf) for c in coins}, index=days)
    F = F.where(has)
    signal = (F - F.shift(7)) * 365 / 7
    fund_daily = F.shift(-1) - F
    adv_p = perp_v.rolling(30, min_periods=20).median()
    adv_s = spot_v.rolling(30, min_periods=20).median()
    hist = has & has.shift(30, fill_value=False)
    uni = (adv_p >= 20e6) & (adv_s >= 5e6) & hist & spot_c.notna() & perp_c.notna() & signal.notna()
    disc = json.loads((ROOT / "discount_rates.json").read_text())
    mmr = json.loads((ROOT / "mmr_tier1.json").read_text())
    return CarryData(days=days, coins=coins, spot=spot_c, perp=perp_c, fund_daily=fund_daily, signal=signal,
                     universe=uni, adv_perp=adv_p, adv_spot=adv_s,
                     discount={c: disc.get(c.split("-")[0], 0.0) for c in coins},
                     mmr={c: mmr.get(c.replace("-SWAP", ""), 0.02) for c in coins})


def _hs(costs: CostModel, adv: float) -> float:
    return float(np.clip(costs.hs_a * (1e8 / max(adv, 1.0)) ** costs.hs_b, costs.hs_min, costs.hs_max))


def _leg_cost(d: CarryData, costs: CostModel, c: str, k: int, notional: float) -> float:
    """Round-trip-half cost (one leg each of spot and perp) for ``notional`` of coin c at day k."""
    hs_p = _hs(costs, d.adv_perp.iat[k, d.coins.index(c)])
    hs_s = 1.5 * _hs(costs, d.adv_spot.iat[k, d.coins.index(c)])
    return notional * (SPOT_FEE + hs_s + PERP_FEE + hs_p) * costs.multiplier


def _margin_ok(d: CarryData, held: dict, prices: dict, cash: float, spike: str | None = None) -> bool:
    """Multi-currency margin: Σ spot·discount + cash + perp uPnL ≥ Σ MMR·perp notional,
    optionally after a +50% spike of coin ``spike`` (others unchanged)."""
    adj, maint = cash, 0.0
    for c, (qty, p_entry) in held.items():
        s, p = prices[c]
        if c == spike:
            s, p = s * 1.5, p * 1.5
        adj += qty * s * d.discount.get(c, 0.0) + qty * (p_entry - p)
        maint += d.mmr.get(c, 0.02) * qty * p
    return adj >= maint


def backtest(d: CarryData, *, k: int | None, theta: float | None, nav: float = 100_000.0,
             costs: CostModel | None = None, mode: str = "select", start: int = START, end: int = END) -> dict:
    """mode: "select" (F1 rule), "btc_eth" (R1, always on), "all_positive" (R2, equal weight, daily)."""
    costs = costs or CostModel()
    days = d.days[(d.days >= start) & (d.days < end)]
    k0 = int(np.where(d.days == days[0])[0][0])
    held: dict[str, list] = {}                     # coin → [qty, perp entry price]
    equity, cash = nav, nav
    rows = []
    liquidations = []
    for j, t in enumerate(days):
        kk = k0 + j
        S, P = d.spot.iloc[kk], d.perp.iloc[kk]
        prices = {c: (S[c], P[c]) for c in d.coins if np.isfinite(S[c]) and np.isfinite(P[c])}
        u = d.universe.iloc[kk]
        sig = d.signal.iloc[kk]
        cost = 0.0
        # ---- decide
        if mode == "btc_eth":
            want = {"BTC-USDT-SWAP": 0.45, "ETH-USDT-SWAP": 0.45} if not held else {}
            exits = []
        elif mode == "all_positive":
            pos = [c for c in d.coins if u[c] and sig[c] > 0 and c in prices]
            want = {c: 0.9 / len(pos) for c in pos} if pos else {}
            exits = [c for c in held if c not in want]
        else:
            ranked = [c for c in sig[u].sort_values(ascending=False).index if c in prices]
            rank = {c: i for i, c in enumerate(ranked)}
            exits = [c for c in held if c not in rank or sig[c] < theta / 2 or rank[c] >= 2 * k]
            want = {}
            room = k - (len(held) - len(exits))
            for c in ranked[:k]:
                if room <= 0:
                    break
                if c in held or sig[c] < theta:
                    continue
                want[c] = 0.9 / k
                room -= 1
        for c in exits:                                   # close both legs at today's prices
            qty, pe = held.pop(c)
            s, p = prices.get(c, (np.nan, np.nan))
            if np.isfinite(s):
                cost += _leg_cost(d, costs, c, kk, qty * s)
                cash += qty * s + qty * (pe - p)
        if mode == "all_positive":                        # resize to equal weight (trade the difference)
            for c, w in want.items():
                s, p = prices[c]
                tgt = w * equity / s
                qty, pe = held.get(c, [0.0, p])
                dq = tgt - qty
                if abs(dq) * s > 1e-9:
                    cost += _leg_cost(d, costs, c, kk, abs(dq) * s)
                    cash += qty * (pe - p) - dq * s       # realise perp P&L, buy/sell spot
                    held[c] = [tgt, p]
        else:
            for c, w in want.items():
                s, p = prices[c]
                qty = w * equity / s
                trial = {**held, c: [qty, p]}
                if mode == "select" and not _margin_ok(d, trial, prices, cash - qty * s, spike=c):
                    continue
                held[c] = [qty, p]
                cash -= qty * s
                cost += _leg_cost(d, costs, c, kk, qty * s)
        cash -= cost
        # ---- hold over (t, t+1]
        if kk + 1 >= len(d.days):
            break
        S1, P1 = d.spot.iloc[kk + 1], d.perp.iloc[kk + 1]
        fd = d.fund_daily.iloc[kk]
        pnl_basis = pnl_fund = 0.0
        for c, (qty, pe) in held.items():
            s0, p0 = prices[c]
            s1 = S1[c] if np.isfinite(S1[c]) else s0
            p1 = P1[c] if np.isfinite(P1[c]) else p0
            pnl_basis += qty * ((s1 - s0) - (p1 - p0))
            pnl_fund += qty * p0 * (fd[c] if np.isfinite(fd[c]) else 0.0)
        cash += pnl_fund
        new_equity = equity + pnl_basis + pnl_fund - cost
        prices1 = {c: (S1[c] if np.isfinite(S1[c]) else prices[c][0], P1[c] if np.isfinite(P1[c]) else prices[c][1])
                   for c in held}
        if held and not _margin_ok(d, held, prices1, cash):     # cash = USDT balance (spot already paid for)
            liquidations.append(int(t))
        held_val = sum(q * prices1[c][0] + q * (pe - prices1[c][1]) for c, (q, pe) in held.items())
        assert abs(cash + held_val - new_equity) <= 1e-6 * max(nav, abs(new_equity)), "carry ledger out of balance"
        rows.append((int(t), (pnl_basis) / equity, pnl_fund / equity, cost / equity,
                     (new_equity - equity) / equity, len(held), sum(q * prices[c][0] for c, (q, _) in held.items()) / equity))
        equity = new_equity
    daily = pd.DataFrame(rows, columns=["t", "basis", "funding", "cost", "net", "n_held", "gross"]).set_index("t")
    return {"daily": daily, "liquidations": liquidations, "final_equity": equity}


def metrics(res: dict) -> dict:
    d = res["daily"]
    s = summary(d["net"].to_numpy())
    q = d["net"].groupby(pd.to_datetime(d.index, unit="ms").to_period("Q")).sum()
    last = d[d.index >= END - 365 * DAY]["net"]
    yr = d["net"].groupby(pd.to_datetime(d.index, unit="ms").year).sum()
    s.update({"ann_funding": float(d["funding"].mean() * 365), "ann_basis": float(d["basis"].mean() * 365),
              "ann_cost": float(d["cost"].mean() * 365), "pct_pos_quarters": float((q > 0).mean()),
              "last12m_ann": float(last.mean() * 365), "avg_held": float(d["n_held"].mean()),
              "avg_gross": float(d["gross"].mean()), "liquidations": len(res["liquidations"]),
              "by_year": {int(k): float(v) for k, v in yr.items()},
              "start": int(d.index.min()), "end": int(d.index.max())})
    return s


TRIALS = {"F1a": (3, 0.10), "F1b": (3, 0.25), "F1c": (8, 0.10), "F1d": (8, 0.25)}


def run_trials(names: list[str] | None = None) -> pd.DataFrame:
    from dataclasses import asdict, replace

    from moneymachine.ml.registry import Registry

    reg = Registry()
    d = load()
    rows = []
    for name in names or list(TRIALS):
        k, theta = TRIALS[name]
        res = backtest(d, k=k, theta=theta)
        m = metrics(res)
        m["sharpe_cost2x"] = metrics(backtest(d, k=k, theta=theta, costs=replace(CostModel(), multiplier=2.0)))["sharpe"]
        cfg = {"name": name, "strategy": "funding_carry", "k": k, "theta": theta, "costs": asdict(CostModel()),
               "spot_fee": SPOT_FEE, "perp_fee": PERP_FEE, "window": "2021-09+", "size_frac": 0.9}
        reg.record("dev-v4", f"{name}:carry_k{k}_th{int(theta * 100)}", cfg, m, res["daily"]["net"])
        log.info("[dev-v4] %s: Sharpe %.2f | ann %.1f%% | maxDD %.1f%% | last 12m %.1f%% | liquidations %d",
                 name, m["sharpe"], m["ann_return"] * 100, m["max_dd"] * 100, m["last12m_ann"] * 100, m["liquidations"])
        rows.append({"name": name, **{k_: m[k_] for k_ in ("sharpe", "ann_return", "max_dd", "ann_funding", "ann_basis",
                                                             "ann_cost", "pct_pos_quarters", "last12m_ann", "avg_held",
                                                             "liquidations", "sharpe_cost2x")}, **m["by_year"]})
    return pd.DataFrame(rows)
