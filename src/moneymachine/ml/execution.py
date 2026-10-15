"""v4 direction E: post-only (maker) vs market (taker) execution of the daily rebalance,
replayed on 1-minute bars (pre-registered in research/2026-10-15_刷钱机v4_研究计划.md).

Data: for every (coin, day) the v2 signal covers (+ BTC, the concentrated book's hedge),
the 1-minute bars 00:00–04:00 UTC are stored under ``store/ml/m1_open/{inst}.parquet``.

Fill model (conservative, same rule as the paper broker's limit orders):
  - decision price p = the panel price at t (00:00 UTC); the first 1-minute bar opens at o
  - taker: o × (1 ± half-spread) × (1 ± 2bp slippage), fee 0.05%
  - post-only: limit at the touch, o × (1 ∓ half-spread); filled only if a later bar (the
    first one included) trades *through* it by ≥ ``ticks`` price ticks; fill at the limit,
    fee 0.02%; all-or-nothing per bar (no queue, no partial fills)
  - M30 / M240: unfilled after 30 / 240 minutes → taker at that minute's open
  - MO: unfilled after 240 minutes → cancelled, the coin is not rebalanced that day
Every cost is measured against p, so drift while waiting counts as cost (or gain).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from moneymachine.ml import data as D

log = logging.getLogger(__name__)
ROOT = D.ROOT / "m1_open"
MIN = 60_000
WINDOW = 240
BTC = "BTC-USDT-SWAP"


def coin_days(signal: pd.Series) -> dict[str, list[int]]:
    """inst → decision days to fetch: every (day, coin) of the signal, plus BTC each day."""
    idx = signal.index
    out: dict[str, set[int]] = {}
    for t, inst in zip(idx.get_level_values("t"), idx.get_level_values("inst")):
        out.setdefault(inst, set()).add(int(t))
    out.setdefault(BTC, set()).update(int(t) for t in idx.get_level_values("t").unique())
    return {k: sorted(v) for k, v in out.items()}


def _have_days(inst: str) -> set[int]:
    p = ROOT / f"{inst}.parquet"
    if not p.exists():
        return set()
    t = pd.read_parquet(p, columns=["time"])["time"]
    return set((t - t % 86_400_000).unique().tolist())


def sync_one(rest, inst: str, days: list[int]) -> int:
    todo = [d for d in days if d not in _have_days(inst)]
    frames = []
    for d in todo:
        df = D._fetch_candles(rest, inst, "1m", d, d + (WINDOW - 1) * MIN)
        if len(df):
            frames.append(df[["time", "open", "high", "low", "close"]])
    if frames:
        ROOT.mkdir(parents=True, exist_ok=True)
        p = ROOT / f"{inst}.parquet"
        new = pd.concat(frames, ignore_index=True)
        if p.exists():
            new = pd.concat([pd.read_parquet(p), new], ignore_index=True)
        new.drop_duplicates("time").sort_values("time").to_parquet(p, index=False)
    return len(todo)


def sync_all(plan: dict[str, list[int]], workers: int = 4) -> dict:
    from moneymachine.market.okx_rest import OkxRest

    clients = [OkxRest(min_interval=0.45) for _ in range(workers)]   # history-candles: 20 req / 2 s / IP
    items = sorted(plan, key=lambda s: -len(plan[s]))
    from concurrent.futures import ThreadPoolExecutor
    import time

    t0 = time.monotonic()
    done = {}

    def job(i_inst):
        i, inst = i_inst
        try:
            return inst, sync_one(clients[i % workers], inst, plan[inst])
        except Exception as e:  # noqa: BLE001
            log.warning("1m %s failed: %s", inst, e)
            return inst, None

    with ThreadPoolExecutor(workers) as ex:
        for n, (inst, res) in enumerate(ex.map(job, enumerate(items)), 1):
            done[inst] = res
            if n % 10 == 0 or n == len(items):
                log.info("1m windows: %d/%d coins (%.0f min)", n, len(items), (time.monotonic() - t0) / 60)
    return done


# ── replay ───────────────────────────────────────────────────────────────
TAKER_FEE, MAKER_FEE, SLIP = 0.0005, 0.0002, 0.0002
POLICIES = {"taker": None, "M30": 30, "M240": 240, "MO": 240}


def load_minutes(insts: list[str]) -> dict[tuple[str, int], np.ndarray]:
    """(inst, day) → array [minute, (open, high, low)] for 00:00–04:00 UTC."""
    out = {}
    for inst in insts:
        p = ROOT / f"{inst}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        day = df["time"] - df["time"] % 86_400_000
        for d, g in df.groupby(day):
            g = g.sort_values("time")
            m = ((g["time"] - d) // MIN).to_numpy()
            a = np.full((WINDOW, 3), np.nan)
            ok = (m >= 0) & (m < WINDOW)
            a[m[ok]] = g[["open", "high", "low"]].to_numpy()[ok]
            out[(inst, int(d))] = a
    return out


@dataclass
class MinuteExecutor:
    """Executes one day's weight changes under ``policy`` (see module docstring).
    Called by portfolio.run / run_concentrated with weights before and after the decision."""
    policy: str
    px: pd.DataFrame                    # decision prices (panel.px)
    minutes: dict
    ticks: dict                         # inst → tick size
    costs: object                       # CostModel (half-spread curve + impact)
    through: int = 1                    # ticks the market must trade through a resting order
    real_spread: dict | None = None     # inst → measured half-spread (sensitivity: replaces the curve)

    def __post_init__(self):
        self.stats = {"orders": 0, "maker": 0, "fallback": 0, "cancelled": 0, "no_data": 0,
                      "notional": 0.0, "maker_notional": 0.0, "fill_minutes": []}

    def _half_spread(self, adv: float) -> float:
        c = self.costs
        return float(np.clip(c.hs_a * (1e8 / max(adv, 1.0)) ** c.hs_b, c.hs_min, c.hs_max))

    def _taker(self, price: float, side: float, hs: float, p: float) -> float:
        f = price * (1 + side * (hs + SLIP))
        return side * (f - p) / p + TAKER_FEE

    def __call__(self, t: int, insts: list[str], w_prev: np.ndarray, w_target: np.ndarray, adv: np.ndarray,
                 vold: np.ndarray, nav: float) -> tuple[np.ndarray, np.ndarray]:
        w_real = w_target.copy()
        cost = np.zeros(len(w_prev))
        wait = POLICIES[self.policy]
        for i in np.nonzero(np.abs(w_target - w_prev) > 1e-12)[0]:
            inst, dw = insts[i], w_target[i] - w_prev[i]
            side = 1.0 if dw > 0 else -1.0
            hs = self.real_spread.get(inst, self._half_spread(adv[i])) if self.real_spread else self._half_spread(adv[i])
            p = self.px.at[t, inst] if inst in self.px.columns else np.nan
            bars = self.minutes.get((inst, t))
            self.stats["orders"] += 1
            self.stats["notional"] += abs(dw)
            impact = float(self.costs.impact(np.array([vold[i]]), np.array([adv[i]]), nav)[0]) * abs(dw) ** 1.5
            if not np.isfinite(p) or bars is None or not np.isfinite(bars[0, 0]):
                self.stats["no_data"] += 1               # research cost model as the fallback
                cost[i] = abs(dw) * (TAKER_FEE + hs) + impact
                continue
            o = bars[0, 0]
            if wait is None:                              # E0: market order at the first minute
                cost[i] = abs(dw) * self._taker(o, side, hs, p) + impact
                continue
            tick = self.ticks.get(inst, 0.0) or o * 1e-5
            lim = o * (1 - side * hs)
            lim = np.floor(lim / tick) * tick if side > 0 else np.ceil(lim / tick) * tick
            w = bars[:wait]
            hit = (w[:, 2] <= lim - self.through * tick) if side > 0 else (w[:, 1] >= lim + self.through * tick)
            hit = np.nan_to_num(hit, nan=0).astype(bool)
            if hit.any():
                self.stats["maker"] += 1
                self.stats["maker_notional"] += abs(dw)
                self.stats["fill_minutes"].append(int(np.argmax(hit)))
                cost[i] = abs(dw) * (side * (lim - p) / p + MAKER_FEE) + impact
            elif self.policy == "MO":
                self.stats["cancelled"] += 1
                w_real[i] = w_prev[i]
            else:
                self.stats["fallback"] += 1
                nxt = bars[wait, 0] if wait < WINDOW and np.isfinite(bars[wait, 0]) else bars[np.isfinite(bars[:, 0])][-1, 0]
                cost[i] = abs(dw) * self._taker(nxt, side, hs, p) + impact
        return w_real, cost

    def summary(self) -> dict:
        s = self.stats
        n = max(s["orders"] - s["no_data"], 1)
        return {"orders": s["orders"], "no_data": s["no_data"], "maker_fill_rate": s["maker"] / n,
                "maker_notional_share": s["maker_notional"] / max(s["notional"], 1e-12),
                "fallback_rate": s["fallback"] / n, "cancel_rate": s["cancelled"] / n,
                "median_fill_minute": float(np.median(s["fill_minutes"])) if s["fill_minutes"] else float("nan")}


def measured_half_spreads(path: str = "store/ml/spread_snapshot.csv") -> dict[str, float]:
    """Half-spreads measured from live order books (2026-10-12 snapshot), at least half a tick."""
    d = pd.read_csv(path)
    return {r.inst: float(max(r.half_spread, r.half_tick)) for r in d.itertuples()}
