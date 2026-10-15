"""Daily cross-sectional panel: features + labels, strictly point-in-time.

Decision time t = 00:00 UTC each day. A feature at t may only use:
  - 4H bars that *closed* at or before t;
  - daily bars whose day ended at or before t;
  - funding settlements at or before t;
  - rubik positioning data with timestamp + 1 day <= t (conservative: OKX daily
    rubik records are stamped at the start of an UTC+8 day).
Labels use prices after t and are only ever used as training targets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from moneymachine.ml import data as D

log = logging.getLogger(__name__)
DAY = 86_400_000
H4 = 4 * 3_600_000
BTC = "BTC-USDT-SWAP"

# Features that are the same for every coin on a day (not rank-normalised).
MARKET_FEATURES = ["btc_r7", "btc_r30", "btc_rv30", "mkt_disp7", "mkt_fund7", "mkt_r30", "mkt_breadth30"]


@dataclass
class Panel:
    days: pd.Index
    insts: list[str]
    px: pd.DataFrame                    # perp price at decision time t
    universe: pd.DataFrame              # bool, tradable at t
    feats: dict[str, pd.DataFrame]      # raw coin features (days × insts)
    market: pd.DataFrame                # market-wide features (days × k)
    fwd_lr: dict[int, pd.DataFrame]     # log return t → t+h
    fwd_fund: dict[int, pd.DataFrame]   # funding a long pays over (t, t+h]
    fund_daily: pd.DataFrame            # funding over (t, t+1d]
    beta: pd.DataFrame
    vol_daily: pd.DataFrame             # daily vol estimate (from 4H, 30d)
    adv: pd.DataFrame                   # 30d median daily quote volume (USD)
    extra: dict = field(default_factory=dict)


_LOAD_FROM: int | None = None  # set by build_panel(load_from=...) to skip old history


def _wide(kind: str, insts: list[str], col: str, root: Path, time_shift: int = 0) -> pd.DataFrame:
    cols = {}
    for inst in insts:
        p = root / kind / f"{inst}.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["time", col])
            if _LOAD_FROM is not None:
                df = df[df["time"] >= _LOAD_FROM]
            cols[inst] = df.set_index(df["time"].astype("int64") + time_shift)[col]
    return pd.DataFrame(cols).sort_index()


def _asof(series_wide: pd.DataFrame, days: pd.Index) -> pd.DataFrame:
    """Last value with index <= t for each day t."""
    return series_wide.reindex(series_wide.index.union(days)).sort_index().ffill().reindex(days)


def _cum_at(events: pd.DataFrame, days: pd.Index) -> pd.DataFrame:
    """Cumulative sum of event values with timestamp <= t."""
    return _asof(events.fillna(0).cumsum(), days).fillna(0)


def build_panel(universe: pd.DataFrame, root: Path = D.ROOT, start: str = "2020-07-01",
                horizons: tuple[int, ...] = (1, 3), load_from: int | None = None) -> Panel:
    """``load_from`` (UTC ms): ignore raw data older than this — used live to keep memory
    small. Leave ≥ 120 days before ``start`` so every rolling window is complete."""
    global _LOAD_FROM
    _LOAD_FROM = load_from
    try:
        return _build_panel(universe, root, start, horizons)
    finally:
        _LOAD_FROM = None


def _build_panel(universe: pd.DataFrame, root: Path, start: str, horizons: tuple[int, ...]) -> Panel:
    insts = [c for c in universe.columns if universe[c].any()]
    if BTC not in insts:
        insts.append(BTC)
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)

    # 4H bars, indexed by bar *end* (= time the bar is known)
    c4 = _wide("bars_4H", insts, "close", root, H4)
    h4 = _wide("bars_4H", insts, "high", root, H4)
    l4 = _wide("bars_4H", insts, "low", root, H4)
    last = int(c4.index.max())
    days = pd.Index(np.arange(start_ms, last - last % DAY + DAY, DAY), dtype="int64", name="t")
    days = days[days <= last]

    px = c4.reindex(days)                                  # close of the 4H bar ending at t
    lp = np.log(px)
    r4 = np.log(c4).diff()
    feats: dict[str, pd.DataFrame] = {}
    for k in (1, 3, 7, 14, 30, 60, 90):
        feats[f"r{k}"] = lp - lp.shift(k)
    feats["mom_30_7"] = feats["r30"] - feats["r7"]
    feats["r4h"] = r4.reindex(days)
    ann = np.sqrt(6 * 365)
    rv7 = r4.rolling(42, min_periods=30).std() * ann
    rv30 = r4.rolling(180, min_periods=120).std() * ann
    feats["rv7"] = rv7.reindex(days)
    feats["rv30"] = rv30.reindex(days)
    feats["rv_ratio"] = feats["rv7"] / feats["rv30"]
    neg = r4.clip(upper=0)
    pos = r4.clip(lower=0)
    feats["dn_vol30"] = (np.sqrt((neg ** 2).rolling(180, min_periods=120).mean()) * ann).reindex(days)
    feats["up_dn_ratio30"] = (np.sqrt((pos ** 2).rolling(180, min_periods=120).mean())
                              / np.sqrt((neg ** 2).rolling(180, min_periods=120).mean())).reindex(days)
    feats["skew30"] = r4.rolling(180, min_periods=120).skew().reindex(days)
    feats["kurt30"] = r4.rolling(180, min_periods=120).kurt().reindex(days)
    r1 = feats["r1"]
    feats["max30"] = r1.rolling(30, min_periods=20).max()
    feats["min30"] = r1.rolling(30, min_periods=20).min()
    hi30 = h4.rolling(180, min_periods=100).max().reindex(days)
    lo30 = l4.rolling(180, min_periods=100).min().reindex(days)
    feats["pos30"] = (px - lo30) / (hi30 - lo30)
    hi90 = h4.rolling(540, min_periods=300).max().reindex(days)
    feats["dh90"] = px / hi90 - 1

    # beta / idiosyncratic
    rb = r1[BTC]
    cov = r1.rolling(60, min_periods=40).cov(rb)
    var = rb.rolling(60, min_periods=40).var()
    beta = cov.div(var, axis=0)
    feats["beta60"] = beta
    resid = r1 - beta.mul(rb, axis=0)
    feats["idio_r30"] = feats["r30"] - beta.mul(feats["r30"][BTC], axis=0)
    feats["idio_r7"] = feats["r7"] - beta.mul(feats["r7"][BTC], axis=0)
    feats["idio_vol60"] = resid.rolling(60, min_periods=40).std() * np.sqrt(365)
    feats["corr_btc60"] = r1.rolling(60, min_periods=40).corr(rb)

    # liquidity (daily bars, known at day end)
    vq = _wide("bars_1D", insts, "vol_quote", root, DAY).reindex(days)
    o1 = _wide("bars_1D", insts, "open", root, DAY).reindex(days)
    c1 = _wide("bars_1D", insts, "close", root, DAY).reindex(days)
    dret = np.log(c1 / o1).abs()
    adv = vq.rolling(30, min_periods=20).median()
    feats["ladv30"] = np.log(adv)
    feats["vtrend"] = vq.rolling(7, min_periods=5).mean() / vq.rolling(30, min_periods=20).mean()
    feats["amihud30"] = np.log1p((dret / vq).rolling(30, min_periods=20).mean() * 1e9)
    feats["turnover_shock"] = np.log(vq / adv)

    # funding (settlements up to t)
    fr = _wide("funding", insts, "rate", root)
    F = _cum_at(fr, days)
    for k in (1, 3, 7, 30):
        feats[f"fund{k}"] = F - F.shift(k)
    feats["fund_last"] = _asof(fr, days)
    feats["fund_chg"] = feats["fund3"] / 3 - feats["fund30"] / 30

    # basis vs spot (spot daily close known at day end)
    spot_cols = {}
    for inst in insts:
        p = root / "spot_1D" / f"{inst.replace('-SWAP', '')}.parquet"
        if p.exists():
            s = pd.read_parquet(p, columns=["time", "close"])
            spot_cols[inst] = s.set_index(s["time"].astype("int64") + DAY)["close"]
    if spot_cols:
        spot = pd.DataFrame(spot_cols).reindex(days)
        basis = np.log(px[spot.columns] / spot)
        feats["basis"] = basis.reindex(columns=insts)
        feats["basis_chg7"] = (basis - basis.shift(7)).reindex(columns=insts)

    # positioning (2024+)
    for kind, col, name in (("oi", "oi_usd", "oi"), ("ls", "ls_ratio", "ls")):
        w = _wide(f"rubik/{kind}", insts, col, root, DAY)
        if w.empty:
            continue
        a = _asof(w, days)
        if name == "oi":
            feats["oi_chg1"] = np.log(a / a.shift(1))
            feats["oi_chg7"] = np.log(a / a.shift(7))
            feats["oi_to_adv"] = np.log(a / adv)
            feats["oi_x_r7"] = feats["oi_chg7"] * np.sign(feats["r7"])
        else:
            feats["ls"] = np.log(a)
            feats["ls_chg7"] = np.log(a / a.shift(7))
    # Taker buy/sell volume is NOT used: OKX's history endpoint returns zeros before
    # ~2026-07 (verified 2026-10-12: 91% of rows are 0), so it has no usable history.
    # The forward collector (ml/collector.py) accumulates it for the future.

    # market-wide context (computed on the universe only)
    u = universe.reindex(index=days, columns=insts).fillna(False).astype(bool)
    market = pd.DataFrame(index=days)
    market["btc_r7"] = feats["r7"][BTC]
    market["btc_r30"] = feats["r30"][BTC]
    market["btc_rv30"] = feats["rv30"][BTC]
    market["mkt_disp7"] = feats["r7"].where(u).std(axis=1)
    market["mkt_fund7"] = feats["fund7"].where(u).median(axis=1)
    market["mkt_r30"] = feats["r30"].where(u).median(axis=1)
    market["mkt_breadth30"] = (feats["r30"].where(u) > 0).sum(axis=1) / u.sum(axis=1).replace(0, np.nan)

    # labels
    fwd_lr, fwd_fund = {}, {}
    for h in horizons:
        fwd_lr[h] = lp.shift(-h) - lp
        fwd_fund[h] = F.shift(-h) - F
    fund_daily = F.shift(-1) - F
    vol_daily = feats["rv30"] / np.sqrt(365)

    return Panel(days=days, insts=insts, px=px, universe=u, feats=feats, market=market, fwd_lr=fwd_lr,
                 fwd_fund=fwd_fund, fund_daily=fund_daily, beta=beta, vol_daily=vol_daily, adv=adv)


def to_long(panel: Panel, horizon: int, *, min_names: int = 10) -> pd.DataFrame:
    """Stack universe rows into a (t, inst) table with cross-sectionally ranked
    features, raw helpers and the training label.

    Label: forward return net of funding, minus beta × BTC forward return, ranked
    cross-sectionally into [-0.5, 0.5].
    """
    u = panel.universe
    keep_days = u.sum(axis=1) >= min_names
    u = u & keep_days.values[:, None]
    idx = u.stack()
    idx = idx[idx].index
    out = pd.DataFrame(index=idx)
    for name, f in panel.feats.items():
        ranked = f.where(u).rank(axis=1, pct=True) - 0.5
        out[name] = ranked.stack(future_stack=True).reindex(idx)
    m = panel.market.reindex(idx.get_level_values(0))
    for c in MARKET_FEATURES:
        out[c] = m[c].to_numpy()

    y = panel.fwd_lr[horizon] - panel.fwd_fund[horizon]
    y_res = y - panel.beta.mul(y[BTC], axis=0)
    out["y_raw"] = y.stack(future_stack=True).reindex(idx)
    out["y_res"] = y_res.stack(future_stack=True).reindex(idx)
    out["y"] = y_res.where(u).rank(axis=1, pct=True).sub(0.5).stack(future_stack=True).reindex(idx)
    for name, df in (("beta", panel.beta), ("vol_daily", panel.vol_daily), ("adv", panel.adv)):
        out[f"_{name}"] = df.stack(future_stack=True).reindex(idx)
    out.index.names = ["t", "inst"]
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith(("y", "_"))]
