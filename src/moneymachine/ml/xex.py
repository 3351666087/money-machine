"""Cross-exchange data (v3 trial C1): Binance + Bybit USDT perpetuals vs OKX.

Stored per OKX instrument under ``store/ml/xex``:
  {venue}_funding/{inst}.parquet   time (settlement, UTC ms), rate
  {venue}_1d/{inst}.parquet        time (UTC day start), close  — price per ONE coin

Symbol mapping handles multiplied contracts (e.g. Binance 1000PEPEUSDT trades
1000 PEPE; Bybit SHIB1000USDT) by dividing prices by the multiplier. Funding
rates are dimensionless and need no scaling.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from moneymachine.ml import data as D

log = logging.getLogger(__name__)
ROOT = D.ROOT / "xex"
START = 1_609_459_200_000  # 2021-01-01
_MULT = re.compile(r"^(1000000|100000|10000|1000|1M)?([A-Z0-9]+?)(1000000|10000|1000)?$")


class _Client:
    def __init__(self, base: str, min_interval: float):
        self.http = httpx.Client(base_url=base, timeout=30, headers={"User-Agent": "moneymachine/0.1"})
        self.min_interval = min_interval
        self._last = 0.0

    def get(self, path: str, params: dict):
        params = {k: v for k, v in params.items() if v is not None}
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.http.get(path, params=params)
                if r.status_code in (418, 429) or r.status_code >= 500:
                    raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                if attempt == 4:
                    raise
                log.warning("%s %s failed (%s), retrying", self.http.base_url, path, e)
                time.sleep(2 ** attempt * 2)


def _split(base: str) -> tuple[str, float]:
    m = _MULT.match(base)
    if not m:
        return base, 1.0
    pre, core, suf = m.groups()
    if pre:
        mult = 1e6 if pre == "1M" else float(pre)
    else:
        mult = float(suf) if suf else 1.0
    return core, mult


def symbol_maps() -> dict[str, dict[str, tuple[str, float]]]:
    """{venue: {coin: (exchange symbol, multiplier)}} for USDT perpetuals."""
    out: dict[str, dict[str, tuple[str, float]]] = {"binance": {}, "bybit": {}}
    bn = _Client("https://fapi.binance.com", 0.2).get("/fapi/v1/exchangeInfo", {})
    for s in bn["symbols"]:
        if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT":
            coin, mult = _split(s["baseAsset"])
            out["binance"].setdefault(coin, (s["symbol"], mult))
    by = _Client("https://api.bybit.com", 0.1)
    cursor = None
    while True:
        j = by.get("/v5/market/instruments-info", {"category": "linear", "limit": 1000, "cursor": cursor})
        for s in j["result"]["list"]:
            if s.get("contractType") == "LinearPerpetual" and s.get("quoteCoin") == "USDT":
                coin, mult = _split(s["baseCoin"])
                out["bybit"].setdefault(coin, (s["symbol"], mult))
        cursor = j["result"].get("nextPageCursor")
        if not cursor:
            break
    return out


def _save(kind: str, inst: str, df: pd.DataFrame) -> None:
    p = ROOT / kind / f"{inst}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
    df.drop_duplicates("time", keep="last").sort_values("time").to_parquet(p, index=False)


def _last_time(kind: str, inst: str) -> int | None:
    p = ROOT / kind / f"{inst}.parquet"
    return int(pd.read_parquet(p, columns=["time"])["time"].max()) if p.exists() else None


def sync_binance(c: _Client, inst: str, sym: str, mult: float) -> int:
    n = 0
    start = (_last_time("binance_funding", inst) or START - 1) + 1
    rows = []
    while True:
        j = c.get("/fapi/v1/fundingRate", {"symbol": sym, "startTime": start, "limit": 1000})
        if not j:
            break
        rows += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in j]
        if len(j) < 1000:
            break
        start = int(j[-1]["fundingTime"]) + 1
    if rows:
        _save("binance_funding", inst, pd.DataFrame(rows, columns=["time", "rate"]))
        n += len(rows)
    start = (_last_time("binance_1d", inst) or START - 86_400_000) + 86_400_000
    rows = []
    now = int(time.time() * 1000)
    while start < now - 86_400_000:
        j = c.get("/fapi/v1/klines", {"symbol": sym, "interval": "1d", "startTime": start, "limit": 1000})
        if not j:
            break
        rows += [(int(k[0]), float(k[4]) / mult) for k in j if int(k[6]) < now]  # closed days only
        start = int(j[-1][0]) + 86_400_000
        if len(j) < 1000:
            break
    if rows:
        _save("binance_1d", inst, pd.DataFrame(rows, columns=["time", "close"]))
        n += len(rows)
    return n


def sync_bybit(c: _Client, inst: str, sym: str, mult: float) -> int:
    n = 0
    stop = _last_time("bybit_funding", inst) or START
    rows, end = [], None
    while True:
        j = c.get("/v5/market/funding/history", {"category": "linear", "symbol": sym, "limit": 200, "endTime": end})
        lst = j.get("result", {}).get("list", [])
        if not lst:
            break
        rows += [(int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in lst]
        oldest = min(int(x["fundingRateTimestamp"]) for x in lst)
        if oldest <= stop or (end is not None and oldest >= end):
            break
        end = oldest - 1
    rows = [r for r in rows if r[0] > stop or stop == START]
    if rows:
        _save("bybit_funding", inst, pd.DataFrame(rows, columns=["time", "rate"]))
        n += len(rows)
    stop = _last_time("bybit_1d", inst) or START
    now = int(time.time() * 1000)
    rows, end = [], now - now % 86_400_000 - 1  # exclude today's unfinished candle
    while True:
        j = c.get("/v5/market/kline", {"category": "linear", "symbol": sym, "interval": "D", "limit": 1000,
                                       "end": end, "start": stop})
        lst = j.get("result", {}).get("list", [])
        if not lst:
            break
        rows += [(int(k[0]), float(k[4]) / mult) for k in lst]
        oldest = min(int(k[0]) for k in lst)
        if oldest <= stop or len(lst) < 1000:
            break
        end = oldest - 1
    if rows:
        _save("bybit_1d", inst, pd.DataFrame(rows, columns=["time", "close"]))
        n += len(rows)
    return n


def sync_all(insts: list[str], binance_interval: float = 0.65) -> dict:
    """Incremental: a re-run only fetches what is missing (e.g. after Binance's WAF answered 403)."""
    maps = symbol_maps()
    bn = _Client("https://fapi.binance.com", binance_interval)  # fundingRate: 500 requests / 5 min / IP
    by = _Client("https://api.bybit.com", 0.12)
    stats = {"binance": 0, "bybit": 0, "missing_binance": [], "missing_bybit": []}
    for k, inst in enumerate(insts, 1):
        coin = inst.split("-")[0]
        for venue, client, fn in (("binance", bn, sync_binance), ("bybit", by, sync_bybit)):
            m = maps[venue].get(coin)
            if m is None:
                stats[f"missing_{venue}"].append(coin)
                continue
            try:
                fn(client, inst, *m)
                stats[venue] += 1
            except Exception as e:  # noqa: BLE001
                log.warning("%s %s: %s", venue, inst, e)
        if k % 20 == 0:
            log.info("cross-exchange: %d/%d", k, len(insts))
    return stats


# ── features (v3 trial C1) ───────────────────────────────────────────────
VENUES = ("binance", "bybit")
MAX_MEDIAN_PREMIUM = 0.02  # a venue whose price differs by >2% in the median is a different token


def features(panel, root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Cross-exchange features known at each decision time t (days × insts):

    xf_fund{1,3,7}  mean funding over the last k days on Binance/Bybit (venues with data)
    xf_fdiff{1,3,7} OKX funding − that mean, same window
    xf_prem         log(OKX price at t / mean Binance/Bybit close of the day ending at t)
    xf_prem_chg3    3-day change of xf_prem
    """
    from moneymachine.ml.features import _cum_at, _wide

    days, insts = panel.days, panel.insts
    okx_F = None
    fund_k: dict[int, list[pd.DataFrame]] = {1: [], 3: [], 7: []}
    closes = []
    for v in VENUES:
        c = _wide(f"{v}_1d", insts, "close", root, 86_400_000).reindex(index=days, columns=insts)
        bad = np.log(panel.px / c).abs().median() > MAX_MEDIAN_PREMIUM   # same ticker, different token
        if bad.any():
            log.warning("%s: dropping %s (median price gap > %.0f%%)", v, list(bad[bad].index),
                        MAX_MEDIAN_PREMIUM * 100)
        closes.append(c.loc[:, ~bad.to_numpy()].reindex(columns=insts))
        fr = _wide(f"{v}_funding", insts, "rate", root).reindex(columns=insts)
        fr.loc[:, bad.to_numpy()] = np.nan
        F = _cum_at(fr, days).where(_asof_has(fr, days))
        for k in fund_k:
            fund_k[k].append(F - F.shift(k))
    out = {}
    for k, frames in fund_k.items():
        x = pd.concat(frames).groupby(level=0).mean()
        x = x.reindex(days)
        out[f"xf_fund{k}"] = x
        out[f"xf_fdiff{k}"] = panel.feats[f"fund{k}"] - x
    ref = pd.concat(closes).groupby(level=0).mean().reindex(days)
    out["xf_prem"] = np.log(panel.px / ref)
    out["xf_prem_chg3"] = out["xf_prem"] - out["xf_prem"].shift(3)
    return out


def _asof_has(events: pd.DataFrame, days: pd.Index) -> pd.DataFrame:
    """True from a coin's first funding settlement on the venue (so missing history is NaN, not 0)."""
    first = events.apply(lambda s: s.first_valid_index())
    return pd.DataFrame({c: days >= (first[c] if pd.notna(first[c]) else np.inf) for c in events.columns},
                        index=days)
