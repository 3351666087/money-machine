"""Continuous forward collection (OKX keeps little history for these):

every ``interval_s`` seconds, two bulk requests cover all USDT perpetuals:
  - /market/tickers  → last, best bid/ask price + size (spread & top-of-book depth), 24h volume
  - /public/open-interest → open interest
Written to ``store/ml/collect/{YYYY-MM-DD}.parquet`` (one file per UTC day).
Used later to calibrate the cost model and as higher-frequency positioning data.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from moneymachine.market.okx_rest import OkxRest
from moneymachine.timeutil import now_ms

log = logging.getLogger(__name__)
OUT = Path("store/ml/collect")


def snapshot(rest: OkxRest) -> pd.DataFrame:
    ts = now_ms()
    tick = {d["instId"]: d for d in rest.get("/api/v5/market/tickers", {"instType": "SWAP"})
            if d["instId"].endswith("-USDT-SWAP")}
    oi = {d["instId"]: d for d in rest.get("/api/v5/public/open-interest", {"instType": "SWAP"})
          if d["instId"].endswith("-USDT-SWAP")}

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    rows = []
    for inst, d in tick.items():
        o = oi.get(inst, {})
        rows.append({
            "ts": ts, "inst": inst, "last": f(d.get("last")), "bid": f(d.get("bidPx")), "ask": f(d.get("askPx")),
            "bid_sz": f(d.get("bidSz")), "ask_sz": f(d.get("askSz")), "vol_ccy_24h": f(d.get("volCcy24h")),
            "oi": f(o.get("oi")), "oi_ccy": f(o.get("oiCcy")), "oi_usd": f(o.get("oiUsd")),
        })
    return pd.DataFrame(rows)


def append(df: pd.DataFrame) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp(int(df["ts"].iloc[0]), unit="ms").strftime("%Y-%m-%d")
    p = OUT / f"{day}.parquet"
    if p.exists():
        df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
    df.to_parquet(p, index=False)
    return p


def run(interval_s: int = 300, max_iters: int | None = None) -> None:
    rest = OkxRest()
    n = 0
    while max_iters is None or n < max_iters:
        t0 = time.monotonic()
        try:
            df = snapshot(rest)
            p = append(df)
            log.info("collected %d instruments → %s", len(df), p)
        except Exception as e:  # noqa: BLE001
            log.warning("collect failed: %s", e)
        n += 1
        if max_iters is not None and n >= max_iters:
            break
        time.sleep(max(1.0, interval_s - (time.monotonic() - t0)))
