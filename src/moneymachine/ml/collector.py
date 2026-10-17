"""Continuous forward collection (OKX keeps little history for these):

every ``interval_s`` seconds, two bulk requests cover all USDT perpetuals:
  - /market/tickers  → last, best bid/ask price + size (spread & top-of-book depth), 24h volume
  - /public/open-interest → open interest
Written to ``store/ml/collect/{YYYY-MM-DD}/{HHMMSS}.parquet`` (one small file per snapshot, so memory
stays flat; until 2026-10-17 it was one file per day, re-read and rewritten every snapshot, which made
the process grow towards its memory limit by midnight). ``load_day`` reads either layout.
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


def append(df: pd.DataFrame, out: Path = OUT) -> Path:
    t = pd.Timestamp(int(df["ts"].iloc[0]), unit="ms")
    d = out / t.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{t.strftime('%H%M%S')}.parquet"
    df.to_parquet(p, index=False)
    return p


def load_day(day: str, out: Path = OUT) -> pd.DataFrame:
    """All snapshots of one UTC day (YYYY-MM-DD), from the old single file and/or the new folder."""
    parts = []
    if (out / f"{day}.parquet").exists():
        parts.append(pd.read_parquet(out / f"{day}.parquet"))
    parts += [pd.read_parquet(f) for f in sorted((out / day).glob("*.parquet"))] if (out / day).is_dir() else []
    return (pd.concat(parts, ignore_index=True).drop_duplicates(["ts", "inst"]).sort_values(["ts", "inst"])
            if parts else pd.DataFrame())


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
