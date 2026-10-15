"""2.0 data foundation: every crypto USDT perpetual on OKX.

Layout under ``store/ml`` (all timestamps UTC ms):
  instruments.parquet                     current crypto USDT perps + specs
  bars_1D/{inst}.parquet                  UTC-aligned daily bars (all crypto perps)
  bars_4H/{inst}.parquet                  4-hour bars (universe candidates)
  spot_1D/{BASE-USDT}.parquet             UTC daily spot bars (for basis)
  funding/{inst}.parquet                  realized funding settlements
  rubik/{oi,ls,taker}/{inst}.parquet      daily open interest / long-short ratio / taker volume (2024+)

Known limitation: OKX serves no data for delisted instruments (e.g. FTT), so
the history is survivorship-biased; reports state this explicitly.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from moneymachine.market.okx_rest import OkxRest
from moneymachine.timeutil import now_ms, tf_ms

log = logging.getLogger(__name__)

ROOT = Path("store/ml")
START_MS = 1_577_836_800_000  # 2020-01-01
RUBIK_START_MS = 1_703_980_800_000  # 2023-12-31 (OKX keeps no earlier daily OI)
BAR_COLS = ["time", "open", "high", "low", "close", "vol_ccy", "vol_quote"]
_TF_MS = {"1Dutc": 86_400_000, "4H": 4 * 3_600_000, "1H": 3_600_000, "1m": 60_000}


def _path(kind: str, name: str, root: Path = ROOT) -> Path:
    p = root / kind / f"{name}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def crypto_instruments(rest: OkxRest, root: Path = ROOT) -> pd.DataFrame:
    rows = []
    for d in rest.get("/api/v5/public/instruments", {"instType": "SWAP"}):
        if d["instId"].endswith("-USDT-SWAP") and d.get("instCategory") == "1":
            rows.append({
                "inst": d["instId"], "base": d["instId"].split("-")[0], "list_ts": int(d["listTime"]),
                "ct_val": float(d["ctVal"]), "lot_sz": float(d["lotSz"]), "min_sz": float(d["minSz"]),
                "tick_sz": float(d["tickSz"]),
            })
    df = pd.DataFrame(rows).sort_values("list_ts").reset_index(drop=True)
    root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "instruments.parquet", index=False)
    return df


def load_instruments(root: Path = ROOT) -> pd.DataFrame:
    return pd.read_parquet(root / "instruments.parquet")


# ── candles ──────────────────────────────────────────────────────────────
def _fetch_candles(rest: OkxRest, inst: str, bar: str, start: int, end: int) -> pd.DataFrame:
    step = _TF_MS[bar]
    cursor = end + step
    got: dict[int, list] = {}
    while cursor > start:
        rows = rest.get("/api/v5/market/history-candles", {"instId": inst, "bar": bar, "after": cursor, "limit": 100})
        if not rows:
            break
        for r in rows:
            ts = int(r[0])
            if str(r[8]) == "1" and start <= ts <= end:
                got[ts] = [ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6]), float(r[7])]
        oldest = min(int(r[0]) for r in rows)
        if oldest >= cursor:
            break
        cursor = oldest
    return pd.DataFrame([got[k] for k in sorted(got)], columns=BAR_COLS)


def sync_candles(rest: OkxRest, inst: str, bar: str, kind: str, *, start: int = START_MS,
                 root: Path = ROOT) -> int:
    p = _path(kind, inst, root)
    step = _TF_MS[bar]
    end = now_ms() - now_ms() % step - step
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=BAR_COLS)
    frm = int(old["time"].max()) + step if len(old) else start
    if frm > end:
        return 0
    new = _fetch_candles(rest, inst, bar, frm, end)
    if len(new):
        df = pd.concat([old, new], ignore_index=True) if len(old) else new
        df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        df.to_parquet(p, index=False)
    return len(new)


def load_bars(kind: str, inst: str, root: Path = ROOT) -> pd.DataFrame:
    return pd.read_parquet(root / kind / f"{inst}.parquet")


# ── funding ──────────────────────────────────────────────────────────────
def sync_funding(rest: OkxRest, inst: str, *, start: int = START_MS, root: Path = ROOT) -> int:
    p = _path("funding", inst, root)
    have: dict[int, float] = {}
    if p.exists():
        df = pd.read_parquet(p)
        have = dict(zip(df["time"].astype("int64"), df["rate"]))
    recent = rest.funding_recent(inst, now_ms() - 120 * 86_400_000)
    have.update(recent)
    oldest_recent = min((t for t, _ in recent), default=now_ms())
    older = sorted(t for t in have if t < oldest_recent)
    need_from = older[-1] if older else start
    if oldest_recent - need_from > 86_400_000:
        try:
            have.update(rest.funding_archive(inst, need_from, oldest_recent))
        except Exception as e:  # noqa: BLE001
            log.warning("funding archive %s: %s", inst, e)
    if have:
        pd.DataFrame(sorted(have.items()), columns=["time", "rate"]).to_parquet(p, index=False)
    return len(have)


# ── rubik (derivatives positioning) ──────────────────────────────────────
RUBIK = {
    "oi": ("/api/v5/rubik/stat/contracts/open-interest-history", ["time", "oi", "oi_ccy", "oi_usd"]),
    "ls": ("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract", ["time", "ls_ratio"]),
    "taker": ("/api/v5/rubik/stat/taker-volume-contract", ["time", "sell_vol", "buy_vol"]),
}


def sync_rubik(rest: OkxRest, inst: str, kind: str, *, root: Path = ROOT) -> int:
    path, cols = RUBIK[kind]
    p = _path(f"rubik/{kind}", inst, root)
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=cols)
    stop = int(old["time"].max()) if len(old) else RUBIK_START_MS
    got: dict[int, list] = {}
    end = None
    while True:
        params = {"instId": inst, "period": "1D", "limit": 100, "end": end}
        if kind == "taker":
            params["unit"] = 2  # USDT
        rows = rest.get(path, params)
        if not rows:
            break
        for r in rows:
            ts = int(r[0])
            got[ts] = [ts] + [float(x) for x in r[1 : len(cols)]]
        oldest = min(int(r[0]) for r in rows)
        if oldest <= stop or (end is not None and oldest >= end):
            break
        end = oldest
    new = pd.DataFrame([got[k] for k in sorted(got)], columns=cols)
    if len(new):
        df = pd.concat([old, new], ignore_index=True) if len(old) else new
        df.drop_duplicates("time", keep="last").sort_values("time").to_parquet(p, index=False)
    return len(new)


# ── orchestration ────────────────────────────────────────────────────────
def run_parallel(fn, items, workers: int = 3, label: str = "") -> dict:
    """Small thread pool; every worker has its own rate-limited OkxRest."""
    results = {}
    t0 = time.monotonic()
    clients = [OkxRest(min_interval=0.25) for _ in range(workers)]

    def job(i_item):
        i, item = i_item
        try:
            return item, fn(clients[i % workers], item)
        except Exception as e:  # noqa: BLE001
            log.warning("%s %s failed: %s", label, item, e)
            return item, None

    with ThreadPoolExecutor(workers) as ex:
        for n, (item, res) in enumerate(ex.map(job, enumerate(items)), 1):
            results[item] = res
            if n % 20 == 0 or n == len(items):
                log.info("%s: %d/%d done (%.0fs)", label, n, len(items), time.monotonic() - t0)
    return results
