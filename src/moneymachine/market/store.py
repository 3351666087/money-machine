"""Local parquet store under ``store/okx``.

Bars: ``{instId}_{tf}.parquet`` with columns time (UTC ms), open, high, low,
close, volume (contracts). Seeded from the archived ``data/OKX_K线数据`` on
first use (that archive is never modified), then extended from OKX REST.
Funding: ``{instId}_funding.parquet`` with columns time, rate.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from moneymachine.market.okx_rest import OkxRest
from moneymachine.timeutil import fmt_ms, now_ms, tf_ms
from moneymachine.types import Bar, Instrument

log = logging.getLogger(__name__)

ARCHIVE_TF = {"5m": "M5", "15m": "M15", "1H": "H1", "1D": "D1"}
COLS = ["time", "open", "high", "low", "close", "volume"]


def archive_symbol(inst: str) -> str:
    """BTC-USDT-SWAP → BTCUSDT (archive naming)."""
    return inst.replace("-SWAP", "").replace("-", "")


class Store:
    def __init__(self, store_dir: Path, archive_dir: Path | None = None):
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = Path(archive_dir) if archive_dir else None

    # ── paths ────────────────────────────────────────────────────────────
    def bars_path(self, inst: str, tf: str) -> Path:
        return self.dir / f"{inst}_{tf}.parquet"

    def funding_path(self, inst: str) -> Path:
        return self.dir / f"{inst}_funding.parquet"

    @property
    def instruments_path(self) -> Path:
        return self.dir / "instruments.json"

    # ── instruments ──────────────────────────────────────────────────────
    def save_instruments(self, inst: dict[str, Instrument]) -> None:
        payload = {k: {"ct_val": v.ct_val, "lot_sz": v.lot_sz, "min_sz": v.min_sz, "tick_sz": v.tick_sz}
                   for k, v in inst.items()}
        self.instruments_path.write_text(json.dumps(payload, indent=1))

    def load_instruments(self) -> dict[str, Instrument]:
        if not self.instruments_path.exists():
            return {}
        raw = json.loads(self.instruments_path.read_text())
        return {k: Instrument(inst_id=k, **v) for k, v in raw.items()}

    # ── bars ─────────────────────────────────────────────────────────────
    def _seed_from_archive(self, inst: str, tf: str, tick: float | None) -> pd.DataFrame:
        if self.archive_dir is None or tf not in ARCHIVE_TF:
            return pd.DataFrame(columns=COLS)
        p = self.archive_dir / f"{archive_symbol(inst)}_{ARCHIVE_TF[tf]}.parquet"
        if not p.exists():
            return pd.DataFrame(columns=COLS)
        df = pd.read_parquet(p)
        t = df["time"].astype("int64")
        out = pd.DataFrame({"time": np.where(t > 1e11, t, t * 1000).astype("int64")})
        for c in ("open", "high", "low", "close"):
            v = df[c].astype("float64").to_numpy()
            if tick:  # archive is float32 → snap back to the exchange tick grid
                v = np.round(v / tick) * tick
            out[c] = v
        out["volume"] = df["tick_volume"].astype("float64")
        log.info("seeded %s %s from archive (%d bars)", inst, tf, len(out))
        return out

    def load_df(self, inst: str, tf: str, tick: float | None = None) -> pd.DataFrame:
        p = self.bars_path(inst, tf)
        if p.exists():
            return pd.read_parquet(p)
        return self._seed_from_archive(inst, tf, tick)

    def save_df(self, inst: str, tf: str, df: pd.DataFrame) -> None:
        df = df.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
        df[COLS].to_parquet(self.bars_path(inst, tf), index=False)

    def sync_bars(self, rest: OkxRest, inst: str, tf: str, since_ms: int | None = None,
                  tick: float | None = None) -> tuple[int, int]:
        """Extend the stored series up to the latest closed bar. Returns (#new, last_ts)."""
        step = tf_ms(tf)
        df = self.load_df(inst, tf, tick)
        end = now_ms() - now_ms() % step - step  # latest fully closed bar open time
        new: list[Bar] = []
        if len(df):
            first, last = int(df["time"].iloc[0]), int(df["time"].iloc[-1])
            if since_ms is not None and since_ms < first:
                new += rest.fetch_range(inst, tf, since_ms, first - step)
            if last < end:
                new += rest.fetch_range(inst, tf, last + step, end)
        else:
            start = since_ms if since_ms is not None else end - 90 * 86_400_000
            new += rest.fetch_range(inst, tf, start, end)
        if new:
            add = pd.DataFrame([(b.ts, b.open, b.high, b.low, b.close, b.volume) for b in new], columns=COLS)
            df = pd.concat([df, add], ignore_index=True) if len(df) else add
        if len(df):
            self.save_df(inst, tf, df)
        last_ts = int(df["time"].max()) if len(df) else 0
        log.info("%s %s: +%d bars, now up to %s", inst, tf, len(new), fmt_ms(last_ts) if last_ts else "-")
        return len(new), last_ts

    def load_bars(self, inst: str, tf: str, start_ms: int | None = None, end_ms: int | None = None) -> dict[str, np.ndarray]:
        p = self.bars_path(inst, tf)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run: mm data sync --config <cfg>")
        df = pd.read_parquet(p)
        if start_ms is not None:
            df = df[df["time"] >= start_ms]
        if end_ms is not None:
            df = df[df["time"] <= end_ms]
        return {c: df[c].to_numpy() for c in COLS}

    # ── funding ──────────────────────────────────────────────────────────
    def load_funding(self, inst: str) -> list[tuple[int, float]]:
        p = self.funding_path(inst)
        if not p.exists():
            return []
        df = pd.read_parquet(p)
        return list(zip(df["time"].astype("int64").tolist(), df["rate"].astype("float64").tolist()))

    def sync_funding(self, rest: OkxRest, inst: str, since_ms: int) -> int:
        have = dict(self.load_funding(inst))
        recent = rest.funding_recent(inst, since_ms)
        have.update(recent)
        oldest_recent = min((t for t, _ in recent), default=now_ms())
        older = sorted(t for t in have if t < oldest_recent)
        need_from = since_ms
        if older and older[0] <= since_ms + 86_400_000:
            need_from = older[-1]  # already have the old part; only fill up to the REST window
        if oldest_recent - need_from > 86_400_000:
            try:
                have.update(rest.funding_archive(inst, need_from, oldest_recent))
            except Exception as e:  # noqa: BLE001
                log.warning("funding archive download failed for %s: %s", inst, e)
        if have:
            df = pd.DataFrame(sorted(have.items()), columns=["time", "rate"])
            df.to_parquet(self.funding_path(inst), index=False)
        log.info("%s funding: %d settlements stored", inst, len(have))
        return len(have)
