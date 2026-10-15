"""Point-in-time tradable universe, decided each day from data known *before* that day."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from moneymachine.ml import data as D

DAY = 86_400_000


def daily_panel(insts: list[str], root: Path = D.ROOT) -> dict[str, pd.DataFrame]:
    """Wide daily frames (index = UTC day start ms, columns = inst)."""
    frames = {k: {} for k in ("open", "high", "low", "close", "vol_quote")}
    for inst in insts:
        p = root / "bars_1D" / f"{inst}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).set_index("time")
        for k in frames:
            frames[k][inst] = df[k]
    return {k: pd.DataFrame(v).sort_index() for k, v in frames.items()}


def pit_universe(vol_quote: pd.DataFrame, list_ts: pd.Series, *, min_adv_usd: float = 20e6,
                 max_names: int = 50, min_age_days: int = 60, lookback: int = 30) -> pd.DataFrame:
    """Boolean frame: True where a coin is tradable on day t.

    Uses the median daily quote volume over the ``lookback`` days *ending the day
    before* t (strictly past data), a minimum listing age, and a top-N cap.
    """
    # Add the day after the last completed bar: that is "today" for live trading.
    nxt = int(vol_quote.index.max()) + DAY
    vol_quote = vol_quote.reindex(vol_quote.index.append(pd.Index([nxt])))
    adv = vol_quote.rolling(lookback, min_periods=int(lookback * 0.8)).median().shift(1)
    idx = vol_quote.index.to_numpy()
    age_ok = pd.DataFrame(
        {c: idx >= list_ts.get(c, np.inf) + min_age_days * DAY for c in vol_quote.columns}, index=vol_quote.index
    )
    # Only bars known at t (the previous day's) — fixed 2026-10-12: this used to also require
    # the bar *starting* at t, which is unknown at t (it only affected listing/delisting days
    # in research, but left the live universe empty).
    has_px = vol_quote.shift(1).notna()
    elig = (adv >= min_adv_usd) & age_ok & has_px
    ranked = adv.where(elig).rank(axis=1, ascending=False, method="first")
    return elig & (ranked <= max_names)


def candidates(universe: pd.DataFrame) -> list[str]:
    """Every coin that is ever in the universe (these need 4H bars, funding, ...)."""
    return sorted(universe.columns[universe.any(axis=0)])
