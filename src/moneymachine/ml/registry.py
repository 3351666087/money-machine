"""Trial registry: every evaluated configuration is recorded, including failures.

The number of trials and the dispersion of their Sharpe ratios feed the
Deflated Sharpe Ratio; their out-of-sample daily returns feed PBO.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from moneymachine.ml.validation import config_hash

DB = Path("runs/ml/trials.sqlite")
RET_DIR = Path("runs/ml/trial_returns")


class Registry:
    def __init__(self, db: Path = DB):
        db.parent.mkdir(parents=True, exist_ok=True)
        RET_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS trials (hash TEXT PRIMARY KEY, created REAL, family TEXT, name TEXT, "
            "config TEXT, metrics TEXT)"
        )

    def record(self, family: str, name: str, cfg: dict, metrics: dict, daily: pd.Series) -> str:
        h = config_hash({"family": family, **cfg})
        self.db.execute(
            "INSERT OR REPLACE INTO trials VALUES (?,?,?,?,?,?)",
            (h, time.time(), family, name, json.dumps(cfg, sort_keys=True, default=str),
             json.dumps(metrics, default=float)),
        )
        self.db.commit()
        daily.rename("ret").to_frame().to_parquet(RET_DIR / f"{h}.parquet")
        return h

    def trials(self, family: str | None = None) -> pd.DataFrame:
        q = "SELECT hash, created, family, name, config, metrics FROM trials"
        df = pd.read_sql(q + (" WHERE family=?" if family else ""), self.db, params=(family,) if family else ())
        if df.empty:
            return df
        m = pd.json_normalize(df["metrics"].map(json.loads))
        return pd.concat([df.drop(columns=["metrics"]), m], axis=1)

    def returns_matrix(self, hashes: list[str]) -> pd.DataFrame:
        cols = {}
        for h in hashes:
            p = RET_DIR / f"{h}.parquet"
            if p.exists():
                cols[h] = pd.read_parquet(p)["ret"]
        return pd.DataFrame(cols).dropna(how="any")

    def per_period_sharpes(self, family: str | None = None) -> np.ndarray:
        t = self.trials(family)
        if t.empty:
            return np.array([])
        return (t["sharpe"].astype(float) / np.sqrt(365)).to_numpy()
