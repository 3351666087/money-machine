"""Leakage-safe evaluation: locked holdout + purged walk-forward splits.

The holdout (``HOLDOUT_START`` onwards) is the final exam. Research code reads
data through ``guard_dev`` which refuses anything at or after it; only
``exam_mode`` with a frozen config hash may open it, once.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86_400_000
HOLDOUT_START = 1_775_001_600_000  # 2026-04-01 00:00 UTC
EXAM_LOG = Path("runs/ml/exam_log.jsonl")


class HoldoutViolation(RuntimeError):
    pass


def guard_dev(index: pd.Index | np.ndarray, *, label_horizon_days: int = 0) -> None:
    """Raise if any decision time (plus its label horizon) reaches the holdout."""
    idx = np.asarray(index, dtype="int64")
    if len(idx) and idx.max() + label_horizon_days * DAY >= HOLDOUT_START:
        raise HoldoutViolation("development code tried to read the locked holdout period (≥ 2026-04-01)")


def dev_slice(df: pd.DataFrame, label_horizon_days: int) -> pd.DataFrame:
    """Rows whose labels end strictly before the holdout."""
    return df.loc[df.index < HOLDOUT_START - label_horizon_days * DAY]


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def open_exam(cfg: dict, note: str) -> str:
    """Record a one-shot holdout evaluation. Refuses a second exam."""
    EXAM_LOG.parent.mkdir(parents=True, exist_ok=True)
    if EXAM_LOG.exists() and EXAM_LOG.read_text().strip():
        prev = EXAM_LOG.read_text().strip().splitlines()[-1]
        raise HoldoutViolation(f"the holdout exam was already taken: {prev}")
    h = config_hash(cfg)
    EXAM_LOG.write_text(json.dumps({"config_hash": h, "note": note, "at": pd.Timestamp.now('UTC').isoformat()}) + "\n")
    return h


@dataclass(frozen=True)
class Split:
    fold: int
    train_end: int      # exclusive; train decision times < train_end
    test_start: int
    test_end: int       # exclusive


def walk_forward(dates: np.ndarray, *, first_test: int, test_days: int = 91, min_train_days: int = 365,
                 horizon_days: int = 1, embargo_days: int = 1, train_days: int | None = None) -> list[Split]:
    """Expanding (or rolling, if ``train_days``) walk-forward folds.

    Purge: training decision times whose label window (``horizon_days``) would
    overlap the test period are dropped, plus an ``embargo_days`` gap.
    """
    dates = np.sort(np.asarray(dates, dtype="int64"))
    out = []
    t = first_test
    k = 0
    last = dates.max()
    while t <= last:
        test_end = t + test_days * DAY
        train_end = t - (horizon_days + embargo_days) * DAY
        if (train_end - dates.min()) >= min_train_days * DAY:
            out.append(Split(k, train_end, t, min(test_end, last + DAY)))
            k += 1
        t = test_end
    return out


def train_mask(dates: np.ndarray, sp: Split, train_days: int | None = None) -> np.ndarray:
    m = dates < sp.train_end
    if train_days:
        m &= dates >= sp.train_end - train_days * DAY
    return m


def test_mask(dates: np.ndarray, sp: Split) -> np.ndarray:
    return (dates >= sp.test_start) & (dates < sp.test_end)
