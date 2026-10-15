"""Kronos (financial K-line foundation model, MIT) as a cross-sectional signal.

For each decision time t and each coin in the universe, Kronos reads the last
``context`` 4H bars (closed by t) and generates the next ``pred_len`` bars;
the signal is the predicted log return of the close over that horizon,
averaged over ``sample_count`` samples.

Contamination: Kronos was pre-trained on data up to June 2024, so its outputs
are only a fair out-of-sample test from 2024-07-01 on. ``compute`` refuses
earlier dates unless explicitly allowed.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from moneymachine.ml import data as D
from moneymachine.ml.validation import guard_dev

log = logging.getLogger(__name__)
KRONOS_SRC = Path("vendor/github/Kronos")
CLEAN_FROM = 1_719_792_000_000  # 2024-07-01
H4 = 4 * 3_600_000
OUT = Path("store/ml/kronos")


def load_predictor(model: str = "NeoQuasar/Kronos-small", tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base",
                   device: str | None = None, max_context: int = 512):
    if str(KRONOS_SRC) not in sys.path:
        sys.path.insert(0, str(KRONOS_SRC))
    from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

    tok = KronosTokenizer.from_pretrained(tokenizer)
    mdl = Kronos.from_pretrained(model)
    tok.eval()
    mdl.eval()
    return KronosPredictor(mdl, tok, device=device, max_context=max_context)


BAR_MS = {"4H": H4, "1D": 86_400_000}


def _bars(insts: list[str], root: Path, bar: str) -> dict[str, pd.DataFrame]:
    out = {}
    kind = "bars_4H" if bar == "4H" else "bars_1D"
    for inst in insts:
        p = root / kind / f"{inst}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["end"] = df["time"] + BAR_MS[bar]   # bar is known once it has closed
            out[inst] = df.set_index("end")
    return out


def _day_inputs(bars: dict[str, pd.DataFrame], members: list[str], t: int, context: int, pred_len: int, step: int):
    """Kronos inputs for decision time t: the last ``context`` bars that closed by t."""
    dfs, xs, ys, last_close, names = [], [], [], [], []
    for c in members:
        b = bars.get(c)
        if b is None:
            continue
        hist = b.loc[:t].tail(context)
        if len(hist) < context or hist.index[-1] != t:
            continue
        x = pd.DataFrame({"open": hist["open"].to_numpy(), "high": hist["high"].to_numpy(),
                          "low": hist["low"].to_numpy(), "close": hist["close"].to_numpy(),
                          "volume": hist["vol_ccy"].to_numpy(), "amount": hist["vol_quote"].to_numpy()})
        if x.isna().any().any():
            continue
        dfs.append(x)
        xs.append(pd.Series(pd.to_datetime(hist["time"].to_numpy(), unit="ms")))
        ys.append(pd.Series(pd.to_datetime(t + np.arange(pred_len) * step, unit="ms")))
        last_close.append(float(hist["close"].iloc[-1]))
        names.append(c)
    return dfs, xs, ys, last_close, names


def _predict(pred, inputs, pred_len: int, sample_count: int, chunk: int | None = None) -> list[tuple[str, float]]:
    """``chunk``: series per forward pass (attention memory grows with batch × samples ×
    context²; 8 keeps the live child process well under 1 GB). None = all at once (research)."""
    dfs, xs, ys, last_close, names = inputs
    step = chunk or len(dfs)
    res = []
    for i in range(0, len(dfs), step):
        out = pred.predict_batch(dfs[i:i + step], xs[i:i + step], ys[i:i + step], pred_len=pred_len, T=1.0,
                                 top_p=0.9, sample_count=sample_count, verbose=False)
        res += [(c, float(np.log(max(o["close"].iloc[-1], 1e-12) / lc)))
                for c, lc, o in zip(names[i:i + step], last_close[i:i + step], out)]
    return res


_LIVE_PREDICTORS: dict[str, object] = {}


def scorable(insts: list[str], t: int, *, context: int = 256, bar: str = "1D", root: Path = D.ROOT) -> list[str]:
    """Coins Kronos can score at t (enough clean history) — without loading the model."""
    bars = _bars(insts, root, bar)
    return _day_inputs(bars, insts, int(t), context, 1, BAR_MS[bar])[4]


def predict_day(insts: list[str], t: int, *, model: str = "NeoQuasar/Kronos-small", context: int = 256,
                pred_len: int = 1, sample_count: int = 5, bar: str = "1D", root: Path = D.ROOT) -> pd.Series:
    """Forward (live) Kronos prediction for one decision time. Not for research on
    past dates — use ``compute`` there, which enforces the contamination / holdout rules."""
    bars = _bars(insts, root, bar)
    inputs = _day_inputs(bars, insts, int(t), context, pred_len, BAR_MS[bar])
    if not inputs[0]:
        return pd.Series(dtype=float)
    if model not in _LIVE_PREDICTORS:
        _LIVE_PREDICTORS[model] = load_predictor(model)
    return pd.Series(dict(_predict(_LIVE_PREDICTORS[model], inputs, pred_len, sample_count, chunk=8)))


def compute(universe: pd.DataFrame, days: pd.Index, *, model: str = "NeoQuasar/Kronos-small", context: int = 256,
            pred_len: int = 1, sample_count: int = 5, bar: str = "1D", allow_contaminated: bool = False,
            exam: bool = False, root: Path = D.ROOT, tag: str = "small") -> pd.DataFrame:
    days = pd.Index(sorted(days))
    if not allow_contaminated and days.min() < CLEAN_FROM:
        raise ValueError("Kronos outputs before 2024-07-01 overlap its pre-training data")
    if not exam:
        guard_dev(days, label_horizon_days=1)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"signal_{tag}_{bar}_c{context}_p{pred_len}_s{sample_count}.parquet"
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["t", "inst", "pred"])
    have = set(zip(done["t"].astype("int64"), done["inst"]))
    insts = [c for c in universe.columns if universe[c].any()]
    bars = _bars(insts, root, bar)
    step = BAR_MS[bar]
    pred = None  # loaded lazily: a fully cached run needs no model (and no network)
    rows = []
    t0 = time.monotonic()
    for k, t in enumerate(days):
        members = [c for c in insts if universe.at[t, c] and c in bars and (int(t), c) not in have] \
            if t in universe.index else []
        inputs = _day_inputs(bars, members, int(t), context, pred_len, step)
        if inputs[0]:
            if pred is None:
                pred = load_predictor(model)
            rows += [(int(t), c, v) for c, v in _predict(pred, inputs, pred_len, sample_count)]
        if rows and (k % 20 == 0 or k == len(days) - 1):
            done = pd.concat([done, pd.DataFrame(rows, columns=["t", "inst", "pred"])], ignore_index=True)
            done.to_parquet(cache, index=False)
            rows = []
            el = time.monotonic() - t0
            log.info("kronos: %d/%d days (%.0fs, ~%.0f min left)", k + 1, len(days), el,
                     el / (k + 1) * (len(days) - k - 1) / 60)
    if rows:
        done = pd.concat([done, pd.DataFrame(rows, columns=["t", "inst", "pred"])], ignore_index=True)
        done.to_parquet(cache, index=False)
    return done.drop_duplicates(["t", "inst"]).pivot(index="t", columns="inst", values="pred")


def predict_day_isolated(insts: list[str], t: int, timeout_s: int = 900, **kw) -> pd.Series:
    """``predict_day`` in a fresh Python process.

    LightGBM and PyTorch each ship their own OpenMP runtime; once LightGBM has run in a
    process, loading Kronos can deadlock (reproduced 2026-10-12 on macOS). A child
    process keeps the two apart on every OS.
    """
    import json
    import os
    import subprocess

    payload = json.dumps({"insts": insts, "t": int(t), "kw": kw})
    r = subprocess.run([sys.executable, "-m", "moneymachine.ml.kronos_signal"], input=payload,
                       capture_output=True, text=True, timeout=timeout_s, env=os.environ.copy())
    if r.returncode != 0:
        raise RuntimeError(f"Kronos subprocess failed: {r.stderr[-2000:]}")
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return pd.Series(out, dtype=float)


if __name__ == "__main__":  # child process entry for predict_day_isolated
    import json

    req = json.loads(sys.stdin.read())
    s = predict_day(req["insts"], req["t"], **req["kw"])
    print(json.dumps({k: float(v) for k, v in s.items()}))
