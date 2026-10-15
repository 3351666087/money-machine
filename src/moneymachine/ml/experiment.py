"""Walk-forward experiments on the development period, recorded in the registry."""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from moneymachine.ml import data as D
from moneymachine.ml import universe as U
from moneymachine.ml.features import Panel, build_panel, feature_columns, to_long
from moneymachine.ml.models import MODELS, zscore_by_date
from moneymachine.ml.portfolio import CostModel, PortfolioConfig, run
from moneymachine.ml.registry import Registry
from moneymachine.ml.stats import summary
from moneymachine.ml.validation import DAY, HOLDOUT_START, dev_slice, guard_dev, test_mask, train_mask, walk_forward

log = logging.getLogger(__name__)
CACHE = Path("store/ml/cache")
FIRST_TEST = 1_656_633_600_000  # 2022-07-01
KRONOS_CLEAN_FROM = 1_719_792_000_000  # 2024-07-01 (Kronos pre-training data ends June 2024)


def load_panel(min_adv: float = 20e6, max_names: int = 50, rebuild: bool = False) -> Panel:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"panel_{int(min_adv / 1e6)}m_{max_names}.pkl"
    if p.exists() and not rebuild:
        return pickle.loads(p.read_bytes())
    ins = D.load_instruments()
    P = U.daily_panel(list(ins["inst"]))
    uni = U.pit_universe(P["vol_quote"], ins.set_index("inst")["list_ts"], min_adv_usd=min_adv, max_names=max_names)
    panel = build_panel(uni)
    p.write_bytes(pickle.dumps(panel))
    return panel


def dev_long(panel: Panel, horizon: int) -> pd.DataFrame:
    """Long table restricted to the development period (labels end before the holdout)."""
    df = to_long(panel, horizon)
    t = df.index.get_level_values("t")
    df = df[t < HOLDOUT_START - horizon * DAY]
    guard_dev(df.index.get_level_values("t").unique(), label_horizon_days=horizon)
    return df.dropna(subset=["y"])


def oos_predict(df: pd.DataFrame, model: str, params: dict, *, horizon: int, first_test: int = FIRST_TEST,
                test_days: int = 91, train_days: int | None = None, features: list[str] | None = None,
                min_train_from: int | None = None) -> tuple[pd.Series, list[dict]]:
    """Out-of-sample predictions from an expanding (or rolling) walk-forward."""
    cols = features or feature_columns(df)
    t = df.index.get_level_values("t").to_numpy()
    splits = walk_forward(np.unique(t), first_test=first_test, test_days=test_days, horizon_days=horizon,
                          embargo_days=1)
    preds = []
    info = []
    for sp in splits:
        trm = train_mask(t, sp, train_days)
        if min_train_from is not None:
            trm &= t >= min_train_from
        tem = test_mask(t, sp)
        if trm.sum() < 2000 or tem.sum() == 0:
            continue
        m = MODELS[model](**params)
        t0 = time.monotonic()
        m.fit(df.loc[trm, cols], df.loc[trm, "y"].to_numpy(), t[trm])
        p = m.predict(df.loc[tem, cols])
        preds.append(pd.Series(zscore_by_date(p, t[tem]), index=df.index[tem]))
        ic = _rank_ic(pd.Series(p, index=df.index[tem]), df.loc[tem, "y_res"])
        info.append({"fold": sp.fold, "test_start": sp.test_start, "n_train": int(trm.sum()),
                     "n_test": int(tem.sum()), "rank_ic": float(ic.mean()), "secs": time.monotonic() - t0,
                     "best_iter": getattr(m, "best_iter", None)})
        log.info("%s fold %d: train %d test %d rankIC %.4f (%.0fs)", model, sp.fold, trm.sum(), tem.sum(),
                 ic.mean(), time.monotonic() - t0)
    return pd.concat(preds), info


def _rank_ic(pred: pd.Series, y: pd.Series) -> pd.Series:
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    return df.groupby(level="t").apply(lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) > 4 else np.nan)


def to_wide(pred: pd.Series) -> pd.DataFrame:
    return pred.unstack("inst")


def baseline_signal(df: pd.DataFrame, name: str) -> pd.Series:
    """Classic, untrained factor signals (same construction pipeline as the models)."""
    t = df.index.get_level_values("t").to_numpy()
    if name == "momentum":
        raw = df["idio_r30"]
    elif name == "reversal":
        raw = -df["r1"]
    elif name == "carry":
        raw = -df["fund7"]
    elif name == "lowvol":
        raw = -df["idio_vol60"]
    elif name == "classic":
        parts = [zscore_by_date(df[c].fillna(0).to_numpy() * s, t)
                 for c, s in (("idio_r30", 1), ("fund7", -1), ("r1", -1))]
        raw = pd.Series(np.mean(parts, axis=0), index=df.index)
    else:
        raise ValueError(name)
    return pd.Series(zscore_by_date(raw.fillna(0).to_numpy(), t), index=df.index)


def evaluate(panel: Panel, signal_long: pd.Series, pcfg: PortfolioConfig, costs: CostModel,
             *, start: int, end: int, runner=None) -> tuple[dict, pd.DataFrame, object]:
    """``runner``: alternative construction with ``run``'s signature (default: the optimizer)."""
    sig = to_wide(signal_long)
    res = (runner or run)(panel, sig, pcfg, costs, start=start, end=end)
    d = res.daily
    s = summary(d["net"].to_numpy())
    q = d["net"].groupby(pd.to_datetime(d.index, unit="ms").to_period("Q")).sum()
    pnl = res.pnl_by_inst[res.pnl_by_inst != 0]
    pos_pnl = pnl[pnl > 0].sort_values(ascending=False)
    monthly = d["net"].groupby(pd.to_datetime(d.index, unit="ms").to_period("M")).sum()
    s.update({
        "gross_sharpe": float(d["gross"].mean() / d["gross"].std() * np.sqrt(365)) if d["gross"].std() > 0 else np.nan,
        "ann_cost": float(d["cost"].mean() * 365),
        "ann_funding": float(d["funding"].mean() * 365),
        "turnover_day": float(d["turnover"].mean()),
        "gross_exp": float(d["gross_exp"].mean()),
        "beta_exp_abs": float(d["beta_exp"].abs().mean()),
        "pct_pos_quarters": float((q > 0).mean()),
        "top2_coin_share": float(pos_pnl.head(2).sum() / pnl.sum()) if pnl.sum() > 0 else np.nan,
        "best_month_share": float(monthly.max() / monthly.sum()) if monthly.sum() > 0 else np.nan,
        "start": int(d.index.min()), "end": int(d.index.max()),
    })
    return s, d, res


def run_trial(reg: Registry, panel: Panel, df: pd.DataFrame, *, family: str, name: str, signal: pd.Series,
              pcfg: PortfolioConfig, costs: CostModel, start: int, end: int, extra: dict | None = None,
              runner=None):
    metrics, daily, res = evaluate(panel, signal, pcfg, costs, start=start, end=end, runner=runner)
    ic = _rank_ic(signal, df["y_res"])
    metrics["rank_ic"] = float(ic.mean())
    metrics["rank_ic_t"] = float(ic.mean() / ic.std() * np.sqrt(ic.count())) if ic.std() > 0 else np.nan
    cfg = {"name": name, "portfolio": asdict(pcfg), "costs": asdict(costs), **(extra or {})}
    h = reg.record(family, name, cfg, metrics, daily["net"])
    log.info("[%s] %s: net Sharpe %.2f | gross %.2f | rankIC %.4f (t=%.1f) | ann ret %.1f%% | maxDD %.1f%% | "
             "turnover %.2f/day", family, name, metrics["sharpe"], metrics["gross_sharpe"], metrics["rank_ic"],
             metrics["rank_ic_t"], metrics["ann_return"] * 100, metrics["max_dd"] * 100, metrics["turnover_day"])
    return h, metrics, daily, res
