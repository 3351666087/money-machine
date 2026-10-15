"""Production model artifacts + the daily live signal.

An artifact = trained model + metadata (features, horizon, training window,
portfolio/cost settings, config hash). Live trading only loads artifacts.

Until the holdout exam has been taken, production models are trained on data
*before* the holdout (2026-04-01) so the exam stays untouched.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from moneymachine.ml import data as D
from moneymachine.ml import universe as U
from moneymachine.ml.experiment import load_panel
from moneymachine.ml.features import Panel, feature_columns, to_long
from moneymachine.ml.models import MODELS, zscore_by_date
from moneymachine.ml.portfolio import CostModel, PortfolioConfig, optimize
from moneymachine.ml.validation import DAY, HOLDOUT_START, EXAM_LOG, config_hash

log = logging.getLogger(__name__)
ART = Path("runs/ml/models")


def train_artifact(name: str, model: str, params: dict, horizon: int, pcfg: PortfolioConfig, costs: CostModel,
                   *, train_until: int | None = None, panel: Panel | None = None,
                   kronos: dict | None = None) -> Path:
    """``kronos``: optional Kronos component, e.g. {"model": "NeoQuasar/Kronos-small",
    "context": 256, "pred_len": 1, "sample_count": 5, "bar": "1D"}. The live signal is then
    z(z(model) + z(kronos)) over the coins Kronos can score — the same combination
    evaluated in research (``lgbm+kronos``)."""
    exam_taken = EXAM_LOG.exists() and EXAM_LOG.read_text().strip()
    limit = HOLDOUT_START if not exam_taken else None
    if train_until is None:
        train_until = limit or int(time.time() * 1000)
    if limit is not None and train_until > limit:
        raise ValueError("holdout exam not taken yet: production model may not train on ≥ 2026-04-01")
    panel = panel or load_panel()
    df = to_long(panel, horizon)
    t = df.index.get_level_values("t")
    df = df[(t < train_until - horizon * DAY)].dropna(subset=["y"])
    cols = feature_columns(df)
    m = MODELS[model](**params).fit(df[cols], df["y"].to_numpy(), df.index.get_level_values("t").to_numpy())
    meta = {
        "name": name, "model": model, "params": params, "horizon": horizon, "features": cols,
        "train_from": int(df.index.get_level_values("t").min()), "train_until": int(train_until),
        "portfolio": asdict(pcfg), "costs": asdict(costs), "kronos": kronos, "created": int(time.time() * 1000),
    }
    meta["hash"] = config_hash({k: v for k, v in meta.items() if k != "created"})
    d = ART / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pkl").write_bytes(pickle.dumps(m))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info("artifact %s trained on %d rows (%s → %s)", name, len(df),
             pd.Timestamp(meta["train_from"], unit="ms").date(), pd.Timestamp(train_until, unit="ms").date())
    return d


def load_artifact(name: str):
    d = ART / name
    meta = json.loads((d / "meta.json").read_text())
    model = pickle.loads((d / "model.pkl").read_bytes())
    return model, meta


def sync_for_live(min_adv: float = 20e6, max_names: int = 50, workers: int = 3) -> list[str]:
    """Incremental update of everything the daily signal needs (≈ 1–3 minutes)."""
    from moneymachine.market.okx_rest import OkxRest

    ins = D.crypto_instruments(OkxRest())
    D.run_parallel(lambda r, i: D.sync_candles(r, i, "1Dutc", "bars_1D"), list(ins["inst"]), workers, "1D")
    P = U.daily_panel(list(ins["inst"]))
    uni = U.pit_universe(P["vol_quote"], ins.set_index("inst")["list_ts"], min_adv_usd=min_adv, max_names=max_names)
    today = uni.iloc[-1]
    members = sorted(today[today].index)
    need = sorted(set(members) | {"BTC-USDT-SWAP"})
    D.run_parallel(lambda r, i: D.sync_candles(r, i, "4H", "bars_4H", start=1590969600000), need, workers, "4H")
    D.run_parallel(lambda r, i: D.sync_funding(r, i), need, workers, "funding")
    D.run_parallel(lambda r, s: D.sync_candles(r, s, "1Dutc", "spot_1D"), [c.replace("-SWAP", "") for c in need],
                   workers, "spot")
    for kind in ("oi", "ls", "taker"):
        D.run_parallel(lambda r, i, k=kind: D.sync_rubik(r, i, k), need, 1, kind)
    return members


def _rows_at(panel: Panel, meta: dict, t: int) -> pd.DataFrame:
    df = to_long(panel, meta["horizon"], min_names=5)
    rows = df.xs(t, level="t", drop_level=False) if t in df.index.get_level_values("t") else None
    if rows is None or rows.empty:
        raise ValueError(f"no feature rows for {pd.Timestamp(t, unit='ms')} — data not up to date?")
    return rows


def model_signal(model, meta: dict, panel: Panel, t: int) -> tuple[pd.Series, dict]:
    """The artifact's signal at t: z(model) or, with a Kronos component,
    z(z(model) + z(kronos)) over the coins Kronos can score (as in research)."""
    rows = _rows_at(panel, meta, t)
    pred = model.predict(rows[meta["features"]])
    insts = list(rows.index.get_level_values("inst"))
    sig = pd.Series(zscore_by_date(pred, np.full(len(pred), t)), index=insts)
    kinfo = {}
    if meta.get("kronos"):
        from moneymachine.ml.kronos_signal import predict_day_isolated as predict_day

        k = meta["kronos"]
        kp = predict_day(insts, t, model=k["model"], context=k["context"], pred_len=k["pred_len"],
                         sample_count=k["sample_count"], bar=k["bar"])
        kp = kp.dropna()
        if len(kp) < 5:
            raise ValueError(f"Kronos scored only {len(kp)} coins — refusing to trade a partial signal")
        kz = pd.Series(zscore_by_date(kp.to_numpy(), np.full(len(kp), t)), index=kp.index)
        both = sig.reindex(kz.index).fillna(0) + kz  # research: restricted to coins Kronos can score
        sig = pd.Series(zscore_by_date(both.to_numpy(), np.full(len(both), t)), index=both.index)
        kinfo = {"kronos_scored": int(len(kp)), "kronos_mean_pred": float(kp.mean())}
    return sig, kinfo


def placebo_signal_at(meta: dict, panel: Panel, t: int, seed: int) -> tuple[pd.Series, dict]:
    """Placebo with the same coin set the model would trade at t (incl. the Kronos
    history requirement), drawn from ``ml.placebo`` — identical to the backtest placebo."""
    from moneymachine.ml.placebo import placebo_signal

    insts = list(_rows_at(panel, meta, t).index.get_level_values("inst"))
    if meta.get("kronos"):
        from moneymachine.ml.kronos_signal import scorable

        k = meta["kronos"]
        insts = scorable(insts, t, context=k["context"], bar=k["bar"])
    if len(insts) < 5:
        raise ValueError(f"placebo: only {len(insts)} coins eligible")
    z = placebo_signal(insts, [t], seed).iloc[0]
    sig = pd.Series(zscore_by_date(z.to_numpy(), np.full(len(z), t)), index=z.index)
    return sig, {"placebo_seed": seed, "placebo_names": len(insts)}


def weights_from_signal(sig: pd.Series, meta: dict, panel: Panel, t: int, current: dict[str, float],
                        nav: float | None = None) -> tuple[dict[str, float], dict]:
    """Cost-aware optimizer with the artifact's portfolio/cost settings."""
    pcfg = PortfolioConfig(**meta["portfolio"])
    if nav:
        pcfg.nav = nav
    costs = CostModel(**meta["costs"])
    universe = list(sig.index)
    held = [s for s, w in current.items() if abs(w) > 0 and s not in sig.index]
    names = universe + held
    k = int(np.where(panel.days == t)[0][0])

    def at(df_):
        return np.array([df_.iloc[k].get(s, np.nan) for s in names], dtype=float)

    beta = at(panel.beta)
    idio = at(panel.feats["idio_vol60"]) / np.sqrt(365)
    vold = at(panel.vol_daily)
    adv = at(panel.adv)
    active = np.array([s in sig.index for s in names]) & np.isfinite(beta) & np.isfinite(vold) & np.isfinite(adv)
    btc_vol = panel.vol_daily["BTC-USDT-SWAP"].iloc[k]
    w0 = np.array([current.get(s, 0.0) for s in names])
    s_vec = np.array([sig.get(s, 0.0) for s in names])
    w = optimize(s_vec, w0, active, np.nan_to_num(beta, nan=1.0),
                 np.nan_to_num(idio, nan=np.nanmedian(idio)) ** 2, float(btc_vol) ** 2,
                 np.nan_to_num(vold, nan=0.05), np.nan_to_num(adv, nan=1e6), pcfg, costs)
    targets = {s: float(x) for s, x in zip(names, w) if abs(x) > 1e-6 or s in current}
    info = {"n_universe": len(universe), "gross": float(np.abs(w).sum()),
            "beta_exp": float(np.nan_to_num(beta, nan=1) @ w), "turnover": float(np.abs(w - w0).sum()),
            "artifact": meta["name"], "artifact_hash": meta["hash"]}
    return targets, info


def target_weights(model, meta: dict, panel: Panel, t: int, current: dict[str, float],
                   nav: float | None = None) -> tuple[dict[str, float], dict]:
    """Today's target weights from the model's signal through the cost-aware optimizer."""
    sig, kinfo = model_signal(model, meta, panel, t)
    targets, info = weights_from_signal(sig, meta, panel, t, current, nav)
    return targets, {**info, **kinfo}


def live_panel(lookback_days: int = 200) -> Panel:
    """Feature panel for today's decision from recent data only (≈10× less memory than
    the research panel). Features at the last day are identical to a full-history build
    (every rolling window is ≤ 90 days); see tests/test_ml.py."""
    ins = D.load_instruments()
    P = U.daily_panel(list(ins["inst"]))
    uni = U.pit_universe(P["vol_quote"], ins.set_index("inst")["list_ts"], min_adv_usd=20e6, max_names=50)
    from moneymachine.ml.features import build_panel

    last = int(uni.index.max())
    start = last - lookback_days * DAY
    uni = uni.loc[uni.index >= start - 40 * DAY]
    return build_panel(uni, start=str(pd.Timestamp(start, unit="ms").date()), load_from=start - 130 * DAY)


# ── v3 exploratory arms (added 2026-10-15; research_v3 trials A1a and B1b) ──
def ema_update(state: dict, sig: pd.Series, t: int, half_life: float, gap_reset: int = 5) -> pd.Series:
    """One day of research_v3.smooth, incrementally. ``state``: inst → [value, last_t] (mutated).
    A coin absent for more than ``gap_reset`` days restarts from today's value; the output
    is re-z-scored over today's coins. Calling it twice for the same t does not update twice."""
    a = 1 - 0.5 ** (1 / half_life)
    out = {}
    for inst, x in sig.items():
        if not np.isfinite(x):
            continue
        prev = state.get(inst)
        if prev is not None and prev[1] == t:
            out[inst] = prev[0]
            continue
        if prev is None or (t - prev[1]) // DAY - 1 > gap_reset:
            v = float(x)
        else:
            v = (1 - a) * prev[0] + a * float(x)
        state[inst] = [v, int(t)]
        out[inst] = v
    s = pd.Series(out, dtype=float)
    return pd.Series(zscore_by_date(s.to_numpy(), np.full(len(s), t)), index=s.index)


def _round_lot(inst, q: float) -> float:
    """Nearest lot (as research's Lots.round); 0 if below the minimum order."""
    step = inst.qty_step
    q = round(q / step) * step if step > 0 else q
    return 0.0 if abs(q) + 1e-12 < inst.min_qty else q


def concentrated_targets(sig: pd.Series, panel: Panel, t: int, *, positions: dict[str, float],
                         marks: dict[str, float], equity: float, instruments: dict, n: int,
                         buffer: int | None = None) -> tuple[dict[str, float], dict]:
    """Live version of portfolio.run_concentrated for one decision (same rules):
    enter from the top / bottom ``n`` at 0.5·equity/n each (rounded to the lot, skipped if it
    rounds to zero), exit when a coin leaves the top / bottom ``buffer`` or the universe, keep
    held quantities unchanged otherwise, and hedge the legs' beta with BTC (rounded to its lot).
    Returns target weights (held coins at exactly their current weight → no trade)."""
    btc = "BTC-USDT-SWAP"
    buffer = buffer or 2 * n
    k = int(np.where(panel.days == t)[0][0])
    beta = {s: float(panel.beta.iloc[k].get(s, np.nan)) for s in set(sig.index) | set(positions)}
    ok = [s for s, v in sig.items() if s != btc and np.isfinite(v) and np.isfinite(beta.get(s, np.nan))
          and marks.get(s, 0) > 0 and s in instruments]
    q = {s: v for s, v in positions.items() if v}
    skipped = 0
    if equity > 0 and len(ok) >= 2 * n:
        order = sorted(ok, key=lambda s: -sig[s])
        rank = {s: i for i, s in enumerate(order)}
        okset = set(ok)
        for s in [s for s in q if s != btc]:
            r = rank.get(s)
            if s not in okset or (q[s] > 0 and r >= buffer) or (q[s] < 0 and len(order) - 1 - r >= buffer):
                q[s] = 0.0
        per = 0.5 * equity / n
        for side, ordered in ((1.0, order), (-1.0, order[::-1])):
            held = sum(1 for s, v in q.items() if s != btc and v * side > 0)
            for s in ordered[:n]:
                if held >= n:
                    break
                if q.get(s, 0.0) != 0:
                    continue
                qi = _round_lot(instruments[s], side * per / marks[s])
                if qi == 0:
                    skipped += 1
                    continue
                q[s] = qi
                held += 1
    expo = sum(v * marks.get(s, 0.0) * (beta.get(s) if np.isfinite(beta.get(s, np.nan)) else 1.0)
               for s, v in q.items() if s != btc)
    q[btc] = _round_lot(instruments[btc], -expo / marks[btc]) if btc in instruments and marks.get(btc) else 0.0
    targets = {}
    for s, v in q.items():
        if s in positions and abs(v - positions[s]) < 1e-12:
            targets[s] = positions[s] * marks[s] / equity          # unchanged: exactly the current weight
        else:
            targets[s] = v * marks.get(s, 0.0) / equity * (1 + 1e-9) if equity > 0 else 0.0
    targets = {s: w for s, w in targets.items() if w or s in positions}
    legs = {s: w for s, w in targets.items() if s != btc}
    info = {"n_long": sum(w > 0 for w in legs.values()), "n_short": sum(w < 0 for w in legs.values()),
            "gross": float(sum(abs(w) for w in targets.values())), "skipped_min_order": skipped,
            "beta_exp": float(sum(w * (beta.get(s) if np.isfinite(beta.get(s, np.nan)) else 1.0)
                                  for s, w in targets.items() if s != btc) + targets.get(btc, 0.0)),
            "n_eligible": len(ok)}
    return targets, info
