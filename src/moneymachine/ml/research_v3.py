"""刷钱机 v3 trials, pre-registered 2026-10-15 in research/2026-10-15_刷钱机v3_研究计划.md.

Family ``dev-v3``. Every trial runs on the Kronos-clean window 2024-07-01 → 2026-03-28
(the same days as v2's candidate and the placebo distribution) and is recorded, so the
Deflated Sharpe Ratio counts it. The definitions below must match the plan's table;
a trial that cannot run (missing data) is reported as "not run", never replaced.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd

from moneymachine.ml.experiment import KRONOS_CLEAN_FROM, dev_long, evaluate, load_panel, run_trial, to_wide
from moneymachine.ml.models import zscore_by_date
from moneymachine.ml.portfolio import CostModel, Lots, PortfolioConfig, run_concentrated
from moneymachine.ml.registry import Registry
from moneymachine.ml.research import KRONOS_FAMILY, OUT, load_pred
from moneymachine.ml.stats import deflated_sharpe, pbo_cscv
from moneymachine.ml.validation import HOLDOUT_START

log = logging.getLogger(__name__)
FAMILY = "dev-v3"
DAY = 86_400_000
START, END = KRONOS_CLEAN_FROM, HOLDOUT_START - 3 * DAY
V2_CFG = replace(PortfolioConfig(), kappa=3.0)
V2_NAME = "lgbm+kronos:clean:k3"
SMALL_NAVS = (500.0, 1000.0, 2000.0)


def _z(s: pd.Series) -> pd.Series:
    return pd.Series(zscore_by_date(s.to_numpy(), s.index.get_level_values("t").to_numpy()), index=s.index)


def combine(base: pd.Series, *others: pd.Series) -> pd.Series:
    """v2's rule: z(base + Σ others) on base's (day, coin) set; a missing other counts as 0."""
    tot = base.copy()
    for o in others:
        tot = tot + o.reindex(base.index).fillna(0)
    return _z(tot)


def v2_signal() -> pd.Series:
    """The deployed v2 signal on the clean window: z(z(LightGBM h1) + z(Kronos h1))."""
    lg = load_pred("lgbm_h1")
    return combine(load_pred("kronos_h1"), lg[lg.index.get_level_values("t") >= START])


def smooth(sig: pd.Series, half_life: float, gap_reset: int = 5) -> pd.Series:
    """Per-coin EMA of a long-form signal, re-z-scored by day. A coin absent for more than
    ``gap_reset`` days restarts from its current value."""
    wide = to_wide(sig).sort_index()
    a = 1 - 0.5 ** (1 / half_life)
    days = wide.index.to_numpy()
    X = wide.to_numpy()
    out = np.full_like(X, np.nan)
    state = np.full(X.shape[1], np.nan)
    last = np.full(X.shape[1], -1, dtype=np.int64)
    for k, t in enumerate(days):
        x = X[k]
        have = np.isfinite(x)
        fresh = have & (np.isnan(state) | ((t - last) // DAY - 1 > gap_reset))
        cont = have & ~fresh
        state[fresh] = x[fresh]
        state[cont] = (1 - a) * state[cont] + a * x[cont]
        last[have] = t
        out[k, have] = state[have]
    # only coins with a signal that day (pandas ≥ 3 keeps NaN in stack(); _z would turn them into 0s —
    # fixed 2026-10-15 after the live/research parity test caught it; A1a/A1b re-run)
    s = pd.DataFrame(out, index=wide.index, columns=wide.columns).stack().dropna().rename("pred")
    s.index.names = ["t", "inst"]
    return _z(s)


def _concentrated(n: int, nav: float):
    return partial(run_concentrated, n=n, nav=nav, lots=None)


# name → builder(panel) → (signal, PortfolioConfig, runner | None, horizon, extra)
def _a1(h):
    return lambda panel: (smooth(v2_signal(), h), V2_CFG, None, 1, {"smooth_half_life": h})


def _a2(panel):
    return v2_signal(), replace(V2_CFG, funding_in_alpha=True), None, 1, {"funding_in_alpha": True}


def kronos_h3(panel) -> pd.Series:
    """Kronos-small, 3 daily bars ahead (log return to the 3rd predicted close), from the GPU cache."""
    from moneymachine.ml import kronos_signal

    days = panel.days[(panel.days >= START) & (panel.days < END)]
    wide = kronos_signal.compute(panel.universe, days, model="NeoQuasar/Kronos-small", pred_len=3,
                                 sample_count=5, tag="small")
    return _z(_long(wide, dev_long(panel, 3).index).dropna())


def _a3(panel):
    lg = load_pred("lgbm_h3")
    sig = combine(kronos_h3(panel), lg[lg.index.get_level_values("t") >= START])
    return sig, replace(V2_CFG, alpha_horizon=3), None, 3, {"model": "lgbm_h3+kronos_p3"}


def _b1(n):
    return lambda panel: (v2_signal(), V2_CFG, _concentrated(n, 1000.0), 1,
                          {"construction": "concentrated", "n": n, "buffer": 2 * n, "nav": 1000.0})


def _long(wide: pd.DataFrame, index: pd.MultiIndex) -> pd.Series:
    s = wide.stack().rename("pred")
    s.index.names = ["t", "inst"]
    return s.reindex(index)


def lgbm_xex() -> pd.Series:
    """LightGBM h1 retrained walk-forward (same folds as v2's) with the cross-exchange features added."""
    from moneymachine.ml import xex
    from moneymachine.ml.experiment import oos_predict
    from moneymachine.ml.research import PRED_DIR, _save_pred

    if not (PRED_DIR / "lgbm_xex_h1.parquet").exists():
        panel = load_panel()
        panel.feats.update(xex.features(panel))
        pred, info = oos_predict(dev_long(panel, 1), "lgbm", {}, horizon=1)
        _save_pred("lgbm_xex_h1", pred)
        (OUT / "folds_lgbm_xex_h1.json").write_text(pd.DataFrame(info).to_json(orient="records", indent=1))
    return load_pred("lgbm_xex_h1")


def _c1a(panel):
    lg = lgbm_xex()
    return (combine(load_pred("kronos_h1"), lg[lg.index.get_level_values("t") >= START]), V2_CFG, None, 1,
            {"model": "lgbm+xex", "features": "xf_*"})


def _c1b(panel):
    """Pre-set direction: OKX funding above the other venues' → expect lower net return."""
    from moneymachine.ml import xex

    base = v2_signal()
    fd = _long(-xex.features(panel)["xf_fdiff3"], base.index).dropna()
    return combine(base, _z(fd)), V2_CFG, None, 1, {"extra_signal": "-xf_fdiff3", "weight": "equal"}


BROAD = {"min_adv": 2e6, "max_names": 150}


def broad_panel():
    return load_panel(**BROAD)


def lgbm_broad() -> pd.Series:
    """LightGBM h1 retrained walk-forward (v2's folds and features) on the broad universe."""
    from moneymachine.ml.experiment import oos_predict
    from moneymachine.ml.research import PRED_DIR, _save_pred

    if not (PRED_DIR / "lgbm_broad_h1.parquet").exists():
        pred, info = oos_predict(dev_long(broad_panel(), 1), "lgbm", {}, horizon=1)
        _save_pred("lgbm_broad_h1", pred)
        (OUT / "folds_lgbm_broad_h1.json").write_text(pd.DataFrame(info).to_json(orient="records", indent=1))
    return load_pred("lgbm_broad_h1")


def broad_signal(panel) -> pd.Series:
    from moneymachine.ml import kronos_signal

    days = panel.days[(panel.days >= START) & (panel.days < END)]
    wide = kronos_signal.compute(panel.universe, days, model="NeoQuasar/Kronos-small", pred_len=1, sample_count=5,
                                 tag="small_u2m150")
    kr = _z(_long(wide, dev_long(panel, 1).index).dropna())
    lg = lgbm_broad()
    return combine(kr, lg[lg.index.get_level_values("t") >= START])


def _b2a(_):
    panel = broad_panel()
    return broad_signal(panel), V2_CFG, None, 1, {"universe": "adv>=2M,top150", "_panel": panel}


def _b2b(_):
    panel = broad_panel()
    return (broad_signal(panel), V2_CFG, _concentrated(8, 1000.0), 1,
            {"universe": "adv>=2M,top150", "construction": "concentrated", "n": 8, "buffer": 16, "nav": 1000.0,
             "_panel": panel})


def lgbm_events() -> pd.Series:
    """LightGBM h1 retrained walk-forward (v2's folds) with the event features added."""
    from moneymachine.ml import events
    from moneymachine.ml.experiment import oos_predict
    from moneymachine.ml.research import PRED_DIR, _save_pred

    if not (PRED_DIR / "lgbm_events_h1.parquet").exists():
        panel = load_panel()
        panel.feats.update(events.features(panel))
        pred, info = oos_predict(dev_long(panel, 1), "lgbm", {}, horizon=1)
        _save_pred("lgbm_events_h1", pred)
        (OUT / "folds_lgbm_events_h1.json").write_text(pd.DataFrame(info).to_json(orient="records", indent=1))
    return load_pred("lgbm_events_h1")


def _c2(panel):
    lg = lgbm_events()
    return (combine(load_pred("kronos_h1"), lg[lg.index.get_level_values("t") >= START]), V2_CFG, None, 1,
            {"model": "lgbm+events", "features": "ev_*"})


FT_MODEL = "store/ml/kronos/finetuned/Kronos-small-okx1d"


def _d1(panel):
    """Kronos-small fine-tuned on OKX daily bars ≤ 2024-06-30 (scripts/kronos_finetune.py), v2's combination."""
    from moneymachine.ml import kronos_signal

    days = panel.days[(panel.days >= START) & (panel.days < END)]
    wide = kronos_signal.compute(panel.universe, days, model=FT_MODEL, pred_len=1, sample_count=5, tag="smallft")
    kr = _z(_long(wide, dev_long(panel, 1).index).dropna())
    lg = load_pred("lgbm_h1")
    return combine(kr, lg[lg.index.get_level_values("t") >= START]), V2_CFG, None, 1, {"model": "lgbm+kronos_ft"}


TRIALS = {
    "A1a:smooth_hl2": _a1(2),
    "A1b:smooth_hl5": _a1(5),
    "A2:funding_in_alpha": _a2,
    "A3:horizon3": _a3,
    "B1a:concentrated_n5": _b1(5),
    "B1b:concentrated_n8": _b1(8),
    "B2a:broad_universe": _b2a,
    "B2b:broad_concentrated_n8": _b2b,
    "C1a:lgbm_xex": _c1a,
    "C1b:funding_diff": _c1b,
    "C2:events": _c2,
    "D1:kronos_finetuned": _d1,
}


def run(names: list[str] | None = None, rerun: bool = False) -> pd.DataFrame:
    """``rerun``: run recorded trials again (same definition → same config hash → the record is
    replaced, the trial count does not change). Only for implementation fixes."""
    reg = Registry()
    panel = load_panel()
    done = set() if rerun or reg.trials(FAMILY).empty else set(reg.trials(FAMILY)["name"])
    out = []
    for name in names or [k for k in TRIALS if k[:2] in ("A1", "A2", "B1")]:
        if name in done:
            log.info("%s already recorded, skipping", name)
            continue
        sig, pcfg, runner, h, extra = TRIALS[name](panel)
        p = extra.pop("_panel", panel)
        _, m, _, res = run_trial(reg, p, dev_long(p, h), family=FAMILY, name=name, signal=sig, pcfg=pcfg,
                                 costs=CostModel(), start=START, end=END, runner=runner,
                                 extra={"signal": "v2", "horizon": h, "window": "2024-07+", **extra})
        out.append({"name": name, **m})
    return pd.DataFrame(out)


def small_account_sensitivity() -> pd.DataFrame:
    """B1 at 500 / 1000 / 2000 USDT (pre-registered as a sensitivity check, not extra trials)."""
    panel = load_panel()
    sig = v2_signal()
    lots = Lots.load(panel.insts)
    rows = []
    for n in (5, 8):
        for nav in SMALL_NAVS:
            m, d, res = evaluate(panel, sig, V2_CFG, CostModel(), start=START, end=END,
                                 runner=partial(run_concentrated, n=n, nav=nav, lots=lots))
            rows.append({"n": n, "nav": nav, "sharpe": m["sharpe"], "ann_return": m["ann_return"],
                         "max_dd": m["max_dd"], "ann_cost": m["ann_cost"], "ann_funding": m["ann_funding"],
                         "turnover_day": m["turnover_day"], "final_equity": res.config["final_equity"],
                         "skipped_min_order": res.config["skipped_min_order"]})
    return pd.DataFrame(rows)


def report() -> str:
    reg = Registry()
    v3 = reg.trials(FAMILY)
    ref = reg.trials(KRONOS_FAMILY)
    ref = ref[ref["name"] == V2_NAME].iloc[0]
    all_srs = reg.per_period_sharpes(None)
    lines = ["# 刷钱机 v3 试验结果（开发期干净窗口 2024-07-01 → 2026-03-28，扣全部成本）", "",
             f"- 基准 v2（LightGBM+Kronos，κ=3）：净 Sharpe {ref['sharpe']:.2f}，年化 {ref['ann_return']:+.1%}，"
             f"最大回撤 {ref['max_dd']:.1%}，换手 {ref['turnover_day']:.3f}/天，"
             f"年化成本 {ref['ann_cost']:.1%}，年化资金费 {ref['ann_funding']:+.1%}",
             f"- 全部登记试验数：{len(all_srs)}（Deflated Sharpe 按此校正）", ""]
    if v3.empty:
        return "\n".join(lines + ["还没有 v3 试验。"])
    rows = []
    for _, r in v3.sort_values("name").iterrows():
        dsr = deflated_sharpe(reg.returns_matrix([r["hash"]])[r["hash"]].to_numpy(), all_srs)
        rows.append(f"| {r['name']} | {r['sharpe']:.2f} | {r['sharpe'] - ref['sharpe']:+.2f} | {r['gross_sharpe']:.2f} | "
                    f"{r['ann_return']:+.1%} | {r['max_dd']:.1%} | {r['turnover_day']:.3f} | {r['ann_cost']:.1%} | "
                    f"{r['ann_funding']:+.1%} | {r['pct_pos_quarters']:.0%} | {dsr['dsr']:.2f} |")
    lines += ["| 试验 | 净 Sharpe | 比 v2 | 毛 Sharpe | 年化收益 | 最大回撤 | 换手/天 | 年化成本 | 年化资金费 | 盈利季度 | DSR |",
              "|---|---|---|---|---|---|---|---|---|---|---|", *rows, ""]
    hashes = list(v3["hash"]) + [ref["hash"]]
    R = reg.returns_matrix(hashes)
    if R.shape[1] >= 2:
        pbo = pbo_cscv(R.to_numpy(), 16)
        lines.append(f"- PBO（v3 全部试验 + v2，同一窗口）：{pbo['pbo']:.2f}")
    lines.append(f"- DSR 门槛：纯运气在 {len(all_srs)} 次试验里能达到的年化 Sharpe ≈ "
                 f"{deflated_sharpe(np.zeros(10), all_srs)['sr0_annual']:.2f}")
    text = "\n".join(lines) + "\n"
    (OUT / "v3_report.md").write_text(text, encoding="utf-8")
    return text



# ── placebo null for a v3 construction (diagnostic, not a trial) ─────────
def placebo_null(trial: str, n_seeds: int = 100) -> pd.DataFrame:
    """Same construction as ``trial``, signal replaced by the placebo process (same
    (day, coin) set and persistence as v2's signal; smoothing applied too for A1)."""
    from moneymachine.ml.placebo import placebo_signal

    panel = load_panel()
    ref, pcfg, runner, _, extra = TRIALS[trial](panel)   # the trial's own (day, coin) set and panel
    panel = extra.pop("_panel", panel)                   # (fixed 2026-10-15: B2a had used the v2 panel)
    coins = sorted(set(ref.index.get_level_values("inst")))
    days = sorted(set(ref.index.get_level_values("t")))
    if runner is not None:
        runner = partial(runner, lots=Lots.load(panel.insts))
    rows = []
    for seed in range(1, n_seeds + 1):
        z = placebo_signal(coins, days, seed).stack()
        z.index.names = ["t", "inst"]
        sig = _z(z.reindex(ref.index).dropna())
        if "smooth_half_life" in extra:
            sig = smooth(sig, extra["smooth_half_life"])
        m, _, _ = evaluate(panel, sig, pcfg, CostModel(), start=START, end=END, runner=runner)
        rows.append({"seed": seed, **{k: m[k] for k in ("sharpe", "ann_return", "max_dd", "turnover_day",
                                                        "ann_cost", "ann_funding")}})
    return pd.DataFrame(rows)


def gate(name: str) -> str:
    """Gate A for one v3 trial — the same checks as 2.0's gate (research.gate_a). The
    baseline is the classic factor signal under the trial's own construction; DSR counts
    every registered trial; PBO is over all v3 trials + v2 on the same window."""
    from moneymachine.ml.experiment import baseline_signal

    reg = Registry()
    v3 = reg.trials(FAMILY)
    row = v3[v3["name"] == name].iloc[0]
    ref = reg.trials(KRONOS_FAMILY)
    ref = ref[ref["name"] == V2_NAME].iloc[0]
    dsr = deflated_sharpe(reg.returns_matrix([row["hash"]])[row["hash"]].to_numpy(), reg.per_period_sharpes(None))
    pbo = pbo_cscv(reg.returns_matrix(list(v3["hash"]) + [ref["hash"]]).to_numpy(), 16)
    panel = load_panel()
    sig, pcfg, runner, h, extra = TRIALS[name](panel)
    panel = extra.pop("_panel", panel)
    stress, _, _ = evaluate(panel, sig, pcfg, replace(CostModel(), multiplier=2.0), start=START, end=END, runner=runner)
    base = baseline_signal(dev_long(panel, 1), "classic")
    base_m, _, _ = evaluate(panel, base[base.index.get_level_values("t") >= START], pcfg, CostModel(),
                            start=START, end=END, runner=runner)
    checks = {
        "净 Sharpe ≥ 1.0": row["sharpe"] >= 1.0,
        "比经典因子基线（同样构建方法）高 ≥ 0.3": row["sharpe"] - base_m["sharpe"] >= 0.3,
        "DSR ≥ 0.95": (dsr["dsr"] or 0) >= 0.95,
        "PBO < 0.3": pbo["pbo"] < 0.3,
        "≥ 70% 季度为正": row["pct_pos_quarters"] >= 0.7,
        "最大回撤 ≤ 20%": row["max_dd"] <= 0.20,
        "成本翻倍后 Sharpe ≥ 0.5": stress["sharpe"] >= 0.5,
        "前两个币贡献 < 50%": (row["top2_coin_share"] or 1) < 0.5,
        "BTC 暴露 |β| 均值 < 0.05": row["beta_exp_abs"] < 0.05,
    }
    lines = [f"## 门槛 A：{name}", "",
             f"- 净 Sharpe {row['sharpe']:.2f}（v2 {ref['sharpe']:.2f}）；经典因子基线 {base_m['sharpe']:.2f}；"
             f"成本翻倍 {stress['sharpe']:.2f}",
             f"- DSR {dsr['dsr']:.3f}（{dsr['n_trials']} 次试验，纯运气 ≈ {dsr['sr0_annual']:.2f}）；"
             f"PBO {pbo['pbo']:.2f}", "",
             *[f"- [{'x' if v else ' '}] {k}" for k, v in checks.items()], "",
             f"**{'通过' if all(checks.values()) else '未通过'}**", ""]
    return "\n".join(lines)
