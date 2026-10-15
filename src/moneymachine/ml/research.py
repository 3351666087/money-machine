"""Pre-registered development grid + Gate A report.

The grid is fixed *before* looking at results (it is written down here), so the
trial count used by the Deflated Sharpe Ratio is honest. Every run is recorded.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from moneymachine.ml.experiment import (
    FIRST_TEST, KRONOS_CLEAN_FROM, baseline_signal, dev_long, evaluate, load_panel, oos_predict, run_trial,
)
from moneymachine.ml.models import zscore_by_date
from moneymachine.ml.portfolio import CostModel, PortfolioConfig
from moneymachine.ml.registry import Registry
from moneymachine.ml.stats import deflated_sharpe, pbo_cscv
from moneymachine.ml.validation import HOLDOUT_START

log = logging.getLogger(__name__)
OUT = Path("runs/ml")
PRED_DIR = OUT / "oos_predictions"
FAMILY = "dev-v1"

BASELINES = ["momentum", "reversal", "carry", "lowvol", "classic"]
MODELS = [("lgbm", {}), ("rfridge", {})]
HORIZONS = [1, 3]
KAPPAS = [1.0, 3.0]


def _save_pred(name: str, s: pd.Series) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    s.rename("pred").to_frame().to_parquet(PRED_DIR / f"{name}.parquet")


def load_pred(name: str) -> pd.Series:
    return pd.read_parquet(PRED_DIR / f"{name}.parquet")["pred"]


def run_grid(rebuild_panel: bool = False) -> None:
    reg = Registry()
    panel = load_panel(rebuild=rebuild_panel)
    end = HOLDOUT_START - 3 * 86_400_000
    base_cfg = PortfolioConfig()
    costs = CostModel()
    for h in HORIZONS:
        df = dev_long(panel, h)
        start = FIRST_TEST
        # 1) untrained baselines
        if h == 1:
            for b in BASELINES:
                sig = baseline_signal(df, b)
                sig = sig[sig.index.get_level_values("t") >= start]
                for k in KAPPAS:
                    run_trial(reg, panel, df, family=FAMILY, name=f"base:{b}:k{k:g}", signal=sig,
                              pcfg=replace(base_cfg, kappa=k), costs=costs, start=start, end=end,
                              extra={"signal": b, "horizon": h})
        # 2) models (walk-forward OOS predictions)
        preds = {}
        for model, params in MODELS:
            pred, info = oos_predict(df, model, params, horizon=h)
            preds[model] = pred
            _save_pred(f"{model}_h{h}", pred)
            (OUT / f"folds_{model}_h{h}.json").write_text(json.dumps(info, indent=1))
        t = preds["lgbm"].index.get_level_values("t").to_numpy()
        ens = pd.Series(zscore_by_date(
            (preds["lgbm"] + preds["rfridge"].reindex(preds["lgbm"].index).fillna(0)).to_numpy(), t),
            index=preds["lgbm"].index)
        preds["ens"] = ens
        _save_pred(f"ens_h{h}", ens)
        for name, sig in preds.items():
            for k in KAPPAS:
                run_trial(reg, panel, df, family=FAMILY, name=f"{name}:h{h}:k{k:g}", signal=sig,
                          pcfg=replace(base_cfg, kappa=k, alpha_horizon=h), costs=costs, start=start, end=end,
                          extra={"model": name, "horizon": h})


def gate_a(out_md: Path = OUT / "gate_a.md") -> str:
    reg = Registry()
    trials = reg.trials(FAMILY)
    if trials.empty:
        return "no trials yet"
    trials["is_base"] = trials["name"].str.startswith("base:")
    models = trials[~trials["is_base"]].sort_values("sharpe", ascending=False)
    bases = trials[trials["is_base"]].sort_values("sharpe", ascending=False)
    best = models.iloc[0]
    best_base = bases.iloc[0] if len(bases) else None
    R = reg.returns_matrix(list(trials["hash"]))
    cand = R[best["hash"]]
    dsr = deflated_sharpe(cand.to_numpy(), reg.per_period_sharpes(FAMILY))
    pbo = pbo_cscv(R.to_numpy(), 16)

    # cost stress: re-run the candidate with every cost doubled
    panel = load_panel()
    cfg = json.loads(best["config"])
    h = int(cfg.get("horizon", 1))
    df = dev_long(panel, h)
    model = cfg.get("model")
    sig = load_pred(f"{model}_h{h}") if model else baseline_signal(df, cfg["signal"])
    pc = PortfolioConfig(**cfg["portfolio"])
    stress, _, _ = evaluate(panel, sig, pc, replace(CostModel(**cfg["costs"]), multiplier=2.0),
                            start=int(best["start"]), end=HOLDOUT_START - 3 * 86_400_000)
    maker, _, _ = evaluate(panel, sig, pc, replace(CostModel(**cfg["costs"]), fee=0.0002),
                           start=int(best["start"]), end=HOLDOUT_START - 3 * 86_400_000)

    base_sr = float(best_base["sharpe"]) if best_base is not None else float("nan")
    checks = {
        "净 Sharpe ≥ 1.0": best["sharpe"] >= 1.0,
        "比最好的基线高 ≥ 0.3": best["sharpe"] - base_sr >= 0.3,
        "DSR ≥ 0.95": (dsr["dsr"] or 0) >= 0.95,
        "PBO < 0.3": pbo["pbo"] < 0.3,
        "≥ 70% 季度为正": best["pct_pos_quarters"] >= 0.7,
        "最大回撤 ≤ 20%": best["max_dd"] <= 0.20,
        "成本翻倍后 Sharpe ≥ 0.5": stress["sharpe"] >= 0.5,
        "前两个币贡献 < 50%": (best["top2_coin_share"] or 1) < 0.5,
        "BTC 暴露 |β| 均值 < 0.05": best["beta_exp_abs"] < 0.05,
    }
    show = trials.sort_values("sharpe", ascending=False)[
        ["name", "sharpe", "gross_sharpe", "rank_ic", "rank_ic_t", "ann_return", "ann_vol", "max_dd",
         "ann_cost", "ann_funding", "turnover_day", "pct_pos_quarters"]]
    fmt = show.copy()
    for c in ("ann_return", "ann_vol", "max_dd", "ann_cost", "ann_funding", "pct_pos_quarters"):
        fmt[c] = fmt[c].map(lambda v: f"{v:+.1%}" if c in ("ann_return", "ann_funding") else f"{v:.1%}")
    for c in ("sharpe", "gross_sharpe", "rank_ic_t", "turnover_day"):
        fmt[c] = fmt[c].map(lambda v: f"{v:.2f}")
    fmt["rank_ic"] = fmt["rank_ic"].map(lambda v: f"{v:.4f}")
    lines = [
        "# 门槛 A：开发期滚动样本外结果（扣全部成本）",
        "",
        f"- 试验总数：{len(trials)}（全部登记，Deflated Sharpe 按此校正）",
        f"- 开发期测试区间：{pd.Timestamp(int(best['start']), unit='ms').date()} → "
        f"{pd.Timestamp(int(best['end']), unit='ms').date()}（锁定考试区 2026-04-01 起未读取）",
        f"- 候选（样本外净 Sharpe 最高的模型配置）：`{best['name']}`",
        f"- 最好的基线：`{best_base['name'] if best_base is not None else '-'}`，Sharpe {base_sr:.2f}",
        f"- Deflated Sharpe：{dsr['dsr']:.3f}（{dsr['n_trials']} 次试验里纯运气能达到的年化 Sharpe ≈ {dsr['sr0_annual']:.2f}）",
        f"- PBO：{pbo['pbo']:.2f}（{pbo['n_splits']} 种组合切分）",
        f"- 成本翻倍：Sharpe {stress['sharpe']:.2f}；全挂单（maker 费率）：Sharpe {maker['sharpe']:.2f}",
        "",
        "## 检查项",
        "",
        "\n".join(f"- [{'x' if v else ' '}] {k}" for k, v in checks.items()),
        "",
        f"**结论：{'通过门槛 A' if all(checks.values()) else '未通过门槛 A'}**",
        "",
        "## 全部试验（按样本外净 Sharpe 排序）",
        "",
        fmt.to_markdown(index=False),
        "",
    ]
    text = "\n".join(lines)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return text


KRONOS_FAMILY = "dev-v1-kronos"


def run_kronos(model: str = "NeoQuasar/Kronos-small", sample_count: int = 5) -> str:
    """Kronos on its clean window only, compared head-to-head with the trained models
    on exactly the same days. All runs are recorded (they count as trials)."""
    from moneymachine.ml import kronos_signal

    reg = Registry()
    panel = load_panel()
    start, end = KRONOS_CLEAN_FROM, HOLDOUT_START - 3 * 86_400_000
    days = panel.days[(panel.days >= start) & (panel.days < end)]
    wide = kronos_signal.compute(panel.universe, days, model=model, sample_count=sample_count,
                                 tag=model.split("-")[-1].lower())
    df = dev_long(panel, 1)
    kr = wide.stack().rename("pred")
    kr.index.names = ["t", "inst"]
    kr = kr.reindex(df.index).dropna()
    kr = pd.Series(zscore_by_date(kr.to_numpy(), kr.index.get_level_values("t").to_numpy()), index=kr.index)
    _save_pred("kronos_h1", kr)
    signals = {"kronos": kr}
    for name in ("lgbm_h1", "ens_h1"):
        try:
            p = load_pred(name)
        except FileNotFoundError:
            continue
        p = p[p.index.get_level_values("t") >= start]
        signals[name.replace("_h1", "")] = p
        both = (p.reindex(kr.index).fillna(0) + kr)
        t = both.index.get_level_values("t").to_numpy()
        signals[f"{name.replace('_h1', '')}+kronos"] = pd.Series(zscore_by_date(both.to_numpy(), t), index=both.index)
    lines = ["| 信号 | κ | 净 Sharpe | 毛 Sharpe | RankIC | 年化收益 | 最大回撤 |", "|---|---|---|---|---|---|---|"]
    for name, sig in signals.items():
        for k in KAPPAS:
            _, m, _, _ = run_trial(reg, panel, df, family=KRONOS_FAMILY, name=f"{name}:clean:k{k:g}", signal=sig,
                                   pcfg=replace(PortfolioConfig(), kappa=k), costs=CostModel(), start=start,
                                   end=end, extra={"signal": name, "horizon": 1, "window": "2024-07+"})
            lines.append(f"| {name} | {k:g} | {m['sharpe']:.2f} | {m['gross_sharpe']:.2f} | {m['rank_ic']:.4f} | "
                         f"{m['ann_return']:+.1%} | {m['max_dd']:.1%} |")
    text = "\n".join(["# Kronos 对比（干净窗口 2024-07-01 → 2026-03-28，同一批天数）", ""] + lines) + "\n"
    (OUT / "kronos_compare.md").write_text(text, encoding="utf-8")
    return text


def run_construction_v2() -> None:
    """Iteration 2 (pre-registered after the Gate A diagnosis): same OOS model
    predictions, rank-scaled alpha instead of Grinold. Recorded in the same family,
    so the Deflated Sharpe Ratio counts these trials too."""
    reg = Registry()
    panel = load_panel()
    end = HOLDOUT_START - 3 * 86_400_000
    for h in HORIZONS:
        df = dev_long(panel, h)
        for name in ("lgbm", "rfridge", "ens"):
            sig = load_pred(f"{name}_h{h}")
            for k in KAPPAS:
                run_trial(reg, panel, df, family=FAMILY, name=f"{name}:h{h}:k{k:g}:rank", signal=sig,
                          pcfg=replace(PortfolioConfig(), kappa=k, alpha_horizon=h, alpha_mode="rank"),
                          costs=CostModel(), start=FIRST_TEST, end=end,
                          extra={"model": name, "horizon": h, "alpha_mode": "rank"})


# ── placebo null distribution ─────────────────────────────────────────────
_PW: dict = {}


def _placebo_worker_init() -> None:
    from moneymachine.ml.experiment import load_panel as _lp

    _PW["panel"] = _lp()
    kr = load_pred("kronos_h1")
    _PW["index"] = kr.index
    _PW["coins"] = sorted(set(kr.index.get_level_values("inst")))
    _PW["days"] = sorted(set(kr.index.get_level_values("t")))


def _placebo_one(seed: int) -> dict:
    from moneymachine.ml.placebo import placebo_signal

    z = placebo_signal(_PW["coins"], _PW["days"], seed).stack()
    z.index.names = ["t", "inst"]
    z = z.reindex(_PW["index"]).dropna()  # same (day, coin) set the model traded
    sig = pd.Series(zscore_by_date(z.to_numpy(), z.index.get_level_values("t").to_numpy()), index=z.index)
    m, _, _ = evaluate(_PW["panel"], sig, replace(PortfolioConfig(), kappa=3.0), CostModel(),
                       start=KRONOS_CLEAN_FROM, end=HOLDOUT_START - 3 * 86_400_000)
    return {"seed": seed, **{k: m[k] for k in ("sharpe", "ann_return", "max_dd", "turnover_day", "gross_exp",
                                               "ann_cost", "ann_funding")}}


def placebo_distribution(n_seeds: int = 100, workers: int = 4) -> str:
    """Run the deployed construction with ``n_seeds`` placebo signals on the clean window
    and place the model's result in that distribution (empirical p-value)."""
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers, initializer=_placebo_worker_init) as ex:
        rows = list(ex.map(_placebo_one, range(1, n_seeds + 1)))
    dist = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    dist.to_parquet(OUT / "placebo_distribution.parquet", index=False)
    ref = Registry().trials(KRONOS_FAMILY)
    ref = ref[ref["name"] == "lgbm+kronos:clean:k3"].iloc[0]
    n_ge = int((dist["sharpe"] >= ref["sharpe"]).sum())
    p = (1 + n_ge) / (len(dist) + 1)
    q = dist["sharpe"].quantile([0.05, 0.5, 0.95])
    text = "\n".join([
        "# 随机对照组的分布（干净窗口 2024-07-01 → 2026-03-28，和模型完全相同的构建方法）",
        "",
        f"- 随机信号次数：{len(dist)}（每个种子一条完整的回测）",
        f"- 随机信号的净 Sharpe：中位数 {q[0.5]:.2f}，5%–95% 区间 [{q[0.05]:.2f}, {q[0.95]:.2f}]，"
        f"均值 {dist['sharpe'].mean():.2f}",
        f"- 随机信号的年化收益中位数 {dist['ann_return'].median():+.1%}，最大回撤中位数 {dist['max_dd'].median():.1%}",
        f"- 换手：随机 {dist['turnover_day'].median():.3f}/天 vs 模型 {ref['turnover_day']:.3f}/天；"
        f"平均仓位：随机 {dist['gross_exp'].median():.2f} vs 模型 {ref['gross_exp']:.2f}",
        f"- **模型（LightGBM+Kronos）净 Sharpe {ref['sharpe']:.2f}，比 {len(dist) - n_ge}/{len(dist)} 个随机信号好，"
        f"经验 p 值 = {p:.3f}**",
        "",
        "说明：这个 p 值回答的是\"同样的构建方法下，模型是否比随机信号好\"，没有校正研究阶段的 47 次试验"
        "（那部分由 Deflated Sharpe 处理）。",
        "",
    ])
    (OUT / "placebo_distribution.md").write_text(text, encoding="utf-8")
    return text
