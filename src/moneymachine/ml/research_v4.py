"""刷钱机 v4 trials, pre-registered 2026-10-15 in research/2026-10-15_刷钱机v4_研究计划.md.

E (execution): the v2 signal under two constructions (v2 optimizer, B1b concentrated book)
executed by the 1-minute replay in ``ml.execution`` — E0 (taker, calibration baseline, not a
trial) vs post-only policies M30 / M240 / MO (6 trials). Every trial is re-run with the stricter
fill rule (2 ticks through) as the pre-registered robustness check (not extra trials).
F (funding carry): ``ml.carry`` (4 trials). Family ``dev-v4`` in runs/ml/trials.sqlite.
"""
from __future__ import annotations

import logging
from functools import partial

import pandas as pd

from moneymachine.ml import execution as X
from moneymachine.ml.experiment import dev_long, evaluate, load_panel, run_trial
from moneymachine.ml.portfolio import CostModel, Lots, run, run_concentrated
from moneymachine.ml.registry import Registry
from moneymachine.ml.research import OUT
from moneymachine.ml.research_v3 import END, START, V2_CFG, v2_signal

log = logging.getLogger(__name__)
FAMILY = "dev-v4"
E_TRIALS = {"E1a": ("v2", "M30"), "E1b": ("v2", "M240"), "E1c": ("v2", "MO"),
            "E1d": ("b1b", "M30"), "E1e": ("b1b", "M240"), "E1f": ("b1b", "MO")}


class _Env:
    def __init__(self):
        self.panel = load_panel()
        self.sig = v2_signal()
        self.df = dev_long(self.panel, 1)
        self.minutes = X.load_minutes(self.panel.insts)
        spec = pd.read_parquet("store/ml/instruments.parquet").set_index("inst")
        self.ticks = spec["tick_sz"].astype(float).to_dict()
        self.lots = Lots.load(self.panel.insts)

    def runner(self, construction: str, policy: str, through: int = 1, real_spread: bool = False):
        ex = X.MinuteExecutor(policy, self.panel.px, self.minutes, self.ticks, CostModel(), through=through,
                              real_spread=X.measured_half_spreads() if real_spread else None)
        if construction == "v2":
            return partial(run, executor=ex), ex
        return partial(run_concentrated, n=8, buffer=16, nav=1000.0, lots=self.lots, executor=ex), ex

    def evaluate(self, construction: str, policy: str, through: int = 1, real_spread: bool = False):
        r, ex = self.runner(construction, policy, through, real_spread)
        m, d, _ = evaluate(self.panel, self.sig, V2_CFG, CostModel(), start=START, end=END, runner=r)
        return m, ex.summary()


def run_e(names: list[str] | None = None) -> pd.DataFrame:
    reg = Registry()
    env = _Env()
    base = {c: env.evaluate(c, "taker") for c in ("v2", "b1b")}
    real_base = {c: env.evaluate(c, "taker", real_spread=True)[0] for c in ("v2", "b1b")}
    for c, (m, s) in base.items():
        log.info("E0 %s (taker baseline): net Sharpe %.2f | ann cost %.2f%% | no-data orders %d/%d",
                 c, m["sharpe"], m["ann_cost"] * 100, s["no_data"], s["orders"])
    done = set() if reg.trials(FAMILY).empty else set(reg.trials(FAMILY)["name"])
    rows = []
    for name in names or list(E_TRIALS):
        construction, policy = E_TRIALS[name]
        full = f"{name}:{construction}_{policy}"
        r, ex = env.runner(construction, policy)
        if full in done:
            log.info("%s already recorded, skipping", full)
            continue
        _, m, _, _ = run_trial(reg, env.panel, env.df, family=FAMILY, name=full, signal=env.sig, pcfg=V2_CFG,
                               costs=CostModel(), start=START, end=END, runner=r,
                               extra={"construction": construction, "execution": policy, "through_ticks": 1,
                                      "window": "2024-07+"})
        strict, sstat = env.evaluate(construction, policy, through=2)
        # not pre-registered, reported as a caveat: both taker and maker at measured spreads
        real, _ = env.evaluate(construction, policy, real_spread=True)
        real0 = real_base[construction]
        b = base[construction][0]
        rows.append({"trial": name, "construction": construction, "policy": policy,
                     "sharpe": m["sharpe"], "sharpe_E0": b["sharpe"], "ann_cost": m["ann_cost"], "ann_cost_E0": b["ann_cost"],
                     "cost_cut": 1 - m["ann_cost"] / b["ann_cost"] if b["ann_cost"] > 0 else float("nan"),
                     "turnover": m["turnover_day"], "max_dd": m["max_dd"], **{f"x_{k}": v for k, v in ex.summary().items()},
                     "strict_sharpe": strict["sharpe"], "strict_ann_cost": strict["ann_cost"],
                     "strict_cost_cut": 1 - strict["ann_cost"] / b["ann_cost"] if b["ann_cost"] > 0 else float("nan"),
                     "strict_fill_rate": sstat["maker_fill_rate"],
                     "realspread_sharpe": real["sharpe"], "realspread_sharpe_E0": real0["sharpe"],
                     "realspread_cost_cut": 1 - real["ann_cost"] / real0["ann_cost"] if real0["ann_cost"] > 0 else float("nan")})
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(OUT / "v4_execution.csv", index=False)
    return out
