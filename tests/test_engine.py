"""End-to-end engine tests on synthetic data: full backtest pipeline, live-style
snapshot/resume equivalence, and the LLM arms with a fake client."""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
import pytest

from conftest import H, SYM
from moneymachine.backtest import run_backtest
from moneymachine.config import Config, StrategyConfig
from moneymachine.engine import Engine
from moneymachine.ledger import Ledger
from moneymachine.llm.client import LLMResult
from moneymachine.market.store import Store
from moneymachine.report import compute, render
from moneymachine.sim.funding import FundingBook
from moneymachine.strategies.base import BarWindow
from moneymachine.types import Bar, ClosePosition, Entry, Instrument

M5 = 5 * 60_000
ETH = "ETH-USDT-SWAP"
T0 = 1_704_067_200_000  # 2024-01-01


def synth(n: int, seed: int, start: float) -> list[Bar]:
    rng = random.Random(seed)
    px, out = start, []
    for i in range(n):
        o = px
        drift = 0.0004 * np.sin(i / 600)  # slow trends so trend rules trade
        c = o * (1 + drift + rng.gauss(0, 0.002))
        h = max(o, c) * (1 + abs(rng.gauss(0, 0.001)))
        l = min(o, c) * (1 - abs(rng.gauss(0, 0.001)))
        out.append(Bar(T0 + i * M5, o, h, l, c, 100.0))
        px = c
    return out


def make_cfg(tmp_path, strategies) -> Config:
    return Config(run_name="t", db_path=tmp_path / "t.sqlite", symbols=[SYM, ETH], strategy_tf="1H",
                  matching_tf="5m", backtest_matching_tf="5m", store_dir=tmp_path / "store", archive_dir=None,
                  warmup_bars=60, strategies=strategies, benchmark="hold")


STRATS = [
    StrategyConfig("hold", "buy_and_hold", [SYM]),
    StrategyConfig("rnd", "random_entry", [SYM, ETH], {"p_enter": 0.2, "max_hold_bars": 6}),
    StrategyConfig("ema", "ema_trend", [SYM, ETH], {"fast": 5, "slow": 20}),
    StrategyConfig("don", "donchian", [SYM, ETH], {"entry_n": 10, "exit_n": 5}),
]
INST = {SYM: Instrument(SYM, 0.01, 0.01, 0.01, 0.1), ETH: Instrument(ETH, 0.1, 0.01, 0.01, 0.01)}


@pytest.fixture
def market(tmp_path):
    store = Store(tmp_path / "store")
    store.save_instruments(INST)
    for sym, seed, p0 in ((SYM, 1, 40_000.0), (ETH, 2, 2_000.0)):
        bars = synth(12 * 24 * 40, seed, p0)
        df = pd.DataFrame([(b.ts, b.open, b.high, b.low, b.close, b.volume) for b in bars],
                          columns=["time", "open", "high", "low", "close", "volume"])
        store.save_df(sym, "5m", df)
    return store


def test_backtest_pipeline_and_report(tmp_path, market):
    cfg = make_cfg(tmp_path, STRATS)
    run_id = run_backtest(cfg, start="2024-01-05", store=market)
    summary, daily, meta = compute(cfg.db_path, run_id)
    assert set(summary["strategy"]) == {"hold", "rnd", "ema", "don"}
    assert (summary[summary.strategy != "hold"]["trades"] > 5).all()
    # equity identity: final equity = start + Σ trade net + open-position effects; check fees non-zero
    assert (summary["fees"] > 0).all()
    text = render(cfg.db_path, run_id, "hold", cfg.gonogo)
    assert "上真钱放行检查" in text


def test_backtest_is_deterministic(tmp_path, market):
    cfg = make_cfg(tmp_path, STRATS)
    a = run_backtest(cfg, start="2024-01-05", store=market)
    b = run_backtest(cfg, start="2024-01-05", store=market)
    sa, _, _ = compute(cfg.db_path, a)
    sb, _, _ = compute(cfg.db_path, b)
    pd.testing.assert_frame_equal(sa, sb)


def _drive(engine: Engine, data: dict[str, list[Bar]], lo: int, hi: int) -> None:
    for i in range(lo, hi):
        closed = []
        for s, bars in data.items():
            closed += [(s, sb) for sb in engine.process_bar(s, bars[i])]
        for s, sb in closed:
            d = sb.ts + engine.strat_ms
            for st, ctx, bw in engine.jobs_for(s, sb):
                engine.commit(st, ctx, engine.run_job(st, ctx, bw), now=d, active_from=d)


def test_snapshot_resume_matches_uninterrupted_run(tmp_path):
    cfg = make_cfg(tmp_path, STRATS)
    data = {SYM: synth(12 * 24 * 20, 1, 40_000.0), ETH: synth(12 * 24 * 20, 2, 2_000.0)}
    n = len(data[SYM])
    fb = FundingBook(0.0001, 8)

    full = Engine(cfg, INST, fb, Ledger(":memory:", "a"), matching_tf="5m")
    _drive(full, data, 0, n)

    part = Engine(cfg, INST, fb, Ledger(":memory:", "b"), matching_tf="5m")
    _drive(part, data, 0, n // 2 + 7)
    blob = json.loads(json.dumps(part.to_state()))  # must survive JSON
    resumed = Engine(cfg, INST, fb, Ledger(":memory:", "c"), matching_tf="5m")
    resumed.load_state(blob)
    _drive(resumed, data, n // 2 + 7, n)

    for sid in full.broker.accounts:
        a, b = full.broker.accounts[sid], resumed.broker.accounts[sid]
        assert a.balance == pytest.approx(b.balance, rel=1e-12), sid
        assert {k: p.qty for k, p in a.positions.items() if p.qty} == \
               {k: p.qty for k, p in b.positions.items() if p.qty}


# ── LLM arms with a fake client ──────────────────────────────────────────
class FakeLLM:
    def __init__(self, reply: dict | None):
        self.reply = reply
        self.calls = 0

    def call_json(self, **kw) -> LLMResult:
        self.calls += 1
        if self.reply is None:
            return LLMResult(ok=False, error="timeout")
        return LLMResult(ok=True, data=dict(self.reply))


def _llm_ctx(engine: Engine, strat_id: str, llm):
    bars = synth(12 * 300, 3, 40_000.0)
    for b in bars:
        engine.process_bar(SYM, b)
    st = next(s for s in engine.strategies if s.id == strat_id)
    ctx = engine._ctx(st, SYM, bars[-1].ts + M5, catchup=False)
    ctx.services = {"llm": llm}
    return st, ctx, BarWindow.from_bars(engine.windows[SYM])


def test_llm_direct_enters_with_atr_stop(tmp_path):
    cfg = make_cfg(tmp_path, [StrategyConfig("s3", "llm_direct", [SYM], {"model": "m", "min_confidence": 60})])
    cfg.warmup_bars = 250
    eng = Engine(cfg, INST, FundingBook(0, 8), Ledger(":memory:", "x"), matching_tf="5m")
    st, ctx, bw = _llm_ctx(eng, "s3", FakeLLM({"action": "enter_short", "stop_atr": 9, "take_profit_atr": 2,
                                               "confidence": 80, "reason": "r"}))
    out = st.on_bar(ctx, bw)
    assert len(out) == 1 and isinstance(out[0], Entry) and out[0].side == "sell"
    assert out[0].stop_loss > bw.close[-1] and out[0].take_profit < bw.close[-1]
    # stop clamped to 5 ATR
    from moneymachine.strategies.indicators import atr
    a = atr(bw.high, bw.low, bw.close, 14)[-1]
    assert out[0].stop_loss == pytest.approx(bw.close[-1] + 5 * a)


@pytest.mark.parametrize("reply,expect", [
    (None, 0),  # failure → nothing
    ({"action": "enter_long", "stop_atr": 2, "take_profit_atr": 0, "confidence": 10, "reason": "r"}, 0),
    ({"action": "hold", "stop_atr": 2, "take_profit_atr": 0, "confidence": 90, "reason": "r"}, 0),
])
def test_llm_direct_fail_safe(tmp_path, reply, expect):
    cfg = make_cfg(tmp_path, [StrategyConfig("s3", "llm_direct", [SYM], {"model": "m"})])
    cfg.warmup_bars = 250
    eng = Engine(cfg, INST, FundingBook(0, 8), Ledger(":memory:", "x"), matching_tf="5m")
    st, ctx, bw = _llm_ctx(eng, "s3", FakeLLM(reply))
    assert len(st.on_bar(ctx, bw)) == expect


def test_llm_veto_only_filters_entries(tmp_path):
    cfg = make_cfg(tmp_path, [StrategyConfig("s2", "llm_veto", [SYM], {"base": "random_entry", "model": "m",
                                                                     "base_params": {"p_enter": 1.0}})])
    cfg.warmup_bars = 250
    eng = Engine(cfg, INST, FundingBook(0, 8), Ledger(":memory:", "x"), matching_tf="5m")
    st, ctx, bw = _llm_ctx(eng, "s2", FakeLLM({"approve": False, "size_multiplier": 1, "reason": "no"}))
    assert st.on_bar(ctx, bw) == []
    ctx.services["llm"] = FakeLLM({"approve": True, "size_multiplier": 0.5, "reason": "ok"})
    out = st.on_bar(ctx, bw)
    assert len(out) == 1 and out[0].size_mult == 0.5
    ctx.services["llm"] = FakeLLM(None)
    assert st.on_bar(ctx, bw) == []
    # exits always pass through without asking the model
    class Closer:
        kind = "x"

        def on_bar(self, ctx, bars):
            return [ClosePosition(SYM)]
    st.base = Closer()
    llm = FakeLLM(None)
    ctx.services["llm"] = llm
    assert isinstance(st.on_bar(ctx, bw)[0], ClosePosition) and llm.calls == 0


class FixedWeights:
    """Portfolio strategy stub: long SYM 50%, short ETH 30%."""
    kind = "fixed"
    portfolio = True
    slow = False
    symbols = [SYM, ETH]

    def __init__(self, sid="pf"):
        self.id = sid
        self.calls = []

    def warmup(self):
        return 1

    def on_portfolio(self, ctx):
        self.calls.append(ctx.now)
        return {SYM: 0.5, ETH: -0.3}

    def get_state(self):
        return {}

    def set_state(self, st):
        pass


def test_portfolio_strategy_rebalances_to_targets(tmp_path):
    cfg = make_cfg(tmp_path, [StrategyConfig("hold", "buy_and_hold", [SYM])])
    eng = Engine(cfg, INST, FundingBook(0, 8), Ledger(":memory:", "pf"), matching_tf="5m")
    st = FixedWeights()
    eng.strategies.append(st)
    eng.broker.add_account(st.id, 10_000.0)
    data = {SYM: synth(12 * 24 * 3, 1, 40_000.0), ETH: synth(12 * 24 * 3, 2, 2_000.0)}
    for i in range(len(data[SYM])):
        for s, bars in data.items():
            eng.process_bar(s, bars[i])
        for pst, pctx in eng.portfolio_jobs():
            eng.commit_portfolio(pst, pctx, pst.on_portfolio(pctx), now=pctx.now, active_from=pctx.now)
    assert len(st.calls) >= 60  # hourly decisions over 3 days
    acct = eng.broker.accounts["pf"]
    eq = eng.broker.equity(acct)
    w = {s: acct.positions[s].qty * eng.broker.marks[s] / eq for s in (SYM, ETH)}
    assert w[SYM] == pytest.approx(0.5, abs=0.03)
    assert w[ETH] == pytest.approx(-0.3, abs=0.03)


def test_rebalance_halted_only_reduces(tmp_path):
    cfg = make_cfg(tmp_path, [StrategyConfig("hold", "buy_and_hold", [SYM])])
    eng = Engine(cfg, INST, FundingBook(0, 8), Ledger(":memory:", "h"), matching_tf="5m")
    b = eng.broker
    b.add_account("pf", 10_000.0)
    b.on_bar(SYM, Bar(T0, 100, 100, 100, 100))
    b.on_bar(ETH, Bar(T0, 10, 10, 10, 10))
    acct = b.accounts["pf"]
    acct.day_start_equity = 20_000.0  # pretend we lost 50% today → halted
    eng.risk.rebalance("pf", {SYM: 0.5}, now=T0 + M5, active_from=T0 + M5)
    assert acct.open_orders() == []  # no new risk while halted


def test_per_strategy_capital_and_held_weights_do_not_trade(tmp_path):
    """initial_usdt param → own starting capital; a target equal to the current weight
    (how the concentrated book keeps a position) sends no order."""
    cfg = make_cfg(tmp_path, [StrategyConfig("hold", "buy_and_hold", [SYM]),
                              StrategyConfig("small", "buy_and_hold", [SYM], params={"initial_usdt": 1000})])
    eng = Engine(cfg, INST, FundingBook(0, 8), Ledger(":memory:", "cap"), matching_tf="5m")
    assert eng.broker.accounts["hold"].balance == cfg.account.initial_usdt
    assert eng.broker.accounts["small"].balance == 1000
    assert eng.services["instruments"] is INST
    b = eng.broker
    b.add_account("pf", 1000.0)
    b.on_bar(SYM, Bar(T0, 100, 100, 100, 100))
    eng.risk.rebalance("pf", {SYM: 0.5}, now=T0 + M5, active_from=T0 + M5)
    b.on_bar(SYM, Bar(T0 + M5, 100, 100, 100, 100))
    acct = b.accounts["pf"]
    qty = acct.positions[SYM].qty
    b.on_bar(SYM, Bar(T0 + 2 * M5, 130, 130, 130, 130))              # price drifts +30%
    w_now = qty * b.marks[SYM] / b.equity(acct)
    eng.risk.rebalance("pf", {SYM: w_now}, now=T0 + 2 * M5, active_from=T0 + 2 * M5)
    assert acct.open_orders() == []
