"""Historical replay through the same Engine used for live paper trading."""
from __future__ import annotations

import logging
import time
import uuid

import numpy as np

from moneymachine.config import Config
from moneymachine.engine import Engine
from moneymachine.ledger import Ledger
from moneymachine.market.store import Store
from moneymachine.sim.funding import FundingBook
from moneymachine.timeutil import fmt_ms, tf_ms, to_ms
from moneymachine.types import Bar

log = logging.getLogger(__name__)


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def run_backtest(cfg: Config, *, start: str | None = None, end: str | None = None,
                 run_id: str | None = None, allow_llm: bool = False, llm_client=None,
                 store: Store | None = None) -> str:
    store = store or Store(cfg.store_dir, cfg.archive_dir)
    instruments = store.load_instruments()
    slow = [s for s in cfg.enabled_strategies if s.kind in ("llm_direct", "llm_veto")]
    if slow and not allow_llm:
        raise ValueError(
            f"LLM strategies {[s.id for s in slow]} are enabled. Historical LLM backtests are unreliable "
            "(the model may have seen this period in training) and cost real money; disable them for "
            "backtests or pass --allow-llm."
        )

    start_s = start or cfg.start
    if not start_s:
        raise ValueError("backtest needs a start date ([run].start or --start)")
    start_ms = to_ms(start_s)
    end_ms = to_ms(end or cfg.end) if (end or cfg.end) else None
    strat_ms = tf_ms(cfg.strategy_tf)
    warm_from = start_ms - (cfg.warmup_bars + 2) * strat_ms

    funding = FundingBook(cfg.costs.default_funding_rate, cfg.costs.default_funding_interval_h, live=False)
    for sym in cfg.symbols:
        funding.add(sym, store.load_funding(sym))

    run_id = run_id or new_run_id("bt")
    ledger = Ledger(cfg.db_path, run_id)
    cfg_text = cfg.source_path.read_text(encoding="utf-8") if cfg.source_path else ""
    ledger.start_run(cfg.run_name, "backtest", cfg_text)
    if llm_client is not None:
        llm_client.ledger = ledger

    engine = Engine(cfg, instruments, funding, ledger, matching_tf=cfg.backtest_matching_tf, llm=llm_client)
    engine.trading_from = start_ms

    data = {s: store.load_bars(s, cfg.backtest_matching_tf, warm_from, end_ms) for s in engine.symbols}
    for s, d in data.items():
        if len(d["time"]) == 0:
            raise ValueError(f"no {cfg.backtest_matching_tf} bars for {s} in range — run: mm data sync")
        cov = funding.coverage(s)
        if cov is None or cov[0] > start_ms:
            ledger.event(start_ms, "warn", None, "funding_assumed",
                         f"{s}: real funding data starts {fmt_ms(cov[0]) if cov else 'never'}; "
                         f"earlier settlements use default rate {cfg.costs.default_funding_rate}")
    all_ts = np.unique(np.concatenate([d["time"] for d in data.values()]))
    idx = {s: 0 for s in data}
    n_bars = 0
    t0 = time.monotonic()
    log.info("backtest %s: %s → %s, %d timestamps, %d strategies", run_id, fmt_ms(start_ms),
             fmt_ms(int(all_ts[-1])), len(all_ts), len(engine.strategies))

    for ts in all_ts.tolist():
        closed: list[tuple[str, Bar]] = []
        for s, d in data.items():
            i = idx[s]
            if i < len(d["time"]) and d["time"][i] == ts:
                b = Bar(ts, float(d["open"][i]), float(d["high"][i]), float(d["low"][i]),
                        float(d["close"][i]), float(d["volume"][i]))
                idx[s] = i + 1
                n_bars += 1
                closed.extend((s, sb) for sb in engine.process_bar(s, b))
        for s, sb in closed:
            decision = sb.ts + strat_ms
            for st, ctx, bw in engine.jobs_for(s, sb):
                intents = engine.run_job(st, ctx, bw)
                engine.commit(st, ctx, intents, now=decision, active_from=decision)
        for st, pctx in engine.portfolio_jobs():
            targets = engine.run_portfolio_job(st, pctx)
            engine.commit_portfolio(st, pctx, targets, now=pctx.now, active_from=pctx.now)
        if closed:
            decision = max(sb.ts for _, sb in closed) + strat_ms
            if decision > start_ms:
                engine.record_equity(decision)

    final_ts = int(all_ts[-1]) + engine.match_ms
    engine.record_equity(final_ts)
    ledger.event(final_ts, "info", None, "backtest_done",
                 f"{n_bars} bars in {time.monotonic() - t0:.1f}s", {"start": start_ms, "end": final_ts})
    ledger.set_status("done", f"{fmt_ms(start_ms)} → {fmt_ms(final_ts)}")
    ledger.close()
    return run_id
