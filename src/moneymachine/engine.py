"""Engine: the part shared by backtest and live paper trading.

matching bar → broker (fills, funding, liquidation) → aggregator → closed
strategy bar → strategies → intents → risk → broker orders.

Running the *same* engine in both modes is what makes "forward result vs
backtest" a meaningful comparison.
"""
from __future__ import annotations

import logging
import traceback
from collections import deque

from moneymachine.config import Config
from moneymachine.market.aggregate import Aggregator
from moneymachine.risk import RiskManager
from moneymachine.sim.broker import EPS, Broker
from moneymachine.sim.funding import FundingBook
from moneymachine.strategies.base import BarWindow, PortfolioContext, Strategy, StrategyContext
from moneymachine.strategies.registry import build_strategy
from moneymachine.timeutil import tf_ms
from moneymachine.types import Bar, Instrument, Intent

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, cfg: Config, instruments: dict[str, Instrument], funding: FundingBook, ledger,
                 *, matching_tf: str, llm=None):
        self.cfg = cfg
        self.ledger = ledger
        self.strat_ms = tf_ms(cfg.strategy_tf)
        self.match_ms = tf_ms(matching_tf)
        self.strategies: list[Strategy] = [build_strategy(sc, cfg.seed) for sc in cfg.enabled_strategies]
        if not self.strategies:
            raise ValueError("no enabled strategies in config")
        self.symbols = sorted(set(cfg.symbols) | {s for st in self.strategies for s in st.symbols})
        missing = [s for s in self.symbols if s not in instruments]
        if missing:
            raise ValueError(f"instrument specs missing for {missing} — run: mm data sync")
        self.broker = Broker(
            maker_fee=cfg.costs.maker_fee, taker_fee=cfg.costs.taker_fee,
            slippage_bps=cfg.costs.slippage_bps, mmr=cfg.account.maintenance_margin_rate,
            instruments=instruments, funding=funding, bar_ms=self.match_ms, ledger=ledger,
        )
        for st in self.strategies:
            self.broker.add_account(st.id, self._initial(st))
        self.risk = RiskManager(cfg.account, self.broker, instruments, self.strat_ms, ledger)
        self.instruments = instruments
        self.windows: dict[str, deque[Bar]] = {s: deque(maxlen=cfg.warmup_bars) for s in self.symbols}
        self.aggs = {s: Aggregator(self.strat_ms, self.match_ms) for s in self.symbols}
        self.services = {"llm": llm} if llm is not None else {}
        self.services["instruments"] = instruments  # lot sizes, for strategies that size their own orders
        self.trading_from = 0
        self.last_equity_ts = 0
        self._closed: dict[int, set[str]] = {}  # decision time -> symbols whose strategy bar closed
        too_long = [st.id for st in self.strategies if st.warmup() > cfg.warmup_bars]
        if too_long:
            raise ValueError(f"[run].warmup_bars={cfg.warmup_bars} is shorter than warmup needed by {too_long}")

    # ── data flow ────────────────────────────────────────────────────────

    def _initial(self, st: Strategy) -> float:
        """Starting capital: the strategy's own ``initial_usdt`` param (e.g. the small-account
        arm), else [account].initial_usdt."""
        return float(st.params.get("initial_usdt", self.cfg.account.initial_usdt))

    def seed_window(self, symbol: str, bars: list[Bar]) -> None:
        self.windows[symbol].extend(bars)

    def process_bar(self, symbol: str, bar: Bar) -> list[Bar]:
        """Feed one closed matching bar; returns strategy bars that just closed."""
        self.broker.on_bar(symbol, bar)
        closed = self.aggs[symbol].push(bar)
        for sb in closed:
            if any(st.portfolio for st in self.strategies):
                self._closed.setdefault(sb.ts + self.strat_ms, set()).add(symbol)
            w = self.windows[symbol]
            if w and sb.ts <= w[-1].ts:
                continue  # already have it (e.g. from warmup)
            w.append(sb)
        return closed

    def jobs_for(self, symbol: str, sbar: Bar, *, catchup: bool = False):
        """(strategy, ctx, window) for every strategy trading ``symbol``."""
        decision_ts = sbar.ts + self.strat_ms
        if decision_ts <= self.trading_from:
            return []
        window = self.windows[symbol]
        jobs = []
        bw = None
        for st in self.strategies:
            if st.portfolio or symbol not in st.symbols or len(window) < st.warmup():
                continue
            if bw is None:
                bw = BarWindow.from_bars(window)
            jobs.append((st, self._ctx(st, symbol, decision_ts, catchup), bw))
        return jobs

    def _ctx(self, st: Strategy, symbol: str, now: int, catchup: bool) -> StrategyContext:
        acct = self.broker.accounts[st.id]
        p = acct.positions.get(symbol)
        open_sym = acct.open_orders(symbol)
        flat = p is None or abs(p.qty) <= EPS
        return StrategyContext(
            strategy_id=st.id, symbol=symbol, now=now, timeframe=self.cfg.strategy_tf, bar_ms=self.strat_ms,
            equity=self.broker.equity(acct),
            position_qty=0.0 if flat else p.qty,
            position_avg_px=0.0 if flat else p.avg_px,
            position_opened_ts=0 if flat else p.opened_ts,
            position_init_stop=None if flat else p.init_stop,
            has_pending_entry=any(o.role == "entry" for o in open_sym),
            instrument=self.instruments[symbol], costs=self.cfg.costs, catchup=catchup,
            services=self.services,
        )

    @staticmethod
    def run_job(st: Strategy, ctx: StrategyContext, bw: BarWindow) -> list[Intent]:
        """Pure decision step (may run in a worker thread for slow strategies)."""
        try:
            return list(st.on_bar(ctx, bw) or [])
        except Exception as e:  # noqa: BLE001 — a broken strategy must not stop the others
            ctx.log("strategy_error", f"{type(e).__name__}: {e}", traceback.format_exc(limit=5))
            return []

    def commit(self, st: Strategy, ctx: StrategyContext, intents: list[Intent], *, now: int, active_from: int) -> None:
        for kind, msg, data in ctx.logs:
            level = "error" if kind == "strategy_error" else "info"
            self.ledger.event(now, level, st.id, kind, msg, data)
        if intents:
            self.risk.apply(st.id, intents, now=now, active_from=active_from)

    # ── portfolio strategies ─────────────────────────────────────────────
    def portfolio_jobs(self, *, now_wall: int | None = None, grace_ms: int = 180_000,
                       catchup: bool = False) -> list[tuple[Strategy, PortfolioContext]]:
        """Jobs for decision times that are complete: every symbol's bar closed, or
        (live) ``grace_ms`` passed since the decision time. ``now_wall=None`` (backtest)
        releases every decision seen so far."""
        jobs = []
        for d in sorted(self._closed):
            done = self._closed[d]
            ready = now_wall is None or done >= set(self.symbols) or now_wall - d >= grace_ms
            if not ready:
                continue
            del self._closed[d]
            if d <= self.trading_from:
                continue
            for st in self.strategies:
                if st.portfolio:
                    jobs.append((st, self._pctx(st, d, catchup)))
        return jobs

    def _pctx(self, st: Strategy, now: int, catchup: bool) -> PortfolioContext:
        acct = self.broker.accounts[st.id]
        return PortfolioContext(
            strategy_id=st.id, now=now, timeframe=self.cfg.strategy_tf, equity=self.broker.equity(acct),
            positions={s: p.qty for s, p in acct.positions.items() if abs(p.qty) > EPS},
            marks=dict(self.broker.marks), catchup=catchup, services=self.services,
        )

    @staticmethod
    def run_portfolio_job(st: Strategy, ctx: PortfolioContext) -> dict[str, float] | None:
        try:
            return st.on_portfolio(ctx)
        except Exception as e:  # noqa: BLE001
            ctx.log("strategy_error", f"{type(e).__name__}: {e}", traceback.format_exc(limit=8))
            return None

    def commit_portfolio(self, st: Strategy, ctx: PortfolioContext, targets: dict[str, float] | None, *,
                         now: int, active_from: int) -> None:
        for kind, msg, data in ctx.logs:
            self.ledger.event(now, "error" if kind == "strategy_error" else "info", st.id, kind, msg, data)
        if targets is not None:
            self.risk.rebalance(st.id, targets, now=now, active_from=active_from)

    def record_equity(self, ts: int) -> None:
        if ts <= self.last_equity_ts:
            return
        self.last_equity_ts = ts
        for acct in self.broker.accounts.values():
            upnl = self.broker.upnl(acct)
            self.ledger.equity(acct.strategy, ts, acct.balance, acct.balance + upnl, upnl,
                               self.broker.gross_exposure(acct))

    # ── persistence ──────────────────────────────────────────────────────
    def to_state(self) -> dict:
        return {
            "broker": self.broker.to_state(),
            "windows": {s: [[b.ts, b.open, b.high, b.low, b.close, b.volume] for b in w] for s, w in self.windows.items()},
            "aggs": {s: a.to_state() for s, a in self.aggs.items()},
            "strategies": {st.id: st.get_state() for st in self.strategies},
            "trading_from": self.trading_from,
            "last_equity_ts": self.last_equity_ts,
        }

    def load_state(self, st: dict) -> None:
        self.broker.load_state(st["broker"])
        for strat in self.strategies:  # accounts for strategies added after the run started
            if strat.id not in self.broker.accounts:
                self.broker.add_account(strat.id, self._initial(strat))
        for s, rows in st["windows"].items():
            if s in self.windows:
                self.windows[s].clear()
                self.windows[s].extend(Bar(*r) for r in rows)
        for s, a in st["aggs"].items():
            if s in self.aggs:
                self.aggs[s].load_state(a)
        for strat in self.strategies:
            strat.set_state(st["strategies"].get(strat.id, {}))
        self.trading_from = st.get("trading_from", 0)
        self.last_equity_ts = st.get("last_equity_ts", 0)
