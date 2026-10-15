"""Live paper trading: OKX public WebSocket 1m candles → Engine → virtual wallets.

- No exchange account or API key is needed (public market data only).
- State is snapshotted to the ledger DB every minute; ``mm paper --resume``
  continues the same run after a crash / reboot / laptop sleep, replaying the
  missed minutes from REST (rule strategies decide on them as in a backtest;
  LLM strategies skip stale bars).
- LLM strategies run in worker threads; their orders become active only at the
  wall-clock time the answer arrived, so model latency is paid realistically.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import signal
import time
from pathlib import Path

import websockets

from moneymachine.backtest import new_run_id
from moneymachine.config import Config, load_env_file
from moneymachine.engine import Engine
from moneymachine.ledger import Ledger
from moneymachine.llm.client import LLMClient
from moneymachine.market.okx_rest import OkxRest
from moneymachine.market.store import Store
from moneymachine.sim.funding import FundingBook
from moneymachine.timeutil import fmt_ms, now_ms, tf_ms
from moneymachine.types import Bar

log = logging.getLogger(__name__)

WS_URL = "wss://ws.okx.com:8443/ws/v5/business"
STALE_DECISION_MS = 10 * 60_000   # older decisions are "catch-up"
FAST_DECISION_MS = 15_000         # bar close → decision latency still counted as "on time"
PORTFOLIO_STALE_MS = 60 * 60_000  # a daily portfolio decision older than this is not acted on
SAVE_EVERY_S = 60
HISTORY_THRESHOLD_MS = 20 * 3_600_000


def _ws_channel(tf: str) -> str:
    return "candle" + tf


class LiveRunner:
    def __init__(self, cfg: Config, engine: Engine, rest: OkxRest, ledger: Ledger, funding: FundingBook,
                 *, max_minutes: float | None = None):
        self.cfg = cfg
        self.engine = engine
        self.rest = rest
        self.ledger = ledger
        self.funding = funding
        self.match_tf = cfg.matching_tf
        self.match_ms = engine.match_ms
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stop = asyncio.Event()
        self.max_minutes = max_minutes
        self.inflight: dict[tuple[str, str], asyncio.Task] = {}
        self._last_save = 0.0
        self._last_bar_wall: dict[str, float] = {s: time.monotonic() for s in engine.symbols}
        self.ws_connected = False
        self._has_portfolio = any(st.portfolio for st in engine.strategies)

    def last_ts(self, sym: str) -> int | None:
        return self.engine.broker.last_bar_ts.get(sym)

    # ── bar handling (event-loop thread only) ────────────────────────────
    async def handle_bar(self, sym: str, bar: Bar, *, replay: bool = False) -> None:
        last = self.last_ts(sym)
        if last is not None and bar.ts <= last:
            return
        if last is not None and bar.ts > last + self.match_ms:
            await self.fill_gap(sym, upto=bar.ts - self.match_ms)
            last = self.last_ts(sym)
            if last is not None and bar.ts <= last:
                return
        closed = self.engine.process_bar(sym, bar)
        self._last_bar_wall[sym] = time.monotonic()
        for sb in closed:
            decision = sb.ts + self.engine.strat_ms
            stale = replay or (now_ms() - decision > STALE_DECISION_MS)
            for st, ctx, bw in self.engine.jobs_for(sym, sb, catchup=stale):
                if st.slow and not stale:
                    key = (st.id, sym)
                    if key in self.inflight and not self.inflight[key].done():
                        self.ledger.event(decision, "warn", st.id, "llm_busy",
                                          f"previous {sym} decision still running; bar skipped")
                        continue
                    self.inflight[key] = asyncio.create_task(self._slow_job(st, ctx, bw, decision))
                else:
                    intents = self.engine.run_job(st, ctx, bw)
                    # Rule strategies answer within milliseconds of the bar close, so their
                    # market orders take the next bar's open — exactly as in a backtest.
                    late = now_ms() - decision > FAST_DECISION_MS
                    active = now_ms() if (late and not stale) else decision
                    self.engine.commit(st, ctx, intents, now=decision, active_from=active)
            if decision > self.engine.trading_from:
                self.engine.record_equity(decision)
            self.save_state(force=True)
        if self._has_portfolio:
            clock = (bar.ts + self.match_ms) if replay else now_ms()
            self._dispatch_portfolio(self.engine.portfolio_jobs(now_wall=clock), replay=replay)
        self.save_state()

    def _dispatch_portfolio(self, jobs, *, replay: bool) -> None:
        for st, pctx in jobs:
            stale = replay or now_ms() - pctx.now > PORTFOLIO_STALE_MS
            pctx.catchup = stale
            if st.slow and not stale:
                key = (st.id, "*")
                if key in self.inflight and not self.inflight[key].done():
                    self.ledger.event(pctx.now, "warn", st.id, "busy", "previous portfolio decision still running")
                    continue
                self.inflight[key] = asyncio.create_task(self._slow_portfolio_job(st, pctx))
            else:
                targets = self.engine.run_portfolio_job(st, pctx)
                active = pctx.now if stale else max(pctx.now, now_ms())
                self.engine.commit_portfolio(st, pctx, targets, now=pctx.now, active_from=active)
        if jobs:
            self.save_state(force=True)

    async def _slow_portfolio_job(self, st, pctx) -> None:
        targets = await asyncio.to_thread(self.engine.run_portfolio_job, st, pctx)
        if self.stop.is_set():
            return
        answered = now_ms()
        self.engine.commit_portfolio(st, pctx, targets, now=answered, active_from=answered)
        self.save_state(force=True)

    async def _slow_job(self, st, ctx, bw, decision: int) -> None:
        intents = await asyncio.to_thread(self.engine.run_job, st, ctx, bw)
        if self.stop.is_set():
            return
        answered = now_ms()
        # Orders go live only after the answer arrived → latency is paid in fills.
        self.engine.commit(st, ctx, intents, now=answered, active_from=answered)
        self.save_state(force=True)

    async def fill_gap(self, sym: str, upto: int) -> None:
        last = self.last_ts(sym)
        start = (last + self.match_ms) if last is not None else upto
        if upto < start:
            return
        history = (upto - start) > HISTORY_THRESHOLD_MS
        bars = await asyncio.to_thread(self.rest.fetch_range, sym, self.match_tf, start, upto, history=history)
        if bars:
            log.info("%s: filled %d missing %s bars (%s → %s)", sym, len(bars), self.match_tf,
                     fmt_ms(bars[0].ts), fmt_ms(bars[-1].ts))
        for b in bars:
            await self.handle_bar(sym, b, replay=(now_ms() - b.ts > STALE_DECISION_MS))

    async def catch_up(self) -> None:
        """Replay everything missed since the last processed bar, symbols interleaved by time."""
        upto = now_ms() - now_ms() % self.match_ms - self.match_ms
        per_sym = {}
        for sym in self.engine.symbols:
            last = self.last_ts(sym)
            if last is None or last >= upto:
                continue
            history = (upto - last) > HISTORY_THRESHOLD_MS
            per_sym[sym] = await asyncio.to_thread(
                self.rest.fetch_range, sym, self.match_tf, last + self.match_ms, upto, history=history)
        merged = sorted(((b.ts, s, b) for s, bars in per_sym.items() for b in bars), key=lambda x: (x[0], x[1]))
        if merged:
            log.info("catch-up: replaying %d bars from %s", len(merged), fmt_ms(merged[0][0]))
        for _, s, b in merged:
            await self.handle_bar(s, b, replay=(now_ms() - b.ts > STALE_DECISION_MS))

    def save_state(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._last_save < SAVE_EVERY_S:
            return
        last = max((v for v in self.engine.broker.last_bar_ts.values()), default=0)
        self.ledger.save_state(last, self.engine.to_state())
        self._last_save = time.monotonic()

    # ── tasks ────────────────────────────────────────────────────────────
    async def ws_loop(self) -> None:
        backoff = 1.0
        channel = _ws_channel(self.match_tf)
        pending: dict[str, int] = {}
        while not self.stop.is_set():
            try:
                async with websockets.connect(WS_URL, ping_interval=None, open_timeout=20) as ws:
                    await ws.send(json.dumps({"op": "subscribe", "args": [
                        {"channel": channel, "instId": s} for s in self.engine.symbols]}))
                    self.ws_connected = True
                    backoff = 1.0
                    log.info("websocket connected (%s × %d)", channel, len(self.engine.symbols))
                    for s in self.engine.symbols:  # fill anything missed while disconnected
                        self.queue.put_nowait(("poll", s))
                    while not self.stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=25)
                        except asyncio.TimeoutError:
                            await ws.send("ping")
                            continue
                        if msg == "pong":
                            continue
                        j = json.loads(msg)
                        if "event" in j:
                            if j["event"] == "error":
                                log.error("websocket error: %s", j)
                                self.ledger.event(now_ms(), "error", None, "ws_error", str(j))
                            continue
                        sym = j.get("arg", {}).get("instId")
                        for r in j.get("data", []):
                            b = Bar(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
                            if str(r[8]) == "1":
                                self.queue.put_nowait(("bar", sym, b))
                                pending.pop(sym, None)
                            else:
                                prev = pending.get(sym)
                                if prev is not None and b.ts > prev:
                                    # previous minute closed without a confirm push → fetch it
                                    self.queue.put_nowait(("poll", sym))
                                pending[sym] = b.ts
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.ws_connected = False
                log.warning("websocket dropped (%s); reconnect in %.0fs", e, backoff)
                self.ledger.event(now_ms(), "warn", None, "ws_reconnect", str(e))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def consumer(self) -> None:
        while not self.stop.is_set():
            item = await self.queue.get()
            try:
                if item[0] == "bar":
                    await self.handle_bar(item[1], item[2])
                elif item[0] == "poll":
                    upto = now_ms() - now_ms() % self.match_ms - self.match_ms
                    await self.fill_gap(item[1], upto)
            except Exception as e:  # noqa: BLE001
                log.exception("bar handling failed")
                self.ledger.event(now_ms(), "error", None, "handler_error", f"{type(e).__name__}: {e}")

    async def funding_poll(self) -> None:
        while not self.stop.is_set():
            for sym in self.engine.symbols:
                try:
                    recs = await asyncio.to_thread(self.rest.funding_recent, sym, now_ms() - 3 * 86_400_000)
                    self.funding.add(sym, recs)
                except Exception as e:  # noqa: BLE001
                    log.warning("funding poll %s failed: %s", sym, e)
            await self._sleep(600)

    async def watchdog(self) -> None:
        started = time.monotonic()
        last_print = 0.0
        while not self.stop.is_set():
            await self._sleep(30)
            for sym, t in self._last_bar_wall.items():
                if time.monotonic() - t > 150:
                    self.queue.put_nowait(("poll", sym))
                    self._last_bar_wall[sym] = time.monotonic()
            if self._has_portfolio:  # release decisions whose slowest symbols never closed
                self._dispatch_portfolio(self.engine.portfolio_jobs(now_wall=now_ms()), replay=False)
            if time.monotonic() - last_print > 3600 or last_print == 0:
                last_print = time.monotonic()
                self.print_status()
            if self.max_minutes and time.monotonic() - started > self.max_minutes * 60:
                log.info("max runtime reached, stopping")
                self.stop.set()

    def print_status(self) -> None:
        b = self.engine.broker
        parts = []
        for sid, acct in b.accounts.items():
            npos = sum(1 for p in acct.positions.values() if abs(p.qty) > 1e-12)
            parts.append(f"{sid}={b.equity(acct):,.0f}({npos}pos)")
        shown = [s for s in ("BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP") if s in b.marks]
        marks = " ".join(f"{s.split('-')[0]}={b.marks[s]:g}" for s in shown) + f" (+{len(b.marks) - len(shown)} more)"
        log.info("status | %s | %s", marks, " ".join(parts))

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop.set)
            except NotImplementedError:
                pass
        await self.catch_up()
        self.save_state(force=True)
        tasks = [asyncio.create_task(c) for c in (self.ws_loop(), self.consumer(), self.funding_poll(), self.watchdog())]
        await self.stop.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.save_state(force=True)


def acquire_paper_lock(db_path: Path):
    """One paper process per ledger DB — two writers on one run would corrupt its state."""
    lock_path = Path(str(db_path) + ".paper.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip() or "?"
        f.close()
        raise ValueError(f"another paper process (pid {holder}) is already using {db_path}; stop it first")
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f  # keep the handle open for the life of the process


async def run_paper(cfg: Config, *, resume_run: str | None = None, max_minutes: float | None = None) -> str:
    _lock = acquire_paper_lock(cfg.db_path)  # noqa: F841 — released when the process exits
    load_env_file()
    rest = OkxRest()
    store = Store(cfg.store_dir, cfg.archive_dir)
    all_inst = await asyncio.to_thread(rest.instruments, "SWAP")
    instruments = {s: all_inst[s] for s in all_inst if s in set(cfg.symbols) | {x for st in cfg.enabled_strategies for x in st.symbols}}
    store.save_instruments({**store.load_instruments(), **instruments})

    funding = FundingBook(cfg.costs.default_funding_rate, cfg.costs.default_funding_interval_h, live=True)
    for sym in instruments:
        funding.add(sym, await asyncio.to_thread(rest.funding_recent, sym, now_ms() - 7 * 86_400_000))

    run_id = resume_run or new_run_id("paper")
    ledger = Ledger(cfg.db_path, run_id, autocommit_every=50)
    state = ledger.load_state() if resume_run else None
    if resume_run and state is None:
        raise ValueError(f"no saved state for run {resume_run} in {cfg.db_path}")
    config_text = cfg.source_path.read_text(encoding="utf-8") if cfg.source_path else ""
    if not resume_run:
        ledger.start_run(cfg.run_name, "paper", config_text)
    elif config_text:
        ledger.update_config(config_text)  # strategies added on resume must be known to the report
    ledger.set_status("running")

    needs_llm = any(s.kind in ("llm_direct", "llm_veto") for s in cfg.enabled_strategies)
    llm = LLMClient(cfg.llm, ledger) if needs_llm else None
    if llm is not None:
        llm.preload_usage(ledger.db, run_id, now_ms())
    if llm is not None and not llm.configured():
        log.warning("LLM strategies enabled but relay not configured (.env) — they will not trade")

    engine = Engine(cfg, instruments, funding, ledger, matching_tf=cfg.matching_tf, llm=llm)
    strat_ms, match_ms = engine.strat_ms, engine.match_ms
    if state:
        engine.load_state(state)
        log.info("resumed run %s (last bar %s)", run_id,
                 fmt_ms(max(engine.broker.last_bar_ts.values(), default=0)))
        ledger.event(now_ms(), "info", None, "paper_resume", "resumed")
    else:
        now = now_ms()
        bucket = now - now % strat_ms
        for sym in engine.symbols:
            warm = await asyncio.to_thread(rest.fetch_range, sym, cfg.strategy_tf,
                                           bucket - cfg.warmup_bars * strat_ms, bucket - strat_ms, history=True)
            engine.seed_window(sym, warm)
            log.info("%s: %d warm-up %s bars", sym, len(warm), cfg.strategy_tf)
        engine.trading_from = bucket
        upto = now - now % match_ms - match_ms
        for sym in engine.symbols:
            for b in await asyncio.to_thread(rest.fetch_range, sym, cfg.matching_tf, bucket, upto, history=False):
                engine.process_bar(sym, b)
        ledger.event(now, "info", None, "paper_start",
                     f"first decision at {fmt_ms(bucket + strat_ms)} UTC",
                     {"strategies": [s.id for s in engine.strategies], "symbols": engine.symbols})
        log.info("paper run %s started; first decision at %s UTC", run_id, fmt_ms(bucket + strat_ms))

    runner = LiveRunner(cfg, engine, rest, ledger, funding, max_minutes=max_minutes)
    try:
        await runner.run()
    finally:
        ledger.set_status("stopped", f"stopped at {fmt_ms(now_ms())}")
        ledger.close()
    return run_id
