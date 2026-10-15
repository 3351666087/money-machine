"""Risk layer: turns strategy intents into broker orders, or rejects them.

Every strategy goes through the same rules, so arms are comparable:
fixed-fraction risk sizing, leverage cap, max open positions, one position per
symbol (no pyramiding), daily loss circuit breaker.
"""
from __future__ import annotations

from moneymachine.config import AccountConfig
from moneymachine.sim.broker import EPS, Broker
from moneymachine.types import CancelPending, ClosePosition, Entry, Instrument, Intent


class RiskManager:
    def __init__(self, cfg: AccountConfig, broker: Broker, instruments: dict[str, Instrument],
                 strategy_bar_ms: int, ledger):
        self.cfg = cfg
        self.broker = broker
        self.instruments = instruments
        self.bar_ms = strategy_bar_ms
        self.ledger = ledger

    def apply(self, strategy: str, intents: list[Intent], *, now: int, active_from: int) -> None:
        for it in intents:
            if isinstance(it, ClosePosition):
                self._cancel_entries(strategy, it.symbol, now, "close requested")
                self.broker.close_position(strategy, it.symbol, now, active_from, it.tag)
            elif isinstance(it, CancelPending):
                self._cancel_entries(strategy, it.symbol, now, "cancel requested")
            elif isinstance(it, Entry):
                reason = self._entry(strategy, it, now=now, active_from=active_from)
                if reason:
                    self.ledger.event(now, "info", strategy, "risk_reject", reason,
                                      {"symbol": it.symbol, "side": it.side, "tag": it.tag})

    def _cancel_entries(self, strategy: str, symbol: str, now: int, why: str) -> None:
        acct = self.broker.accounts[strategy]
        for o in acct.open_orders(symbol):
            if o.role == "entry":
                self.broker.cancel(acct, o, now, why)

    def _entry(self, strategy: str, it: Entry, *, now: int, active_from: int) -> str | None:
        b = self.broker
        acct = b.accounts[strategy]
        sym = it.symbol
        inst = self.instruments[sym]
        mark = b.marks.get(sym)
        if mark is None:
            return "no market price yet"
        eq = b.equity(acct)
        if eq <= 0:
            return "account equity <= 0"
        if eq < acct.day_start_equity * (1 - self.cfg.daily_loss_limit):
            return f"daily loss limit hit ({eq:.2f} < {acct.day_start_equity:.2f} -{self.cfg.daily_loss_limit:.0%})"

        open_sym = acct.open_orders(sym)
        if any(o.role == "entry" for o in open_sym):
            return "pending entry already exists"
        closing = any(o.role == "close" for o in open_sym)
        pos = acct.positions.get(sym)
        if pos is not None and abs(pos.qty) > EPS and not closing:
            return "position already open (no pyramiding)"
        others = [s for s, p in acct.positions.items() if abs(p.qty) > EPS and s != sym]
        if len(others) >= self.cfg.max_open_positions:
            return f"max open positions ({self.cfg.max_open_positions})"

        ref = it.price if it.order_type == "limit" and it.price else mark
        long = it.side == "buy"
        if it.stop_loss is not None and (it.stop_loss >= ref if long else it.stop_loss <= ref):
            return f"stop {it.stop_loss} on wrong side of entry {ref}"
        if it.take_profit is not None and (it.take_profit <= ref if long else it.take_profit >= ref):
            return f"take-profit {it.take_profit} on wrong side of entry {ref}"

        if it.sizing == "risk":
            if it.stop_loss is None:
                return "risk sizing requires stop_loss"
            risk_pct = it.risk_pct if it.risk_pct is not None else self.cfg.risk_per_trade
            qty = eq * risk_pct / abs(ref - it.stop_loss)
        else:
            qty = eq * it.notional_frac / ref
        qty *= max(0.0, it.size_mult)

        exposure = sum(abs(p.qty) * b.marks.get(s, p.avg_px) for s, p in acct.positions.items() if s != sym)
        room = eq * self.cfg.max_leverage - exposure
        if room <= 0:
            return "leverage cap reached"
        qty = min(qty, room / ref)
        qty = inst.floor_qty(qty)
        if qty < inst.min_qty - EPS:
            return f"qty below instrument minimum ({qty} < {inst.min_qty})"

        expire_at = None
        if it.order_type == "limit" and it.expire_bars:
            expire_at = now + it.expire_bars * self.bar_ms
        b.submit(
            strategy, symbol=sym, side=it.side, type=it.order_type, qty=qty,
            price=inst.round_px(it.price) if it.price else None, role="entry",
            active_from=active_from, now=now,
            stop_loss=inst.round_px(it.stop_loss) if it.stop_loss is not None else None,
            take_profit=inst.round_px(it.take_profit) if it.take_profit is not None else None,
            expire_at=expire_at, tag=it.tag,
        )
        return None

    # ── portfolio rebalancing ────────────────────────────────────────────
    def rebalance(self, strategy: str, targets: dict[str, float], *, now: int, active_from: int,
                  min_trade_frac: float = 0.002) -> None:
        """Move positions toward target weights with market orders.

        Gross exposure is capped at ``max_leverage``; after the daily loss limit
        is hit, positions may only shrink. Trades smaller than ``min_trade_frac``
        of equity are skipped.
        """
        b = self.broker
        acct = b.accounts[strategy]
        eq = b.equity(acct)
        if eq <= 0:
            return
        halted = eq < acct.day_start_equity * (1 - self.cfg.daily_loss_limit)
        for o in list(acct.open_orders()):
            if o.role == "rebalance":
                b.cancel(acct, o, now, "superseded by new rebalance")
        tg = {s: float(w) for s, w in targets.items() if s in self.instruments and s in b.marks and w == w}
        gross = sum(abs(w) for w in tg.values())
        if gross > self.cfg.max_leverage:
            tg = {s: w * self.cfg.max_leverage / gross for s, w in tg.items()}
        symbols = set(tg) | {s for s, p in acct.positions.items() if abs(p.qty) > EPS}
        skipped = []
        for sym in sorted(symbols):
            mark = b.marks.get(sym)
            if not mark:
                skipped.append(sym)
                continue
            cur = acct.positions[sym].qty if sym in acct.positions else 0.0
            w = tg.get(sym, 0.0)
            if halted and abs(w * eq / mark) > abs(cur):
                w = cur * mark / eq  # no new risk while halted
            dq = w * eq / mark - cur
            if abs(dq) * mark < min_trade_frac * eq:
                continue
            inst = self.instruments[sym]
            qty = inst.floor_qty(abs(dq))
            if qty < inst.min_qty - EPS:
                continue
            reducing = abs(w * eq / mark) < abs(cur) and (w == 0 or (w > 0) == (cur > 0))
            b.submit(strategy, symbol=sym, side="buy" if dq > 0 else "sell", type="market", qty=qty, price=None,
                     role="rebalance", active_from=active_from, now=now, reduce_only=reducing, tag="rebalance")
        if halted:
            self.ledger.event(now, "warn", strategy, "risk_halt", "daily loss limit: rebalance may only reduce")
        if skipped:
            self.ledger.event(now, "warn", strategy, "rebalance_skip", f"no price for {skipped}")
