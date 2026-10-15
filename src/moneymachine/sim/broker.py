"""Paper broker: virtual USDT wallets + bar-driven order matching.

One ``Account`` per strategy (independent wallets, same starting capital), all
sharing the same market data. Orders are matched against *matching bars*
(1m live / 5m backtest) with deliberately pessimistic rules:

- market orders fill at the **open of the first bar that starts after the
  decision**, plus slippage, taker fee;
- limit orders need a strict **trade-through** (buy: low < price), fill at the
  limit price, maker fee; a limit that is marketable on submission is treated
  as a market order (taker);
- stop orders trigger on touch and fill at the stop price, or at the bar open
  if the bar gapped through it, plus slippage, taker fee;
- within one bar, stop-loss is assumed to happen **before** take-profit; on the
  bar where an entry fills, its stop-loss is live but its take-profit is not;
- funding is charged on positions held at each settlement time;
- liquidation is checked against the worst price inside each bar.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from moneymachine.sim.funding import FundingBook
from moneymachine.timeutil import day_start
from moneymachine.types import Bar, Instrument, Liquidity, OrdStatus, OrdType, Side, opposite, side_sign

EPS = 1e-12
FUNDING_FALLBACK_MS = 2 * 3_600_000  # live: use last known rate if realized one is late


@dataclass
class Order:
    id: int
    strategy: str
    symbol: str
    side: Side
    type: OrdType
    qty: float
    price: float | None
    role: str  # entry | sl | tp | close
    reduce_only: bool
    active_from: int
    created_ts: int
    expire_at: int | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    tag: str = ""
    status: OrdStatus = "new"
    parent_id: int | None = None


@dataclass
class Position:
    qty: float = 0.0
    side: str = ""
    avg_px: float = 0.0
    opened_ts: int = 0
    entry_tag: str = ""
    init_stop: float | None = None
    max_abs_qty: float = 0.0
    entry_notional: float = 0.0
    entry_qty: float = 0.0
    exit_notional: float = 0.0
    exit_qty: float = 0.0
    realized: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    sl_order: int | None = None
    tp_order: int | None = None


@dataclass
class Account:
    strategy: str
    balance: float
    positions: dict[str, Position] = field(default_factory=dict)
    orders: dict[int, Order] = field(default_factory=dict)
    day_ts: int = 0
    day_start_equity: float = 0.0
    fees_total: float = 0.0
    funding_total: float = 0.0
    liquidations: int = 0

    def pos(self, symbol: str) -> Position:
        p = self.positions.get(symbol)
        if p is None:
            p = self.positions[symbol] = Position()
        return p

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return [o for o in self.orders.values() if o.status == "new" and (symbol is None or o.symbol == symbol)]


class _NullLedger:
    def __getattr__(self, name):
        return lambda *a, **k: None


class Broker:
    def __init__(
        self,
        *,
        maker_fee: float,
        taker_fee: float,
        slippage_bps: float,
        mmr: float,
        instruments: dict[str, Instrument],
        funding: FundingBook,
        bar_ms: int,
        ledger=None,
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slip = slippage_bps / 10_000
        self.mmr = mmr
        self.instruments = instruments
        self.funding = funding
        self.bar_ms = bar_ms
        self.ledger = ledger or _NullLedger()
        self.accounts: dict[str, Account] = {}
        self.marks: dict[str, float] = {}
        self.last_bar_ts: dict[str, int] = {}
        self._funding_cursor: dict[str, int] = {}
        self._pending_funding: list[dict[str, Any]] = []
        self._next_id = 1

    # ── accounts ─────────────────────────────────────────────────────────
    def add_account(self, strategy: str, initial: float) -> Account:
        acct = Account(strategy=strategy, balance=initial, day_start_equity=initial)
        self.accounts[strategy] = acct
        return acct

    def upnl(self, acct: Account, prices: dict[str, float] | None = None) -> float:
        px = prices or self.marks
        total = 0.0
        for sym, p in acct.positions.items():
            if abs(p.qty) > EPS:
                m = px.get(sym, self.marks.get(sym, p.avg_px))
                total += p.qty * (m - p.avg_px)
        return total

    def equity(self, acct: Account) -> float:
        return acct.balance + self.upnl(acct)

    def gross_exposure(self, acct: Account) -> float:
        return sum(abs(p.qty) * self.marks.get(s, p.avg_px) for s, p in acct.positions.items())

    # ── order entry ──────────────────────────────────────────────────────
    def submit(self, strategy: str, *, symbol: str, side: Side, type: OrdType, qty: float,
               price: float | None, role: str, active_from: int, now: int,
               reduce_only: bool = False, stop_loss: float | None = None,
               take_profit: float | None = None, expire_at: int | None = None,
               tag: str = "", parent_id: int | None = None) -> Order:
        acct = self.accounts[strategy]
        if type == "limit" and price is not None and symbol in self.marks:
            mark = self.marks[symbol]
            if (side == "buy" and price >= mark) or (side == "sell" and price <= mark):
                # Marketable limit → crosses the book immediately → taker.
                type = "market"
                tag = (tag + " [marketable-limit]").strip()
        o = Order(
            id=self._next_id, strategy=strategy, symbol=symbol, side=side, type=type, qty=qty,
            price=price, role=role, reduce_only=reduce_only, active_from=active_from,
            created_ts=now, expire_at=expire_at, stop_loss=stop_loss, take_profit=take_profit,
            tag=tag, parent_id=parent_id,
        )
        self._next_id += 1
        acct.orders[o.id] = o
        self.ledger.order(o, now)
        return o

    def cancel(self, acct: Account, o: Order, ts: int, reason: str, status: OrdStatus = "canceled") -> None:
        if o.status != "new":
            return
        o.status = status
        acct.orders.pop(o.id, None)
        self.ledger.order(o, ts, reason)
        p = acct.positions.get(o.symbol)
        if p is not None:
            if p.sl_order == o.id:
                p.sl_order = None
            if p.tp_order == o.id:
                p.tp_order = None

    def close_position(self, strategy: str, symbol: str, now: int, active_from: int, tag: str = "close") -> Order | None:
        acct = self.accounts[strategy]
        p = acct.positions.get(symbol)
        if p is None or abs(p.qty) <= EPS:
            return None
        for o in acct.open_orders(symbol):
            if o.role == "close":
                return o  # already closing
        return self.submit(
            strategy, symbol=symbol, side="sell" if p.qty > 0 else "buy", type="market",
            qty=abs(p.qty), price=None, role="close", active_from=active_from, now=now,
            reduce_only=True, tag=tag,
        )

    # ── market data ──────────────────────────────────────────────────────
    def on_bar(self, symbol: str, bar: Bar) -> None:
        """Process one *closed* matching bar for ``symbol``."""
        prev = self.last_bar_ts.get(symbol)
        if prev is not None and bar.ts <= prev:
            return  # duplicate / out of order
        self._settle_funding(symbol, bar)
        for acct in self.accounts.values():
            if day_start(bar.ts) != acct.day_ts:
                acct.day_ts = day_start(bar.ts)
                acct.day_start_equity = self.equity(acct)
            self._match(acct, symbol, bar)
        self.marks[symbol] = bar.close
        self.last_bar_ts[symbol] = bar.ts
        for acct in self.accounts.values():
            self._check_liquidation(acct, symbol, bar)

    # ── funding ──────────────────────────────────────────────────────────
    def _settle_funding(self, symbol: str, bar: Bar) -> None:
        cursor = self._funding_cursor.get(symbol)
        if cursor is None:
            cursor = bar.ts - 1  # nothing held before the first bar
        for ft in self.funding.settlements(symbol, cursor, bar.ts):
            snaps = {}
            for acct in self.accounts.values():
                p = acct.positions.get(symbol)
                if p is not None and abs(p.qty) > EPS:
                    snaps[acct.strategy] = p.qty
            if snaps:
                self._pending_funding.append(
                    {"symbol": symbol, "ts": ft, "mark": bar.open, "qty": snaps}
                )
        self._funding_cursor[symbol] = bar.ts
        if self._pending_funding:
            self._resolve_funding(bar.ts)

    def _resolve_funding(self, now: int) -> None:
        still = []
        for pf in self._pending_funding:
            sym, ft = pf["symbol"], pf["ts"]
            rate = self.funding.rate(sym, ft)
            assumed = self.funding.is_assumed(sym, ft)
            if rate is None:
                if now - ft < FUNDING_FALLBACK_MS:
                    still.append(pf)
                    continue
                rate, assumed = self.funding.last_rate(sym), True
            for strategy, qty in pf["qty"].items():
                acct = self.accounts.get(strategy)
                if acct is None:
                    continue
                amount = -qty * pf["mark"] * rate  # longs pay when rate > 0
                acct.balance += amount
                acct.funding_total += amount
                p = acct.positions.get(sym)
                if p is not None and abs(p.qty) > EPS:
                    p.funding += amount
                self.ledger.funding(strategy, sym, ft, rate, assumed, qty, pf["mark"], amount)
        self._pending_funding = still

    # ── matching ─────────────────────────────────────────────────────────
    def _match(self, acct: Account, symbol: str, bar: Bar) -> None:
        orders = [o for o in acct.orders.values() if o.symbol == symbol and o.status == "new"]
        if not orders:
            return
        for o in orders:
            if o.expire_at is not None and bar.ts >= o.expire_at and o.role == "entry":
                self.cancel(acct, o, bar.ts, "expired", status="expired")
        live = [o for o in orders if o.status == "new" and o.active_from <= bar.ts]
        if not live:
            return

        # 1) market orders at the open
        for o in sorted((o for o in live if o.type == "market"), key=lambda x: x.id):
            if o.status != "new":
                continue
            px = bar.open * (1 + self.slip) if o.side == "buy" else bar.open * (1 - self.slip)
            self._fill(acct, o, px, "taker", bar)

        # 2) entry limits
        for o in sorted((o for o in live if o.type == "limit" and o.role == "entry"), key=lambda x: x.id):
            if o.status == "new" and self._limit_hit(o, bar):
                self._fill(acct, o, o.price, "maker", bar)

        # 3) stops (incl. stop-losses created by fills in this bar)
        for o in sorted(acct.open_orders(symbol), key=lambda x: x.id):
            if o.type != "stop" or o.active_from > bar.ts:
                continue
            trig = (o.side == "sell" and bar.low <= o.price) or (o.side == "buy" and bar.high >= o.price)
            if trig:
                if o.side == "sell":
                    base = min(o.price, bar.open)
                    px = base * (1 - self.slip)
                else:
                    base = max(o.price, bar.open)
                    px = base * (1 + self.slip)
                self._fill(acct, o, px, "taker", bar)

        # 4) take-profit / other reduce-only limits
        for o in sorted(acct.open_orders(symbol), key=lambda x: x.id):
            if o.type == "limit" and o.role != "entry" and o.active_from <= bar.ts and self._limit_hit(o, bar):
                self._fill(acct, o, o.price, "maker", bar)

    @staticmethod
    def _limit_hit(o: Order, bar: Bar) -> bool:
        return (o.side == "buy" and bar.low < o.price) or (o.side == "sell" and bar.high > o.price)

    def _fill(self, acct: Account, o: Order, px: float, liq: Liquidity, bar: Bar) -> None:
        sym = o.symbol
        p = acct.pos(sym)
        qty = o.qty
        if o.reduce_only:
            # Only reduce in the opposite direction of the current position.
            if abs(p.qty) <= EPS or side_sign(o.side) == (1 if p.qty > 0 else -1):
                self.cancel(acct, o, bar.ts, "reduce-only: no position")
                return
            qty = min(qty, abs(p.qty))
        fee_rate = self.maker_fee if liq == "maker" else self.taker_fee
        fee = qty * px * fee_rate
        realized = self._apply_fill(acct, p, sym, o.side, qty, px, fee, bar.ts, o)
        o.status = "filled"
        acct.orders.pop(o.id, None)
        self.ledger.order(o, bar.ts, f"filled @{px:.8g} {liq}")
        self.ledger.fill(acct.strategy, o.id, sym, o.side, o.role, qty, px, fee, liq, realized, bar.ts)

        if o.role == "entry" and abs(p.qty) > EPS:
            self._attach_brackets(acct, o, p, bar)
        if abs(p.qty) <= EPS:
            # Flat → cancel any remaining brackets / closes on this symbol.
            for other in acct.open_orders(sym):
                if other.reduce_only:
                    self.cancel(acct, other, bar.ts, "position closed (OCO)")

    def _attach_brackets(self, acct: Account, entry: Order, p: Position, bar: Bar) -> None:
        exit_side = opposite(entry.side)
        if entry.stop_loss is not None:
            sl = self.submit(
                acct.strategy, symbol=entry.symbol, side=exit_side, type="stop", qty=abs(p.qty),
                price=entry.stop_loss, role="sl", active_from=bar.ts, now=bar.ts,
                reduce_only=True, tag=entry.tag, parent_id=entry.id,
            )
            p.sl_order = sl.id
            p.init_stop = entry.stop_loss
        if entry.take_profit is not None:
            tp = self.submit(
                acct.strategy, symbol=entry.symbol, side=exit_side, type="limit", qty=abs(p.qty),
                price=entry.take_profit, role="tp", active_from=bar.ts + self.bar_ms, now=bar.ts,
                reduce_only=True, tag=entry.tag, parent_id=entry.id,
            )
            p.tp_order = tp.id

    def _apply_fill(self, acct: Account, p: Position, sym: str, side: Side, qty: float, px: float,
                    fee: float, ts: int, o: Order | None, reason: str | None = None) -> float:
        """Net the fill into the position; returns realized gross PnL."""
        signed = qty * side_sign(side)
        acct.balance -= fee
        acct.fees_total += fee
        realized = 0.0
        if abs(p.qty) <= EPS:
            self._open(p, signed, px, ts, o)
            p.fees += fee
            return 0.0
        same_dir = (p.qty > 0) == (signed > 0)
        if same_dir:
            new_qty = p.qty + signed
            p.avg_px = (p.avg_px * abs(p.qty) + px * abs(signed)) / abs(new_qty)
            p.qty = new_qty
            p.entry_notional += abs(signed) * px
            p.entry_qty += abs(signed)
            p.max_abs_qty = max(p.max_abs_qty, abs(new_qty))
            p.fees += fee
            return 0.0
        close_qty = min(abs(signed), abs(p.qty))
        realized = close_qty * (px - p.avg_px) * (1 if p.qty > 0 else -1)
        acct.balance += realized
        p.realized += realized
        p.exit_notional += close_qty * px
        p.exit_qty += close_qty
        remaining = abs(signed) - close_qty
        # attribute fee proportionally between the closing and the opening part
        p.fees += fee * (close_qty / abs(signed))
        p.qty += close_qty * (1 if signed > 0 else -1)
        if abs(p.qty) <= EPS:
            p.qty = 0.0
            self._record_trade(acct, sym, p, ts, reason or (o.role if o else "close"))
            acct.positions[sym] = Position()
            if remaining > EPS:  # flip
                np_ = acct.positions[sym]
                self._open(np_, remaining * (1 if signed > 0 else -1), px, ts, o)
                np_.fees += fee * (remaining / abs(signed))
        return realized

    @staticmethod
    def _open(p: Position, signed: float, px: float, ts: int, o: Order | None) -> None:
        p.qty = signed
        p.side = "long" if signed > 0 else "short"
        p.avg_px = px
        p.opened_ts = ts
        p.entry_tag = o.tag if o else ""
        p.max_abs_qty = abs(signed)
        p.entry_notional = abs(signed) * px
        p.entry_qty = abs(signed)

    def _record_trade(self, acct: Account, sym: str, p: Position, ts: int, reason: str) -> None:
        entry_px = p.entry_notional / p.entry_qty if p.entry_qty else p.avg_px
        exit_px = p.exit_notional / p.exit_qty if p.exit_qty else entry_px
        side = p.side or "long"
        net = p.realized - p.fees + p.funding
        r = None
        if p.init_stop is not None:
            risk = abs(entry_px - p.init_stop) * p.max_abs_qty
            if risk > EPS:
                r = net / risk
        self.ledger.trade(acct.strategy, {
            "symbol": sym, "side": side, "entry_ts": p.opened_ts, "exit_ts": ts,
            "qty": p.max_abs_qty, "entry_px": entry_px, "exit_px": exit_px,
            "gross_pnl": p.realized, "fees": p.fees, "funding": p.funding,
            "net_pnl": net, "r_multiple": r, "exit_reason": reason, "entry_tag": p.entry_tag,
        })

    # ── liquidation ──────────────────────────────────────────────────────
    def _check_liquidation(self, acct: Account, symbol: str, bar: Bar) -> None:
        p = acct.positions.get(symbol)
        if p is None or abs(p.qty) <= EPS:
            return
        worst = dict(self.marks)
        worst[symbol] = bar.low if p.qty > 0 else bar.high
        eq = acct.balance + self.upnl(acct, worst)
        maint = sum(abs(q.qty) * worst.get(s, q.avg_px) * self.mmr for s, q in acct.positions.items())
        if eq > maint:
            return
        acct.liquidations += 1
        self.ledger.event(bar.ts, "error", acct.strategy, "liquidation",
                          f"equity {eq:.2f} <= maintenance {maint:.2f}; closing all positions")
        for o in list(acct.open_orders()):
            self.cancel(acct, o, bar.ts, "liquidation")
        for sym, q in list(acct.positions.items()):
            if abs(q.qty) <= EPS:
                continue
            px = worst.get(sym, q.avg_px)
            side: Side = "sell" if q.qty > 0 else "buy"
            qty = abs(q.qty)
            fee = qty * px * self.taker_fee
            realized = self._apply_fill(acct, q, sym, side, qty, px, fee, bar.ts, None, reason="liquidation")
            self.ledger.fill(acct.strategy, 0, sym, side, "liquidation", qty, px, fee, "taker", realized, bar.ts)
        if acct.balance < 0:
            acct.balance = 0.0  # isolated loss: wallet can't go below zero

    # ── persistence ──────────────────────────────────────────────────────
    def to_state(self) -> dict[str, Any]:
        return {
            "accounts": {
                k: {
                    **{f: getattr(a, f) for f in ("strategy", "balance", "day_ts", "day_start_equity",
                                                  "fees_total", "funding_total", "liquidations")},
                    "positions": {s: asdict(p) for s, p in a.positions.items() if abs(p.qty) > EPS},
                    "orders": {str(i): asdict(o) for i, o in a.orders.items()},
                }
                for k, a in self.accounts.items()
            },
            "marks": self.marks,
            "last_bar_ts": self.last_bar_ts,
            "funding_cursor": self._funding_cursor,
            "pending_funding": self._pending_funding,
            "next_id": self._next_id,
        }

    def load_state(self, st: dict[str, Any]) -> None:
        self.accounts = {}
        for k, a in st["accounts"].items():
            acct = Account(strategy=a["strategy"], balance=a["balance"])
            for f in ("day_ts", "day_start_equity", "fees_total", "funding_total", "liquidations"):
                setattr(acct, f, a[f])
            acct.positions = {s: Position(**p) for s, p in a["positions"].items()}
            acct.orders = {int(i): Order(**o) for i, o in a["orders"].items()}
            self.accounts[k] = acct
        self.marks = dict(st["marks"])
        self.last_bar_ts = {k: int(v) for k, v in st["last_bar_ts"].items()}
        self._funding_cursor = {k: int(v) for k, v in st["funding_cursor"].items()}
        self._pending_funding = list(st["pending_funding"])
        self._next_id = int(st["next_id"])
