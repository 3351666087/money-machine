"""Core data types shared by market, sim, strategies and ledger.

Conventions
-----------
- All timestamps are **UTC milliseconds**. A bar's ``ts`` is its *open* time.
- Quantities are in **base currency** (e.g. BTC), not contracts. Signed position
  quantity: ``> 0`` long, ``< 0`` short.
- Prices are quote currency (USDT).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

Side = Literal["buy", "sell"]
OrdType = Literal["market", "limit", "stop"]
OrdStatus = Literal["new", "filled", "canceled", "expired", "rejected"]
Liquidity = Literal["maker", "taker"]


@dataclass(slots=True)
class Bar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(slots=True)
class Instrument:
    inst_id: str
    ct_val: float = 1.0      # base ccy per contract
    lot_sz: float = 1.0      # contracts step
    min_sz: float = 1.0      # min contracts
    tick_sz: float = 0.01

    @property
    def qty_step(self) -> float:
        return self.ct_val * self.lot_sz

    @property
    def min_qty(self) -> float:
        return self.ct_val * self.min_sz

    def floor_qty(self, qty: float) -> float:
        step = self.qty_step
        n = math.floor(qty / step + 1e-9)
        return round(n * step, 12)

    def round_px(self, px: float) -> float:
        t = self.tick_sz
        return round(round(px / t) * t, 12)


def opposite(side: Side) -> Side:
    return "sell" if side == "buy" else "buy"


def side_sign(side: Side) -> int:
    return 1 if side == "buy" else -1


# ── Strategy intents ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class Entry:
    """Open a new position (strategy → risk → broker).

    sizing:
      - ``"risk"``: qty so that hitting ``stop_loss`` loses ``risk_pct`` of equity
        (``risk_pct`` None → account default). Requires ``stop_loss``.
      - ``"notional"``: qty = equity * ``notional_frac`` / price (e.g. buy & hold).
    ``size_mult`` scales the final quantity (used by the LLM veto layer).
    """

    symbol: str
    side: Side
    order_type: Literal["market", "limit"] = "market"
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    sizing: Literal["risk", "notional"] = "risk"
    risk_pct: float | None = None
    notional_frac: float = 1.0
    size_mult: float = 1.0
    expire_bars: int | None = None
    tag: str = ""
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class ClosePosition:
    symbol: str
    tag: str = "close"


@dataclass(slots=True)
class CancelPending:
    """Cancel pending (unfilled) entry orders on ``symbol``."""

    symbol: str


Intent = Entry | ClosePosition | CancelPending
