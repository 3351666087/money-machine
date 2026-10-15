"""Strategy interface. Strategies see only *closed* strategy-timeframe bars and a
read-only snapshot of their own account; they return intents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from moneymachine.config import CostConfig, StrategyConfig
from moneymachine.types import Bar, Instrument, Intent


@dataclass
class BarWindow:
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_bars(cls, bars) -> "BarWindow":
        bars = list(bars)
        return cls(
            ts=np.fromiter((b.ts for b in bars), dtype=np.int64, count=len(bars)),
            open=np.fromiter((b.open for b in bars), dtype=float, count=len(bars)),
            high=np.fromiter((b.high for b in bars), dtype=float, count=len(bars)),
            low=np.fromiter((b.low for b in bars), dtype=float, count=len(bars)),
            close=np.fromiter((b.close for b in bars), dtype=float, count=len(bars)),
            volume=np.fromiter((b.volume for b in bars), dtype=float, count=len(bars)),
        )

    def __len__(self) -> int:
        return len(self.close)


@dataclass
class StrategyContext:
    """Immutable snapshot handed to a strategy at decision time."""

    strategy_id: str
    symbol: str
    now: int                  # decision time (close of the latest strategy bar)
    timeframe: str
    bar_ms: int
    equity: float
    position_qty: float
    position_avg_px: float
    position_opened_ts: int
    position_init_stop: float | None
    has_pending_entry: bool
    instrument: Instrument
    costs: CostConfig
    catchup: bool = False     # live: replaying bars missed while offline
    services: dict[str, Any] = field(default_factory=dict)  # e.g. {"llm": LLMClient}
    logs: list[tuple[str, str, Any]] = field(default_factory=list)

    @property
    def flat(self) -> bool:
        return abs(self.position_qty) < 1e-12

    @property
    def long(self) -> bool:
        return self.position_qty > 1e-12

    @property
    def short(self) -> bool:
        return self.position_qty < -1e-12

    def bars_held(self) -> int:
        if self.flat:
            return 0
        return max(0, (self.now - self.position_opened_ts) // self.bar_ms)

    def log(self, kind: str, message: str, data: Any = None) -> None:
        self.logs.append((kind, message, data))


@dataclass
class PortfolioContext:
    """Snapshot for portfolio strategies: one call per decision time, all symbols at once."""

    strategy_id: str
    now: int
    timeframe: str
    equity: float
    positions: dict[str, float]          # symbol -> signed qty (base ccy)
    marks: dict[str, float]              # symbol -> latest price
    catchup: bool = False
    services: dict[str, Any] = field(default_factory=dict)
    logs: list[tuple[str, str, Any]] = field(default_factory=list)

    def weights(self) -> dict[str, float]:
        if self.equity <= 0:
            return {}
        return {s: q * self.marks.get(s, 0.0) / self.equity for s, q in self.positions.items() if q}

    def log(self, kind: str, message: str, data: Any = None) -> None:
        self.logs.append((kind, message, data))


class Strategy:
    kind = "base"
    #: True → may block on network (LLM). Live runner executes it in a thread.
    slow = False
    #: True → ``on_portfolio`` is called once per decision time with every symbol,
    #: and returns target weights instead of per-symbol intents.
    portfolio = False

    def __init__(self, cfg: StrategyConfig, seed: int = 0):
        self.cfg = cfg
        self.params = dict(cfg.params)
        self.seed = seed

    @property
    def id(self) -> str:
        return self.cfg.id

    @property
    def symbols(self) -> list[str]:
        return self.cfg.symbols

    def warmup(self) -> int:
        """Minimum number of closed strategy bars before on_bar is called."""
        return 50

    def on_bar(self, ctx: StrategyContext, bars: BarWindow) -> list[Intent]:
        raise NotImplementedError

    def on_portfolio(self, ctx: PortfolioContext) -> dict[str, float] | None:
        """Target weights (fraction of equity, signed) per symbol; None = keep positions."""
        raise NotImplementedError

    def get_state(self) -> dict:
        return {}

    def set_state(self, st: dict) -> None:
        pass

    def p(self, key: str, default):
        return type(default)(self.params.get(key, default)) if default is not None else self.params.get(key)


def last_bar(bars: BarWindow) -> Bar:
    return Bar(int(bars.ts[-1]), float(bars.open[-1]), float(bars.high[-1]), float(bars.low[-1]),
               float(bars.close[-1]), float(bars.volume[-1]))
