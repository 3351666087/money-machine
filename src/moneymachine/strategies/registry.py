from __future__ import annotations

from moneymachine.config import StrategyConfig
from moneymachine.strategies.base import Strategy
from moneymachine.strategies.llm_strategies import LLMDirect, LLMVeto
from moneymachine.strategies.ml_portfolio import (
    MLConcentrated, MLPlacebo, MLPlaceboConcentrated, MLPlaceboSmoothed, MLPortfolio, MLSmoothed,
)
from moneymachine.strategies.rules import BuyAndHold, DonchianBreakout, EmaTrend, RandomEntry
from moneymachine.strategies.trend import TrendTSMOM

KINDS: dict[str, type[Strategy]] = {
    cls.kind: cls for cls in (BuyAndHold, RandomEntry, EmaTrend, DonchianBreakout, LLMDirect, LLMVeto, MLPortfolio, MLPlacebo,
                MLSmoothed, MLPlaceboSmoothed, MLConcentrated, MLPlaceboConcentrated, TrendTSMOM)
}


def build_strategy(cfg: StrategyConfig, seed: int = 0) -> Strategy:
    try:
        cls = KINDS[cfg.kind]
    except KeyError:
        raise ValueError(f"unknown strategy kind {cfg.kind!r}; available: {sorted(KINDS)}") from None
    return cls(cfg, seed)
