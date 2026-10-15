"""v5 trend following (research/2026-10-15_刷钱机v5_趋势跟踪研究计划.md, trial T1b passed Gate):
BTC + ETH, 20/60/120-day time-series momentum, vol-scaled, long or flat, no leverage.
The paper account checks that live execution follows the backtest rule; performance needs ≥ 6 months."""
from __future__ import annotations

from moneymachine.strategies.base import PortfolioContext, Strategy


class TrendTSMOM(Strategy):
    kind = "trend_tsmom"
    slow = True          # syncs two daily bars over REST at decision time
    portfolio = True

    def warmup(self) -> int:
        return 1

    def on_portfolio(self, ctx: PortfolioContext):
        if ctx.catchup:
            ctx.log("ml_skip", "stale decision (catch-up) — positions left unchanged")
            return None
        from moneymachine.ml.trend import live_targets

        targets, info = live_targets(self.symbols, ctx.now, ctx.weights(),
                                     long_only=bool(self.params.get("long_only", True)),
                                     sync=bool(self.params.get("sync", True)))
        if targets is None:
            ctx.log("ml_skip", info["reason"])
            return None
        sig = ", ".join(f"{s.split('-')[0]} {v:+.2f}" for s, v in info["signal"].items())
        ctx.log("ml_targets", f"trend signal {sig}; gross {info['gross']:.2f}, turnover {info['turnover']:.2f}",
                {"targets": targets, **info})
        return targets
