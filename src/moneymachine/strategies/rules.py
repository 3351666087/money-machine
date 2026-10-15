"""Deterministic strategies: S0 baselines (buy & hold, random placebo) and S1
classic rule-based trend followers. All are fully reproducible in backtests."""
from __future__ import annotations

import random

import numpy as np

from moneymachine.strategies.base import BarWindow, Strategy, StrategyContext
from moneymachine.strategies.indicators import atr, ema
from moneymachine.types import ClosePosition, Entry, Intent


class BuyAndHold(Strategy):
    """S0: go long once with ``notional_frac`` of equity and never sell.

    On perpetual swaps this also pays funding — that is part of the benchmark.
    """

    kind = "buy_and_hold"

    def warmup(self) -> int:
        return 1

    def on_bar(self, ctx: StrategyContext, bars: BarWindow) -> list[Intent]:
        if ctx.flat and not ctx.has_pending_entry:
            return [Entry(ctx.symbol, "buy", sizing="notional",
                          notional_frac=self.p("notional_frac", 1.0), tag="hold")]
        return []


class RandomEntry(Strategy):
    """S0 placebo: random direction, same risk rules (ATR stop / target, time exit).

    If a "real" strategy can't beat this, its entries carry no information.
    """

    kind = "random_entry"

    def __init__(self, cfg, seed: int = 0):
        super().__init__(cfg, seed)
        self.rng = random.Random(f"{seed}:{cfg.id}")

    def warmup(self) -> int:
        return self.p("atr_n", 14) + 1

    def on_bar(self, ctx, bars):
        hold = self.p("max_hold_bars", 48)
        if not ctx.flat:
            if ctx.bars_held() >= hold:
                return [ClosePosition(ctx.symbol, tag="time_exit")]
            return []
        if ctx.has_pending_entry or self.rng.random() >= self.p("p_enter", 0.02):
            return []
        a = atr(bars.high, bars.low, bars.close, self.p("atr_n", 14))[-1]
        px = float(bars.close[-1])
        side = "buy" if self.rng.random() < 0.5 else "sell"
        k_sl, k_tp = self.p("sl_atr", 2.0), self.p("tp_atr", 3.0)
        sl = px - k_sl * a if side == "buy" else px + k_sl * a
        tp = px + k_tp * a if side == "buy" else px - k_tp * a
        return [Entry(ctx.symbol, side, stop_loss=sl, take_profit=tp, tag="random")]

    def get_state(self) -> dict:
        v, internal, gauss = self.rng.getstate()
        return {"rng": [v, list(internal), gauss]}

    def set_state(self, st: dict) -> None:
        if "rng" in st:
            v, internal, gauss = st["rng"]
            self.rng.setstate((v, tuple(internal), gauss))


class EmaTrend(Strategy):
    """S1: EMA fast/slow crossover trend follower with an ATR initial stop.

    Enters only on a fresh cross; exits (and optionally reverses) on the
    opposite cross. No take-profit — trend following lets winners run.
    """

    kind = "ema_trend"

    def warmup(self) -> int:
        return self.p("slow", 100) * 2

    def on_bar(self, ctx, bars):
        fast, slow = self.p("fast", 20), self.p("slow", 100)
        allow_short = bool(self.params.get("allow_short", True))
        c = bars.close
        ef, es = ema(c, fast), ema(c, slow)
        now_sig = np.sign(ef[-1] - es[-1])
        prev_sig = np.sign(ef[-2] - es[-2])
        out: list[Intent] = []
        if ctx.long and now_sig < 0:
            out.append(ClosePosition(ctx.symbol, tag="ema_cross_exit"))
        if ctx.short and now_sig > 0:
            out.append(ClosePosition(ctx.symbol, tag="ema_cross_exit"))
        crossed = now_sig != prev_sig and now_sig != 0
        if crossed and not ctx.has_pending_entry:
            want_long = now_sig > 0
            if (want_long and not ctx.long) or (not want_long and allow_short and not ctx.short):
                a = atr(bars.high, bars.low, c, self.p("atr_n", 14))[-1]
                px = float(c[-1])
                k = self.p("sl_atr", 3.0)
                if want_long:
                    out.append(Entry(ctx.symbol, "buy", stop_loss=px - k * a, tag="ema_long"))
                else:
                    out.append(Entry(ctx.symbol, "sell", stop_loss=px + k * a, tag="ema_short"))
        return out


class DonchianBreakout(Strategy):
    """S1: Turtle-style channel breakout. Long above the N-bar high, exit below
    the M-bar low (symmetric for shorts), ATR initial stop."""

    kind = "donchian"

    def warmup(self) -> int:
        return max(self.p("entry_n", 55), self.p("exit_n", 20), self.p("atr_n", 14)) + 2

    def on_bar(self, ctx, bars):
        n_in, n_out = self.p("entry_n", 55), self.p("exit_n", 20)
        allow_short = bool(self.params.get("allow_short", True))
        h, l, c = bars.high, bars.low, bars.close
        px = float(c[-1])
        hi_in, lo_in = h[-n_in - 1 : -1].max(), l[-n_in - 1 : -1].min()
        hi_out, lo_out = h[-n_out - 1 : -1].max(), l[-n_out - 1 : -1].min()
        out: list[Intent] = []
        if ctx.long and px < lo_out:
            return [ClosePosition(ctx.symbol, tag="donchian_exit")]
        if ctx.short and px > hi_out:
            return [ClosePosition(ctx.symbol, tag="donchian_exit")]
        if ctx.flat and not ctx.has_pending_entry:
            a = atr(h, l, c, self.p("atr_n", 14))[-1]
            k = self.p("sl_atr", 2.0)
            if px > hi_in:
                out.append(Entry(ctx.symbol, "buy", stop_loss=px - k * a, tag="donchian_long"))
            elif allow_short and px < lo_in:
                out.append(Entry(ctx.symbol, "sell", stop_loss=px + k * a, tag="donchian_short"))
        return out
