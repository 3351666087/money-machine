"""Prompt builders. Market data is given as compact *relative* numbers so the
prompt is short, and (with ``anonymize``) the model can't anchor on a symbol or
a date it may remember from training."""
from __future__ import annotations

import numpy as np

from moneymachine.strategies.base import BarWindow, StrategyContext
from moneymachine.strategies.indicators import atr, ema, rsi
from moneymachine.timeutil import fmt_ms

DIRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["enter_long", "enter_short", "close", "hold"]},
        "stop_atr": {"type": "number"},
        "take_profit_atr": {"type": "number"},
        "confidence": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["action", "stop_atr", "take_profit_atr", "confidence", "reason"],
    "additionalProperties": False,
}

VETO_SCHEMA = {
    "type": "object",
    "properties": {
        "approve": {"type": "boolean"},
        "size_multiplier": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["approve", "size_multiplier", "reason"],
    "additionalProperties": False,
}

DIRECT_SYSTEM = """You are a disciplined systematic crypto trader managing one USDT-margined perpetual swap position.
You are evaluated on net profit AFTER fees, slippage and funding, compared with a buy-and-hold benchmark and a random-entry placebo with identical risk rules.

Rules:
- "hold" is the default. Only enter when you see a concrete, specific edge large enough to beat the round-trip cost shown below.
- Position size is fixed by the risk engine from your stop distance (~1% of equity at risk); you only choose direction, stop and target.
- stop_atr: stop distance in ATR units, between 1.0 and 5.0. take_profit_atr: target in ATR units between 1.0 and 10.0, or 0 for no fixed target.
- "close" exits the current position; "enter_long"/"enter_short" while holding the opposite side reverses it.
- confidence: integer 0-100, your honest probability-weighted conviction.
- reason: one short sentence in Chinese.
Respond with a single JSON object only: {"action", "stop_atr", "take_profit_atr", "confidence", "reason"}."""

VETO_SYSTEM = """You are a risk reviewer for a systematic crypto trading strategy on USDT-margined perpetual swaps.
A rule-based strategy proposes an entry. You may only approve it (optionally at reduced size) or reject it; you can never create a trade.
Reject when the market context makes this specific entry unusually poor (e.g. entering right into an obvious exhaustion, extreme volatility spike, or immediately against a strong opposite move). Otherwise approve — the rule strategy is evaluated on many trades, and excessive vetoing destroys its edge.
size_multiplier: 1.0 for full size, down to 0.25 for reduced size (ignored when rejecting).
reason: one short sentence in Chinese.
Respond with a single JSON object only: {"approve", "size_multiplier", "reason"}."""


def _pct(a: float, b: float) -> float:
    return (a / b - 1.0) * 100.0 if b else 0.0


def market_snapshot(ctx: StrategyContext, bars: BarWindow, *, lookback: int = 48, anonymize: bool = False) -> str:
    c, h, l, o, v = bars.close, bars.high, bars.low, bars.open, bars.volume
    px = float(c[-1])
    a = float(atr(h, l, c, 14)[-1])
    e20, e50, e200 = (float(ema(c, n)[-1]) for n in (20, 50, 200))
    r = float(rsi(c, 14)[-1])
    rt = ctx.costs
    rt_cost = (rt.maker_fee + rt.taker_fee) * 100 + 2 * rt.slippage_bps / 100  # % of notional, rough

    def ret(n: int) -> str:
        return f"{_pct(px, float(c[-n - 1])):+.2f}%" if len(c) > n else "n/a"

    hi72, lo72 = float(h[-72:].max()), float(l[-72:].min())
    name = "ASSET-X (USDT perpetual swap)" if anonymize else f"{ctx.symbol} (USDT perpetual swap)"
    lines = [
        f"Instrument: {name}",
        f"Timeframe: {ctx.timeframe}" + ("" if anonymize else f" | decision time {fmt_ms(ctx.now)} UTC"),
        f"Costs: maker {rt.maker_fee:.3%}, taker {rt.taker_fee:.3%}, slippage ~{rt.slippage_bps:.0f}bp per side "
        f"→ round trip ≈ {rt_cost:.3f}% ≈ {rt_cost / (a / px * 100):.2f} ATR. Funding is charged every few hours on open positions.",
        "",
        "Indicators (latest closed bar):",
        (f"  close = 100.00 (prices below are relative, close=100)" if anonymize else f"  close = {px:.8g}"),
        f"  ATR14 = {a / px * 100:.3f}% of price",
        f"  EMA20 {_pct(px, e20):+.2f}% | EMA50 {_pct(px, e50):+.2f}% | EMA200 {_pct(px, e200):+.2f}% (close vs EMA)",
        f"  RSI14 = {r:.1f}",
        f"  returns: 1 bar {ret(1)}, 6 bars {ret(6)}, 24 bars {ret(24)}, 72 bars {ret(72)}",
        f"  72-bar range: high {_pct(hi72, px):+.2f}% / low {_pct(lo72, px):+.2f}% from close",
        "",
        f"Last {lookback} bars, oldest first. OHLC as % vs latest close; vol relative to {lookback}-bar mean:",
        "  idx    open    high     low   close   vol",
    ]
    n = min(lookback, len(c))
    vmean = float(v[-n:].mean()) or 1.0
    for i in range(len(c) - n, len(c)):
        k = i - len(c)
        lines.append(
            f"  {k:>4} {_pct(o[i], px):+7.2f} {_pct(h[i], px):+7.2f} {_pct(l[i], px):+7.2f} "
            f"{_pct(c[i], px):+7.2f} {v[i] / vmean:5.2f}"
        )
    lines.append("")
    if ctx.flat:
        lines.append("Current position: flat.")
    else:
        side = "LONG" if ctx.long else "SHORT"
        upnl_pct = _pct(px, ctx.position_avg_px) * (1 if ctx.long else -1)
        stop_txt = ""
        if ctx.position_init_stop:
            risk = abs(ctx.position_avg_px - ctx.position_init_stop)
            if risk > 0:
                r_mult = (px - ctx.position_avg_px) / risk * (1 if ctx.long else -1)
                stop_txt = f", initial stop {_pct(ctx.position_init_stop, px):+.2f}% from close, now {r_mult:+.2f}R"
        lines.append(
            f"Current position: {side} for {ctx.bars_held()} bars, entry {_pct(ctx.position_avg_px, px):+.2f}% "
            f"from close, unrealized {upnl_pct:+.2f}%{stop_txt}."
        )
    return "\n".join(lines)


def veto_request(ctx: StrategyContext, bars: BarWindow, entry, base_kind: str, *, lookback: int, anonymize: bool) -> str:
    px = float(bars.close[-1])
    a = float(atr(bars.high, bars.low, bars.close, 14)[-1])
    side = "LONG" if entry.side == "buy" else "SHORT"
    stop = f"{abs(entry.stop_loss - px) / a:.2f} ATR" if entry.stop_loss else "none"
    tp = f"{abs(entry.take_profit - px) / a:.2f} ATR" if entry.take_profit else "none (trend-following, exit on signal)"
    return (
        f"Proposed by rule strategy '{base_kind}': {side} market entry at the next bar open, "
        f"stop {stop}, target {tp}.\n\n" + market_snapshot(ctx, bars, lookback=lookback, anonymize=anonymize)
    )


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))
