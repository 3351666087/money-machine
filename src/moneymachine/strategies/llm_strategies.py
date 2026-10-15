"""S2 / S3: LLM arms. Both are fail-safe: any LLM failure → no new trade."""
from __future__ import annotations

from moneymachine.config import StrategyConfig
from moneymachine.llm.prompts import (
    DIRECT_SCHEMA, DIRECT_SYSTEM, VETO_SCHEMA, VETO_SYSTEM, clamp, market_snapshot, veto_request,
)
from moneymachine.strategies.base import BarWindow, Strategy, StrategyContext
from moneymachine.strategies.indicators import atr
from moneymachine.types import ClosePosition, Entry, Intent


class LLMDirect(Strategy):
    """S3: the model decides direction / stop / target every bar."""

    kind = "llm_direct"
    slow = True

    def warmup(self) -> int:
        return 210  # EMA200 in the snapshot

    def on_bar(self, ctx: StrategyContext, bars: BarWindow) -> list[Intent]:
        if ctx.catchup:
            ctx.log("llm_skip", "catch-up bar: LLM not called (decision would be stale)")
            return []
        if ctx.has_pending_entry:
            return []
        llm = ctx.services.get("llm")
        if llm is None:
            ctx.log("llm_skip", "no LLM client")
            return []
        model = self.params["model"]
        user = market_snapshot(ctx, bars, lookback=int(self.params.get("lookback", 48)),
                               anonymize=bool(self.params.get("anonymize", False)))
        res = llm.call_json(strategy=self.id, purpose="direct", symbol=ctx.symbol, model=model,
                            system=DIRECT_SYSTEM, user=user, schema=DIRECT_SCHEMA, now=ctx.now)
        if not res.ok:
            ctx.log("llm_fail", f"{res.error} → no action")
            return []
        d = res.data
        action = d["action"]
        if int(d["confidence"]) < int(self.params.get("min_confidence", 60)) and action.startswith("enter"):
            ctx.log("llm_low_conf", f"confidence {d['confidence']} below threshold → hold", d)
            return []
        if action == "hold":
            return []
        if action == "close":
            return [] if ctx.flat else [ClosePosition(ctx.symbol, tag=f"llm_close:{model}")]

        want_long = action == "enter_long"
        if (want_long and ctx.long) or (not want_long and ctx.short):
            return []
        px = float(bars.close[-1])
        a = float(atr(bars.high, bars.low, bars.close, 14)[-1])
        k_sl = clamp(float(d["stop_atr"]), 1.0, 5.0)
        k_tp = float(d["take_profit_atr"])
        k_tp = 0.0 if k_tp <= 0 else clamp(k_tp, 1.0, 10.0)
        sl = px - k_sl * a if want_long else px + k_sl * a
        tp = (px + k_tp * a if want_long else px - k_tp * a) if k_tp else None
        out: list[Intent] = []
        if not ctx.flat:
            out.append(ClosePosition(ctx.symbol, tag=f"llm_reverse:{model}"))
        out.append(Entry(ctx.symbol, "buy" if want_long else "sell", stop_loss=sl, take_profit=tp,
                         risk_pct=self.params.get("risk_pct"), tag=f"llm:{model}",
                         meta={"confidence": d["confidence"], "reason": d["reason"]}))
        return out


class LLMVeto(Strategy):
    """S2: a rule strategy whose *entries* must be approved by the model.
    Exits always pass through. The model can only remove or shrink trades."""

    kind = "llm_veto"
    slow = True

    def __init__(self, cfg: StrategyConfig, seed: int = 0):
        super().__init__(cfg, seed)
        from moneymachine.strategies.registry import build_strategy

        base_cfg = StrategyConfig(id=cfg.id, kind=self.params["base"], symbols=cfg.symbols,
                                  params=dict(self.params.get("base_params", {})))
        self.base = build_strategy(base_cfg, seed)

    def warmup(self) -> int:
        return max(self.base.warmup(), 210)

    def on_bar(self, ctx: StrategyContext, bars: BarWindow) -> list[Intent]:
        intents = self.base.on_bar(ctx, bars)
        out: list[Intent] = []
        for it in intents:
            if not isinstance(it, Entry):
                out.append(it)
                continue
            if ctx.catchup:
                ctx.log("llm_skip", "catch-up bar: entry dropped (cannot review in time)")
                continue
            llm = ctx.services.get("llm")
            if llm is None:
                ctx.log("llm_skip", "no LLM client → entry dropped")
                continue
            model = self.params["model"]
            user = veto_request(ctx, bars, it, self.base.kind, lookback=int(self.params.get("lookback", 48)),
                                anonymize=bool(self.params.get("anonymize", False)))
            res = llm.call_json(strategy=self.id, purpose="veto", symbol=ctx.symbol, model=model,
                                system=VETO_SYSTEM, user=user, schema=VETO_SCHEMA, now=ctx.now)
            if not res.ok:
                ctx.log("llm_fail", f"{res.error} → entry dropped (fail-safe)")
                continue
            if not res.data["approve"]:
                ctx.log("llm_veto", f"vetoed: {res.data['reason']}", res.data)
                continue
            it.size_mult *= clamp(float(res.data["size_multiplier"]), 0.25, 1.0)
            it.tag = f"{it.tag}+veto:{model}"
            it.meta["veto_reason"] = res.data["reason"]
            out.append(it)
        return out

    def get_state(self) -> dict:
        return {"base": self.base.get_state()}

    def set_state(self, st: dict) -> None:
        self.base.set_state(st.get("base", {}))
