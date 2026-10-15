"""2.0 strategies: the daily cross-sectional ML portfolio and its placebo, plus the
v3 exploratory arms (2026-10-15): signal smoothing (A1a) and the small-account
concentrated book (B1b), each with a placebo built the same way.

All use the same artifact, data, universe and costs. Decisions are serialised and
share one data sync + one feature panel per decision time (running them concurrently
would write the same data files twice and could double peak memory on a small
server). The model signal is computed once per decision time and shared: Kronos
samples, so separate calls would give the arms different signals.
"""
from __future__ import annotations

import logging
import threading

from moneymachine.strategies.base import PortfolioContext, Strategy

log = logging.getLogger(__name__)

_DECISION_LOCK = threading.Lock()
_SHARED: dict = {"synced_t": None, "panel_t": None, "panel": None, "uses": 0, "instances": 0,
                 "sig_key": None, "sig": None}


class MLPortfolio(Strategy):
    """Each day at 00:00 UTC: refresh data → features → model signal → cost-aware
    optimizer → target weights. Stale (catch-up) decisions are skipped."""

    kind = "ml_portfolio"
    slow = True
    portfolio = True

    def __init__(self, cfg, seed: int = 0):
        super().__init__(cfg, seed)
        self._model = None
        self._meta = None
        _SHARED["instances"] += 1

    def warmup(self) -> int:
        return 1

    def _load(self):
        if self._meta is None:
            from moneymachine.ml.production import load_artifact

            self._model, self._meta = load_artifact(self.params["artifact"])
        return self._model, self._meta

    def _panel_for(self, t: int):
        from moneymachine.ml.production import live_panel, sync_for_live

        if self.params.get("sync", True) and _SHARED["synced_t"] != t:
            sync_for_live()
            _SHARED["synced_t"] = t
        if _SHARED["panel_t"] != t:
            _SHARED.update(panel=live_panel(), panel_t=t, uses=0)
        return _SHARED["panel"]

    def _release_panel(self) -> None:
        _SHARED["uses"] += 1
        if _SHARED["uses"] >= _SHARED["instances"]:
            _SHARED.update(panel=None, panel_t=None, uses=0, sig_key=None, sig=None)  # free memory until tomorrow

    def _signal(self, model, meta, panel, t):
        """The artifact's model signal at t, computed once and shared by every model arm."""
        from moneymachine.ml.production import model_signal

        key = (meta["hash"], t)
        if _SHARED["sig_key"] != key:
            _SHARED.update(sig_key=key, sig=model_signal(model, meta, panel, t))
        sig, info = _SHARED["sig"]
        return sig.copy(), dict(info)

    def _transform(self, sig, t):
        """Hook: signal → signal (identity; smoothing arms override)."""
        return sig, {}

    def _construct(self, sig, meta, panel, ctx):
        """Hook: signal → target weights (the cost-aware optimizer; concentrated arms override)."""
        from moneymachine.ml.production import weights_from_signal

        return weights_from_signal(sig, meta, panel, ctx.now, ctx.weights(), nav=max(ctx.equity, 1.0))

    def on_portfolio(self, ctx: PortfolioContext):
        if ctx.catchup:
            ctx.log("ml_skip", "stale decision (catch-up) — positions left unchanged")
            return None
        with _DECISION_LOCK:
            model, meta = self._load()
            panel = self._panel_for(ctx.now)
            try:
                if ctx.now not in set(panel.days):
                    ctx.log("ml_skip", f"feature data does not reach decision time {ctx.now}")
                    return None
                sig, sinfo = self._signal(model, meta, panel, ctx.now)
                sig, tinfo = self._transform(sig, ctx.now)
                targets, info = self._construct(sig, meta, panel, ctx)
                info = {**info, **tinfo}
            finally:
                self._release_panel()
        ctx.log("ml_targets", f"{sum(1 for w in targets.values() if w)} names, gross {info['gross']:.2f}, "
                              f"turnover {info['turnover']:.2f}", {"targets": targets, **info, **sinfo})
        return targets


class MLPlacebo(MLPortfolio):
    """Placebo arm: identical construction, random signal with the model signal's
    persistence (see ``ml.placebo``). Answers "does the model beat chance under
    exactly the same costs and constraints?"."""

    kind = "ml_placebo"

    def _signal(self, model, meta, panel, t):
        from moneymachine.ml.production import placebo_signal_at

        return placebo_signal_at(meta, panel, t, int(self.params.get("seed", 1)))


# ── v3 exploratory arms (pre-registered trials A1a / B1b; not past Gate A) ──
class _Smoothed:
    """Per-coin EMA of the signal (research_v3 trial A1a). State persists with the run."""

    def __init__(self, cfg, seed: int = 0):
        super().__init__(cfg, seed)
        self._ema: dict = {}

    def _transform(self, sig, t):
        from moneymachine.ml.production import ema_update

        out = ema_update(self._ema, sig, t, float(self.params.get("half_life", 2)),
                         int(self.params.get("gap_reset", 5)))
        return out, {"smooth_half_life": float(self.params.get("half_life", 2)), "smooth_names": len(out)}

    def get_state(self) -> dict:
        return {"ema": self._ema}

    def set_state(self, st: dict) -> None:
        self._ema = {k: list(v) for k, v in (st or {}).get("ema", {}).items()}


class _Concentrated:
    """Small-account book (research_v3 trial B1b): n longs / n shorts with a rank buffer,
    lot-rounded entries, BTC beta hedge, no drift rebalancing."""

    def _construct(self, sig, meta, panel, ctx):
        from moneymachine.ml.production import concentrated_targets

        n = int(self.params.get("n", 8))
        targets, info = concentrated_targets(sig, panel, ctx.now, positions=ctx.positions, marks=ctx.marks,
                                             equity=ctx.equity, instruments=ctx.services["instruments"], n=n,
                                             buffer=int(self.params.get("buffer", 2 * n)))
        return targets, {**info, "turnover": float(sum(abs(w - ctx.weights().get(s, 0.0))
                                                       for s, w in targets.items())),
                         "artifact": meta["name"], "artifact_hash": meta["hash"]}


class MLSmoothed(_Smoothed, MLPortfolio):
    kind = "ml_smoothed"


class MLPlaceboSmoothed(_Smoothed, MLPlacebo):
    kind = "ml_placebo_smoothed"


class MLConcentrated(_Concentrated, MLPortfolio):
    kind = "ml_concentrated"


class MLPlaceboConcentrated(_Concentrated, MLPlacebo):
    kind = "ml_placebo_concentrated"
