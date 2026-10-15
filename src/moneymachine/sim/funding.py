"""Funding-rate schedule for perpetual swaps.

Inside the period covered by real data we use the real settlement times and
realized rates. Outside it:
  - backtest mode: settlements continue on the inferred interval with the
    configured default rate (and the report flags it);
  - live mode: settlements after the latest known one are *pending* (rate None)
    until the realized rate is fetched.
"""
from __future__ import annotations

import bisect

HOUR_MS = 3_600_000


class FundingBook:
    def __init__(self, default_rate: float, default_interval_h: int = 8, live: bool = False):
        self.default_rate = default_rate
        self.default_interval = default_interval_h * HOUR_MS
        self.live = live
        self._times: dict[str, list[int]] = {}
        self._rates: dict[str, dict[int, float]] = {}

    # ── data ──────────────────────────────────────────────────────────────
    def add(self, symbol: str, records) -> int:
        """records: iterable of (funding_time_ms, rate). Returns #new."""
        rates = self._rates.setdefault(symbol, {})
        before = len(rates)
        for ts, rate in records:
            rates[int(ts)] = float(rate)
        self._times[symbol] = sorted(rates)
        return len(rates) - before

    def has_data(self, symbol: str) -> bool:
        return bool(self._times.get(symbol))

    def coverage(self, symbol: str) -> tuple[int, int] | None:
        t = self._times.get(symbol)
        return (t[0], t[-1]) if t else None

    def interval(self, symbol: str) -> int:
        t = self._times.get(symbol)
        if not t or len(t) < 2:
            return self.default_interval
        tail = t[-50:]
        diffs = sorted(b - a for a, b in zip(tail, tail[1:]))
        return diffs[len(diffs) // 2] or self.default_interval

    def last_rate(self, symbol: str) -> float:
        t = self._times.get(symbol)
        return self._rates[symbol][t[-1]] if t else self.default_rate

    # ── queries ───────────────────────────────────────────────────────────
    def settlements(self, symbol: str, after: int, upto: int) -> list[int]:
        """Settlement times in the half-open interval (after, upto]."""
        if upto <= after:
            return []
        t = self._times.get(symbol)
        iv = self.interval(symbol)
        if not t:
            first = (after // iv + 1) * iv
            return list(range(first, upto + 1, iv))
        out: list[int] = []
        lo, hi = t[0], t[-1]
        # before coverage
        if after < lo:
            k_first = -((lo - after - 1) // iv)  # smallest k<=0 with lo+k*iv > after
            ts = lo + k_first * iv
            while ts < lo and ts <= upto:
                if ts > after:
                    out.append(ts)
                ts += iv
        # inside coverage
        i = bisect.bisect_right(t, after)
        j = bisect.bisect_right(t, upto)
        out.extend(t[i:j])
        # after coverage
        if upto > hi:
            ts = hi + iv
            if ts <= after:
                ts += ((after - ts) // iv + 1) * iv
            while ts <= upto:
                out.append(ts)
                ts += iv
        return out

    def rate(self, symbol: str, ts: int) -> float | None:
        """Realized rate at settlement ``ts``; None = not known yet (live only)."""
        r = self._rates.get(symbol, {}).get(ts)
        if r is not None:
            return r
        cov = self.coverage(symbol)
        if self.live and (cov is None or ts > cov[1]):
            return None
        return self.default_rate

    def is_assumed(self, symbol: str, ts: int) -> bool:
        return ts not in self._rates.get(symbol, {})
