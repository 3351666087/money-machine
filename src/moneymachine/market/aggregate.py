from __future__ import annotations

from moneymachine.types import Bar


class Aggregator:
    """Builds strategy-timeframe bars (e.g. 1H) from matching bars (1m/5m).

    A bucket is emitted when its last sub-bar arrives, or — if data has a gap —
    as soon as a sub-bar of a later bucket arrives. Emitted bar ``ts`` is the
    bucket open time; its decision time is ``ts + tf_ms``.
    """

    def __init__(self, tf_ms: int, sub_ms: int):
        if tf_ms % sub_ms:
            raise ValueError("strategy timeframe must be a multiple of matching timeframe")
        self.tf_ms = tf_ms
        self.sub_ms = sub_ms
        self.cur: Bar | None = None

    def push(self, bar: Bar) -> list[Bar]:
        out: list[Bar] = []
        bucket = bar.ts - bar.ts % self.tf_ms
        if self.cur is not None and self.cur.ts != bucket:
            out.append(self.cur)
            self.cur = None
        if self.cur is None:
            self.cur = Bar(bucket, bar.open, bar.high, bar.low, bar.close, bar.volume)
        else:
            c = self.cur
            c.high = max(c.high, bar.high)
            c.low = min(c.low, bar.low)
            c.close = bar.close
            c.volume += bar.volume
        if bar.ts + self.sub_ms >= bucket + self.tf_ms:
            out.append(self.cur)
            self.cur = None
        return out

    def to_state(self) -> list | None:
        c = self.cur
        return [c.ts, c.open, c.high, c.low, c.close, c.volume] if c else None

    def load_state(self, st: list | None) -> None:
        self.cur = Bar(*st) if st else None
