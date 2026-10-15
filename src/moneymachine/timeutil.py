from __future__ import annotations

import re
import time
from datetime import datetime, timezone

_TF_RE = re.compile(r"^(\d+)([mHDW])$")
_UNIT_MS = {"m": 60_000, "H": 3_600_000, "D": 86_400_000, "W": 604_800_000}

DAY_MS = 86_400_000


def tf_ms(tf: str) -> int:
    """'1m' '5m' '15m' '1H' '4H' '1D' → milliseconds (OKX bar notation)."""
    m = _TF_RE.match(tf)
    if not m:
        raise ValueError(f"unsupported timeframe: {tf!r} (use e.g. 1m 5m 15m 1H 4H 1D)")
    return int(m.group(1)) * _UNIT_MS[m.group(2)]


def now_ms() -> int:
    return int(time.time() * 1000)


def to_ms(value: str | int | float | datetime) -> int:
    """Parse '2024-01-01', '2024-01-01T08:00', unix seconds or ms → UTC ms."""
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    if isinstance(value, (int, float)):
        return int(value if value > 1e11 else value * 1000)
    s = str(value).strip()
    if s.isdigit():
        return to_ms(int(s))
    dt = datetime.fromisoformat(s)
    return to_ms(dt)


def fmt_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def day_start(ms: int) -> int:
    return ms - ms % DAY_MS
