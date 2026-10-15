"""OKX public REST (no API key needed): candles, instruments, funding rates."""
from __future__ import annotations

import csv
import io
import logging
import time
import zipfile
from typing import Any

import httpx

from moneymachine.timeutil import tf_ms
from moneymachine.types import Bar, Instrument

log = logging.getLogger(__name__)

BASE = "https://www.okx.com"


class OkxError(RuntimeError):
    pass


class OkxRest:
    def __init__(self, timeout: float = 20.0, min_interval: float = 0.12, http: httpx.Client | None = None):
        self.http = http or httpx.Client(base_url=BASE, timeout=timeout,
                                         headers={"User-Agent": "moneymachine/0.1"})
        self.min_interval = min_interval
        self._last = 0.0

    def get(self, path: str, params: dict[str, Any] | None = None, retries: int = 4) -> list:
        params = {k: str(v) for k, v in (params or {}).items() if v is not None}
        delay = 1.0
        for attempt in range(retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.http.get(path, params=params)
                if r.status_code == 429 or r.status_code >= 500:
                    raise OkxError(f"HTTP {r.status_code}")
                j = r.json()
                if str(j.get("code")) == "50011":  # rate limited
                    raise OkxError("rate limited")
                if str(j.get("code")) != "0":
                    raise OkxError(f"{path} code={j.get('code')} msg={j.get('msg')}")
                return j.get("data") or []
            except (httpx.HTTPError, OkxError) as e:
                if attempt == retries or (isinstance(e, OkxError) and "code=" in str(e)):
                    raise
                log.warning("OKX %s failed (%s), retry in %.1fs", path, e, delay)
                time.sleep(delay)
                delay = min(delay * 2, 20)
        return []

    # ── instruments ──────────────────────────────────────────────────────
    def instruments(self, inst_type: str = "SWAP") -> dict[str, Instrument]:
        out = {}
        for d in self.get("/api/v5/public/instruments", {"instType": inst_type}):
            try:
                out[d["instId"]] = Instrument(
                    inst_id=d["instId"], ct_val=float(d.get("ctVal") or 1), lot_sz=float(d["lotSz"]),
                    min_sz=float(d["minSz"]), tick_sz=float(d["tickSz"]),
                )
            except (KeyError, ValueError):
                continue
        return out

    # ── candles ──────────────────────────────────────────────────────────
    @staticmethod
    def _parse(rows: list) -> list[tuple[Bar, bool]]:
        out = []
        for r in rows:
            out.append((Bar(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])),
                        str(r[8]) == "1" if len(r) > 8 else True))
        return out

    def candles(self, inst: str, bar: str, *, after: int | None = None, limit: int = 300,
                history: bool = False) -> list[tuple[Bar, bool]]:
        """Newest-first page of bars with ts < ``after`` (or latest)."""
        path = "/api/v5/market/history-candles" if history else "/api/v5/market/candles"
        lim = min(limit, 100 if history else 300)
        return self._parse(self.get(path, {"instId": inst, "bar": bar, "after": after, "limit": lim}))

    def fetch_range(self, inst: str, bar: str, start_ms: int, end_ms: int, *, history: bool = True,
                    progress=None) -> list[Bar]:
        """All *confirmed* bars with start_ms <= ts <= end_ms, ascending."""
        step = tf_ms(bar)
        cursor = end_ms + step
        got: dict[int, Bar] = {}
        while cursor > start_ms:
            page = self.candles(inst, bar, after=cursor, history=history)
            if not page:
                break
            for b, confirmed in page:
                if confirmed and start_ms <= b.ts <= end_ms:
                    got[b.ts] = b
            oldest = min(b.ts for b, _ in page)
            if oldest >= cursor:
                break
            cursor = oldest
            if progress:
                progress(len(got))
        return [got[k] for k in sorted(got)]

    # ── funding ──────────────────────────────────────────────────────────
    def funding_recent(self, inst: str, since_ms: int = 0) -> list[tuple[int, float]]:
        """Realized funding rates via REST (OKX keeps ~3 months here)."""
        out: dict[int, float] = {}
        after = None
        while True:
            rows = self.get("/api/v5/public/funding-rate-history", {"instId": inst, "limit": 100, "after": after})
            if not rows:
                break
            for r in rows:
                ts = int(r["fundingTime"])
                rate = r.get("realizedRate") or r.get("fundingRate")
                if rate not in (None, ""):
                    out[ts] = float(rate)
            oldest = min(int(r["fundingTime"]) for r in rows)
            if oldest <= since_ms or (after is not None and oldest >= after):
                break
            after = oldest
        return sorted((t, r) for t, r in out.items() if t >= since_ms)

    def funding_archive(self, inst: str, begin_ms: int, end_ms: int) -> list[tuple[int, float]]:
        """Older realized funding from OKX's monthly historical-data files."""
        family = inst.replace("-SWAP", "")
        out: dict[int, float] = {}
        chunk_ms = 270 * 86_400_000  # API limit: <= 10 months per request
        b = begin_ms
        while b < end_ms:
            e = min(b + chunk_ms, end_ms)
            data = self.get("/api/v5/public/market-data-history", {
                "module": 3, "instType": "SWAP", "instFamilyList": family,
                "dateAggrType": "monthly", "begin": b, "end": e,
            })
            for block in data:
                for det in block.get("details", []):
                    for g in det.get("groupDetails", []):
                        url = g.get("url")
                        if url:
                            for ts, rate in self._read_funding_zip(url, inst):
                                out[ts] = rate
            b = e
        return sorted(out.items())

    def _read_funding_zip(self, url: str, inst: str) -> list[tuple[int, float]]:
        r = httpx.get(url, timeout=60)
        r.raise_for_status()
        rows = []
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for name in z.namelist():
                text = z.read(name).decode("utf-8-sig")
                for rec in csv.DictReader(io.StringIO(text)):
                    if rec.get("instrument_name") == inst:
                        rows.append((int(rec["funding_time"]), float(rec["funding_rate"])))
        return rows
