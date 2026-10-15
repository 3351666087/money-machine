"""v4 E2: forward shadow measurement of post-only execution — observation only, never trades.

The paper arms rebalance with market orders just after 00:00 UTC. For every such fill of
ML_v2 / A1a_smoothed / B1b_concentrated this process places a *virtual* post-only order of the
same side and size at the best bid (buy) / best ask (sell) seen when it notices the fill, and
watches OKX's public trade stream for 240 minutes:

  filled      a trade printed through the limit by ≥ 1 tick (the E1 replay's rule)
  through_sz  contracts traded through the limit (could our size really have filled?)
  mid30/240   mid price after 30 / 240 minutes (cost of falling back to a market order)

Results: ``runs/execlab.sqlite`` (table ``shadow``), compared with the E1 replay of the same
orders in the v4 report. Reads the paper ledger read-only; never touches the paper accounts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd
import websockets

from moneymachine.market.okx_rest import OkxRest

log = logging.getLogger(__name__)
WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
ARMS = ("ML_v2", "A1a_smoothed", "B1b_concentrated")
WINDOW_MS = 240 * 60_000
CHECK_MS = 30 * 60_000
SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow (
    id INTEGER PRIMARY KEY, arm TEXT, symbol TEXT, side TEXT, qty REAL, contracts REAL, notional REAL,
    taker_ts INTEGER, taker_px REAL, placed_ts INTEGER, bid REAL, ask REAL, limit_px REAL, tick REAL,
    filled_ts INTEGER, filled_min REAL, through_sz REAL DEFAULT 0, mid30 REAL, mid240 REAL,
    status TEXT, note TEXT, UNIQUE(arm, symbol, taker_ts, side)
);
"""


def _now() -> int:
    return int(time.time() * 1000)


class ExecLab:
    def __init__(self, ledger_db: Path, out_db: Path = Path("runs/execlab.sqlite"),
                 instruments: Path = Path("store/ml/instruments.parquet")):
        self.ledger_db = ledger_db
        # network calls run in worker threads (awaited one at a time), the trade handler in the loop
        # thread: never concurrently, so one connection shared across threads is safe
        self.db = sqlite3.connect(str(out_db), check_same_thread=False)
        self.db.executescript(SCHEMA)
        spec = pd.read_parquet(instruments).set_index("inst")
        self.tick = spec["tick_sz"].astype(float).to_dict()
        self.ct_val = spec["ct_val"].astype(float).to_dict()
        self.rest = OkxRest(min_interval=0.15)
        self.last_seen = self.db.execute("SELECT COALESCE(MAX(taker_ts), 0) FROM shadow").fetchone()[0] or \
            _now() - 10 * 60_000
        self.open: dict[int, dict] = {}
        for row in self.db.execute("SELECT id, symbol, side, limit_px, tick, placed_ts FROM shadow "
                                   "WHERE status='open'"):
            i, sym, side, lim, tick, placed = row
            if _now() - placed > WINDOW_MS:
                self._close(i, "expired_during_downtime")
            else:
                self.open[i] = {"symbol": sym, "side": side, "limit": lim, "tick": tick, "placed": placed,
                                "note": "restarted: trades during downtime unseen"}
        self.subscribed: set[str] = set()

    # ── new paper fills → virtual post-only orders ─────────────────────────
    def poll_fills(self) -> list[tuple]:
        db = sqlite3.connect(f"file:{self.ledger_db}?mode=ro", uri=True)
        try:
            run = db.execute("SELECT run_id FROM runs WHERE mode='paper' ORDER BY created_at DESC LIMIT 1").fetchone()
            if not run:
                return []
            q = (f"SELECT strategy, symbol, side, qty, price, ts FROM fills WHERE run_id=? AND role='rebalance' "
                 f"AND ts>? AND strategy IN ({','.join('?' * len(ARMS))}) ORDER BY ts")
            return db.execute(q, (run[0], self.last_seen, *ARMS)).fetchall()
        finally:
            db.close()

    def _mid(self, sym: str) -> tuple[float, float]:
        t = self.rest.get("/api/v5/market/ticker", {"instId": sym})[0]
        return float(t["bidPx"]), float(t["askPx"])

    def place(self, arm, sym, side, qty, px, ts) -> None:
        bid, ask = self._mid(sym)
        tick = self.tick.get(sym, 0.0) or bid * 1e-5
        lim = bid if side == "buy" else ask
        now = _now()
        cur = self.db.execute(
            "INSERT OR IGNORE INTO shadow (arm, symbol, side, qty, contracts, notional, taker_ts, taker_px, placed_ts, "
            "bid, ask, limit_px, tick, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, 'open')",
            (arm, sym, side, qty, qty / self.ct_val.get(sym, 1.0), qty * px, ts, px, now, bid, ask, lim, tick))
        self.db.commit()
        if cur.rowcount:
            self.open[cur.lastrowid] = {"symbol": sym, "side": side, "limit": lim, "tick": tick, "placed": now}
            log.info("shadow %s %s %s %.6g @ %.6g (taker filled @ %.6g)", arm, side, sym, qty, lim, px)

    # ── trade stream ─────────────────────────────────────────────────────
    def on_trade(self, sym: str, px: float, sz: float, ts: int) -> None:
        for i, o in list(self.open.items()):
            if o["symbol"] != sym or ts < o["placed"]:
                continue
            through = px <= o["limit"] - o["tick"] if o["side"] == "buy" else px >= o["limit"] + o["tick"]
            if through:
                self.db.execute("UPDATE shadow SET through_sz = through_sz + ? WHERE id=?", (sz, i))
                if "filled" not in o:
                    o["filled"] = ts
                    self.db.execute("UPDATE shadow SET filled_ts=?, filled_min=? WHERE id=?",
                                    (ts, (ts - o["placed"]) / 60_000, i))

    def _close(self, i: int, status: str, note: str | None = None) -> None:
        self.db.execute("UPDATE shadow SET status=?, note=COALESCE(?, note) WHERE id=?", (status, note, i))
        self.db.commit()
        self.open.pop(i, None)

    def housekeeping(self) -> None:
        now = _now()
        for i, o in list(self.open.items()):
            age = now - o["placed"]
            if age >= CHECK_MS and "mid30" not in o:
                b, a = self._mid(o["symbol"])
                o["mid30"] = (b + a) / 2
                self.db.execute("UPDATE shadow SET mid30=? WHERE id=?", (o["mid30"], i))
            if age >= WINDOW_MS:
                b, a = self._mid(o["symbol"])
                self.db.execute("UPDATE shadow SET mid240=? WHERE id=?", ((b + a) / 2, i))
                self._close(i, "filled" if "filled" in o else "unfilled", o.get("note"))
        self.db.commit()

    async def run(self, stop: asyncio.Event | None = None) -> None:
        stop = stop or asyncio.Event()
        backoff = 1.0
        while not stop.is_set():
            try:
                async with websockets.connect(WS_URL, ping_interval=None, open_timeout=20) as ws:
                    self.subscribed = set()
                    backoff = 1.0
                    next_poll = 0.0
                    while not stop.is_set():
                        if time.monotonic() >= next_poll:
                            next_poll = time.monotonic() + 20
                            for arm, sym, side, qty, px, ts in await asyncio.to_thread(self.poll_fills):
                                self.last_seen = max(self.last_seen, ts)
                                try:
                                    await asyncio.to_thread(self.place, arm, sym, side, qty, px, ts)
                                except Exception as e:  # noqa: BLE001
                                    log.warning("shadow %s %s not placed: %s", arm, sym, e)
                            await asyncio.to_thread(self.housekeeping)
                            want = {o["symbol"] for o in self.open.values()}
                            add, drop = want - self.subscribed, self.subscribed - want
                            if add:
                                await ws.send(json.dumps({"op": "subscribe",
                                                          "args": [{"channel": "trades", "instId": s} for s in add]}))
                            if drop:
                                await ws.send(json.dumps({"op": "unsubscribe",
                                                          "args": [{"channel": "trades", "instId": s} for s in drop]}))
                            self.subscribed = want
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            await ws.send("ping")
                            continue
                        if msg == "pong":
                            continue
                        j = json.loads(msg)
                        if "event" in j:
                            if j["event"] == "error":
                                log.error("websocket error: %s", j)
                            continue
                        for r in j.get("data", []):
                            self.on_trade(r["instId"], float(r["px"]), float(r["sz"]), int(r["ts"]))
                        self.db.commit()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("execlab websocket dropped (%s); reconnect in %.0fs", e, backoff)
                for o in self.open.values():
                    o["note"] = "websocket gap: some trades unseen"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
