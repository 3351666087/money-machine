"""SQLite ledger: every order, fill, funding payment, trade, equity snapshot,
LLM call and event is recorded here. One DB file can hold several runs."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, name TEXT, mode TEXT, created_at INTEGER,
    config_toml TEXT, status TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    run_id TEXT, strategy TEXT, order_id INTEGER, symbol TEXT, side TEXT, type TEXT,
    role TEXT, qty REAL, price REAL, stop_loss REAL, take_profit REAL,
    reduce_only INTEGER, status TEXT, created_ts INTEGER, updated_ts INTEGER,
    tag TEXT, reason TEXT,
    PRIMARY KEY (run_id, strategy, order_id)
);
CREATE TABLE IF NOT EXISTS fills (
    run_id TEXT, strategy TEXT, order_id INTEGER, symbol TEXT, side TEXT, role TEXT,
    qty REAL, price REAL, fee REAL, liquidity TEXT, realized_pnl REAL, ts INTEGER
);
CREATE TABLE IF NOT EXISTS funding (
    run_id TEXT, strategy TEXT, symbol TEXT, ts INTEGER, rate REAL, assumed INTEGER,
    position_qty REAL, mark REAL, amount REAL
);
CREATE TABLE IF NOT EXISTS trades (
    run_id TEXT, strategy TEXT, symbol TEXT, side TEXT, entry_ts INTEGER, exit_ts INTEGER,
    qty REAL, entry_px REAL, exit_px REAL, gross_pnl REAL, fees REAL, funding REAL,
    net_pnl REAL, r_multiple REAL, exit_reason TEXT, entry_tag TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    run_id TEXT, strategy TEXT, ts INTEGER, balance REAL, equity REAL, upnl REAL,
    gross_exposure REAL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id TEXT, strategy TEXT, ts INTEGER, purpose TEXT, symbol TEXT,
    model_requested TEXT, model_returned TEXT, ok INTEGER, error TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER, latency_ms INTEGER,
    cost_usd REAL, request TEXT, response TEXT, parsed TEXT
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT, ts INTEGER, level TEXT, strategy TEXT, kind TEXT, message TEXT, data TEXT
);
CREATE TABLE IF NOT EXISTS state (
    run_id TEXT PRIMARY KEY, updated_at INTEGER, last_ts INTEGER, blob TEXT
);
CREATE INDEX IF NOT EXISTS ix_equity ON equity(run_id, strategy, ts);
CREATE INDEX IF NOT EXISTS ix_trades ON trades(run_id, strategy);
CREATE INDEX IF NOT EXISTS ix_fills ON fills(run_id, strategy);
"""


class Ledger:
    def __init__(self, path: str | Path, run_id: str, *, autocommit_every: int = 2000):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA)
        self.run_id = run_id
        self._pending = 0
        self._every = autocommit_every
        # LLM strategies run in worker threads in live mode.
        self._lock = threading.RLock()

    # ── run bookkeeping ──────────────────────────────────────────────────
    def start_run(self, name: str, mode: str, config_toml: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?)",
            (self.run_id, name, mode, int(time.time() * 1000), config_toml, "running", ""),
        )
        self.commit()

    def update_config(self, config_toml: str) -> None:
        """On resume: store the current config if it changed, keeping the previous one in an event."""
        row = self.db.execute("SELECT config_toml FROM runs WHERE run_id=?", (self.run_id,)).fetchone()
        if row is None or row[0] == config_toml:
            return
        self.event(int(time.time() * 1000), "info", None, "config_update", "config changed on resume",
                   {"previous_config_toml": row[0]})
        self.db.execute("UPDATE runs SET config_toml=? WHERE run_id=?", (config_toml, self.run_id))
        self.commit()

    def set_status(self, status: str, note: str = "") -> None:
        self.db.execute("UPDATE runs SET status=?, note=? WHERE run_id=?", (status, note, self.run_id))
        self.commit()

    # ── writers ──────────────────────────────────────────────────────────
    def _w(self, sql: str, row: tuple) -> None:
        with self._lock:
            self.db.execute(sql, row)
            self._pending += 1
            if self._pending >= self._every:
                self.commit()

    def order(self, o, ts: int, reason: str = "") -> None:
        self._w(
            "INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, o.strategy, o.id, o.symbol, o.side, o.type, o.role, o.qty, o.price,
             o.stop_loss, o.take_profit, int(o.reduce_only), o.status, o.created_ts, ts, o.tag, reason),
        )

    def fill(self, strategy, order_id, symbol, side, role, qty, price, fee, liquidity, realized, ts) -> None:
        self._w(
            "INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, strategy, order_id, symbol, side, role, qty, price, fee, liquidity, realized, ts),
        )

    def funding(self, strategy, symbol, ts, rate, assumed, qty, mark, amount) -> None:
        self._w(
            "INSERT INTO funding VALUES (?,?,?,?,?,?,?,?,?)",
            (self.run_id, strategy, symbol, ts, rate, int(assumed), qty, mark, amount),
        )

    def trade(self, strategy: str, rec: dict[str, Any]) -> None:
        self._w(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, strategy, rec["symbol"], rec["side"], rec["entry_ts"], rec["exit_ts"],
             rec["qty"], rec["entry_px"], rec["exit_px"], rec["gross_pnl"], rec["fees"],
             rec["funding"], rec["net_pnl"], rec["r_multiple"], rec["exit_reason"], rec["entry_tag"]),
        )

    def equity(self, strategy, ts, balance, equity, upnl, gross_exposure) -> None:
        self._w(
            "INSERT INTO equity VALUES (?,?,?,?,?,?,?)",
            (self.run_id, strategy, ts, balance, equity, upnl, gross_exposure),
        )

    def llm_call(self, rec: dict[str, Any]) -> None:
        self._w(
            "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, rec.get("strategy"), rec.get("ts"), rec.get("purpose"), rec.get("symbol"),
             rec.get("model_requested"), rec.get("model_returned"), int(bool(rec.get("ok"))),
             rec.get("error"), rec.get("prompt_tokens"), rec.get("completion_tokens"),
             rec.get("latency_ms"), rec.get("cost_usd"), rec.get("request"), rec.get("response"),
             json.dumps(rec.get("parsed"), ensure_ascii=False) if rec.get("parsed") is not None else None),
        )

    def event(self, ts: int, level: str, strategy: str | None, kind: str, message: str, data: Any = None) -> None:
        self._w(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
            (self.run_id, ts, level, strategy, kind, message,
             json.dumps(data, ensure_ascii=False, default=str) if data is not None else None),
        )

    # ── state snapshots (live resume) ────────────────────────────────────
    def save_state(self, last_ts: int, blob: dict) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO state VALUES (?,?,?,?)",
                (self.run_id, int(time.time() * 1000), last_ts, json.dumps(blob)),
            )
            self.commit()

    def load_state(self) -> dict | None:
        row = self.db.execute("SELECT blob FROM state WHERE run_id=?", (self.run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def commit(self) -> None:
        with self._lock:
            self.db.commit()
            self._pending = 0

    def close(self) -> None:
        self.commit()
        self.db.close()
