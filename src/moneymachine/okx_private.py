"""OKX private REST — READ-ONLY by construction (stage 1 of the real-trading work).

- Only GET requests, and only to an explicit allow-list of query endpoints: nothing here can
  place, amend or cancel an order, transfer or withdraw.
- On connect it reads the key's own permission (account/config ``perm``) and refuses to run if
  the key can trade or withdraw (``require_read_only``).
- Credentials come from the environment (.env): MM_OKX_API_KEY, MM_OKX_SECRET_KEY,
  MM_OKX_PASSPHRASE. They are never logged or printed.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import os
from urllib.parse import urlencode

import httpx

BASE = "https://www.okx.com"
ALLOWED = (
    "/api/v5/account/config",
    "/api/v5/account/balance",
    "/api/v5/account/positions",
    "/api/v5/trade/orders-pending",
    "/api/v5/trade/fills-history",
    "/api/v5/asset/balances",
)


class OkxApiError(RuntimeError):
    pass


class ReadOnlyOkx:
    def __init__(self, key: str | None = None, secret: str | None = None, passphrase: str | None = None,
                 http: httpx.Client | None = None, require_read_only: bool = True):
        self._key = key if key is not None else os.environ.get("MM_OKX_API_KEY", "")
        self._secret = secret if secret is not None else os.environ.get("MM_OKX_SECRET_KEY", "")
        self._pass = passphrase if passphrase is not None else os.environ.get("MM_OKX_PASSPHRASE", "")
        missing = [n for n, v in (("MM_OKX_API_KEY", self._key), ("MM_OKX_SECRET_KEY", self._secret),
                                  ("MM_OKX_PASSPHRASE", self._pass)) if not v]
        if missing:
            raise OkxApiError(f"OKX credentials missing in .env: {', '.join(missing)}")
        self.http = http or httpx.Client(timeout=15)
        self.require_read_only = require_read_only
        self._checked = False

    def __repr__(self) -> str:  # never show credentials
        return "ReadOnlyOkx(<credentials hidden>)"

    @staticmethod
    def sign(secret: str, ts: str, method: str, path: str, body: str = "") -> str:
        mac = hmac.new(secret.encode(), f"{ts}{method}{path}{body}".encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def get(self, path: str, params: dict | None = None) -> list:
        if path not in ALLOWED:
            raise PermissionError(f"read-only client: {path} is not an allowed query endpoint")
        if self.require_read_only and not self._checked and path != "/api/v5/account/config":
            self.permissions()
        query = ("?" + urlencode(params)) if params else ""
        ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        headers = {"OK-ACCESS-KEY": self._key, "OK-ACCESS-SIGN": self.sign(self._secret, ts, "GET", path + query),
                   "OK-ACCESS-TIMESTAMP": ts, "OK-ACCESS-PASSPHRASE": self._pass}
        r = self.http.get(BASE + path + query, headers=headers)
        try:
            j = r.json()
        except ValueError:
            raise OkxApiError(f"HTTP {r.status_code} from OKX (not JSON)") from None
        if str(j.get("code")) != "0":
            raise OkxApiError(f"OKX error {j.get('code')}: {j.get('msg')}")
        return j.get("data", [])

    def permissions(self) -> dict:
        cfg = self.get("/api/v5/account/config")[0]
        perm = cfg.get("perm", "")
        if self.require_read_only and any(p in perm for p in ("trade", "withdraw")):
            raise OkxApiError(f"this key has permission '{perm}': stage 1 needs a read-only key")
        self._checked = True
        return cfg

    def snapshot(self) -> dict:
        """Account summary for reconciliation: equity, balances, positions, open orders."""
        cfg = self.permissions()
        bal = self.get("/api/v5/account/balance")[0]
        pos = self.get("/api/v5/account/positions")
        pend = self.get("/api/v5/trade/orders-pending")
        return {
            "label": cfg.get("label"), "perm": cfg.get("perm"), "ip_bound": bool(cfg.get("ip")),
            "account_level": cfg.get("acctLv"), "position_mode": cfg.get("posMode"),
            "total_equity_usd": float(bal.get("totalEq") or 0),
            "balances": {d["ccy"]: float(d.get("eq") or 0) for d in bal.get("details", []) if float(d.get("eq") or 0)},
            "positions": [{"inst": p["instId"], "pos": float(p.get("pos") or 0), "avg_px": float(p.get("avgPx") or 0),
                           "upl": float(p.get("upl") or 0)} for p in pos],
            "open_orders": len(pend),
        }
