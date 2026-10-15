"""Event data for v3 trial C2: listings, delistings and token unlocks.

Sources (all public, no key):
  - Binance announcements: catalog 48 "New Cryptocurrency Listing", 161 "Delisting"
  - OKX announcements: new listings (from 2023-03), delistings (from 2024-01)
  - DefiLlama's public unlock datasets (defillama-datasets.llama.fi/emissions/<slug>):
    daily cumulative unlocked tokens per allocation; schedules are published in advance

Stored under ``store/ml/events``. Coins are matched to announcement titles by ticker
(upper-case words equal to an OKX base currency, also after stripping a USDT/USDC suffix
or a 1000-style multiplier; tickers right after "/" are quote currencies and ignored; if the
title has "(TICKER)" groups, only those count).
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from moneymachine.ml import data as D

log = logging.getLogger(__name__)
ROOT = D.ROOT / "events"
DAY = 86_400_000
BINANCE = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
OKX = "https://www.okx.com/api/v5/support/announcements"
LLAMA = "https://defillama-datasets.llama.fi"
KINDS = {("binance", 48): "bn_list", ("binance", 161): "bn_delist",
         ("okx", "announcements-new-listings"): "okx_list", ("okx", "announcements-delistings"): "okx_delist"}


def _client() -> httpx.Client:
    return httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)


def _get(c: httpx.Client, url: str, params: dict | None = None, pause: float = 0.4):
    for attempt in range(5):
        try:
            r = c.get(url, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
            r.raise_for_status()
            time.sleep(pause)
            return r.json()
        except httpx.HTTPError as e:
            if attempt == 4:
                raise
            log.warning("%s failed (%s), retrying", url, e)
            time.sleep(2 ** attempt * 2)


def sync_announcements() -> pd.DataFrame:
    c = _client()
    rows = []
    for (src, cat), kind in KINDS.items():
        page = 1
        while True:
            if src == "binance":
                j = _get(c, BINANCE, {"type": 1, "catalogId": cat, "pageNo": page, "pageSize": 50})
                arts = j["data"]["catalogs"][0]["articles"]
                rows += [(kind, int(a["releaseDate"]), a["title"]) for a in arts]
                done = len(arts) < 50
            else:
                j = _get(c, OKX, {"annType": cat, "page": page})["data"][0]
                rows += [(kind, int(a["pTime"]), a["title"]) for a in j["details"]]
                done = page >= int(j["totalPage"])
            if done:
                break
            page += 1
        log.info("%s: %d pages", kind, page)
    df = pd.DataFrame(rows, columns=["kind", "time", "title"]).drop_duplicates().sort_values("time")
    ROOT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ROOT / "announcements.parquet", index=False)
    return df


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def sync_unlocks(bases: set[str]) -> dict[str, str]:
    """Download unlock schedules for DefiLlama protocols whose token is one of ``bases``.
    Matching: CoinGecko entry with that symbol whose id or name equals the protocol slug."""
    c = _client()
    ROOT.mkdir(parents=True, exist_ok=True)
    cg_path = ROOT / "coingecko_list.json"
    if not cg_path.exists():
        cg_path.write_text(json.dumps(_get(c, "https://api.coingecko.com/api/v3/coins/list")))
    by_key: dict[str, set[str]] = {}
    for e in json.loads(cg_path.read_text()):
        sym = e["symbol"].upper()
        if sym in bases:
            for k in (_norm(e["id"]), _norm(e["name"])):
                by_key.setdefault(k, set()).add(sym)
    slugs = _get(c, f"{LLAMA}/emissionsProtocolsList")
    matched = {}
    for slug in slugs:
        syms = by_key.get(_norm(slug), set())
        if len(syms) == 1:
            matched[slug] = syms.pop()
    out_dir = ROOT / "unlocks"
    out_dir.mkdir(exist_ok=True)
    for slug, sym in matched.items():
        j = _get(c, f"{LLAMA}/emissions/{slug}", pause=0.2)
        cats = j.get("categories") or {}
        if isinstance(cats, str):
            cats = json.loads(cats.replace("'", '"'))
        skip = set(cats.get("noncirculating", []))
        frames = [pd.DataFrame(a["data"]).assign(label=a["label"]) for a in j["documentedData"]["data"]
                  if a["label"] not in skip and a.get("data")]
        if not frames:
            continue
        d = pd.concat(frames).groupby("timestamp")["unlocked"].sum().reset_index()
        d["time"] = d["timestamp"].astype("int64") * 1000
        d[["time", "unlocked"]].to_parquet(out_dir / f"{sym}.parquet", index=False)
    (ROOT / "unlock_map.json").write_text(json.dumps(matched, indent=1, sort_keys=True))
    log.info("unlock schedules: %d protocols matched to OKX coins", len(matched))
    return matched


_TOKEN = re.compile(r"(?<![/A-Za-z0-9])([A-Z0-9]{2,15})(?![a-z])")
_PAREN = re.compile(r"\(([A-Z0-9]{1,15})\)")
_QUOTES = {"USDT", "USDC", "USD", "FDUSD", "TRY", "EUR", "BRL"}


def tickers(title: str, bases: set[str]) -> set[str]:
    """Coins named in a title. "Name (TICKER)" is the listing format: when a title has it,
    only the parenthesised tickers count (project names like "DeAgent AI" are not tickers)."""
    out = set()
    for tok in _PAREN.findall(title) or _TOKEN.findall(title):
        bare = re.sub(r"(USDT|USDC|USD)$", "", tok)
        for cand in (tok, bare, re.sub(r"^(1000000|10000|1000|1M)", "", bare)):
            if cand in bases and cand not in _QUOTES:
                out.add(cand)
                break
    return out


def features(panel, root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Event features known at each decision time t (days × insts):

    ev_{kind}_{3,7}   announcements of that kind naming the coin in the last 3 / 7 days
    ev_unlock_next7   tokens scheduled to unlock in (t, t+7d] / tokens unlocked at t
    ev_unlock_past7   tokens unlocked in (t−7d, t] / tokens unlocked at t−7d
    """
    days, insts = panel.days, panel.insts
    base_of = {i: i.split("-")[0] for i in insts}
    bases = set(base_of.values())
    ann = pd.read_parquet(root / "announcements.parquet")
    out = {}
    for kind in sorted(set(KINDS.values())):
        a = ann[ann["kind"] == kind]
        hits = [(t, b) for t, title in zip(a["time"], a["title"]) for b in tickers(title, bases)]
        ev = pd.DataFrame(hits, columns=["time", "base"])
        for k in (3, 7):
            m = pd.DataFrame(0.0, index=days, columns=insts)
            for inst, b in base_of.items():
                ts = np.sort(ev.loc[ev["base"] == b, "time"].to_numpy())
                if len(ts):
                    d = days.to_numpy()
                    m[inst] = np.searchsorted(ts, d, side="right") - np.searchsorted(ts, d - k * DAY, side="right")
            out[f"ev_{kind}_{k}"] = m
    nxt = pd.DataFrame(np.nan, index=days, columns=insts)
    past = nxt.copy()
    for inst, b in base_of.items():
        p = root / "unlocks" / f"{b}.parquet"
        if not p.exists():
            continue
        u = pd.read_parquet(p).set_index("time")["unlocked"].sort_index()
        u = u[~u.index.duplicated()]
        at = lambda x: pd.Series(np.interp(x, u.index.to_numpy(), u.to_numpy(), left=np.nan), index=days)  # noqa: E731
        now, ahead, before = at(days.to_numpy()), at(days.to_numpy() + 7 * DAY), at(days.to_numpy() - 7 * DAY)
        nxt[inst] = ((ahead - now) / now.where(now > 0)).to_numpy()
        past[inst] = ((now - before) / before.where(before > 0)).to_numpy()
    out["ev_unlock_next7"] = nxt
    out["ev_unlock_past7"] = past
    return out
