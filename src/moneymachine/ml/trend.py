"""v5: time-series trend following, pre-registered in research/2026-10-15_刷钱机v5_趋势跟踪研究计划.md.

Rule (all trials): at 00:00 UTC, per coin
  s = mean over L ∈ {20, 60, 120} of sign(ln P_t − ln P_{t−L})        ∈ {−1, −⅓, ⅓, 1}
  w = s · min(1, 40% / σ₆₀) / n      (σ₆₀ = annualised 60-day vol, n = coins in the universe)
  long/flat variants use max(w, 0); trade only when the target differs from the drifted
  position by > 25% (relative) or the sign changes; gross ≤ 1, no leverage.
Costs: OKX perp taker 0.05% + half-spread curve + 2bp slippage; funding: longs pay, shorts receive.

T1 (BTC+ETH): Binance spot daily closes (2017-08+) and Binance perp funding (BTC 2019-09+, ETH 2019-11+).
T2 (top 10): OKX perp daily closes, 30-day median volume ranking among perps listed ≥ 180 days;
funding from OKX where present, else Binance, else 0 (counted).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from moneymachine.ml import data as D
from moneymachine.ml.features import _cum_at, _wide
from moneymachine.ml.portfolio import CostModel
from moneymachine.ml.stats import summary
from moneymachine.ml.validation import HOLDOUT_START

log = logging.getLogger(__name__)
DAY = 86_400_000
ROOT = D.ROOT / "trend"
END = HOLDOUT_START - 3 * DAY            # 2026-03-29 exclusive, as in v3/v4
T1_START = 1_575_158_400_000             # 2019-12-01
T2_START = 1_598_918_400_000             # 2020-09-01
SPOT_ONLY = (1_512_086_400_000, T1_START)  # 2017-12-01 → 2019-11-30: long/flat only, reported apart
LOOKBACKS = (20, 60, 120)
VOL_TARGET = 0.40
BAND = 0.25
FEE, SLIP = 0.0005, 0.0002


# ── data ─────────────────────────────────────────────────────────────────
def _get(c: httpx.Client, url: str, params: dict, pause: float = 1.0):
    for attempt in range(6):
        try:
            r = c.get(url, params=params)
            r.raise_for_status()
            time.sleep(pause)
            return r.json()
        except httpx.HTTPError as e:
            if attempt == 5:
                raise
            waf = isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (403, 418, 429)
            log.warning("%s failed (%s), retrying", url.split("/")[-1], type(e).__name__)
            time.sleep(60 if waf else 2 ** attempt * 2)   # Binance's WAF answers 403 when we are too fast


def sync_binance_spot(symbols=("BTCUSDT", "ETHUSDT")) -> None:
    c = httpx.Client(timeout=30)
    out = ROOT / "binance_spot_1d"
    out.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        rows, start = [], 1_500_000_000_000
        while True:
            k = _get(c, "https://api.binance.com/api/v3/klines",
                     {"symbol": sym, "interval": "1d", "startTime": start, "limit": 1000})
            rows += [(int(x[0]), float(x[4]), float(x[7])) for x in k if int(x[6]) < time.time() * 1000]
            if len(k) < 1000:
                break
            start = int(k[-1][0]) + DAY
        pd.DataFrame(rows, columns=["time", "close", "vol_quote"]).drop_duplicates("time").to_parquet(out / f"{sym}.parquet")
        log.info("binance spot %s: %d days", sym, len(rows))


def sync_binance_funding(symbols) -> None:
    c = httpx.Client(timeout=30)
    out = ROOT / "binance_funding"
    out.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        if (out / f"{sym}.parquet").exists():
            continue
        rows, start = [], 1_567_296_000_000   # 2019-09-01
        while True:
            j = _get(c, "https://fapi.binance.com/fapi/v1/fundingRate", {"symbol": sym, "startTime": start, "limit": 1000})
            if not isinstance(j, list) or not j:
                break
            rows += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in j]
            if len(j) < 1000:
                break
            start = int(j[-1]["fundingTime"]) + 1
        if rows:
            pd.DataFrame(rows, columns=["time", "rate"]).drop_duplicates("time").to_parquet(out / f"{sym}.parquet")
        log.info("binance funding %s: %d settlements", sym, len(rows))


# ── inputs ───────────────────────────────────────────────────────────────
def _binance_frame(kind: str, syms: list[str], col: str, shift: int = 0) -> pd.DataFrame:
    cols = {}
    for s in syms:
        p = ROOT / kind / f"{s}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            cols[s] = d.set_index(d["time"].astype("int64") + shift)[col]
    return pd.DataFrame(cols)


def _daily_funding(settle: pd.DataFrame, days: pd.Index) -> pd.DataFrame:
    """Funding rate sum settled in (t, t+1] (NaN before a coin's first settlement)."""
    F = _cum_at(settle, days)
    first = settle.apply(lambda s: s.first_valid_index())
    has = pd.DataFrame({c: days >= (first[c] if pd.notna(first[c]) else np.inf) for c in settle.columns}, index=days)
    return (F.shift(-1) - F).where(has)


def inputs_t1() -> dict:
    syms = ["BTCUSDT", "ETHUSDT"]
    px = _binance_frame("binance_spot_1d", syms, "close", DAY)          # close of the day ending at t
    vq = _binance_frame("binance_spot_1d", syms, "vol_quote", DAY)
    days = pd.Index(np.arange(px.index.min(), END + DAY, DAY), dtype="int64")
    px, vq = px.reindex(days), vq.reindex(days)
    fund = _daily_funding(_binance_frame("binance_funding", syms, "rate"), days)
    return {"px": px, "adv": vq.rolling(30, min_periods=20).median(), "fund": fund,
            "universe": px.notna(), "n": 2}


def _merge_funding(okx: pd.Series, bn: pd.Series | None) -> pd.Series:
    """OKX settlements; Binance ones where OKX has none within 12 hours (e.g. 2021-10 → 2022-01)."""
    if bn is None or bn.empty:
        return okx
    ot = np.sort(okx.index.to_numpy()) if len(okx) else np.array([], dtype=np.int64)
    pos = np.searchsorted(ot, bn.index.to_numpy())
    near = np.full(len(bn), np.inf)
    for off in (-1, 0):
        j = np.clip(pos + off, 0, max(len(ot) - 1, 0))
        if len(ot):
            near = np.minimum(near, np.abs(ot[j] - bn.index.to_numpy()))
    return pd.concat([okx, bn[near > 12 * 3_600_000]]).sort_index()


def inputs_t2(top: int = 10, min_age_days: int = 180) -> dict:
    ins = D.load_instruments()
    insts = [i for i in ins["inst"] if (D.ROOT / "bars_1D" / f"{i}.parquet").exists()]
    px = _wide("bars_1D", insts, "close", D.ROOT, DAY)
    vq = _wide("bars_1D", insts, "vol_quote", D.ROOT, DAY)
    days = pd.Index(np.arange(T2_START - 200 * DAY, END + DAY, DAY), dtype="int64")
    px, vq = px.reindex(index=days, columns=insts), vq.reindex(index=days, columns=insts)
    adv = vq.rolling(30, min_periods=20).median()
    first = px.apply(lambda s: s.first_valid_index())
    age_ok = pd.DataFrame({c: days >= (first[c] + min_age_days * DAY if pd.notna(first[c]) else np.inf)
                           for c in insts}, index=days)
    ranked = adv.where(age_ok & px.notna()).rank(axis=1, ascending=False, method="first")
    uni = ranked <= top
    used = list(uni.columns[uni.loc[T2_START:END].any()])
    okx = _wide("funding", used, "rate", D.ROOT).reindex(columns=used)
    bmap = binance_symbols(used)
    bn = _binance_frame("binance_funding", sorted(set(bmap.values())), "rate")
    merged = {}
    for u in used:
        o = okx[u].dropna() if u in okx else pd.Series(dtype=float)
        b = bn[bmap[u]].dropna() if bmap.get(u) in bn else None
        merged[u] = _merge_funding(o, b)
    settle = pd.DataFrame(merged)
    return {"px": px[used], "adv": adv[used], "fund": _daily_funding(settle, days), "universe": uni[used], "n": top}


def binance_symbols(insts: list[str]) -> dict[str, str]:
    """OKX perp → Binance USDT perp symbol (handles 1000PEPEUSDT-style multipliers); cached."""
    import json

    p = ROOT / "binance_symbols.json"
    cache = json.loads(p.read_text()) if p.exists() else {}
    if any(i not in cache for i in insts):
        from moneymachine.ml.xex import symbol_maps

        bn = symbol_maps()["binance"]
        for i in insts:
            m = bn.get(i.split("-")[0])
            cache[i] = m[0] if m else None
        ROOT.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, indent=0, sort_keys=True))
    return {i: cache[i] for i in insts if cache.get(i)}


# ── backtest ─────────────────────────────────────────────────────────────
def signal(px: pd.DataFrame) -> pd.DataFrame:
    lp = np.log(px)
    s = sum(np.sign(lp - lp.shift(L)) for L in LOOKBACKS) / len(LOOKBACKS)
    return s.where(lp.shift(max(LOOKBACKS)).notna())


def target_weights(px: pd.DataFrame, n: int, long_only: bool, universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Target weight per day and coin, before the trading band (shared by backtest and live)."""
    lr = np.log(px)
    vol = (lr - lr.shift(1)).rolling(60, min_periods=40).std() * np.sqrt(365)
    tgt = signal(px) * np.minimum(1.0, VOL_TARGET / vol) / n
    if universe is not None:
        tgt = tgt.where(universe)
    tgt = tgt.fillna(0.0)
    return tgt.clip(lower=0.0) if long_only else tgt


def apply_band(target: pd.Series, current: pd.Series) -> pd.Series:
    """Trade a coin only if its sign changes or it is > BAND (relative) away from the target."""
    current = current.reindex(target.index).fillna(0.0)
    flip = np.sign(target) != np.sign(current)
    big = (target - current).abs() > BAND * np.maximum(target.abs(), current.abs())
    return current.where(~(flip | big), target)


def backtest(inp: dict, *, long_only: bool, start: int, end: int = END, costs: CostModel | None = None,
             funding: bool = True) -> dict:
    costs = costs or CostModel()
    px, uni, n = inp["px"], inp["universe"], inp["n"]
    days = px.index[(px.index >= start) & (px.index < end)]
    tgt = target_weights(px, n, long_only, uni)
    r_fwd = (px.shift(-1) / px - 1).fillna(0.0)
    fund = inp["fund"].reindex(index=px.index, columns=px.columns).fillna(0.0) if funding else px * 0.0
    no_fund = inp["fund"].reindex(index=px.index, columns=px.columns).isna()
    hs = np.clip(costs.hs_a * (1e8 / inp["adv"].clip(lower=1.0)) ** costs.hs_b, costs.hs_min, costs.hs_max)
    w = pd.Series(0.0, index=px.columns)
    rows, missing_fund = [], 0
    for t in days:
        new = apply_band(tgt.loc[t], w)
        trade = (new - w).abs()
        cost = float((trade * (FEE + hs.loc[t].fillna(costs.hs_max) + SLIP)).sum()) * costs.multiplier
        r, f = r_fwd.loc[t], fund.loc[t]
        missing_fund += int(((new != 0) & no_fund.loc[t]).sum()) if funding else 0
        gross = float((new * r).sum())
        fpay = float(-(new * f).sum())
        net = gross + fpay - cost
        rows.append((int(t), gross, fpay, cost, net, float(new.abs().sum()), float(trade.sum())))
        w = new * (1 + r) / (1 + net) if 1 + net > 0 else new * 0
    daily = pd.DataFrame(rows, columns=["t", "gross", "funding", "cost", "net", "gross_exp", "turnover"]).set_index("t")
    return {"daily": daily, "missing_funding_days": missing_fund}


def metrics(res: dict, bench_px: pd.Series) -> dict:
    d = res["daily"]
    m = summary(d["net"].to_numpy())
    yr = d["net"].groupby(pd.to_datetime(d.index, unit="ms").year).sum()
    b = bench_px.reindex(d.index)
    bdd = float((1 - b / b.cummax()).max())
    m.update({"ann_cost": float(d["cost"].mean() * 365), "ann_funding": float(d["funding"].mean() * 365),
              "avg_gross": float(d["gross_exp"].mean()), "turnover_day": float(d["turnover"].mean()),
              "trades_per_year": float((d["turnover"] > 0).sum() / (len(d) / 365)),
              "btc_hold_max_dd": bdd, "by_year": {int(k): float(v) for k, v in yr.items()},
              "missing_funding_days": res["missing_funding_days"], "start": int(d.index.min()), "end": int(d.index.max())})
    return m


# ── trials + gate ────────────────────────────────────────────────────────
FAMILY = "dev-v5-trend"
TRIALS = {"T1a": ("t1", False), "T1b": ("t1", True), "T2a": ("t2", False), "T2b": ("t2", True)}


def _setup(universe: str):
    if universe == "t1":
        inp = inputs_t1()
        return inp, T1_START, inp["px"]["BTCUSDT"]
    inp = inputs_t2()
    return inp, T2_START, inp["px"]["BTC-USDT-SWAP"]


def run_trials(names: list[str] | None = None) -> pd.DataFrame:
    from dataclasses import asdict, replace

    from moneymachine.ml.registry import Registry

    reg = Registry()
    rows, cache = [], {}
    for name in names or list(TRIALS):
        universe, long_only = TRIALS[name]
        if universe not in cache:
            cache[universe] = _setup(universe)
        inp, start, bench = cache[universe]
        res = backtest(inp, long_only=long_only, start=start)
        m = metrics(res, bench)
        m["sharpe_cost2x"] = metrics(backtest(inp, long_only=long_only, start=start,
                                              costs=replace(CostModel(), multiplier=2.0)), bench)["sharpe"]
        full = f"{name}:{'btc_eth' if universe == 't1' else 'top10'}_{'long_flat' if long_only else 'long_short'}"
        cfg = {"name": full, "strategy": "trend", "universe": universe, "long_only": long_only, "lookbacks": LOOKBACKS,
               "vol_target": VOL_TARGET, "band": BAND, "costs": asdict(CostModel()), "fee": FEE, "slippage": SLIP}
        reg.record(FAMILY, full, cfg, m, res["daily"]["net"])
        log.info("[%s] %s: Sharpe %.2f | ann %.1f%% | maxDD %.1f%% (BTC hold %.1f%%) | cost×2 %.2f",
                 FAMILY, full, m["sharpe"], m["ann_return"] * 100, m["max_dd"] * 100, m["btc_hold_max_dd"] * 100,
                 m["sharpe_cost2x"])
        rows.append({"name": full, **{k: m[k] for k in ("sharpe", "ann_return", "ann_vol", "max_dd", "btc_hold_max_dd",
                                                        "ann_cost", "ann_funding", "avg_gross", "trades_per_year",
                                                        "sharpe_cost2x", "missing_funding_days")}, **m["by_year"]})
    return pd.DataFrame(rows)


def gate() -> str:
    from moneymachine.ml.registry import Registry
    from moneymachine.ml.stats import deflated_sharpe

    reg = Registry()
    tr = reg.trials(FAMILY)
    v2 = reg.trials("dev-v1-kronos")
    v2 = reg.returns_matrix([v2[v2["name"] == "lgbm+kronos:clean:k3"].iloc[0]["hash"]]).iloc[:, 0]
    fam, allsr = reg.per_period_sharpes(FAMILY), reg.per_period_sharpes(None)
    lines = [f"# 刷钱机 v5 趋势跟踪结果（{len(tr)} 次试验；DSR 按本组校正，另列全部 {len(allsr)} 次作参考）", ""]
    for _, r in tr.sort_values("name").iterrows():
        net = reg.returns_matrix([r["hash"]]).iloc[:, 0]
        dsr_f = deflated_sharpe(net.to_numpy(), fam)["dsr"]
        dsr_a = deflated_sharpe(net.to_numpy(), allsr)["dsr"]
        both = pd.concat([net, v2], axis=1, join="inner").dropna()
        corr = float(both.corr().iloc[0, 1]) if len(both) > 30 else float("nan")
        combo = (both.iloc[:, 0] / both.iloc[:, 0].std() + both.iloc[:, 1] / both.iloc[:, 1].std()) / 2
        sr = lambda x: float(x.mean() / x.std() * np.sqrt(365))  # noqa: E731
        years = {int(k): v for k, v in json_years(r).items()}
        full = [y for y in years if (2020 if r["name"].startswith("T1") else 2021) <= y <= 2025]
        pos = sum(years[y] > 0 for y in full)
        checks = {
            "净 Sharpe ≥ 0.7": r["sharpe"] >= 0.7,
            "DSR ≥ 0.95（本组 4 次）": dsr_f >= 0.95,
            "最大回撤 ≤ 30% 且 ≤ BTC 持有的一半": r["max_dd"] <= 0.30 and r["max_dd"] <= 0.5 * r["btc_hold_max_dd"],
            "和 v2 相关性 ≤ 0.3": corr <= 0.3,
            "成本翻倍后 Sharpe ≥ 0.5": r["sharpe_cost2x"] >= 0.5,
            "≥ 2/3 完整年份盈利": pos >= np.ceil(len(full) * 2 / 3),
        }
        lines += [f"## {r['name']}", "",
                  f"- 净 Sharpe {r['sharpe']:.2f}，年化 {r['ann_return']:+.1%}，波动 {r['ann_vol']:.1%}，最大回撤 {r['max_dd']:.1%}"
                  f"（BTC 持有 {r['btc_hold_max_dd']:.1%}），每年交易约 {r['trades_per_year']:.0f} 天，"
                  f"年化成本 {r['ann_cost']:.2%}，年化资金费 {r['ann_funding']:+.2%}",
                  f"- DSR：本组 {dsr_f:.3f}；全部 {len(allsr)} 次参考 {dsr_a:.3f}",
                  f"- 和 v2 的相关性 {corr:+.2f}；v2 单独 Sharpe {sr(both.iloc[:, 1]):.2f} → v2 + 趋势各半风险 {sr(combo):.2f}"
                  f"（重叠 {len(both)} 天）",
                  "- 各年：" + "，".join(f"{y} {v:+.1%}" for y, v in sorted(years.items())),
                  "", *[f"- [{'x' if v else ' '}] {k}" for k, v in checks.items()], "",
                  f"**{'通过' if all(checks.values()) else '未通过'}**", ""]
    return "\n".join(lines)


def json_years(row) -> dict:
    import json

    m = json.loads(row["metrics"]) if "metrics" in row and isinstance(row["metrics"], str) else None
    if m is not None:
        return m["by_year"]
    return {int(c.split(".")[-1]): row[c] for c in row.index if str(c).startswith("by_year.") and pd.notna(row[c])}


# ── live (paper account T1b_trend) ───────────────────────────────────────
def live_targets(insts: list[str], t: int, current: dict[str, float], *, long_only: bool = True,
                 sync: bool = True) -> tuple[dict[str, float] | None, dict]:
    """Today's targets from OKX perp daily closes (the traded instruments; the backtest on OKX
    data matches the Binance-based one, see the v5 report). Same functions as the backtest."""
    if sync:
        from moneymachine.market.okx_rest import OkxRest

        rest = OkxRest()
        for i in insts:
            D.sync_candles(rest, i, "1Dutc", "bars_1D")
    px = _wide("bars_1D", insts, "close", D.ROOT, DAY).sort_index()
    if t not in px.index or px.loc[t].isna().any():
        return None, {"reason": f"daily close for decision time {t} not available yet"}
    hist = px.loc[:t]
    tgt = target_weights(hist, len(insts), long_only).loc[t]
    new = apply_band(tgt, pd.Series(current, dtype=float))
    lp = np.log(hist)
    info = {"signal": {i: float(signal(hist)[i].iloc[-1]) for i in insts},
            "vol60": {i: float((lp[i] - lp[i].shift(1)).rolling(60, min_periods=40).std().iloc[-1] * np.sqrt(365))
                      for i in insts},
            "target": {i: float(tgt[i]) for i in insts}, "gross": float(new.abs().sum()),
            "turnover": float((new - pd.Series(current, dtype=float).reindex(new.index).fillna(0.0)).abs().sum())}
    return {i: float(new[i]) for i in insts}, info
