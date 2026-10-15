"""Per-strategy performance report + go/no-go checklist, from the ledger DB."""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from moneymachine.config import GoNoGo
from moneymachine.timeutil import fmt_ms

PLACEBO_KINDS = ("random_entry", "ml_placebo", "ml_placebo_smoothed", "ml_placebo_concentrated")
BASELINE_KINDS = ("buy_and_hold",) + PLACEBO_KINDS
PORTFOLIO_KINDS = ("ml_portfolio", "ml_placebo", "ml_smoothed", "ml_placebo_smoothed", "ml_concentrated",
                   "ml_placebo_concentrated", "trend_tsmom")


def list_runs(db_path: str | Path) -> pd.DataFrame:
    with sqlite3.connect(str(db_path)) as db:
        return pd.read_sql("SELECT run_id, name, mode, created_at, status, note FROM runs ORDER BY created_at", db)


def _max_dd(eq: np.ndarray) -> float:
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(np.max((peak - eq) / np.where(peak > 0, peak, 1)))


def _strategies(config_toml: str) -> list[dict]:
    import tomllib

    try:
        return tomllib.loads(config_toml or "").get("strategy", [])
    except tomllib.TOMLDecodeError:
        return []


def strategy_kinds(config_toml: str) -> dict[str, str]:
    return {s["id"]: s["kind"] for s in _strategies(config_toml)}


def strategy_controls(config_toml: str) -> dict[str, str]:
    """id → the placebo arm it is compared with (``params.control``); unset → every placebo."""
    return {s["id"]: s["params"]["control"] for s in _strategies(config_toml) if s.get("params", {}).get("control")}


def compute(db_path: str | Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    with sqlite3.connect(str(db_path)) as db:
        run = pd.read_sql("SELECT * FROM runs WHERE run_id=?", db, params=(run_id,))
        if run.empty:
            raise ValueError(f"run {run_id} not found in {db_path}")
        eq = pd.read_sql("SELECT strategy, ts, equity, gross_exposure FROM equity WHERE run_id=? ORDER BY ts",
                         db, params=(run_id,))
        tr = pd.read_sql("SELECT * FROM trades WHERE run_id=?", db, params=(run_id,))
        fu = pd.read_sql("SELECT strategy, SUM(amount) amt, SUM(assumed) n_assumed, COUNT(*) n FROM funding "
                         "WHERE run_id=? GROUP BY strategy", db, params=(run_id,))
        fees = pd.read_sql("SELECT strategy, SUM(fee) fee, COUNT(*) n_fills FROM fills WHERE run_id=? GROUP BY strategy",
                           db, params=(run_id,))
        llm = pd.read_sql("SELECT strategy, COUNT(*) calls, SUM(ok) ok, SUM(cost_usd) cost, "
                          "SUM(prompt_tokens+completion_tokens) tokens, AVG(latency_ms) lat "
                          "FROM llm_calls WHERE run_id=? GROUP BY strategy", db, params=(run_id,))
        liq = pd.read_sql("SELECT strategy, COUNT(*) n FROM events WHERE run_id=? AND kind='liquidation' "
                          "GROUP BY strategy", db, params=(run_id,))
    meta = run.iloc[0].to_dict()
    kinds = strategy_kinds(meta.get("config_toml", ""))
    controls = strategy_controls(meta.get("config_toml", ""))
    rows = []
    daily_cols = {}
    for strat, g in eq.groupby("strategy", sort=False):
        g = g.sort_values("ts")
        e = g["equity"].to_numpy(dtype=float)
        start_eq, end_eq = e[0], e[-1]
        t0, t1 = int(g["ts"].iloc[0]), int(g["ts"].iloc[-1])
        days = max((t1 - t0) / 86_400_000, 1e-9)
        daily = g.assign(day=pd.to_datetime(g["ts"], unit="ms", utc=True).dt.floor("D")).groupby("day")["equity"].last()
        daily_cols[strat] = daily
        dr = daily.pct_change().dropna()
        sharpe = float(dr.mean() / dr.std() * math.sqrt(365)) if len(dr) > 2 and dr.std() > 0 else float("nan")
        total = end_eq / start_eq - 1
        cagr = (end_eq / start_eq) ** (365 / days) - 1 if end_eq > 0 and days >= 30 else float("nan")
        t = tr[tr["strategy"] == strat]
        wins, losses = t[t["net_pnl"] > 0]["net_pnl"].sum(), t[t["net_pnl"] <= 0]["net_pnl"].sum()
        r = t["r_multiple"].dropna()
        tstat = float(r.mean() / r.std() * math.sqrt(len(r))) if len(r) > 2 and r.std() > 0 else float("nan")
        f = fu[fu["strategy"] == strat]
        fe = fees[fees["strategy"] == strat]
        lm = llm[llm["strategy"] == strat]
        lq = liq[liq["strategy"] == strat]
        llm_cost = float(lm["cost"].iloc[0] or 0) if len(lm) else 0.0
        rows.append({
            "strategy": strat,
            "kind": kinds.get(strat, ""),
            "control": controls.get(strat, ""),
            "start_equity": start_eq,
            "days": round(days, 1),
            "end_equity": end_eq,
            "total_return": total,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_dd": _max_dd(e),
            "time_in_mkt": float((g["gross_exposure"] > 0).mean()),
            "trades": len(t),
            "fills": int(fe["n_fills"].iloc[0]) if len(fe) else 0,
            "win_rate": float((t["net_pnl"] > 0).mean()) if len(t) else float("nan"),
            "avg_R": float(r.mean()) if len(r) else float("nan"),
            "t_stat_R": tstat,
            "profit_factor": float(wins / -losses) if losses < 0 else float("nan"),
            "fees": float(fe["fee"].iloc[0]) if len(fe) else 0.0,
            "funding": float(f["amt"].iloc[0]) if len(f) else 0.0,
            "funding_assumed": int(f["n_assumed"].iloc[0] or 0) if len(f) else 0,
            "llm_calls": int(lm["calls"].iloc[0]) if len(lm) else 0,
            "llm_fail_rate": float(1 - lm["ok"].iloc[0] / lm["calls"].iloc[0]) if len(lm) and lm["calls"].iloc[0] else float("nan"),
            "llm_cost_usd": llm_cost,
            "net_after_llm": end_eq - llm_cost,
            "liquidations": int(lq["n"].iloc[0]) if len(lq) else 0,
            "first_ts": t0,
        })
    summary = pd.DataFrame(rows)
    daily_df = pd.DataFrame(daily_cols)
    return summary, daily_df, meta


def window_return(daily: pd.DataFrame, strategy: str, start) -> float:
    """Return of ``strategy`` from its equity at ``start`` (first day ≥ start) to the last day."""
    e = daily[strategy].dropna()
    e = e[e.index >= start]
    return float(e.iloc[-1] / e.iloc[0] - 1) if len(e) >= 2 else float("nan")


def gonogo(summary: pd.DataFrame, benchmark: str, crit: GoNoGo, mode: str,
           daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pre-committed release checklist. Placebo comparisons use the window both
    strategies were running (an arm added later is compared from its first day)."""
    if summary.empty:
        return pd.DataFrame()
    bench = summary[summary["strategy"] == benchmark]
    bench_net = float(bench["net_after_llm"].iloc[0]) if len(bench) else float("nan")
    placebo = summary[summary["kind"].isin(PLACEBO_KINDS)]
    out = []
    for _, r in summary.iterrows():
        if r["kind"] in BASELINE_KINDS:
            continue
        n_trades = r["fills"] if r["kind"] in PORTFOLIO_KINDS else r["trades"]
        checks = {
            "forward_days": bool(mode == "paper" and r["days"] >= crit.min_forward_days),
            "trades": bool(n_trades >= crit.min_trades),
        }
        if crit.beat_benchmark:
            checks["beats_benchmark"] = bool((not math.isnan(bench_net)) and r["net_after_llm"] > bench_net)
        if crit.require_positive_net:
            checks["net_positive"] = bool(r["net_after_llm"] > r["end_equity"] / (1 + r["total_return"]))
        beats = []
        mine_placebos = placebo[placebo["strategy"] == r["control"]] if r.get("control") else placebo
        if r.get("control") == "none":
            mine_placebos = placebo.iloc[0:0]
        for _, p in mine_placebos.iterrows():
            if daily is not None and r["strategy"] in daily and p["strategy"] in daily:
                common = max(daily[r["strategy"]].first_valid_index(), daily[p["strategy"]].first_valid_index())
                mine, theirs = window_return(daily, r["strategy"], common), window_return(daily, p["strategy"], common)
                beats.append(bool(mine > theirs) if not (math.isnan(mine) or math.isnan(theirs)) else False)
            else:
                beats.append(bool(r["net_after_llm"] > p["net_after_llm"]))
        if r.get("control") != "none":   # control = "none": no placebo arm (e.g. the trend account)
            checks["beats_placebo"] = bool(beats) and all(beats)
        checks["max_dd"] = bool(r["max_dd"] <= crit.max_drawdown)
        row = {"strategy": r["strategy"], **checks, "PASS": all(checks.values())}
        if not crit.beat_benchmark and not math.isnan(bench_net):
            row["vs_benchmark(参考)"] = "领先" if r["net_after_llm"] > bench_net else "落后"
        out.append(row)
    return pd.DataFrame(out)


def criteria_text(crit: GoNoGo, benchmark: str) -> str:
    parts = [f"前向 ≥ {crit.min_forward_days} 天（只认 paper 模式）",
             f"成交 ≥ {crit.min_trades} 笔（组合型策略按成交笔数）"]
    if crit.beat_benchmark:
        parts.append(f"扣除全部成本后跑赢 `{benchmark}`")
    if crit.require_positive_net:
        parts.append("扣除全部成本后净收益为正")
    parts.append("跑赢自己的随机对照组（同样构建方法；只比较双方都在运行的同一段时间）")
    parts.append(f"最大回撤 ≤ {crit.max_drawdown:.0%}")
    text = "、".join(parts)
    if not crit.beat_benchmark:
        text += f"；`{benchmark}` 只作参考"
    return text


def _fmt(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary.copy()
    pct = ["total_return", "cagr", "max_dd", "time_in_mkt", "win_rate", "llm_fail_rate"]
    for c in pct:
        s[c] = s[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}" if c in ("total_return", "cagr") else f"{v:.1%}")
    for c in ["start_equity", "end_equity", "fees", "funding", "net_after_llm"]:
        s[c] = s[c].map(lambda v: f"{v:,.0f}")
    for c in ["sharpe", "avg_R", "t_stat_R", "profit_factor"]:
        s[c] = s[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    s["llm_cost_usd"] = s["llm_cost_usd"].map(lambda v: f"{v:.2f}")
    return s


def render(db_path: str | Path, run_id: str, benchmark: str, crit: GoNoGo, out_dir: Path | None = None) -> str:
    summary, daily, meta = compute(db_path, run_id)
    mode = meta.get("mode", "")
    lines = [
        f"# 报告：{meta.get('name')}  ({run_id})",
        "",
        f"- 模式：{mode}　状态：{meta.get('status')}　{meta.get('note') or ''}",
        f"- 生成于：{fmt_ms(int(pd.Timestamp.now(tz='UTC').timestamp() * 1000))} UTC",
        f"- 基准：`{benchmark}`；所有策略风控规则相同、成本相同；起始资金见 start_equity（小资金组 1,000 USDT，其余 100,000）",
        "",
        "## 各策略表现（已扣手续费、滑点、资金费；net_after_llm 另扣 LLM 官方价费用）",
        "",
    ]
    if summary.empty:
        lines.append("（还没有权益记录）")
    else:
        cols = ["strategy", "kind", "days", "end_equity", "total_return", "cagr", "sharpe", "max_dd",
                "time_in_mkt", "trades", "fills", "win_rate", "avg_R", "t_stat_R", "profit_factor", "fees",
                "funding", "llm_calls", "llm_fail_rate", "llm_cost_usd", "net_after_llm", "liquidations"]
        lines.append(_fmt(summary)[cols].to_markdown(index=False))
        assumed = summary[summary["funding_assumed"] > 0]
        if len(assumed):
            lines += ["", f"> 注：有 {int(assumed['funding_assumed'].sum())} 次资金费结算没有真实数据，按默认费率估算。"]
        g = gonogo(summary, benchmark, crit, mode, daily)
        lines += [
            "",
            "## 上真钱放行检查（全部为 True 才算 PASS）",
            "",
            "标准：" + criteria_text(crit, benchmark),
            "",
            g.to_markdown(index=False) if not g.empty else "（没有非基线策略）",
            "",
            "t_stat_R：每笔交易 R 倍数均值的 t 统计量。|t| < 2 说明结果和运气区分不开。",
        ]
    text = "\n".join(lines) + "\n"
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{run_id}_report.md").write_text(text, encoding="utf-8")
        if not daily.empty:
            daily.to_csv(out_dir / f"{run_id}_daily_equity.csv")
    return text
