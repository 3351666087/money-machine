"""Server watchdog: run every few minutes (systemd timer); push alerts + a daily summary.

Checks (each yields an alert key, so the same problem is not re-sent every run):
  - systemd services active
  - market data fresh (last processed 1m bar per symbol)
  - today's portfolio decisions executed (after 00:30 UTC), no strategy errors
  - errors / liquidations in the ledger since the last run
  - service memory vs its MemoryMax, free disk
  - drawdown of each strategy from its peak
  - forward-collector files still growing

Channels (set in the server's .env; none set → journal only):
  MM_ALERT_FEISHU_WEBHOOK   Feishu custom bot webhook URL
  MM_ALERT_PUSHPLUS_TOKEN   PushPlus token (WeChat; pushplus.plus is unreachable from some overseas servers)
  MM_ALERT_WEBHOOK          any URL that accepts a plain-text POST (e.g. an ntfy topic)
  MM_ALERT_SMTP_USER        e-mail via SMTP over SSL (default smtp.gmail.com:465); with Gmail use an
  MM_ALERT_SMTP_PASSWORD    app password (myaccount.google.com/apppasswords), not the account password
  MM_ALERT_EMAIL_TO         recipient (default: MM_ALERT_SMTP_USER); MM_ALERT_SMTP_HOST overrides the host
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import smtplib
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)
DAY = 86_400_000
RESEND_AFTER_S = 6 * 3600
NAMES = {"ML_v2": "机器学习策略", "S0_hold_btc": "买入持有BTC", "P0_placebo": "随机对照组",
         "A1a_smoothed": "探索组·信号平滑", "P2_placebo_smoothed": "随机对照·平滑",
         "B1b_concentrated": "探索组·小资金集中", "P1_placebo_concentrated": "随机对照·小资金集中",
         "T1b_trend": "趋势跟踪 BTC+ETH"}
PORTFOLIO_IDS = ("ML_v2", "P0_placebo", "A1a_smoothed", "P2_placebo_smoothed", "B1b_concentrated",
                 "P1_placebo_concentrated", "T1b_trend")


@dataclass(frozen=True)
class Alert:
    key: str
    level: str  # warn | crit
    message: str


# ── probes (thin wrappers so tests can replace them) ─────────────────────
def _systemctl(*args: str) -> str:
    try:
        return subprocess.run(["systemctl", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def service_state(name: str) -> str:
    return _systemctl("is-active", name) or "unknown"


def service_memory(name: str) -> tuple[int, int]:
    """(memory the process really uses, MemoryMax). systemd's MemoryCurrent also counts the page cache
    of files the service wrote or read (e.g. 518 of the paper process's 936 MB on 2026-10-21), which the
    kernel reclaims under pressure, so it over-states the risk; the cgroup's ``anon`` memory does not."""
    out = _systemctl("show", name, "-p", "MemoryCurrent", "-p", "MemoryMax", "--value").split()
    try:
        cur, mx = int(out[0]), int(out[1])
    except (IndexError, ValueError):
        return 0, 0
    try:
        stat = Path(f"/sys/fs/cgroup/system.slice/{name}.service/memory.stat").read_text().split("\n")
        cur = next(int(line.split()[1]) for line in stat if line.startswith("anon "))
    except (OSError, StopIteration, ValueError, IndexError):
        pass
    return cur, mx


# ── checks ───────────────────────────────────────────────────────────────
def check(db_path: Path, *, now_ms: int, since_ms: int,
          services=("money-machine-paper", "money-machine-collect", "money-machine-execlab"),
          collect_dir: Path | None = None, portfolio_strategies=PORTFOLIO_IDS,
          lag_min: float = 10, mem_frac: float = 0.85, mem_clear: float = 0.75, disk_min_gb: float = 2.0,
          dd_warn: float = 0.10, probe=None, open_keys=()) -> list[Alert]:
    probe = probe or {"state": service_state, "memory": service_memory,
                      "disk_free": lambda p: shutil.disk_usage(p).free}
    alerts: list[Alert] = []
    for s in services:
        st = probe["state"](s)
        if st != "active":
            alerts.append(Alert(f"svc:{s}", "crit", f"服务 {s} 状态是 {st}"))
        cur, mx = probe["memory"](s)
        # hysteresis: raise above mem_frac, clear only below mem_clear (no flapping around the line)
        if mx and cur / mx > (mem_clear if f"mem:{s}" in open_keys else mem_frac):
            alerts.append(Alert(f"mem:{s}", "warn", f"{s} 实际内存 {cur / 2**20:.0f}MB（不含文件缓存），已到上限的 {cur / mx:.0%}"))
    free = probe["disk_free"](str(db_path.parent))
    if free < disk_min_gb * 2**30:
        alerts.append(Alert("disk", "warn", f"磁盘只剩 {free / 2**30:.1f}GB"))

    if not db_path.exists():
        return alerts + [Alert("ledger", "crit", f"找不到账本 {db_path}")]
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row = db.execute("SELECT run_id FROM runs WHERE mode='paper' ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        return alerts + [Alert("ledger", "crit", "账本里没有模拟盘记录")]
    run_id = row[0]
    st = db.execute("SELECT blob FROM state WHERE run_id=?", (run_id,)).fetchone()
    if st:
        last = json.loads(st[0])["broker"]["last_bar_ts"]
        if last:
            lag = (now_ms - max(last.values())) / 60_000
            stale = sum((now_ms - v) / 60_000 > lag_min for v in last.values())
            if lag > lag_min:
                alerts.append(Alert("data:stale", "crit", f"行情停止更新 {lag:.0f} 分钟"))
            elif stale > len(last) * 0.2:
                alerts.append(Alert("data:partial", "warn", f"{stale}/{len(last)} 个币的行情超过 {lag_min:.0f} 分钟没更新"))

    today = now_ms - now_ms % DAY
    if now_ms - today > 30 * 60_000:
        done = {s for (s,) in db.execute(
            "SELECT DISTINCT strategy FROM events WHERE run_id=? AND kind='ml_targets' AND ts>=?", (run_id, today))}
        known = {s for (s,) in db.execute("SELECT DISTINCT strategy FROM equity WHERE run_id=?", (run_id,))}
        for s in portfolio_strategies:
            if s in known and s not in done:
                alerts.append(Alert(f"decision:{s}", "crit", f"{NAMES.get(s, s)} 今天的决策没有执行"))
    for ts, s, kind, msg in db.execute(
            "SELECT ts, strategy, kind, message FROM events WHERE run_id=? AND ts>? AND (level='error' OR kind "
            "IN ('liquidation','risk_halt','ml_skip')) ORDER BY ts", (run_id, since_ms)):
        lvl = "crit" if kind in ("liquidation", "strategy_error", "handler_error") else "warn"
        alerts.append(Alert(f"event:{kind}:{s}:{ts}", lvl, f"{NAMES.get(s, s or '-')} {kind}: {msg[:200]}"))

    for s, in db.execute("SELECT DISTINCT strategy FROM equity WHERE run_id=?", (run_id,)):
        eq = [e for (e,) in db.execute("SELECT equity FROM equity WHERE run_id=? AND strategy=? ORDER BY ts",
                                       (run_id, s))]
        if len(eq) >= 2:
            peak = max(eq)
            dd = 1 - eq[-1] / peak
            if dd > dd_warn and s != "S0_hold_btc":
                alerts.append(Alert(f"dd:{s}", "warn", f"{NAMES.get(s, s)} 从高点回撤 {dd:.1%}"))
    db.close()

    if collect_dir is not None and collect_dir.exists():
        files = list(collect_dir.glob("**/*.parquet"))
        if not files or time.time() - max(f.stat().st_mtime for f in files) > 20 * 60:
            alerts.append(Alert("collect:stale", "warn", "数据采集超过 20 分钟没有新快照"))
    return alerts


def daily_summary(db_path: Path) -> str:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    run_id, created = db.execute(
        "SELECT run_id, created_at FROM runs WHERE mode='paper' ORDER BY created_at DESC LIMIT 1").fetchone()
    st = json.loads(db.execute("SELECT blob FROM state WHERE run_id=?", (run_id,)).fetchone()[0])["broker"]
    marks = st["marks"]
    lines = [f"刷钱机日报（第 {int((time.time() * 1000 - created) // DAY) + 1} 天）"]
    for sid, a in st["accounts"].items():
        upnl = sum(p["qty"] * (marks.get(k, p["avg_px"]) - p["avg_px"]) for k, p in a["positions"].items())
        eq = a["balance"] + upnl
        first = db.execute("SELECT equity FROM equity WHERE run_id=? AND strategy=? ORDER BY ts LIMIT 1",
                           (run_id, sid)).fetchone()
        start = first[0] if first and first[0] else eq
        lines.append(f"{NAMES.get(sid, sid)}：{eq:,.0f}（{eq / start - 1:+.2%}），持仓 {len(a['positions'])} 个，"
                     f"手续费 {a['fees_total']:,.1f}，资金费 {a['funding_total']:+,.1f}")
    for ts, s, msg in db.execute("SELECT ts, strategy, message FROM events WHERE run_id=? AND kind='ml_targets' "
                                 "ORDER BY ts DESC LIMIT 6", (run_id,)):
        lines.append(f"今日决策 {NAMES.get(s, s)}：{msg}")
    db.close()
    okx = okx_line()
    if okx:
        lines.append(okx)
    return "\n".join(lines)


def okx_line() -> str | None:
    """Real OKX account, read-only (stage 1 of real trading); None when no key is configured."""
    if not os.environ.get("MM_OKX_API_KEY"):
        return None
    try:
        from moneymachine.okx_private import ReadOnlyOkx

        s = ReadOnlyOkx().snapshot()
        return (f"OKX 实盘账户（只读）：权益 {s['total_equity_usd']:,.2f} USD，持仓 {len(s['positions'])} 个，"
                f"挂单 {s['open_orders']} 个，密钥权限 {s['perm']}")
    except Exception as e:  # noqa: BLE001 — message never contains the credentials
        return f"OKX 实盘账户（只读）：查询失败 — {type(e).__name__}: {str(e)[:120]}"


# ── delivery ─────────────────────────────────────────────────────────────
def _post(url: str, **kw) -> None:
    """POST and check the service's own result code: Feishu and PushPlus answer HTTP 200
    even when they reject the message (bad token, wrong webhook), with code ≠ 0/200."""
    r = httpx.post(url, timeout=10, **kw)
    r.raise_for_status()
    try:
        j = r.json()
    except ValueError:
        return
    code = j.get("code", 0) if isinstance(j, dict) else 0
    if code not in (0, 200):
        raise RuntimeError(f"rejected: code {code} {j.get('msg', '')}")


def _email(title: str, text: str) -> None:
    from email.message import EmailMessage

    user = os.environ["MM_ALERT_SMTP_USER"]
    msg = EmailMessage()
    msg["Subject"], msg["From"] = title, user
    msg["To"] = os.environ.get("MM_ALERT_EMAIL_TO") or user
    msg.set_content(text)
    with smtplib.SMTP_SSL(os.environ.get("MM_ALERT_SMTP_HOST", "smtp.gmail.com"), 465, timeout=20) as s:
        s.login(user, os.environ["MM_ALERT_SMTP_PASSWORD"].replace(" ", ""))  # Gmail shows it in groups of 4
        s.send_message(msg)


def send(title: str, text: str) -> list[str]:
    body = f"{title}\n{text}"
    channels = []
    if url := os.environ.get("MM_ALERT_FEISHU_WEBHOOK"):
        channels.append(("feishu", lambda u=url: _post(u, json={"msg_type": "text", "content": {"text": body}})))
    if tok := os.environ.get("MM_ALERT_PUSHPLUS_TOKEN"):
        channels.append(("pushplus", lambda: _post("https://www.pushplus.plus/send",
                                                   json={"token": tok, "title": title, "content": text})))
    if url := os.environ.get("MM_ALERT_WEBHOOK"):
        channels.append(("webhook", lambda u=url: _post(u, content=body.encode(), headers={
            "Title": title.encode("utf-8").decode("latin-1", "ignore")})))
    if os.environ.get("MM_ALERT_SMTP_USER") and os.environ.get("MM_ALERT_SMTP_PASSWORD"):
        channels.append(("email", lambda: _email(title, text)))
    sent = []
    for name, deliver in channels:
        try:
            deliver()
            sent.append(name)
        except (httpx.HTTPError, RuntimeError, smtplib.SMTPException, OSError) as e:
            # log the error type only: URLs and SMTP replies may contain secrets
            log.warning("alert delivery via %s failed: %s", name, e if isinstance(e, RuntimeError)
                        else type(e).__name__)
    log.info("[%s] %s\n%s", ",".join(sent) or "journal-only", title, text)
    return sent


def run_once(db_path: Path, state_path: Path, *, collect_dir: Path | None = None, summary_after_min: int = 40,
             now_ms: int | None = None, probe=None, sender=send) -> dict:
    now_ms = now_ms or int(time.time() * 1000)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"open": {}, "last_run": now_ms - 600_000}
    alerts = check(db_path, now_ms=now_ms, since_ms=state.get("last_run", now_ms - 600_000),
                   collect_dir=collect_dir, probe=probe, open_keys=tuple(state["open"]))
    if probe is None and now_ms - state.get("okx_checked", 0) > 3_600_000:   # real account: hourly, read-only
        state["okx_checked"] = now_ms
        line = okx_line()
        state["okx_alert"] = line if line and ("查询失败" in line or "read_only" not in line) else None
    if state.get("okx_alert"):
        alerts.append(Alert("okx:api", "warn", state["okx_alert"]))
    current = {a.key: a for a in alerts}
    fresh = [a for a in alerts if a.key not in state["open"] or now_ms / 1000 - state["open"][a.key] > RESEND_AFTER_S]
    recovered = [k for k in state["open"] if k not in current and not k.startswith("event:")]
    if fresh:
        crit = any(a.level == "crit" for a in fresh)
        sender("【刷钱机告警】" if crit else "【刷钱机提醒】", "\n".join(f"- {a.message}" for a in fresh))
    if recovered:
        sender("【刷钱机恢复】", "\n".join(f"- 已恢复：{k}" for k in recovered))
    open_ = {k: v for k, v in state["open"].items() if k in current}
    open_.update({a.key: now_ms / 1000 for a in fresh})
    day = time.strftime("%Y-%m-%d", time.gmtime(now_ms / 1000))
    if now_ms % DAY > summary_after_min * 60_000 and state.get("summary_day") != day:
        try:
            sender("【刷钱机日报】", daily_summary(db_path))
            state["summary_day"] = day
        except Exception as e:  # noqa: BLE001
            log.warning("daily summary failed: %s", e)
    state.update(open=open_, last_run=now_ms)
    state_path.write_text(json.dumps(state))
    return {"alerts": len(alerts), "sent": len(fresh), "recovered": len(recovered)}
