from __future__ import annotations

import httpx
import pytest

from conftest import H, M, SYM, bar, make_broker
from moneymachine.config import AccountConfig, LLMConfig, LLMModel
from moneymachine.llm.client import LLMClient, extract_json, same_model
from moneymachine.llm.prompts import DIRECT_SCHEMA
from moneymachine.market.aggregate import Aggregator
from moneymachine.risk import RiskManager
from moneymachine.sim.funding import FundingBook
from moneymachine.types import ClosePosition, Entry


# ── aggregation ──────────────────────────────────────────────────────────
def test_aggregator_emits_on_last_subbar():
    agg = Aggregator(H, 5 * M)
    out = []
    for k in range(12):
        out += agg.push(bar(k * 5 * M, 100 + k, 101 + k, 99 + k, 100.5 + k))
    assert len(out) == 1
    h = out[0]
    assert (h.ts, h.open, h.high, h.low, h.close) == (0, 100, 112, 99, 111.5)


def test_aggregator_emits_incomplete_bucket_after_gap():
    agg = Aggregator(H, 5 * M)
    assert agg.push(bar(0, 1, 1, 1, 1)) == []
    out = agg.push(bar(H + 5 * M, 2, 2, 2, 2))
    assert [b.ts for b in out] == [0]


# ── funding schedule ─────────────────────────────────────────────────────
def test_funding_schedule_extends_outside_coverage():
    fb = FundingBook(0.0001, 8)
    iv = 8 * H
    fb.add(SYM, [(10 * iv, 0.001), (11 * iv, 0.002), (12 * iv, 0.003)])
    assert fb.settlements(SYM, 8 * iv, 14 * iv) == [9 * iv, 10 * iv, 11 * iv, 12 * iv, 13 * iv, 14 * iv]
    assert fb.rate(SYM, 11 * iv) == 0.002
    assert fb.rate(SYM, 14 * iv) == 0.0001 and fb.is_assumed(SYM, 14 * iv)
    live = FundingBook(0.0001, 8, live=True)
    live.add(SYM, [(10 * iv, 0.001)])
    assert live.rate(SYM, 11 * iv) is None


# ── risk sizing ──────────────────────────────────────────────────────────
def _risk(inst, ledger, **acct):
    b = make_broker(inst, ledger)
    b.on_bar(SYM, bar(0, 100, 100, 100, 100))
    return b, RiskManager(AccountConfig(**acct), b, inst, H, ledger)


def test_risk_sizing_one_percent(inst, ledger):
    b, rm = _risk(inst, ledger, risk_per_trade=0.01, max_leverage=10)
    rm.apply("s", [Entry(SYM, "buy", stop_loss=98.0)], now=M, active_from=M)
    o = b.accounts["s"].open_orders()[0]
    assert o.qty == pytest.approx(50.0)  # 100 USDT risk / 2 USDT stop distance


def test_risk_leverage_cap(inst, ledger):
    b, rm = _risk(inst, ledger, risk_per_trade=0.05, max_leverage=2)
    rm.apply("s", [Entry(SYM, "buy", stop_loss=99.9)], now=M, active_from=M)
    o = b.accounts["s"].open_orders()[0]
    assert o.qty * 100 <= 20_000 + 1e-6


def test_risk_rejects_wrong_side_stop_and_pyramiding(inst, ledger):
    b, rm = _risk(inst, ledger)
    rm.apply("s", [Entry(SYM, "buy", stop_loss=101.0)], now=M, active_from=M)
    assert b.accounts["s"].open_orders() == []
    rm.apply("s", [Entry(SYM, "buy", stop_loss=95.0)], now=M, active_from=M)
    b.on_bar(SYM, bar(M, 100, 100, 100, 100))
    rm.apply("s", [Entry(SYM, "buy", stop_loss=95.0)], now=2 * M, active_from=2 * M)
    assert [o.role for o in b.accounts["s"].open_orders()] == ["sl"]
    # close + reverse in one batch is allowed
    rm.apply("s", [ClosePosition(SYM), Entry(SYM, "sell", stop_loss=105.0)], now=2 * M, active_from=2 * M)
    b.on_bar(SYM, bar(2 * M, 100, 100, 100, 100))
    assert b.accounts["s"].positions[SYM].qty < 0


def test_daily_loss_limit_blocks_entries(inst, ledger):
    b, rm = _risk(inst, ledger, daily_loss_limit=0.05)
    acct = b.accounts["s"]
    acct.day_start_equity = 10_000
    acct.balance = 9_400
    rm.apply("s", [Entry(SYM, "buy", stop_loss=95.0)], now=M, active_from=M)
    assert acct.open_orders() == []


# ── LLM client ───────────────────────────────────────────────────────────
def _llm(handler, **cfg):
    c = LLMConfig(models=[LLMModel(name="deepseek-chat", protocol="openai", price_in=1, price_out=2)], **cfg)
    import os
    os.environ["MM_LLM_BASE_URL"] = "https://relay.test/v1"
    os.environ["MM_LLM_API_KEY"] = "k"
    return LLMClient(c, http=httpx.Client(transport=httpx.MockTransport(handler)))


def _reply(content, model="deepseek-chat"):
    return httpx.Response(200, json={
        "model": model, "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
    })


GOOD = '{"action":"enter_long","stop_atr":2,"take_profit_atr":3,"confidence":70,"reason":"测试"}'


def test_llm_ok_and_cost():
    c = _llm(lambda req: _reply("```json\n" + GOOD + "\n```"))
    r = c.call_json(strategy="x", purpose="t", symbol=SYM, model="deepseek-chat", system="s", user="u",
                    schema=DIRECT_SCHEMA, now=0)
    assert r.ok and r.data["action"] == "enter_long"
    assert r.cost_usd == pytest.approx((1000 * 1 + 100 * 2) / 1e6)


@pytest.mark.parametrize("handler,needle", [
    (lambda req: _reply(GOOD, model="gpt-4o-mini"), "model mismatch"),
    (lambda req: _reply('{"action":"buy"}'), "schema"),
    (lambda req: _reply("I think you should buy"), "bad JSON"),
    (lambda req: httpx.Response(502, text="bad gateway"), "HTTP 502"),
])
def test_llm_failures_are_not_ok(handler, needle):
    r = _llm(handler).call_json(strategy="x", purpose="t", symbol=SYM, model="deepseek-chat", system="s",
                                user="u", schema=DIRECT_SCHEMA, now=0)
    assert not r.ok and needle in r.error


def test_llm_budget():
    c = _llm(lambda req: _reply(GOOD), max_calls_per_day=2)
    res = [c.call_json(strategy="x", purpose="t", symbol=SYM, model="deepseek-chat", system="s", user="u",
                       schema=DIRECT_SCHEMA, now=0).ok for _ in range(3)]
    assert res == [True, True, False]


def test_same_model_and_extract():
    assert same_model("claude-opus-5-5", "claude-opus-5-5")
    assert same_model("deepseek-chat", "deepseek/deepseek-chat")
    assert not same_model("claude-opus-5-5", "claude-sonnet-5-5")
    assert extract_json('sure: {"a": 1} ok') == {"a": 1}


def test_llm_budget_survives_restart():
    from moneymachine.ledger import Ledger

    lg = Ledger(":memory:", "r")
    c = _llm(lambda req: _reply(GOOD), max_calls_per_day=2)
    c.ledger = lg
    assert c.call_json(strategy="x", purpose="t", symbol=SYM, model="deepseek-chat", system="s", user="u",
                       schema=DIRECT_SCHEMA, now=1000).ok
    c2 = _llm(lambda req: _reply(GOOD), max_calls_per_day=2)
    c2.preload_usage(lg.db, "r", 2000)
    oks = [c2.call_json(strategy="x", purpose="t", symbol=SYM, model="deepseek-chat", system="s", user="u",
                        schema=DIRECT_SCHEMA, now=2000).ok for _ in range(2)]
    assert oks == [True, False]


def test_paper_lock_is_exclusive(tmp_path):
    import multiprocessing as mp

    from moneymachine.live import acquire_paper_lock

    db = tmp_path / "x.sqlite"
    held = acquire_paper_lock(db)
    # flock is per open file description: a second process must be refused
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_try_lock, args=(str(db), q))
    p.start()
    p.join(20)
    assert q.get(timeout=5) == "refused"
    held.close()


def _try_lock(db, q):
    from pathlib import Path

    from moneymachine.live import acquire_paper_lock

    try:
        acquire_paper_lock(Path(db))
        q.put("acquired")
    except ValueError:
        q.put("refused")


def _fake_run(tmp_path, eq: dict, fills: dict, kinds: dict, mode="paper"):
    from moneymachine.ledger import Ledger

    lg = Ledger(tmp_path / "g.sqlite", "r1")
    toml = "\n".join(f'[[strategy]]\nid = "{s}"\nkind = "{k}"' for s, k in kinds.items())
    lg.start_run("t", mode, toml)
    for s, series in eq.items():
        for ts, v in series:
            lg.equity(s, ts, v, v, 0.0, 0.0)
    for s, n in fills.items():
        for i in range(n):
            lg.fill(s, i, SYM, "buy", "rebalance", 1.0, 100.0, 0.01, "taker", 0.0, i)
    lg.close()
    return tmp_path / "g.sqlite"


def test_gonogo_market_neutral_mode_and_aligned_placebo(tmp_path):
    from moneymachine.config import GoNoGo
    from moneymachine.report import compute, gonogo

    D = 86_400_000
    days = [1_760_000_000_000 + i * D for i in range(70)]
    # ML: +4% overall but flat after day 10; placebo starts day 10 and gains 1% → placebo wins
    ml = [(t, 100_000 * (1 + 0.004 * min(i, 10))) for i, t in enumerate(days)]
    pl = [(t, 100_000 * (1 + 0.01 * (i - 10) / 59)) for i, t in enumerate(days) if i >= 10]
    btc = [(t, 100_000 * (1 + 0.003 * i)) for i, t in enumerate(days)]  # BTC +20%
    db = _fake_run(tmp_path, {"ML": ml, "P0": pl, "BTC": btc}, {"ML": 150, "P0": 150},
                   {"ML": "ml_portfolio", "P0": "ml_placebo", "BTC": "buy_and_hold"})
    summary, daily, meta = compute(db, "r1")
    crit = GoNoGo(beat_benchmark=False, require_positive_net=True)
    g = gonogo(summary, "BTC", crit, "paper", daily).set_index("strategy")
    assert list(g.index) == ["ML"]
    row = g.loc["ML"]
    assert row["trades"] and row["net_positive"] and row["forward_days"]   # 150 fills, +4%, 69 days
    assert not row["beats_placebo"]          # aligned window: ML flat vs placebo +1%
    assert "beats_benchmark" not in g.columns and row["vs_benchmark(参考)"] == "落后"
    assert not row["PASS"]
    # without alignment the whole-period +4% would have "beaten" the placebo's +1%
    g2 = gonogo(summary, "BTC", crit, "paper", None).set_index("strategy")
    assert g2.loc["ML", "beats_placebo"]


def test_monitor_alerts_dedup_and_recovery(tmp_path):
    import json as _json

    from moneymachine.ledger import Ledger
    from moneymachine.monitor import run_once

    D = 86_400_000
    now = 1_791_000_000_000 - 1_791_000_000_000 % D + 3_600_000  # 01:00 UTC
    lg = Ledger(tmp_path / "v2.sqlite", "r")
    lg.start_run("v2", "paper", "")
    lg.equity("ML_v2", now - D, 100_000, 100_000, 0, 0)
    lg.equity("ML_v2", now - 2 * 3_600_000, 88_000, 88_000, 0, 0)       # 12% drawdown
    lg.save_state(now, {"broker": {"last_bar_ts": {"BTC": now - 30 * 60_000}, "accounts": {}, "marks": {}}})
    lg.close()
    sent = []
    probe = {"state": lambda s: "failed" if s == "money-machine-collect" else "active",
             "memory": lambda s: (0, 0), "disk_free": lambda p: 50 * 2**30}
    r = run_once(tmp_path / "v2.sqlite", tmp_path / "m.json", now_ms=now, probe=probe,
                 sender=lambda title, text: sent.append((title, text)))
    text = "\n".join(t for _, t in sent)
    assert "money-machine-collect" in text and "行情停止更新" in text and "今天的决策没有执行" in text and "回撤" in text
    n_first = len(sent)
    run_once(tmp_path / "v2.sqlite", tmp_path / "m.json", now_ms=now + 300_000, probe=probe,
             sender=lambda title, text: sent.append((title, text)))
    assert all("告警" not in t and "提醒" not in t for t, _ in sent[n_first:])  # no repeats within 6h
    probe["state"] = lambda s: "active"
    run_once(tmp_path / "v2.sqlite", tmp_path / "m.json", now_ms=now + 600_000, probe=probe,
             sender=lambda title, text: sent.append((title, text)))
    assert any("恢复" in t and "svc:money-machine-collect" in x for t, x in sent)
    assert _json.loads((tmp_path / "m.json").read_text())["summary_day"]


def test_monitor_send_detects_rejected_pushplus(monkeypatch):
    import httpx
    from moneymachine import monitor

    calls = []

    def fake_post(url, timeout=10, **kw):
        calls.append(url)
        body = {"code": 903, "msg": "无效的用户token"} if "pushplus" in url else {"code": 0}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(monitor.httpx, "post", fake_post)
    monkeypatch.setenv("MM_ALERT_PUSHPLUS_TOKEN", "bad")
    monkeypatch.setenv("MM_ALERT_FEISHU_WEBHOOK", "https://open.feishu.cn/hook/x")
    monkeypatch.delenv("MM_ALERT_WEBHOOK", raising=False)
    assert monitor.send("t", "x") == ["feishu"]       # PushPlus rejected, Feishu still delivered
    assert len(calls) == 2


def test_monitor_email_channel(monkeypatch):
    from moneymachine import monitor

    got = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            got["host"] = (host, port)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, user, pw):
            got["login"] = (user, pw)

        def send_message(self, msg):
            got["msg"] = (msg["Subject"], msg["To"], msg.get_content().strip())

    monkeypatch.setattr(monitor.smtplib, "SMTP_SSL", FakeSMTP)
    for k in ("MM_ALERT_PUSHPLUS_TOKEN", "MM_ALERT_FEISHU_WEBHOOK", "MM_ALERT_WEBHOOK", "MM_ALERT_EMAIL_TO",
              "MM_ALERT_SMTP_HOST"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MM_ALERT_SMTP_USER", "me@example.com")
    monkeypatch.setenv("MM_ALERT_SMTP_PASSWORD", "abcd efgh ijkl mnop")
    assert monitor.send("【刷钱机测试】", "ok") == ["email"]
    assert got == {"host": ("smtp.gmail.com", 465), "login": ("me@example.com", "abcdefghijklmnop"),
                   "msg": ("【刷钱机测试】", "me@example.com", "ok")}


def test_gonogo_uses_each_strategys_own_control(tmp_path):
    from moneymachine.config import GoNoGo
    from moneymachine.ledger import Ledger
    from moneymachine.report import compute, gonogo

    D = 86_400_000
    days = [1_760_000_000_000 + i * D for i in range(70)]
    lg = Ledger(tmp_path / "c.sqlite", "r1")
    toml = """
[[strategy]]
id = "ML"
kind = "ml_portfolio"
params = { control = "P0" }
[[strategy]]
id = "P0"
kind = "ml_placebo"
[[strategy]]
id = "P1"
kind = "ml_placebo_concentrated"
"""
    lg.start_run("t", "paper", toml)
    for s, gain in (("ML", 0.05), ("P0", 0.01), ("P1", 0.20)):
        for i, t in enumerate(days):
            lg.equity(s, t, 1e5 * (1 + gain * i / 69), 1e5 * (1 + gain * i / 69), 0.0, 0.0)
    for i in range(150):
        lg.fill("ML", i, SYM, "buy", "rebalance", 1.0, 100.0, 0.01, "taker", 0.0, i)
    lg.close()
    summary, daily, _ = compute(tmp_path / "c.sqlite", "r1")
    g = gonogo(summary, "P0", GoNoGo(beat_benchmark=False, require_positive_net=True), "paper", daily)
    g = g.set_index("strategy")
    assert list(g.index) == ["ML"]              # placebos of every construction are baselines
    assert g.loc["ML", "beats_placebo"]          # compared with P0 only (P1's +20% does not count)


def test_resume_updates_stored_config_with_audit_event(tmp_path):
    import sqlite3

    from moneymachine.ledger import Ledger

    lg = Ledger(tmp_path / "u.sqlite", "r1")
    lg.start_run("t", "paper", "old")
    lg.update_config("old")                       # unchanged → nothing logged
    lg.update_config("new")
    lg.close()
    db = sqlite3.connect(tmp_path / "u.sqlite")
    assert db.execute("SELECT config_toml FROM runs").fetchone()[0] == "new"
    ev = db.execute("SELECT kind, data FROM events").fetchall()
    assert len(ev) == 1 and ev[0][0] == "config_update" and '"old"' in ev[0][1]


def test_execlab_shadow_fill_rule(tmp_path, monkeypatch):
    import pandas as pd

    from moneymachine.execlab import ExecLab
    from moneymachine.ledger import Ledger

    pd.DataFrame({"inst": ["X-USDT-SWAP"], "tick_sz": [0.01], "ct_val": [10.0]}).to_parquet(tmp_path / "i.parquet")
    lg = Ledger(tmp_path / "l.sqlite", "r1")
    lg.start_run("t", "paper", "")
    now = int(__import__("time").time() * 1000)
    lg.fill("ML_v2", 1, "X-USDT-SWAP", "buy", "rebalance", 5.0, 100.2, 0.25, "taker", 0.0, now)
    lg.fill("S0_hold_btc", 2, "X-USDT-SWAP", "buy", "rebalance", 5.0, 100.2, 0.25, "taker", 0.0, now)   # not an arm
    lg.close()
    lab = ExecLab(tmp_path / "l.sqlite", tmp_path / "x.sqlite", tmp_path / "i.parquet")
    monkeypatch.setattr(lab, "_mid", lambda s: (100.0, 100.1))
    fills = lab.poll_fills()
    assert [f[0] for f in fills] == ["ML_v2"]
    for f in fills:
        lab.place(*f)
        lab.place(*f)                                      # same fill seen twice → one shadow order
    (i, o), = lab.open.items()
    assert o["limit"] == 100.0 and o["side"] == "buy"
    lab.on_trade("X-USDT-SWAP", 100.0, 3, o["placed"] + 1)      # at the limit: not through
    assert "filled" not in o
    lab.on_trade("X-USDT-SWAP", 99.99, 2, o["placed"] + 60_000)  # one tick through → filled
    lab.on_trade("X-USDT-SWAP", 99.98, 4, o["placed"] + 90_000)
    row = lab.db.execute("SELECT filled_min, through_sz, contracts FROM shadow WHERE id=?", (i,)).fetchone()
    assert row == (1.0, 6.0, 0.5)


def test_okx_read_only_client():
    import base64
    import hashlib
    import hmac

    import httpx
    import pytest as _pt

    from moneymachine.okx_private import OkxApiError, ReadOnlyOkx

    sig = ReadOnlyOkx.sign("s3cret", "2020-12-08T09:08:57.715Z", "GET", "/api/v5/account/balance?ccy=BTC")
    want = base64.b64encode(hmac.new(b"s3cret", b"2020-12-08T09:08:57.715ZGET/api/v5/account/balance?ccy=BTC",
                                     hashlib.sha256).digest()).decode()
    assert sig == want
    perm = {"v": "read_only"}
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.method, req.url.path, req.headers.get("OK-ACCESS-PASSPHRASE")))
        if req.url.path == "/api/v5/account/config":
            return httpx.Response(200, json={"code": "0", "data": [{"perm": perm["v"], "label": "t", "ip": ""}]})
        return httpx.Response(200, json={"code": "0", "data": [{"totalEq": "12.5", "details": []}]})

    c = ReadOnlyOkx("KEY-zzz", "SECRET-zzz", "PASS-zzz", http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert "zzz" not in repr(c) and "zzz" not in str(vars(c).get("http"))
    assert c.get("/api/v5/account/balance")[0]["totalEq"] == "12.5"
    assert seen[0][1] == "/api/v5/account/config"                       # permission checked first
    assert all(m == "GET" for m, _, _ in seen)
    with _pt.raises(PermissionError):
        c.get("/api/v5/trade/order")                                    # order endpoint: refused locally
    perm["v"] = "read_only,trade"
    c2 = ReadOnlyOkx("k", "s", "p", http=httpx.Client(transport=httpx.MockTransport(handler)))
    with _pt.raises(OkxApiError, match="read-only"):
        c2.get("/api/v5/account/balance")                               # trading key: refused
    with _pt.raises(OkxApiError, match="MM_OKX_PASSPHRASE"):
        ReadOnlyOkx("k", "s", "")


def test_execlab_sees_fills_committed_late_with_the_same_timestamp(tmp_path, monkeypatch):
    import pandas as pd

    from moneymachine.execlab import ExecLab
    from moneymachine.ledger import Ledger

    pd.DataFrame({"inst": ["X-USDT-SWAP", "Y-USDT-SWAP"], "tick_sz": [0.01, 0.01], "ct_val": [1.0, 1.0]}).to_parquet(
        tmp_path / "i.parquet")
    lg = Ledger(tmp_path / "l.sqlite", "r1")
    lg.start_run("t", "paper", "")
    ts = int(__import__("time").time() * 1000)
    lg.fill("ML_v2", 1, "X-USDT-SWAP", "buy", "rebalance", 1.0, 10.0, 0.0, "taker", 0.0, ts)
    lg.commit()
    lab = ExecLab(tmp_path / "l.sqlite", tmp_path / "x.sqlite", tmp_path / "i.parquet")
    monkeypatch.setattr(lab, "_mid", lambda s: (10.0, 10.01))
    for f in lab.poll_fills():
        lab.last_seen = max(lab.last_seen, f[5])
        lab.place(*f)
    lg.fill("B1b_concentrated", 2, "Y-USDT-SWAP", "sell", "rebalance", 1.0, 10.0, 0.0, "taker", 0.0, ts)  # same ts,
    lg.commit()                                                                                            # later commit
    late = lab.poll_fills()
    assert [f[0] for f in late] == ["B1b_concentrated"]
    for f in late:
        lab.place(*f)
    assert lab.poll_fills() == []
    lg.close()


def test_collector_part_files_and_memory_alert_hysteresis(tmp_path):
    import pandas as pd

    from moneymachine.ledger import Ledger
    from moneymachine.ml.collector import append, load_day
    from moneymachine.monitor import check

    t0 = 1_792_195_200_000                                   # 2026-10-17 00:00 UTC
    for k in range(3):
        append(pd.DataFrame({"ts": [t0 + k * 300_000] * 2, "inst": ["A", "B"], "last": [1.0, 2.0]}), tmp_path)
    pd.DataFrame({"ts": [t0 - 1], "inst": ["A"], "last": [0.5]}).to_parquet(tmp_path / "2026-10-17.parquet")  # old layout
    d = load_day("2026-10-17", tmp_path)
    assert len(d) == 7 and d["ts"].is_monotonic_increasing
    assert len(list((tmp_path / "2026-10-17").glob("*.parquet"))) == 3

    lg = Ledger(tmp_path / "m.sqlite", "r1")
    lg.start_run("t", "paper", "")
    lg.close()
    mem = {"v": 0.80}
    probe = {"state": lambda s: "active", "memory": lambda s: (int(mem["v"] * 100), 100), "disk_free": lambda p: 10 * 2**30}
    keys = lambda open_keys: {a.key for a in check(tmp_path / "m.sqlite", now_ms=t0, since_ms=t0, services=("svc",),  # noqa: E731
                                                   probe=probe, open_keys=open_keys)}
    assert "mem:svc" not in keys(())                         # 80%: below the 85% alert line
    mem["v"] = 0.86
    assert "mem:svc" in keys(())                             # 86%: alert
    mem["v"] = 0.80
    assert "mem:svc" in keys(("mem:svc",))                   # back to 80% while open: still open (no flapping)
    mem["v"] = 0.70
    assert "mem:svc" not in keys(("mem:svc",))               # below 75%: recovered


def test_execlab_marks_stale_fills(tmp_path, monkeypatch):
    import pandas as pd

    from moneymachine.execlab import STALE_MS, ExecLab
    from moneymachine.ledger import Ledger

    pd.DataFrame({"inst": ["X-USDT-SWAP"], "tick_sz": [0.01], "ct_val": [1.0]}).to_parquet(tmp_path / "i.parquet")
    lg = Ledger(tmp_path / "l.sqlite", "r1")
    lg.start_run("t", "paper", "")
    lg.close()
    lab = ExecLab(tmp_path / "l.sqlite", tmp_path / "x.sqlite", tmp_path / "i.parquet")
    monkeypatch.setattr(lab, "_mid", lambda s: (_ for _ in ()).throw(AssertionError("must not query the book")))
    old = int(__import__("time").time() * 1000) - STALE_MS - 60_000
    lab.place("ML_v2", "X-USDT-SWAP", "buy", 1.0, 10.0, old)
    assert lab.db.execute("SELECT status FROM shadow").fetchone()[0] == "stale" and not lab.open


def test_save_parquet_is_atomic(tmp_path, monkeypatch):
    import pandas as pd
    import pytest as _pt

    from moneymachine.ml.data import save_parquet

    p = tmp_path / "x.parquet"
    save_parquet(pd.DataFrame({"a": [1, 2]}), p)
    real = pd.DataFrame.to_parquet

    def half_write(self, path, **kw):
        real(self, path, **kw)
        with open(path, "r+b") as f:
            f.truncate(4)                                     # what a crash mid-write leaves behind
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", half_write)
    with _pt.raises(OSError):
        save_parquet(pd.DataFrame({"a": [9]}), p)
    monkeypatch.undo()
    assert pd.read_parquet(p)["a"].tolist() == [1, 2]          # the old file is untouched
    assert [f.name for f in tmp_path.iterdir()] == ["x.parquet"]  # no temp file left


def test_sync_candles_keeps_the_old_file_if_a_write_dies(tmp_path, monkeypatch):
    import pandas as pd
    import pytest as _pt

    from moneymachine.ml import data as Dmod

    (tmp_path / "bars_1D").mkdir()
    p = tmp_path / "bars_1D" / "X-USDT-SWAP.parquet"
    now = int(__import__("time").time() * 1000)
    day = 86_400_000
    t0 = now - now % day - 5 * day
    old = pd.DataFrame({"time": [t0], "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "vol_ccy": 1.0, "vol_quote": 1.0})
    old.to_parquet(p, index=False)
    monkeypatch.setattr(Dmod, "_fetch_candles", lambda *a, **k: old.assign(time=t0 + day))
    real = pd.DataFrame.to_parquet

    def half_write(self, path, **kw):
        real(self, path, **kw)
        with open(path, "r+b") as f:
            f.truncate(4)
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", half_write)
    with _pt.raises(OSError):
        Dmod.sync_candles(None, "X-USDT-SWAP", "1Dutc", "bars_1D", root=tmp_path)
    monkeypatch.undo()
    assert pd.read_parquet(p)["time"].tolist() == [t0]        # the incident's call site cannot leave a torn file


def _monitor_env(tmp_path):
    from moneymachine.ledger import Ledger

    lg = Ledger(tmp_path / "v2.sqlite", "r1")
    lg.start_run("t", "paper", "")
    lg.save_state(0, {"broker": {"accounts": {"ML_v2": {"balance": 1e5, "positions": {}, "fees_total": 0, "funding_total": 0},
                                           "T1b_trend": {"balance": 1e3, "positions": {}, "fees_total": 0, "funding_total": 0}},
                              "marks": {}, "last_bar_ts": {}}})
    probe = {"state": lambda s: "active", "memory": lambda s: (0, 0), "disk_free": lambda p: 10 * 2**30}
    return lg, probe


def test_failed_delivery_is_retried_and_late_events_are_seen(tmp_path):
    from moneymachine.monitor import DAY, run_once

    lg, probe = _monitor_env(tmp_path)
    now = 1_793_000_000_000 - 1_793_000_000_000 % DAY + 2 * 3_600_000    # 02:00 UTC
    st = tmp_path / "m.json"
    got, ok = [], {"v": False}

    def sender(title, text):
        got.append((title, text))
        return ["email"] if ok["v"] else []

    lg.event(now - 60_000, "error", "T1b_trend", "strategy_error", "boom")
    lg.commit()
    kw = dict(now_ms=now, probe=probe, sender=sender, summary_after_min=24 * 60)
    run_once(tmp_path / "v2.sqlite", st, **{**kw})
    assert any("boom" in t for _, t in got)                   # tried, failed
    ok["v"] = True
    got.clear()
    run_once(tmp_path / "v2.sqlite", st, **{**kw, "now_ms": now + 300_000})
    assert any("boom" in t for _, t in got)                   # the same event is sent again and now arrives
    got.clear()
    run_once(tmp_path / "v2.sqlite", st, **{**kw, "now_ms": now + 600_000})
    assert not any("boom" in t for _, t in got)               # and only once after that
    lg.event(now + 590_000, "error", None, "handler_error", "late commit")     # ts before the last run
    lg.commit()
    got.clear()
    run_once(tmp_path / "v2.sqlite", st, **{**kw, "now_ms": now + 900_000})
    assert any("late commit" in t for _, t in got)
    lg.close()


def test_recurring_identical_events_alert_once(tmp_path):
    from moneymachine.monitor import DAY, run_once

    lg, probe = _monitor_env(tmp_path)
    now = 1_793_000_000_000 - 1_793_000_000_000 % DAY + 2 * 3_600_000    # 02:00 UTC
    st = tmp_path / "m.json"
    got = []
    kw = dict(probe=probe, sender=lambda t, x: got.append((t, x)), summary_after_min=24 * 60)
    lg.event(now - 120_000, "error", None, "handler_error", "OkxError: code=51001")   # two identical in one window
    for i in range(4):                                   # same error every 5 min, like a delisted-symbol poll
        lg.event(now + i * 300_000 - 60_000, "error", None, "handler_error", "OkxError: code=51001")
        lg.commit()
        run_once(tmp_path / "v2.sqlite", st, now_ms=now + i * 300_000, **kw)
    assert sum(x.count("51001") for _, x in got) == 1  # one line total: dedup within a batch and across runs
    lg.event(now + 4 * 300_000 - 60_000, "error", None, "handler_error", "something else")
    lg.commit()
    got.clear()
    run_once(tmp_path / "v2.sqlite", st, now_ms=now + 4 * 300_000, **kw)
    assert any("something else" in x for _, x in got)    # a different error still alerts
    lg.close()


def test_missed_decision_is_not_reported_recovered_at_midnight(tmp_path):
    import json

    from moneymachine.monitor import DAY, run_once

    lg, probe = _monitor_env(tmp_path)
    midnight = 1_793_000_000_000 - 1_793_000_000_000 % DAY
    st = tmp_path / "m.json"
    st.write_text(json.dumps({"open": {"decision:T1b_trend": midnight / 1000 - 3600}, "last_run": midnight - 300_000}))
    got = []
    run_once(tmp_path / "v2.sqlite", st, now_ms=midnight + 300_000, probe=probe,
             sender=lambda t, x: got.append(t), summary_after_min=24 * 60)
    assert got == []                                          # 00:05: nothing is known about today yet
    st.write_text("{broken")
    run_once(tmp_path / "v2.sqlite", st, now_ms=midnight + 600_000, probe=probe,
             sender=lambda t, x: got.append(t), summary_after_min=24 * 60)     # damaged state: still runs
    assert json.loads(st.read_text())["last_run"] == midnight + 600_000
    lg.close()


def test_daily_summary_lists_today_per_arm(tmp_path):
    import time as _t

    from moneymachine.monitor import daily_summary

    lg, _ = _monitor_env(tmp_path)
    now = int(_t.time() * 1000)
    lg.event(now - 2 * 86_400_000, "info", "ML_v2", "ml_targets", "old decision")
    lg.event(now, "info", "T1b_trend", "ml_targets", "trend ok")
    lg.commit()
    text = daily_summary(tmp_path / "v2.sqlite")
    assert "trend ok" in text and "old decision" not in text
    assert "今日决策 机器学习策略：今日没有决策" in text
    lg.close()


def test_execlab_restart_keeps_what_an_open_row_saw(tmp_path, monkeypatch):
    import pandas as pd

    from moneymachine.execlab import ExecLab, _now
    from moneymachine.ledger import Ledger

    pd.DataFrame({"inst": ["X-USDT-SWAP"], "tick_sz": [0.01], "ct_val": [1.0]}).to_parquet(tmp_path / "i.parquet")
    lg = Ledger(tmp_path / "l.sqlite", "r1")
    lg.start_run("t", "paper", "")
    lg.close()
    args = (tmp_path / "l.sqlite", tmp_path / "x.sqlite", tmp_path / "i.parquet")
    lab = ExecLab(*args)
    monkeypatch.setattr(lab, "_mid", lambda s: (100.0, 100.1))
    lab.place("ML_v2", "X-USDT-SWAP", "buy", 1.0, 100.2, _now())
    (i, o), = lab.open.items()
    lab.on_trade("X-USDT-SWAP", 99.9, 2, o["placed"] + 1)
    lab.db.execute("UPDATE shadow SET mid30=50.0 WHERE id=?", (i,))
    lab.db.commit()
    again = ExecLab(*args)
    assert again.open[i]["filled"] == o["placed"] + 1 and again.open[i]["mid30"] == 50.0
    assert ("through" not in again.open[i]) and again.db.execute("SELECT filled_ts FROM shadow WHERE id=?", (i,)).fetchone()[0]
