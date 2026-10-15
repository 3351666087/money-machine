#!/usr/bin/env bash
# 管理服务器上的刷钱机模拟盘。用法：
#   scripts/server.sh root@服务器IP status|check|report|logs|restart|stop|start|pull|collect-status [v2|v1]
set -euo pipefail
HOST="${1:-}"; CMD="${2:-status}"; RUN="${3:-v2}"
if [[ -z "${HOST}" ]]; then
  echo "用法: scripts/server.sh root@服务器IP status|report|logs|restart|stop|start|pull" >&2
  exit 1
fi
cd "$(dirname "$0")/.."
# 有项目专用密钥就用它（scripts/server.sh 不再需要输入密码）
KEY="${HOME}/.ssh/money_machine_ed25519"
if [[ -f "${KEY}" ]]; then
  ssh() { command ssh -i "${KEY}" "$@"; }
  scp() { command scp -i "${KEY}" "$@"; }
fi
SVC=money-machine-paper.service
AS_MM='su -s /bin/bash mm -c'

case "${CMD}" in
  status)
    ssh "${HOST}" "systemctl --no-pager --lines=0 status ${SVC} | head -6; echo; journalctl -u ${SVC} -n 200 --no-pager -o cat | grep 'status |' | tail -1"
    ;;
  report)
    ssh "${HOST}" "${AS_MM} 'cd /opt/money-machine/app && HOME=/opt/money-machine .venv/bin/mm report --config configs/${RUN}.toml'"
    ;;
  logs)
    ssh -t "${HOST}" "journalctl -u ${SVC} -f -n 50 -o cat"
    ;;
  restart|stop|start)
    ssh "${HOST}" "systemctl ${CMD} ${SVC} && systemctl --no-pager --lines=0 status ${SVC} | head -4"
    ;;
  check)
    # 一次看全：服务状态 + 最近决策 + 持仓/权益 + 成交 + 报错
    ssh "${HOST}" "systemctl is-active money-machine-paper money-machine-collect | paste -sd' '; journalctl -u ${SVC} --since '-26h' --no-pager -o cat | grep -E 'status \||ERROR|Traceback|error' | tail -4"
    ssh "${HOST}" "su -s /bin/bash mm -c 'cd /opt/money-machine/app && .venv/bin/python - ${RUN}'" <<'PY'
import json, sqlite3, sys, time
db = sqlite3.connect(f"runs/{sys.argv[1]}.sqlite")
fmt = lambda ms: time.strftime("%m-%d %H:%M", time.gmtime(ms / 1000))
run_id, status = db.execute("SELECT run_id, status FROM runs WHERE mode='paper' ORDER BY created_at DESC LIMIT 1").fetchone()
print(f"\n== 运行 {run_id}（{status}）")
print("== 每个策略最新权益")
for s, ts, eq, exp in db.execute("SELECT strategy, MAX(ts), equity, gross_exposure FROM equity WHERE run_id=? GROUP BY strategy", (run_id,)):
    print(f"  {s:<14} {fmt(ts)}  权益 {eq:,.2f}  总敞口 {exp:,.0f}")
st = json.loads(db.execute("SELECT blob FROM state WHERE run_id=?", (run_id,)).fetchone()[0])
marks = st["broker"]["marks"]
for sid, a in st["broker"]["accounts"].items():
    pos = {k: v["qty"] for k, v in a["positions"].items()}
    upnl = sum(q * (marks.get(k, v["avg_px"]) - v["avg_px"]) for k, v in a["positions"].items() for q in [v["qty"]])
    eq = a["balance"] + upnl
    print(f"== {sid}: 余额 {a['balance']:,.2f} 浮盈 {upnl:+,.2f} 手续费 {a['fees_total']:,.2f} 资金费 {a['funding_total']:+,.2f} 挂单 {len(a['orders'])}")
    rows = sorted(((k, q * marks.get(k, 0) / eq) for k, q in pos.items()), key=lambda x: x[1])
    if rows:
        print("   空头:", ", ".join(f"{k.replace('-USDT-SWAP','')} {w:+.1%}" for k, w in rows if w < 0) or "-")
        print("   多头:", ", ".join(f"{k.replace('-USDT-SWAP','')} {w:+.1%}" for k, w in rows if w > 0) or "-")
print("== 最近的决策和事件")
for ts, lvl, s, kind, msg in db.execute("SELECT ts, level, strategy, kind, message FROM events WHERE run_id=? ORDER BY ts DESC LIMIT 8", (run_id,)):
    print(f"  {fmt(ts)} [{lvl}] {s or '-'} {kind}: {msg[:150]}")
n, fees = db.execute("SELECT COUNT(*), COALESCE(SUM(fee),0) FROM fills WHERE run_id=?", (run_id,)).fetchone()
print(f"== 成交 {n} 笔，手续费合计 {fees:,.2f}")
for s, sym, side, q, px, fee, ts in db.execute("SELECT strategy, symbol, side, qty, price, fee, ts FROM fills WHERE run_id=? ORDER BY ts DESC LIMIT 6", (run_id,)):
    print(f"  {fmt(ts)} {s} {side} {q:g} {sym.replace('-USDT-SWAP','')} @ {px:g} 费 {fee:.2f}")
PY
    ;;
  collect-status)
    ssh "${HOST}" "systemctl --no-pager --lines=0 status money-machine-collect | head -4; ls -la /opt/money-machine/app/store/ml/collect | tail -3"
    ;;
  pull)
    # 在线备份（不打断运行），再拷回本地
    ssh "${HOST}" "su -s /bin/bash mm -c 'cd /opt/money-machine/app && .venv/bin/python - ${RUN}'" <<'PY'
import sqlite3
import sys
src = sqlite3.connect(f"runs/{sys.argv[1]}.sqlite")
dst = sqlite3.connect("runs/pull_backup.sqlite")
src.backup(dst)
dst.close()
print("backup ok")
PY
    mkdir -p runs
    scp "${HOST}:/opt/money-machine/app/runs/pull_backup.sqlite" "runs/server_${RUN}.sqlite"
    echo "已保存到 runs/server_${RUN}.sqlite；本地查看： uv run mm report --config configs/${RUN}.toml --db runs/server_${RUN}.sqlite"
    ;;
  *)
    echo "未知命令: ${CMD}" >&2; exit 1 ;;
esac
