#!/usr/bin/env bash
# 刷钱机只读查看脚本：在服务器上运行（需要 root 或 sudo），不会修改任何东西。
#
#   sudo bash /opt/money-machine/view.sh          # 总览
#   sudo bash /opt/money-machine/view.sh logs     # 实时日志（Ctrl+C 退出）
set -uo pipefail
APP=/opt/money-machine/app

if [[ "${1:-}" == "logs" ]]; then
  exec journalctl -u money-machine-paper -f -n 30 -o cat
fi

echo "================ 刷钱机 2.0 模拟盘（只用虚拟资金，不连交易账户）================"
echo
echo "【服务状态】"
for s in money-machine-paper money-machine-collect; do
  printf "  %-24s %s\n" "${s}" "$(systemctl is-active "${s}")"
done
echo
echo "【资源占用】（两个服务都有上限：模拟盘 内存1.8G/CPU 1核，采集 内存300M/CPU 0.25核）"
for s in money-machine-paper money-machine-collect; do
  mem=$(systemctl show "${s}" -p MemoryCurrent --value)
  peak=$(systemctl show "${s}" -p MemoryPeak --value)
  cpu=$(systemctl show "${s}" -p CPUUsageNSec --value)
  printf "  %-24s 内存 %4d MB（峰值 %4d MB） 累计CPU %6d 秒\n" "${s}" $((mem / 1048576)) $((peak / 1048576)) $((cpu / 1000000000))
done
printf "  磁盘占用 %s\n" "$(du -sh /opt/money-machine 2>/dev/null | cut -f1)"
echo
echo "【最近几小时的净值】（每小时一行；BTC/ETH/SOL 是当时价格，后面是两个虚拟账户的净值和持仓数）"
journalctl -u money-machine-paper --since '-6h' --no-pager -o cat | grep 'status |' | tail -6 | sed 's/^/  /'
echo
su -s /bin/bash mm -c "cd ${APP} && .venv/bin/python -" <<'PY'
import json, sqlite3, time
db = sqlite3.connect("runs/v2.sqlite")
fmt = lambda ms: time.strftime("%m-%d %H:%M", time.gmtime(ms / 1000))
row = db.execute("SELECT run_id, status, created_at FROM runs WHERE mode='paper' ORDER BY created_at DESC LIMIT 1").fetchone()
if not row:
    raise SystemExit("还没有模拟盘记录")
run_id, status, created = row
print(f"【当前这一轮】{run_id}，开始于 {fmt(created)} UTC，状态 {status}")
st = json.loads(db.execute("SELECT blob FROM state WHERE run_id=?", (run_id,)).fetchone()[0])
marks = st["broker"]["marks"]
names = {"ML_v2": "机器学习策略", "S0_hold_btc": "对照组：买入持有BTC", "P0_placebo": "随机对照组（同样的构建方法，随机信号）"}
for sid, a in st["broker"]["accounts"].items():
    upnl = sum(p["qty"] * (marks.get(k, p["avg_px"]) - p["avg_px"]) for k, p in a["positions"].items())
    eq = a["balance"] + upnl
    print(f"\n【{names.get(sid, sid)}】净值 {eq:,.0f} USDT（起始 100,000）  已付手续费 {a['fees_total']:,.1f}  资金费 {a['funding_total']:+,.1f}")
    pos = sorted(((k.replace("-USDT-SWAP", ""), p["qty"] * marks.get(k, 0) / eq) for k, p in a["positions"].items()),
                 key=lambda x: x[1])
    if pos:
        print("  空头：" + ("、".join(f"{k} {w:.1%}" for k, w in pos if w < 0) or "无"))
        print("  多头：" + ("、".join(f"{k} {w:.1%}" for k, w in pos if w > 0) or "无"))
print("\n【最近的决策 / 事件】（时间为 UTC，北京时间 +8 小时）")
for ts, lvl, s, kind, msg in db.execute(
        "SELECT ts, level, strategy, kind, message FROM events WHERE run_id=? ORDER BY ts DESC LIMIT 6", (run_id,)):
    print(f"  {fmt(ts)} [{lvl}] {s or '-'} {kind}: {msg[:120]}")
n, fees = db.execute("SELECT COUNT(*), COALESCE(SUM(fee), 0) FROM fills WHERE run_id=?", (run_id,)).fetchone()
print(f"\n【成交】共 {n} 笔，手续费合计 {fees:,.1f} USDT")
PY
echo
echo "实时日志： sudo bash /opt/money-machine/view.sh logs"
