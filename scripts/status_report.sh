#!/bin/zsh
# Multi-day health + performance snapshot of the paper run (read-only), plus services and alerts.
#   scripts/status_report.sh root@SERVER_IP
set -euo pipefail
HOST=${1:?usage: scripts/status_report.sh root@SERVER_IP}
cd "$(dirname "$0")/.."
SSH=(ssh -i ~/.ssh/money_machine_ed25519 "${HOST}")
"${SSH[@]}" 'cd /opt/money-machine/app && runuser -u mm -- .venv/bin/python -' < scripts/status_report.py
"${SSH[@]}" 'echo "== services"; systemctl is-active money-machine-paper money-machine-collect money-machine-execlab money-machine-monitor.timer | tr "\n" " "; echo
echo "== memory (MB): paper $(( $(systemctl show money-machine-paper -p MemoryCurrent --value) / 1048576 )), collect $(( $(systemctl show money-machine-collect -p MemoryCurrent --value) / 1048576 )), execlab $(( $(systemctl show money-machine-execlab -p MemoryCurrent --value) / 1048576 ))"
echo "== disk: $(df -h / | tail -1 | awk "{print \$4\" free (\"\$5\" used)\"}")"
echo "== paper errors in log (21 days): $(journalctl -u money-machine-paper --since -21d --no-pager -o cat | grep -c -i -E "traceback|error")"
echo "== e-mails sent (21 days), daily summaries counted, others listed:"; journalctl -u money-machine-monitor --since -21d --no-pager -o cat | grep -A1 "\[email\]" > /tmp/mm_mail.txt; echo "  daily summaries: $(grep -c "\[email\].*\xe6\x97\xa5\xe6\x8a\xa5" /tmp/mm_mail.txt)"; grep -v "\xe6\x97\xa5\xe6\x8a\xa5" /tmp/mm_mail.txt | grep -v "^--" | sed "s/^/  /" | tail -24; rm -f /tmp/mm_mail.txt
echo "== uptime: $(uptime -p)"; echo "== service restarts: paper $(systemctl show money-machine-paper -p NRestarts --value), since $(systemctl show money-machine-paper -p ActiveEnterTimestamp --value)"
echo "== open alerts:"; head -c 400 /opt/money-machine/app/runs/monitor_state.json; echo'
