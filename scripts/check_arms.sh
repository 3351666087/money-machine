#!/bin/zsh
# After a redeploy: are all paper accounts loaded, and was the stored config updated?
#   scripts/check_arms.sh root@SERVER_IP
set -euo pipefail
HOST=${1:?usage: scripts/check_arms.sh root@SERVER_IP}
ssh -i ~/.ssh/money_machine_ed25519 "${HOST}" 'journalctl -u money-machine-paper --since "-10min" --no-pager -o cat \
  | grep -v -i warn | tail -5; cd /opt/money-machine/app && runuser -u mm -- .venv/bin/python -c "
import json, sqlite3
db = sqlite3.connect(\"file:runs/v2.sqlite?mode=ro\", uri=True)
run = db.execute(\"SELECT run_id, config_toml FROM runs WHERE mode=\x27paper\x27 ORDER BY created_at DESC LIMIT 1\").fetchone()
print(\"run\", run[0], \"| stored config has v3 arms:\", \"B1b_concentrated\" in run[1])
for ts, kind, msg in db.execute(\"SELECT ts, kind, message FROM events WHERE run_id=? AND kind IN (\x27config_update\x27, \x27paper_resume\x27) ORDER BY ts DESC LIMIT 3\", (run[0],)):
    print(\"event\", ts, kind, msg)
st = json.loads(db.execute(\"SELECT blob FROM state WHERE run_id=?\", (run[0],)).fetchone()[0])
for sid, a in st[\"broker\"][\"accounts\"].items():
    print(\"  %-24s balance %12.2f  positions %d\" % (sid, a.get(\"balance\"), len(a.get(\"positions\"))))
"'
