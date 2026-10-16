"""Today's decisions per paper account + the shadow post-only orders (E2), read-only.
Run on the server via scripts/check_today.sh."""
import json
import sqlite3
import time

db = sqlite3.connect("file:runs/v2.sqlite?mode=ro", uri=True)
run = db.execute("SELECT run_id FROM runs WHERE mode=? ORDER BY created_at DESC LIMIT 1", ("paper",)).fetchone()[0]
day = int(time.time() * 1000) // 86_400_000 * 86_400_000
print("decisions today (UTC):")
for s, kind, msg in db.execute("SELECT strategy, kind, message FROM events WHERE run_id=? AND ts>=? AND kind IN (?,?,?) "
                               "ORDER BY ts", (run, day, "ml_targets", "ml_skip", "strategy_error")):
    print(f"  {s:24s} {kind:14s} {msg[:110]}")
fills = db.execute("SELECT strategy, COUNT(*), ROUND(SUM(qty*price), 0) FROM fills WHERE run_id=? AND ts>=? "
                   "GROUP BY strategy", (run, day)).fetchall()
print("fills today:", {s: (c, v) for s, c, v in fills})
st = json.loads(db.execute("SELECT blob FROM state WHERE run_id=?", (run,)).fetchone()[0])["broker"]
for sid, a in st["accounts"].items():
    print(f"  {sid:24s} balance {a['balance']:>12,.2f}  positions {len(a['positions'])}")
x = sqlite3.connect("file:runs/execlab.sqlite?mode=ro", uri=True)
rows = x.execute("SELECT arm, status, COUNT(*), SUM(filled_ts IS NOT NULL), ROUND(AVG(filled_min), 1) FROM shadow "
                 "GROUP BY arm, status").fetchall()
print("shadow orders (arm, status, n, filled, avg minutes to fill):", rows)
