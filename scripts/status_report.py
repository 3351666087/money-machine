"""Multi-day health + performance snapshot of the paper run, read-only. Run on the server via
scripts/status_report.sh (prints plain text; no secrets)."""
import json
import sqlite3
import time
from collections import defaultdict

DAY = 86_400_000
db = sqlite3.connect("file:runs/v2.sqlite?mode=ro", uri=True)
run = db.execute("SELECT run_id FROM runs WHERE mode=? ORDER BY created_at DESC LIMIT 1", ("paper",)).fetchone()[0]
now = int(time.time() * 1000)
today = now // DAY * DAY

print("== decisions per day (last 6 UTC days): ok / skipped / errors")
per = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
for ts, s, kind in db.execute("SELECT ts, strategy, kind FROM events WHERE run_id=? AND ts>=? AND kind IN (?,?,?,?)",
                              (run, today - 5 * DAY, "ml_targets", "ml_skip", "strategy_error", "handler_error")):
    i = {"ml_targets": 0, "ml_skip": 1}.get(kind, 2)
    per[time.strftime("%m-%d", time.gmtime(ts / 1000))][s][i] += 1
for d in sorted(per):
    print(f"  {d}: " + "  ".join(f"{s or '(engine)'}={v[0]}/{v[1]}/{v[2]}"
                                 for s, v in sorted(per[d].items(), key=lambda kv: str(kv[0]))))
errs = db.execute("SELECT ts, strategy, kind, message FROM events WHERE run_id=? AND ts>=? AND level=? ORDER BY ts DESC "
                  "LIMIT 5", (run, today - 5 * DAY, "error")).fetchall()
print("== error events (last 6 days):", len(errs))
for ts, s, kind, msg in errs:
    print(f"  {time.strftime('%m-%d %H:%M', time.gmtime(ts / 1000))} {s} {kind}: {msg[:120]}")

print("== accounts: equity, return since start, max drawdown, fills, fees, funding")
st = json.loads(db.execute("SELECT blob FROM state WHERE run_id=?", (run,)).fetchone()[0])["broker"]
marks = st["marks"]
for sid, a in st["accounts"].items():
    upnl = sum(p["qty"] * (marks.get(k, p["avg_px"]) - p["avg_px"]) for k, p in a["positions"].items())
    eq = a["balance"] + upnl
    series = [e for (e,) in db.execute("SELECT equity FROM equity WHERE run_id=? AND strategy=? ORDER BY ts", (run, sid))]
    start, peak, mdd = (series[0] if series else eq), 0.0, 0.0
    for e in series:
        peak = max(peak, e)
        mdd = max(mdd, 1 - e / peak if peak else 0)
    first = db.execute("SELECT MIN(ts) FROM equity WHERE run_id=? AND strategy=?", (run, sid)).fetchone()[0] or now
    fills = db.execute("SELECT COUNT(*) FROM fills WHERE run_id=? AND strategy=?", (run, sid)).fetchone()[0]
    print(f"  {sid:24s} {eq:>12,.2f}  {eq / start - 1:+7.2%}  maxDD {mdd:6.2%}  {(now - first) / DAY:4.1f} days  "
          f"fills {fills:4d}  fees {a['fees_total']:>8,.2f}  funding {a['funding_total']:>+8,.2f}  positions {len(a['positions'])}")

print("== shadow post-only orders (E2), placed within 5 min of the paper fill")
x = sqlite3.connect("file:runs/execlab.sqlite?mode=ro", uri=True)
for arm, n, filled, avg_min, stale in x.execute(
        "SELECT arm, SUM(status!='stale' AND placed_ts-taker_ts<=300000), "
        "SUM(status!='stale' AND placed_ts-taker_ts<=300000 AND filled_ts IS NOT NULL), "
        "ROUND(AVG(CASE WHEN placed_ts-taker_ts<=300000 THEN filled_min END),1), SUM(status='stale') FROM shadow GROUP BY arm"):
    print(f"  {arm:24s} orders {n or 0:4d}  filled {filled or 0:4d} ({(filled or 0) / max(n or 1, 1):.0%})  "
          f"avg {avg_min} min to fill  stale {stale or 0}")
