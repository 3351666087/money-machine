"""Data for v3 trial B2: the broad universe (30d median volume >= $2M, top 150)."""
import logging
from pathlib import Path

from moneymachine.market.okx_rest import OkxRest
from moneymachine.ml import data as D, universe as U

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
ins = D.load_instruments()
P = U.daily_panel(list(ins["inst"]))
uni = U.pit_universe(P["vol_quote"], ins.set_index("inst")["list_ts"], min_adv_usd=2e6, max_names=150)
cands = U.candidates(uni)
have = {p.stem for p in (D.ROOT / "bars_4H").glob("*.parquet")}
new = [c for c in cands if c not in have]
Path("store/ml/candidates_broad.txt").write_text("\n".join(cands) + "\n")
spot = {d["instId"] for d in OkxRest().get("/api/v5/public/instruments", {"instType": "SPOT"})}
logging.info("broad candidates %d, new %d", len(cands), len(new))
D.run_parallel(lambda r, i: D.sync_candles(r, i, "4H", "bars_4H", start=1590969600000), new, workers=3, label="4H")
D.run_parallel(lambda r, s: D.sync_candles(r, s, "1Dutc", "spot_1D"),
               [c.replace("-SWAP", "") for c in new if c.replace("-SWAP", "") in spot], workers=3, label="spot")
D.run_parallel(lambda r, i: D.sync_funding(r, i), new, workers=2, label="funding")
for kind in ("oi", "ls"):
    D.run_parallel(lambda r, i, k=kind: D.sync_rubik(r, i, k), new, workers=1, label=f"rubik {kind}")
logging.info("ALL DONE")
