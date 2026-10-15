"""Compute the Kronos signal cache on the GPU (run outside the sandbox):

    uv run --no-sync python scripts/kronos_signal.py [Kronos-small|Kronos-base|<local dir>] [pred_len] [min_adv_musd] [max_names]

A local directory is a fine-tuned checkpoint (scripts/kronos_finetune.py); its cache tag is "smallft".

Clean window only: 2024-07-01 (after Kronos's pre-training cutoff) → before the
locked holdout. Resumable: results are cached in store/ml/kronos/.
"""
import logging
import sys
from pathlib import Path

import pandas as pd

from moneymachine.ml import data as D, universe as U
from moneymachine.ml import kronos_signal as K
from moneymachine.ml.validation import DAY, HOLDOUT_START

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
name = sys.argv[1] if len(sys.argv) > 1 else "Kronos-small"
pred_len = int(sys.argv[2]) if len(sys.argv) > 2 else 1
min_adv = float(sys.argv[3]) * 1e6 if len(sys.argv) > 3 else 20e6
max_names = int(sys.argv[4]) if len(sys.argv) > 4 else 50
ins = D.load_instruments()
P = U.daily_panel(list(ins["inst"]))
uni = U.pit_universe(P["vol_quote"], ins.set_index("inst")["list_ts"], min_adv_usd=min_adv, max_names=max_names)
days = uni.index[(uni.index >= K.CLEAN_FROM) & (uni.index < HOLDOUT_START - 3 * DAY)]
print(f"{name}: pred_len={pred_len}, {len(days)} decision days, {int(uni.loc[days].sum().sum())} coin-days", flush=True)
local = Path(name).is_dir()
tag = ("smallft" if local else name.split("-")[-1].lower()) + (
    "" if (min_adv, max_names) == (20e6, 50) else f"_u{int(min_adv / 1e6)}m{max_names}")
sig = K.compute(uni, pd.Index(days), model=name if local else f"NeoQuasar/{name}", pred_len=pred_len, tag=tag)
print("done:", sig.shape, flush=True)
