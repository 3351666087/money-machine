"""Download Kronos weights (Hugging Face) and benchmark batch inference on MPS vs CPU.

Run outside the sandbox (Terminal panel):  uv run --no-sync python scripts/kronos_bench.py
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "vendor/github/Kronos")
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
models = {name: Kronos.from_pretrained(f"NeoQuasar/{name}") for name in sys.argv[1:] or ["Kronos-small"]}
print("weights ready:", {k: f"{sum(p.numel() for p in m.parameters()) / 1e6:.1f}M" for k, m in models.items()}, flush=True)

bars = pd.read_parquet("store/ml/bars_4H/BTC-USDT-SWAP.parquet").tail(400 + 60)
n_series, ctx, pred_len = 50, 400, 6
dfs, xs, ys = [], [], []
for i in range(n_series):
    h = bars.iloc[i : i + ctx]
    dfs.append(pd.DataFrame({"open": h["open"].values, "high": h["high"].values, "low": h["low"].values,
                             "close": h["close"].values, "volume": h["vol_ccy"].values, "amount": h["vol_quote"].values}))
    xs.append(pd.Series(pd.to_datetime(h["time"].values, unit="ms")))
    last = int(h["time"].iloc[-1])
    ys.append(pd.Series(pd.to_datetime(last + (np.arange(pred_len) + 1) * 4 * 3_600_000, unit="ms")))

for name, m in models.items():
    for device in ("mps", "cpu"):
        p = KronosPredictor(m, tok, device=device, max_context=512)
        t = time.time()
        p.predict_batch(dfs, xs, ys, pred_len=pred_len, sample_count=3, verbose=False)
        if device == "mps":
            torch.mps.synchronize()
        print(f"{name} on {device}: {n_series} series x 3 samples in {time.time() - t:.1f}s", flush=True)
