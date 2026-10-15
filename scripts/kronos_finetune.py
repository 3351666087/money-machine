"""v3 trial D1: fine-tune Kronos-small on OKX daily bars that closed on or before 2024-06-30.

Run on the Mac GPU, outside the sandbox:
    uv run --no-sync python scripts/kronos_finetune.py [steps] [batch]

Follows the official recipe (vendor/github/Kronos/finetune/train_predictor.py): frozen
tokenizer, next-token cross-entropy on both token streams, AdamW(lr 4e-5, betas 0.9/0.95,
weight decay 0.1), OneCycle schedule, gradient clip 3. Each sample is 256 context bars + the
next bar, normalised on the context (as KronosPredictor does at inference) and clipped at ±5.
Training windows end by 2023-12-31; validation windows end in 2024-01-01 → 2024-06-30.
The checkpoint with the lowest validation loss is saved to store/ml/kronos/finetuned/.
Nothing after 2024-06-30 is read, so the clean evaluation window stays out of sample.
"""
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "vendor/github/Kronos")
from model import Kronos, KronosTokenizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("finetune")

DAY = 86_400_000
CONTEXT = 256
WIN = CONTEXT + 1
TRAIN_END = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)   # window last bar ends ≤ this
DATA_END = int(pd.Timestamp("2024-07-01", tz="UTC").value // 1_000_000)    # nothing after 2024-06-30
OUT = Path("store/ml/kronos/finetuned/Kronos-small-okx1d")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 32
EVAL_EVERY, EVAL_BATCHES, SEED = 250, 60, 20261015


def load_series() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """inst → (features [n, 6], stamps [n, 5], bar end times [n]) for bars closed by 2024-06-30."""
    out = {}
    for p in sorted(Path("store/ml/bars_1D").glob("*.parquet")):
        df = pd.read_parquet(p).sort_values("time")
        df = df[df["time"] + DAY <= DATA_END]
        x = df[["open", "high", "low", "close", "vol_ccy", "vol_quote"]].to_numpy(np.float32)
        if len(df) < WIN or not np.isfinite(x).all():
            continue
        ts = pd.to_datetime(df["time"].to_numpy(), unit="ms")
        st = np.stack([ts.minute, ts.hour, ts.weekday, ts.day, ts.month], axis=1).astype(np.float32)
        out[p.stem] = (x, st, df["time"].to_numpy() + DAY)
    return out


def windows(series, lo: int, hi: int) -> list[tuple[str, int]]:
    """(inst, start) for windows whose last bar ends in (lo, hi]."""
    idx = []
    for inst, (_, _, end) in series.items():
        for s in range(len(end) - WIN + 1):
            if lo < end[s + WIN - 1] <= hi:
                idx.append((inst, s))
    return idx


def batch(series, idx, rng: random.Random, n: int, device):
    xs, sts = [], []
    for inst, s in (idx[rng.randrange(len(idx))] for _ in range(n)):
        x, st, _ = series[inst]
        w = x[s:s + WIN]
        m, sd = w[:CONTEXT].mean(0), w[:CONTEXT].std(0)
        xs.append(np.clip((w - m) / (sd + 1e-5), -5, 5))
        sts.append(st[s:s + WIN])
    return torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(sts)).to(device)


def loss_of(model, tok, x, st):
    with torch.no_grad():
        z0, z1 = tok.encode(x, half=True)
    logits = model(z0[:, :-1], z1[:, :-1], st[:, :-1, :])
    loss, _, _ = model.head.compute_loss(logits[0], logits[1], z0[:, 1:], z1[:, 1:])
    return loss


def main():
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    series = load_series()
    tr, va = windows(series, 0, TRAIN_END), windows(series, TRAIN_END, DATA_END)
    log.info("device %s | %d coins | %d train windows | %d validation windows | %d steps × %d",
             device, len(series), len(tr), len(va), STEPS, BATCH)
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device).eval()
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-5, total_steps=STEPS, pct_start=0.03, div_factor=10)

    def evaluate() -> float:
        model.eval()
        rng = random.Random(SEED)  # the same validation batches every time
        with torch.no_grad():
            v = [loss_of(model, tok, *batch(series, va, rng, BATCH, device)).item() for _ in range(EVAL_BATCHES)]
        model.train()
        return float(np.mean(v))

    best = evaluate()
    hist = [{"step": 0, "val": best}]
    log.info("step 0 (pre-trained): validation loss %.4f", best)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT))
    rng = random.Random(SEED + 1)
    model.train()
    t0, run = time.time(), []
    for step in range(1, STEPS + 1):
        loss = loss_of(model, tok, *batch(series, tr, rng, BATCH, device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % EVAL_EVERY == 0 or step == STEPS:
            v = evaluate()
            hist.append({"step": step, "train": float(np.mean(run)), "val": v})
            run = []
            mark = ""
            if v < best:
                best, mark = v, " (best, saved)"
                model.save_pretrained(str(OUT))
            log.info("step %d/%d: train %.4f | validation %.4f%s | %.0f min", step, STEPS, hist[-1]["train"], v,
                     mark, (time.time() - t0) / 60)
    (OUT / "finetune_log.json").write_text(json.dumps({"best_val": best, "history": hist, "steps": STEPS,
                                                       "batch": BATCH, "train_windows": len(tr),
                                                       "val_windows": len(va)}, indent=1))
    log.info("done: best validation loss %.4f → %s", best, OUT)


if __name__ == "__main__":
    main()
