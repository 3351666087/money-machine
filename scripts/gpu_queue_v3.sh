#!/bin/zsh
# v3 GPU queue (run in a terminal outside the sandbox, Mac GPU). Every step resumes from its
# cache, so the script can simply be started again after an interruption.
#   A3: Kronos-small, pred_len=3 (resumes from store/ml/kronos/signal_small_1D_c256_p3_s5.parquet)
#   D1: fine-tune Kronos-small on bars <= 2024-06-30 (skipped if already done), then score
#   B2: Kronos-small on the broad universe (ADV >= $2M, top 150)
cd ~/Code/money-machine
FT=store/ml/kronos/finetuned/Kronos-small-okx1d
step() { echo "$(date -u '+%F %T') $*"; }
quiet() { grep --line-buffered -v -i -E "warn|HTTP Request"; }

step "A3 Kronos pred_len=3: start"
uv run --no-sync python scripts/kronos_signal.py Kronos-small 3 2>&1 | quiet
step "A3 Kronos done"

if [[ -f ${FT}/finetune_log.json ]]; then
  step "D1 fine-tune already done, skipping"
else
  step "D1 fine-tune: start"
  uv run --no-sync python scripts/kronos_finetune.py 3000 32 2>&1 | quiet
fi
if [[ -f ${FT}/finetune_log.json ]]; then  # written only when training completed
  step "D1 fine-tune done"
  uv run --no-sync python scripts/kronos_signal.py ${FT} 1 2>&1 | quiet
  step "D1 scoring done"
else
  step "D1 fine-tune FAILED (no finetune_log.json); skipping D1 scoring"
fi

uv run --no-sync python scripts/kronos_signal.py Kronos-small 1 2 150 2>&1 | quiet
step "B2 broad Kronos done"
step "GPU queue finished"
