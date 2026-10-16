#!/bin/zsh
# Today's decisions per paper account + the shadow post-only orders (E2), read-only.
#   scripts/check_today.sh root@SERVER_IP
set -euo pipefail
HOST=${1:?usage: scripts/check_today.sh root@SERVER_IP}
cd "$(dirname "$0")/.."
ssh -i ~/.ssh/money_machine_ed25519 "${HOST}" 'cd /opt/money-machine/app && runuser -u mm -- .venv/bin/python -' \
  < scripts/check_today.py
