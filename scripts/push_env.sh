#!/bin/zsh
# Upload the local .env (and the monitor module) to the server, then send a test alert.
# No service is restarted: the monitor timer starts a fresh process every 5 minutes and
# reads /opt/money-machine/app/.env each time.
#   scripts/push_env.sh root@SERVER_IP
set -euo pipefail
HOST=${1:?usage: scripts/push_env.sh root@SERVER_IP}
KEY="${HOME}/.ssh/money_machine_ed25519"
SSH=(ssh -o ServerAliveInterval=30)
[[ -f "${KEY}" ]] && SSH+=(-i "${KEY}")
cd "$(dirname "$0")/.."
[[ -f .env ]] || { echo "no local .env"; exit 1; }
APP=/opt/money-machine/app
echo "==> uploading .env (mode 600, owner mm)"
"${SSH[@]}" "${HOST}" "umask 077 && cat > ${APP}/.env && chown mm:mm ${APP}/.env && chmod 600 ${APP}/.env" < .env
echo "==> uploading src/moneymachine/monitor.py"
"${SSH[@]}" "${HOST}" "cat > ${APP}/src/moneymachine/monitor.py && chown mm:mm ${APP}/src/moneymachine/monitor.py" \
  < src/moneymachine/monitor.py
echo "==> test alert"
"${SSH[@]}" "${HOST}" "cd ${APP} && runuser -u mm -- env HOME=/opt/money-machine LD_LIBRARY_PATH=/opt/money-machine/lib \
  .venv/bin/python -c 'import moneymachine.monitor as m; print(\"module:\", m.__file__)' && \
  runuser -u mm -- env HOME=/opt/money-machine LD_LIBRARY_PATH=/opt/money-machine/lib \
  .venv/bin/mm monitor --config configs/v2.toml --test 2>&1 | grep -v -i 'token' ; \
  ls -l ${APP}/.env | cut -c1-10"
