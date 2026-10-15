#!/usr/bin/env bash
# 把刷钱机部署到 Linux 服务器（systemd）。在本机终端运行：
#
#   scripts/deploy_server.sh root@服务器IP                      # 2.0 模拟盘 + 连续数据采集
#   scripts/deploy_server.sh root@服务器IP --collect-only       # 只跑连续数据采集
#   scripts/deploy_server.sh root@服务器IP --config configs/v1.toml   # 换成别的模拟盘配置
#   scripts/deploy_server.sh root@服务器IP --with-env           # 同时上传本地 .env
#
# 只会新增这些东西，不碰服务器上已有的系统：
#   - 系统用户 mm（无登录 shell），目录 /opt/money-machine
#   - 该用户自己的 uv 和 Python（装在 /opt/money-machine 下，不动系统 Python / apt；PyTorch 用 CPU 版）
#   - systemd 服务 money-machine-collect（每 5 分钟采集全部永续的价差和持仓量；限内存 300M）
#   - systemd 服务 money-machine-paper（模拟盘；限 CPU 1 核、内存 1.8G、低优先级）
# 不开放任何端口：只需要主动连 OKX 的公共行情接口。
# 重复运行是安全的：账本 runs/*.sqlite 和采集数据会保留，服务从上次的状态继续。
set -euo pipefail

HOST="${1:-}"
if [[ -z "${HOST}" ]]; then
  echo "用法: scripts/deploy_server.sh root@服务器IP [--with-env]" >&2
  exit 1
fi
shift
WITH_ENV=0
COLLECT_ONLY=0
CONFIG=configs/v2.toml
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-env) WITH_ENV=1 ;;
    --collect-only) COLLECT_ONLY=1 ;;
    --config) CONFIG="$2"; shift ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
  shift
done

cd "$(dirname "$0")/.."
CTL="${HOME}/.ssh/mm-deploy-%C"
SSH=(ssh -o ControlMaster=auto -o ControlPath="$CTL" -o ControlPersist=180 -o ServerAliveInterval=30)
KEY="${HOME}/.ssh/money_machine_ed25519"
[[ -f "${KEY}" ]] && SSH+=(-i "${KEY}")   # 有项目专用密钥就不用再输密码
cleanup() { ssh -o ControlPath="$CTL" -O exit "${HOST}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> 连接 ${HOST}（整个部署只需要输入一次密码）"
"${SSH[@]}" "${HOST}" true

echo "==> 预检：系统、资源、名称冲突、OKX 连通性、时钟"
"${SSH[@]}" "${HOST}" 'bash -s' <<'REMOTE'
set -e
[ "$(id -u)" = 0 ] || { echo "需要 root（或能 sudo 的用户直接以 root 登录）"; exit 2; }
echo "系统: $(uname -sm) | $( (. /etc/os-release && echo "$PRETTY_NAME") 2>/dev/null)"
echo "磁盘(/opt): $(df -h /opt | awk 'NR==2{print $4" 可用"}') | 内存可用: $(free -m | awk 'NR==2{print $7" MB"}')"
command -v systemctl >/dev/null || { echo "ABORT: 没有 systemd"; exit 3; }
command -v curl >/dev/null || { echo "ABORT: 服务器上没有 curl"; exit 3; }
if id mm >/dev/null 2>&1 && [ ! -f /opt/money-machine/.mm-owned ]; then
  echo "ABORT: 用户 mm 已存在且不是本项目创建的，为避免冲突已停止"; exit 3
fi
for svc in money-machine-paper money-machine-collect; do
  if [ -f /etc/systemd/system/${svc}.service ] && [ ! -f /opt/money-machine/.mm-owned ]; then
    echo "ABORT: 服务名 ${svc} 已被占用"; exit 3
  fi
done
okx=$(curl -s -m 10 https://www.okx.com/api/v5/public/time || true)
echo "${okx}" | grep -q '"code":"0"' || { echo "ABORT: 这台服务器访问不了 OKX 公共接口: ${okx:-无响应}"; exit 4; }
okx_ms=$(echo "${okx}" | sed -E 's/.*"ts":"([0-9]+)".*/\1/')
skew=$(( $(date +%s) - okx_ms / 1000 ))   # whole seconds (some date builds lack %N)
echo "OKX 可访问；本机时钟与 OKX 相差约 ${skew} 秒"
if [ "${skew#-}" -gt 3 ]; then echo "WARN: 时钟偏差超过 3 秒，建议开启 NTP 同步（timedatectl set-ntp true）"; fi
REMOTE

echo "==> 创建用户 mm 和目录 /opt/money-machine"
"${SSH[@]}" "${HOST}" 'bash -s' <<'REMOTE'
set -e
NOLOGIN=$(command -v nologin || echo /bin/false)
id mm >/dev/null 2>&1 || useradd --system --home-dir /opt/money-machine --create-home --shell "${NOLOGIN}" mm
mkdir -p /opt/money-machine/app
touch /opt/money-machine/.mm-owned
rm -rf /opt/money-machine/app/src   # 删除旧代码（账本 runs/ 保留）
REMOTE

echo "==> 上传代码、2.0 数据（约 100MB，省得服务器从头下载历史）和训练好的模型"
EXTRA=()
# 2.0 数据只在首次部署时上传：之后服务器每天自己更新，覆盖会让它退回到旧数据
if "${SSH[@]}" "${HOST}" 'test -f /opt/money-machine/app/store/ml/instruments.parquet'; then
  echo "    服务器已有 2.0 数据（每天自动更新），跳过数据上传"
else
  [[ -d store/ml ]] && EXTRA+=(store/ml)
fi
[[ -d runs/ml/models ]] && EXTRA+=(runs/ml/models)
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --exclude '__pycache__' --exclude 'store/ml/kronos' \
    --exclude 'store/ml/collect' --exclude 'store/ml/cache' -czf - \
    pyproject.toml uv.lock .python-version README.md src configs scripts vendor/github/Kronos/model "${EXTRA[@]}" \
  | "${SSH[@]}" "${HOST}" 'tar -xzf - -C /opt/money-machine/app'
if [[ ${WITH_ENV} == 1 ]]; then
  [[ -f .env ]] || { echo "本地没有 .env"; exit 1; }
  echo "==> 上传 .env（权限 600，仅 mm 可读）"
  "${SSH[@]}" "${HOST}" 'cat > /opt/money-machine/app/.env && chmod 600 /opt/money-machine/app/.env' < .env
fi
"${SSH[@]}" "${HOST}" 'chown -R mm:mm /opt/money-machine'

echo "==> 安装 uv 和依赖（装在 /opt/money-machine 下，首次需要 1~3 分钟）"
"${SSH[@]}" "${HOST}" 'bash -s' <<'REMOTE'
set -e
su -s /bin/bash mm <<'AS_MM'
set -e
export HOME=/opt/money-machine
cd "${HOME}/app"
if [ ! -x "${HOME}/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="${HOME}/.local/bin" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh
fi
"${HOME}/.local/bin/uv" sync --frozen --no-dev
.venv/bin/mm --help >/dev/null && echo "依赖安装完成"
# LightGBM 需要 OpenMP 运行库 libgomp.so.1。系统没有的话，不用 apt 装，
# 而是链接 PyTorch 自带的那份到 /opt/money-machine/lib（两个库共用一份运行库，也避免死锁）。
mkdir -p "${HOME}/lib"
if ! ldconfig -p 2>/dev/null | grep -q 'libgomp.so.1 '; then
  g=$(ls .venv/lib/python3.12/site-packages/torch/lib/libgomp*.so* 2>/dev/null | head -1)
  if [ -n "${g}" ]; then ln -sf "$(pwd)/${g}" "${HOME}/lib/libgomp.so.1" && echo "libgomp: 使用 PyTorch 自带的 ${g##*/}"; fi
fi
export LD_LIBRARY_PATH="${HOME}/lib"
.venv/bin/python -c "import lightgbm, cvxpy, torch; print('lightgbm', lightgbm.__version__, '| torch', torch.__version__)"
# 预先下载 Kronos 权重（约 120MB），免得第一次决策时才下载
.venv/bin/python -c "import sys; sys.path.insert(0, 'vendor/github/Kronos'); from model import Kronos, KronosTokenizer; KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base'); Kronos.from_pretrained('NeoQuasar/Kronos-small'); print('Kronos 权重就绪')" 2>&1 | grep -v -i warn | tail -1
AS_MM
REMOTE

echo "==> 写入并启动 systemd 服务"
"${SSH[@]}" "${HOST}" "CONFIG='${CONFIG}' COLLECT_ONLY=${COLLECT_ONLY} bash -s" <<'REMOTE'
set -e
unit() {  # name description exec memory cpu
cat > /etc/systemd/system/$1.service <<UNIT
[Unit]
Description=$2
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=mm
Group=mm
WorkingDirectory=/opt/money-machine/app
Environment=PYTHONUNBUFFERED=1
Environment=HOME=/opt/money-machine
Environment=LD_LIBRARY_PATH=/opt/money-machine/lib
ExecStart=$3
Restart=always
RestartSec=30
TimeoutStopSec=60
Nice=10
CPUQuota=$5
MemoryMax=$4
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/opt/money-machine
PrivateTmp=true

[Install]
WantedBy=multi-user.target
UNIT
}
unit money-machine-collect "money-machine forward data collection (spreads + open interest)" \
     "/opt/money-machine/app/.venv/bin/mm ml collect" 300M 25%
SERVICES="money-machine-collect"
if [ "${COLLECT_ONLY}" != 1 ]; then
  unit money-machine-paper "money-machine paper trading (${CONFIG}, virtual wallets, OKX public data)" \
       "/opt/money-machine/app/.venv/bin/mm paper --config ${CONFIG} --resume latest" 1800M 100%
  # v4 E2：只观察、不下单的影子挂单测量（读模拟盘账本，只读）
  unit money-machine-execlab "money-machine shadow post-only orders (observation only)" \
       "/opt/money-machine/app/.venv/bin/mm execlab --config ${CONFIG}" 300M 15%
  SERVICES="${SERVICES} money-machine-paper money-machine-execlab"
fi
install -o root -g root -m 755 /opt/money-machine/app/scripts/view_on_server.sh /opt/money-machine/view.sh
# 监控：每 5 分钟检查一次，有问题推送提醒（推送通道在 /opt/money-machine/app/.env 里配置）
if [ "${COLLECT_ONLY}" != 1 ]; then
cat > /etc/systemd/system/money-machine-monitor.service <<UNIT
[Unit]
Description=money-machine health checks + alerts

[Service]
Type=oneshot
User=mm
Group=mm
WorkingDirectory=/opt/money-machine/app
Environment=HOME=/opt/money-machine
Environment=LD_LIBRARY_PATH=/opt/money-machine/lib
ExecStart=/opt/money-machine/app/.venv/bin/mm monitor --config ${CONFIG}
Nice=10
MemoryMax=300M
UNIT
cat > /etc/systemd/system/money-machine-monitor.timer <<UNIT
[Unit]
Description=run money-machine health checks every 5 minutes

[Timer]
OnBootSec=3min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now money-machine-monitor.timer >/dev/null
fi
systemctl daemon-reload
for svc in ${SERVICES}; do
  systemctl enable ${svc}.service >/dev/null
  systemctl restart ${svc}.service
done
sleep 10
for svc in ${SERVICES}; do
  systemctl --no-pager --lines=0 status ${svc}.service | head -4
  echo "---- ${svc} 最近日志 ----"
  journalctl -u ${svc}.service -n 10 --no-pager -o cat
done
REMOTE

echo
echo "部署完成。常用命令："
echo "  scripts/server.sh ${HOST} status    # 运行状态"
echo "  scripts/server.sh ${HOST} report    # 各策略表现 + 放行检查"
echo "  scripts/server.sh ${HOST} logs      # 实时日志（Ctrl+C 退出）"
echo "  scripts/server.sh ${HOST} pull      # 把账本拉回本地"
