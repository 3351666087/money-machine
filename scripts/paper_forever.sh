#!/bin/zsh
# 长期运行模拟盘：防止 Mac 空闲休眠，进程意外退出后 30 秒自动续跑同一个 run。
# 停止：连按两次 Ctrl+C。注意合上笔记本盖子仍会休眠，长期跑建议放在常开的机器上。
cd "${0:A:h}/.."
CONFIG="${1:-configs/v1.toml}"
while true; do
  caffeinate -i uv run mm paper --config "${CONFIG}" --resume latest
  echo "$(date '+%F %T') paper exited ($?) — restarting in 30s (Ctrl+C again to quit)"
  sleep 30
done
