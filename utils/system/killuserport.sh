#!/usr/bin/env bash
set -euo pipefail

CURRENT_USER="$(whoami)"

echo "== 当前用户($CURRENT_USER)占用的监听端口 =="
lsof -i -P -n | awk -v user="$CURRENT_USER" '
NR==1 {next}
$3 == user && /LISTEN/ {
  printf "USER=%s PID=%s CMD=%s PORT=%s\n", $3, $2, $1, $9
}
'

echo
read -p "是否杀掉上述所有由用户($CURRENT_USER)启动的监听进程？输入 YES 确认: " answer
if [[ "$answer" != "YES" ]]; then
  echo "已取消操作。"
  exit 0
fi

# 获取当前用户所有监听端口的 PID（去重）
PIDS=$(lsof -i -P -n | awk -v user="$CURRENT_USER" '
NR==1 {next}
$3 == user && /LISTEN/ {print $2}
' | sort -u)

if [[ -z "${PIDS}" ]]; then
  echo "没有发现需要清理的进程。"
  exit 0
fi

echo "开始杀进程: $PIDS"
for pid in $PIDS; do
  cmd=$(ps -p "$pid" -o comm= || echo "unknown")
  echo "杀掉 PID=$pid CMD=$cmd"
  kill "$pid" 2>/dev/null || sudo kill "$pid" 2>/dev/null || echo "无法杀死 $pid"
done

echo "完成。"
