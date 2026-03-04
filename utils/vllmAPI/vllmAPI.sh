#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/remote-home1/yli/Workspace/DiagVerse/data/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0}"                     # 默认只用 0 号卡；支持 "all" 或 "0,2" 这种列表
TP_OVERRIDE="${TP_OVERRIDE:-}"        # 可手工指定 TP，否则按选中的 GPU 数量
GPU_UTIL="${GPU_UTIL:-0.95}"
MAX_LEN="${MAX_LEN:-32768}"
# MAX_LEN="${MAX_LEN:-65536}"
# MAX_LEN="${MAX_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"     # batch=1
HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ">>> HOST: $HOST" # 10.176.58.103
echo ">>> PORT: $PORT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;       # 例：--gpus all 或 --gpus 0,1
    --tp) TP_OVERRIDE="$2"; shift 2 ;;
    --max-len) MAX_LEN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

command -v vllm >/dev/null 2>&1 || { echo "未找到 vLLM（命令 vllm）。请先安装 vllm"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi 不存在"; exit 1; }
gpu_total=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
if [[ "$gpu_total" -eq 0 ]]; then
  echo "未检测到可用 GPU"; exit 1
fi
# 解析 GPU 选择
if [[ "$GPUS" == "all" ]]; then
  # 选择所有 GPU
  if [[ "$gpu_total" -gt 0 ]]; then
    GPUS=$(seq -s, 0 $((gpu_total - 1)))
  else
    echo "没有可用 GPU"; exit 1
  fi
fi
# 计算 TP 大小
IFS=',' read -r -a _GPU_ARR <<< "$GPUS"
TP_SIZE="${TP_OVERRIDE:-${#_GPU_ARR[@]}}"
if [[ "$TP_SIZE" -lt 1 ]]; then TP_SIZE=1; fi
export CUDA_VISIBLE_DEVICES="$GPUS"
echo ">>> 使用 GPU: $CUDA_VISIBLE_DEVICES"
echo ">>> tensor-parallel-size: $TP_SIZE"

# 端口占用检查与自动处理
is_port_in_use() {
  # 优先用 ss，无则退回 lsof
  if command -v ss >/dev/null 2>&1; then
    ss -ltnH | awk '{print $4}' | grep -qE "(:|\\.)${1}\$"
  else
    command -v lsof >/dev/null 2>&1 && lsof -iTCP -sTCP:LISTEN -P -n | grep -q ":${1} "
  fi
}
if is_port_in_use "$PORT"; then
  echo "端口 $PORT 已被占用。"
  # 也可以选择直接退出：exit 1
  # 这里给你一个自动递增寻找空闲端口的策略（8000-8099）
  base_port="$PORT"
  found=0
  for p in $(seq "$base_port" 8099); do
    if ! is_port_in_use "$p"; then PORT="$p"; found=1; break; fi
  done
  if [[ "$found" -eq 1 ]]; then
    echo "自动切换到空闲端口：$PORT"
  else
    echo "未找到可用端口（$base_port-8099）"; exit 1
  fi
fi

vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --device cuda \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL"
