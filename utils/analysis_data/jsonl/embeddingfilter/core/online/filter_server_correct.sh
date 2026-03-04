#!/bin/bash

########################################
# 自动识别显卡（支持 strict / non-strict）
# 用法：
#   ./run.sh                # 默认 strict 模式
#   ./run.sh non-strict     # 非严格模式：自动剔除脏卡
########################################

# 模式：默认 strict，可以通过第一个参数改成 non-strict
MODE="${1:-strict}"   # strict / non-strict

MIN_FREE_PCT=80       # 要求空闲显存比例 >= 80%

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: 找不到 nvidia-smi，无法检测显卡信息" >&2
    exit 1
fi

# 1. 决定候选 GPU 列表：
#    - 如果已经有 CUDA_VISIBLE_DEVICES（比如 Slurm 设的），只检查这些
#    - 否则检查整机所有 GPU index
candidate_gpus=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a candidate_gpus <<< "${CUDA_VISIBLE_DEVICES}"
else
    mapfile -t candidate_gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi

# 清理一下空白和空字符串
tmp=()
for gid in "${candidate_gpus[@]}"; do
    gid="${gid//[[:space:]]/}"
    [ -n "${gid}" ] && tmp+=("${gid}")
done
candidate_gpus=("${tmp[@]}")

if [ "${#candidate_gpus[@]}" -eq 0 ]; then
    echo "没有检测到候选显卡" >&2
    exit 1
fi

echo "候选显卡 (来自 CUDA_VISIBLE_DEVICES 或全局): ${candidate_gpus[*]}"

# 2. 查询每块候选 GPU 的显存空闲比例
good_gpus=()
bad_gpus=()

while IFS=',' read -r idx total used; do
    idx="${idx//[[:space:]]/}"
    total="${total//[[:space:]]/}"
    used="${used//[[:space:]]/}"

    # 只处理候选列表里的 GPU
    in_candidates=false
    for gid in "${candidate_gpus[@]}"; do
        gid="${gid//[[:space:]]/}"
        if [ "${gid}" = "${idx}" ]; then
            in_candidates=true
            break
        fi
    done
    [ "${in_candidates}" = false ] && continue

    free=$(( total - used ))
    free_pct=$(( 100 * free / total ))

    if [ "${free_pct}" -ge "${MIN_FREE_PCT}" ]; then
        echo "GPU ${idx}: OK (空闲 ${free_pct}% = ${free}/${total} MiB)"
        good_gpus+=("${idx}")
    else
        echo "GPU ${idx}: BUSY (空闲 ${free_pct}% = ${free}/${total} MiB)" >&2
        bad_gpus+=("${idx}")
    fi
done < <(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits)

# 3. 根据模式决定最终使用哪些 GPU
final_gpus=()

if [ "${MODE}" = "strict" ]; then
    # strict：只要有一块候选 GPU 空闲率不达标就直接报错退出
    if [ "${#bad_gpus[@]}" -ne 0 ]; then
        echo "STRICT 模式：以下候选 GPU 空闲显存低于 ${MIN_FREE_PCT}% : ${bad_gpus[*]}" >&2
        exit 1
    fi
    final_gpus=("${candidate_gpus[@]}")
else
    # non-strict：只使用空闲率达标的 GPU
    if [ "${#good_gpus[@]}" -eq 0 ]; then
        echo "NON-STRICT 模式：没有任何显卡空闲显存 >= ${MIN_FREE_PCT}%，无法继续" >&2
        exit 1
    fi
    final_gpus=("${good_gpus[@]}")
fi

gpu_count=${#final_gpus[@]}
if [ "${gpu_count}" -le 0 ]; then
    echo "没有可用的显卡（final_gpus 为空）" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${final_gpus[*]}")
tensor_parallel_size=${gpu_count}

echo "最终使用的 CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "tensor-parallel-size: ${tensor_parallel_size}"
echo "gpu_count: ${gpu_count}"

export CUDA_VISIBLE_DEVICES

export TOKENIZERS_PARALLELISM=false

# 自动识别 host
host=$(hostname -I | awk '{print $1}')
echo "host: $host"

python filter_server.py \
  --host "$host" \
  --port 8000 \
  --correct-key "reward_extra_infos.acc" \
  --kept-or-dropped kept \
  --cls correct \
  --subset-mapping \
  --cpu-procs 8 \
  --chunk-lines 512 \
  --pattern-workers 16 \
  --embed-model-name "/remote-home1/yli/Workspace/BandPO/data/models/BAAI/bge-m3" \
  --embed-backend hf \
  --hf-device cuda \
  --hf-max-workers "$gpu_count" \
  --hf-batch-size 128 \
  --hf-max-length 1024 \
  --hf-torch-dtype bfloat16 \
  --trust-remote-code \
  --dup-threshold 0.90 \
  --cluster-threshold 0.85 \
  --plateau-min-len 3 \
  --max-sent-per-sample 128 \
  --rambling-weight-self 1.0 \
  --rambling-weight-dup 1.0 \
  --rambling-weight-loop 1.0 \
  --rambling-weight-plateau 1.0 \
  --rambling-weight-lang 0.1 \
  --rambling-loopiness-cap 5.0 \
  --rambling-plateau-k 4 \
  --debug \
  --log-level "info" \
  --debug-out-dir "/remote-home1/yli/Workspace/BandPO/data/records/rambling_filter_debug108/"
