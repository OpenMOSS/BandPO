#!/usr/bin/env bash
set -euo pipefail

project_name="RLtraining_DapoMathDataset_DeepseekR1_1.5B"
steps=(125 135 145 165)
exps=("dapo_official")
BASE=$BandPODir/data
# CKPTS=${BASE}/ckpts
CKPTS=/remote-home1/yli/Workspace/BandPO/data/ckpts
OUT=${BASE}/ckpt2hf
PYTHON=python
BACKEND=fsdp
USE_CPU="--use_cpu_initialization"      # CPU 初始化，避免占用 GPU
TRUST_RC="--trust_remote_code"          # 某些模型家族（如 Qwen2.5）建议打开；若不需要可设为空字符串：TRUST_RC=""

set -x

date_tag="$(date +%F_%H-%M-%S)"
log_dir="${OUT}/logs"
mkdir -p "${log_dir}"
log_file="${log_dir}/merge_${project_name}_${date_tag}.log"



for exp in "${exps[@]}"; do
  for step in "${steps[@]}"; do
    local_dir="${CKPTS}/${project_name}/${exp}/global_step_${step}/actor"
    target_dir="${OUT}/${project_name}/${exp}/global_step_${step}"

    # 1) 基本存在性检查
    if [[ ! -d "${local_dir}" ]]; then
      echo "[SKIP] local_dir 不存在：${local_dir}" | tee -a "${log_file}"
      continue
    fi
    mkdir -p "${target_dir}"

    # 2) 关键文件提示（不是硬性要求，但缺失时合并大概率失败）
    if [[ ! -f "${local_dir}/fsdp_config.json" ]]; then
      echo "[WARN] 缺少 fsdp_config.json：${local_dir}/fsdp_config.json ；旧 checkpoint 可能需要 legacy 合并脚本。" | tee -a "${log_file}"
    fi
    if [[ ! -f "${local_dir}/huggingface/config.json" ]]; then
      echo "[WARN] 缺少 huggingface/config.json：${local_dir}/huggingface/config.json ；请确认训练时已正确保存 HF 配置/分词器。" | tee -a "${log_file}"
    fi

    echo "=== MERGING: ${project_name}/${exp} @ step ${step} ===" | tee -a "${log_file}"

    # 3) CPU-only 执行（不占 GPU）
    CUDA_VISIBLE_DEVICES="" \
    ${PYTHON} -m verl.model_merger merge \
      --backend "${BACKEND}" \
      --local_dir  "${local_dir}" \
      --target_dir "${target_dir}" \
      ${USE_CPU} 2>&1 | tee -a "${log_file}"

    rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then
      echo "[FAIL] ${exp} step ${step} 合并失败（退出码 ${rc) }）→ 请检查上方日志" | tee -a "${log_file}"
    else
      echo "[OK]   ${exp} step ${step} 合并完成 → ${target_dir}" | tee -a "${log_file}"
    fi
  done
done

set +x
echo "全部处理完成。日志：${log_file}"
