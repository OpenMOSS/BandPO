#!/usr/bin/env bash
set -euo pipefail
set -x
# 统计整个脚本总用时（秒）
SECONDS=0
trap '{
  set +x
  h=$((SECONDS/3600))
  m=$(((SECONDS%3600)/60))
  s=$((SECONDS%60))
  printf "\n==> 总用时：%02d:%02d:%02d（%ds）\n" "$h" "$m" "$s" "$SECONDS"
}' EXIT

export RAY_ADDRESS=local    # ray.init() 将启动本地实例
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export WANDB_MODE=offline
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
validate_root_dir="/remote-home1/yli/Workspace/BandPO/data/validate"
dapo_train=/remote-home1/yli/Workspace/BandPO/data/dataset/dapo/dapo-math-17k.parquet
dapo_train_processed=/remote-home1/yli/Workspace/BandPO/data/dataset/dapo_processed/modified/DAPO-Math-17k-Processed_modified.parquet
math_500_l35_train=/remote-home1/yli/Workspace/BandPO/data/dataset/math-500/train/train_L3-5.parquet
gsm8k_test=/remote-home1/yli/Workspace/BandPO/data/dataset/gsm8k/verl2dapo/test.parquet
math_500_test=/remote-home1/yli/Workspace/BandPO/data/dataset/math-500/test/test.parquet
math_test=/remote-home1/yli/Workspace/BandPO/data/dataset/math/verl2dapo/test/test.parquet # 5000 for test (7500 for train)
aime2024_test=/remote-home1/yli/Workspace/BandPO/data/dataset/aime2024_dapo/test.parquet
aime2024_16_test=/remote-home1/yli/Workspace/BandPO/data/dataset/aime2024_dapo/test_16_shuffle24.parquet
aime2025_test=/remote-home1/yli/Workspace/BandPO/data/dataset/aime2025_dapo/test.parquet
aime2025_16_test=/remote-home1/yli/Workspace/BandPO/data/dataset/aime2025_dapo/test_16_shuffle24.parquet
amc2023_test=/remote-home1/yli/Workspace/BandPO/data/dataset/amc2023_dapo/test.parquet
amc2023_16_test=/remote-home1/yli/Workspace/BandPO/data/dataset/amc2023_dapo/test_16_shuffle24.parquet

model_list=(
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo/global_step_64
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo/global_step_120
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo/global_step_128
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo/global_step_176
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo/global_step_184
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo/global_step_203
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_64
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_120
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_128
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_176
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_184
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_203
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10/global_step_64
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10/global_step_120
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10/global_step_128
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10/global_step_176
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10/global_step_184
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10/global_step_203
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_64
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_120
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_128
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_176
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_184
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dcpo_on_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_204
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01/global_step_64
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01/global_step_120
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01/global_step_128
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01/global_step_176
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01/global_step_184
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01/global_step_204
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01_rmax10/global_step_64
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01_rmax10/global_step_120
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01_rmax10/global_step_128
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01_rmax10/global_step_176
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01_rmax10/global_step_184
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl01_rmax10/global_step_204
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10/global_step_64
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10/global_step_120
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10/global_step_128
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10/global_step_176
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10/global_step_184
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10/global_step_204
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho01/global_step_64
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho01/global_step_120
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho01/global_step_128
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho01/global_step_176
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho01/global_step_184
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho01/global_step_203
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho034/global_step_64    
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho034/global_step_120
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho034/global_step_128
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho034/global_step_176
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho034/global_step_184
    /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_trgppo_on_dapo_relax_bisect_nolog_kl005_rmax10_soft_clip_by_3seg_rho034/global_step_204
)
temp_list=(0 1.0)
# datasets=("gsm8k_test" "math_500_test" "math_test" "aime2024_test" "aime2025_test" "amc2023_test")
datasets=("math_500_test" "aime2024_test" "aime2025_test" "amc2023_test")
# datasets=("aime2025_test" "amc2023_test")

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 3))
date_tag="$(date +%F_%H-%M-%S)"
for model_path in "${model_list[@]}"; do
    [[ -d "$model_path" ]] || { echo "[SKIP] 模型目录不存在：$model_path"; continue; }
    model_path="${model_path%/}" # 去掉可能的结尾斜杠
    d3="$(basename "$model_path")"
    d2="$(basename "$(dirname "$model_path")")"
    d1="$(basename "$(dirname "$(dirname "$model_path")")")"
    last_three="${d1}/${d2}/${d3}" # 提取倒数 3 级目录名
    validate_dir="${validate_root_dir}/${last_three}" # 组合 validate_dir，并创建所需目录
    mkdir -p "${validate_dir}/logs/wandb_runs" \
            "${validate_dir}/logs/wandb_artifacts" \
            "${validate_dir}/tmp/wandb_cache" \
            "${validate_dir}/logs/console"
    export WANDB_DIR="${validate_dir}/logs/wandb_runs"
    export WANDB_ARTIFACT_DIR="${validate_dir}/logs/wandb_artifacts"
    export WANDB_CACHE_DIR="${validate_dir}/tmp/wandb_cache"
    console_log_dir="${validate_dir}/logs/console/${date_tag}" # 控制台日志按时间戳分文件保存
    mkdir -p "${console_log_dir}"
    project_name="validate_${d1}"
    exp_name_prefix="${d2}_${d3}"
    for temperature in "${temp_list[@]}"; do
        # 判断是否为 0 / 0.0 / 0.000 等
        if [[ "$temperature" =~ ^0+(\.0+)?$ ]]; then
            greedy=true
            echo "温度=${temperature} → 贪婪解码（do_sample=False，所有数据集 n=1）"
        else
            greedy=false
            echo "温度=${temperature} → 采样解码（do_sample=True，按数据集规则设定 n）"
        fi
        # 温度字符串用于命名（把小数点替换为 p）
        t_tag="t${temperature//./p}"
        for dataset in "${datasets[@]}"; do
            # 设定 rollout_n
            if $greedy; then
                rollout_n=1
                rule_note="greedy"
            else
                if [[ "$dataset" == "math_500_test" ]]; then
                rollout_n=2
                elif [[ "$dataset" == "math_test" ]]; then
                rollout_n=1
                else
                rollout_n=32
                fi
                rule_note="by-dataset"
            fi
            echo "运行数学数据集: $dataset (rollout_n=${rollout_n}, ${rule_note})"
            dataset_path=${!dataset}
            exp_name="${exp_name_prefix}_${dataset}_${t_tag}_n${rollout_n}"
            log_file="${console_log_dir}/${exp_name}.log"
            do_sample_flag=$($greedy && echo False || echo True)
            python3 -m verl.trainer.main_ppo \
                algorithm.adv_estimator=grpo \
                data.train_files="['$math_500_l35_train']" \
                data.val_files="['$dataset_path']" \
                data.train_batch_size=8 \
                data.prompt_key=prompt \
                data.max_prompt_length=${max_prompt_length} \
                data.max_response_length=${max_response_length} \
                data.filter_overlong_prompts=True \
                actor_rollout_ref.model.path="$model_path" \
                actor_rollout_ref.model.use_remove_padding=True \
                actor_rollout_ref.actor.ppo_mini_batch_size=8 \
                actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.actor.fsdp_config.param_offload=True \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\
                actor_rollout_ref.actor.use_kl_loss=False \
                algorithm.use_kl_in_reward=False \
                actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
                actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
                actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
                actor_rollout_ref.rollout.val_kwargs.do_sample=${do_sample_flag} \
                actor_rollout_ref.rollout.val_kwargs.n=${rollout_n} \
                actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.861 \
                actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.ref.fsdp_config.param_offload=True \
                actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
                trainer.logger='["console","wandb"]' \
                trainer.n_gpus_per_node=8 \
                trainer.nnodes=1 \
                trainer.val_before_train=True \
                trainer.val_only=True \
                trainer.project_name="${project_name}" \
                trainer.experiment_name="${exp_name}" \
                trainer.total_epochs=1 "$@" \
                2>&1 | tee -a "${log_file}"
            echo "完成数据集: $dataset @ 温度=${temperature}"
            echo "---"
        done
    done
    echo "完成模型评测：${d3}"
done
echo "全部模型评测完成。批次：${date_tag}"