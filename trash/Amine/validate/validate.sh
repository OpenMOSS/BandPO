set -x
export RAY_ADDRESS=local    # ray.init() 将启动本地实例
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="0"


export WANDB_DIR=/remote-home1/yli/Workspace/BandPO/data/test/data/logs/wandb_runs
export WANDB_ARTIFACT_DIR=/remote-home1/yli/Workspace/BandPO/data/test/data/logs/wandb_artifacts
export WANDB_CACHE_DIR=/remote-home1/yli/Workspace/BandPO/data/test/data/tmp/wandb_cache

project_name="test_validate"
exp_name_prefix="qwen2.5-math-1.5b"
# model_path=/remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo/global_step_64
model_path=/remote-home1/yli/Workspace/BandPO/data/models/qwen25math/instruct/Qwen2.5-Math-1.5B-Instruct

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
datasets=("gsm8k_test" "math_500_test" "math_test" "aime2024_test" "aime2025_test" "amc2023_test")
datasets=("aime2025_test")

max_prompt_length=$((1024 * 1))
# max_response_length=$((1024 * 2))
max_response_length=$((1024 * 3))
        # actor_rollout_ref.actor.optim.lr=1e-6 \
        # actor_rollout_ref.actor.use_kl_loss=True \
        # actor_rollout_ref.actor.kl_loss_coef=0.001 \
        # actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        # algorithm.use_kl_in_reward=False \
        # trainer.critic_warmup=0 \

for dataset in "${datasets[@]}"; do
    if [[ "$dataset" == "math_500_test" ]]; then
        rollout_n=4
        echo "运行数学数据集: $dataset (rollout_n=4)"
    elif [[ "$dataset" == "math_test" ]]; then
        rollout_n=1
        echo "运行数学数据集: $dataset (rollout_n=1)"
    else
        rollout_n=64
        echo "运行数学数据集: $dataset (rollout_n=64)"
    fi
    dataset_path=${!dataset}
    exp_name="${exp_name_prefix}_${dataset}_n${rollout_n}"
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="['$math_500_l35_train']" \
        data.val_files="['$dataset_path']" \
        data.train_batch_size=1 \
        data.prompt_key=prompt \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.filter_overlong_prompts=True \
        actor_rollout_ref.model.path="$model_path" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=1 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_kl_loss=False \
        algorithm.use_kl_in_reward=False \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=${rollout_n} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.861 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        trainer.logger='["console","wandb"]' \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.total_epochs=1 $@
    
    echo "完成数据集: $dataset"
    echo "---"
done

echo "完成所有数据集评测！"