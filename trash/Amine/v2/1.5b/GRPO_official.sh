set -x
: "${BandPODir:?未检测到 BandPODir。请先在 bash 中执行：source /path_to_BandPO/init.sh}"

export RAY_ADDRESS=local    # ray.init() 将启动本地实例
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
# export WANDB_MODE=offline
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

dapo_train=$BandPODir/data/dataset/dapo/dapo-math-17k.parquet
dapo_train_processed=$BandPODir/data/dataset/dapo_processed/modified/DAPO-Math-17k-Processed_modified.parquet
dapo_train_processed_grpo=$BandPODir/data/dataset/dapo_processed/modified_math/DAPO-Math-17k-Processed_modified_modified.parquet
math_500_l35_train=$BandPODir/data/dataset/math-500/train/train_L3-5.parquet
math_500_l35_train_grpo=$BandPODir/data/dataset/math-500_math/train/train_L3-5.parquet
gsm8k_test=$BandPODir/data/dataset/gsm8k/verl2dapo/test.parquet
math_500_test=$BandPODir/data/dataset/math-500/test/test.parquet
math_test=$BandPODir/data/dataset/math/verl2dapo/test/test.parquet
aime2024_test=$BandPODir/data/dataset/aime2024_dapo/test.parquet
aime2024_16_test=$BandPODir/data/dataset/aime2024_dapo/test_16_shuffle24.parquet
aime2025_test=$BandPODir/data/dataset/aime2025_dapo/test.parquet
aime2025_16_test=$BandPODir/data/dataset/aime2025_dapo/test_16_shuffle24.parquet
amc2023_test=$BandPODir/data/dataset/amc2023_dapo/test.parquet
amc2023_16_test=$BandPODir/data/dataset/amc2023_dapo/test_16_shuffle24.parquet
minerva_test=$BandPODir/data/dataset/minerva_dapo/test.parquet
olympiad_test=$BandPODir/data/dataset/olympiad_dapo/OE_TO_maths_en_COMP/test.parquet

# to modify
project_name="clip_improve_v3"
exp_name="DapoMathDataset_DeepseekR1_1.5B_grpo_official" # no dual-clip, no clip higher
model_path="$BandPODir/data/models/deepseek/R1/DeepSeek-R1-Distill-Qwen-1.5B"

max_prompt_length=$((1024 * 2)) # 1024*1时会有部分overlong被filter，DAPO设置了1024 * 2
max_response_length_real=$((1024 * 8)) # 没有overlong_buffer_len的要设置为这个
overlong_buffer_len=$((1024 * 1))
max_response_length=$((max_response_length_real)) # max_response_length=$((max_response_length_real + overlong_buffer_len))
# actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
# infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$((max_prompt_length + max_response_length))
train_batch_size=256
ppo_mini_batch_size=64
ppo_micro_batch_size=8
tensor_model_parallel_size=1 # actor_rollout_ref.rollout.tensor_model_parallel_size=2->1
# ulysses_sequence_parallel_size=1
# gen_prompt_bsz=$((train_batch_size * 1))
# max_num_gen_batches=100
rollout_n=16
n_gpus_per_node=8
validate_n=32
test_freq=5
save_freq=5
total_epochs=5 # 10
val_before_train=True
# clip_ratio_high=0.2
kl_delta=0.05
# actor_rollout_ref.rollout.gpu_memory_utilization=0.6->0.8
# actor_rollout_ref.actor.fsdp_config.param_offload=False->True
# actor_rollout_ref.actor.fsdp_config.optimizer_offload=False->True

CKPTS_DIR="$BandPODir/data/ckpts/${project_name}/${exp_name}"
debug_record_path="$BandPODir/data/records/${project_name}/${exp_name}/record.txt"
mkdir -p "$(dirname "$debug_record_path")"
: > "$debug_record_path"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$dapo_train_processed_grpo', '$math_500_l35_train_grpo']" \
    data.val_files="['$aime2024_test', '$aime2025_test', '$amc2023_test']" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    +actor_rollout_ref.actor.clip_use_dual_clip=True \
    +actor_rollout_ref.actor.use_tokenwise_ratio_bounds=False \
    +actor_rollout_ref.actor.tokenwise_ratio_bounds_method="trgppo" \
    +actor_rollout_ref.actor.does_relax_high_p_bound=True \
    +actor_rollout_ref.actor.rmax=10.0 \
    +actor_rollout_ref.actor.trgppo_solve_method="bisect" \
    +actor_rollout_ref.actor.trgppo_use_log_domain=False \
    +actor_rollout_ref.actor.kl_delta=${kl_delta} \
    +actor_rollout_ref.actor.use_soft_clip=False \
    +actor_rollout_ref.actor.soft_clip_methods="3seg" \
    +actor_rollout_ref.actor.use_decaying_clip=False \
    +actor_rollout_ref.actor.decaying_clip_methods="hold_linear" \
    +actor_rollout_ref.actor.decaying_clip_start_value=0.2 \
    +actor_rollout_ref.actor.decaying_clip_end_value=0.02 \
    +actor_rollout_ref.actor.delta_mapping_ratio_method="sub" \
    +actor_rollout_ref.actor.use_one_minus_p_for_mapping_in_div=True \
    +actor_rollout_ref.actor.mapping_eps=1e-12 \
    +actor_rollout_ref.actor.mapping_ratio_min_cap=1e-6 \
    +actor_rollout_ref.actor.mapping_ratio_max_cap=1e6 \
    +actor_rollout_ref.actor.is_debug=True \
    +actor_rollout_ref.actor.debug_record_path="${debug_record_path}" \
    +actor_rollout_ref.actor.debug_clip_print_max_pairs=256 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1  \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${validate_n} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto $@
