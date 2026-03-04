#!/usr/bin/env bash
: "${BandPODir:?未检测到 BandPODir。请先在 bash 中执行：source /path_to_BandPO/init.sh}"

set -xeuo pipefail
TS="${1:-$(date +'%Y-%m-%d_%H-%M-%S')}"

dapo_train=$BandPODir/data/dataset/dapo/dapo-math-17k.parquet
dapo_train_processed=$BandPODir/data/dataset/dapo_processed/modified/DAPO-Math-17k-Processed_modified.parquet
math_500_l35_train=$BandPODir/data/dataset/math-500/train/train_L3-5.parquet
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
RUNTIME_ENV="$BandPODir/RLtraining/verl/recipe/dapo/runtime_env.yaml"

# to modify
project_name="RLtraining_DapoMathDataset_DeepseekR1_1.5B"
exp_name="grpo_plus_ray_trgppo01_tokenmean_df" # no dual-clip, no clip higher
model_path="$BandPODir/data/models/deepseek/R1/DeepSeek-R1-Distill-Qwen-1.5B"

max_prompt_length=$((1024 * 1)) # 1024*1时会有部分overlong被filter，DAPO设置了1024 * 2
max_response_length_real=$((1024 * 4)) # 没有overlong_buffer_len的要设置为这个
overlong_buffer_len=$((512 * 1))
max_response_length=$((max_response_length_real)) # max_response_length=$((max_response_length_real + overlong_buffer_len))
# actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
# infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$((max_prompt_length + max_response_length))
train_batch_size=256
ppo_mini_batch_size=64
ppo_micro_batch_size=8
tensor_model_parallel_size=1 #  2->1
ulysses_sequence_parallel_size=1
gen_prompt_bsz=$((train_batch_size * 1))
max_num_gen_batches=100
rollout_n=16
n_gpus_per_node=8
validate_n=32
test_freq=5
save_freq=5
total_epochs=8 # 10
val_before_train=True
clip_ratio_high=0.28 # Modify but 无效
CKPTS_DIR="$BandPODir/data/ckpts/${project_name}/${exp_name}"
loss_agg_mode="token-mean" # "seq-mean-token-mean" 
rollout_record_path="$BandPODir/data/records/${project_name}/${exp_name}/sample_traces+${TS}.jsonl"
mkdir -p "$(dirname "$rollout_record_path")"
: > "$rollout_record_path"
rollout_record_detail=false

# 和trgppo相比
# 新增：
#     data.gen_batch_size=${gen_prompt_bsz} \（必须有）
#     algorithm.filter_groups.enable=True \（必须有）
#     algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \（必须有）
#     algorithm.filter_groups.metric=acc \（必须有）
#     二选一环节：
#         官方推荐：
#             actor_rollout_ref.actor.use_dynamic_bsz=True \（建议有）
#             actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \（建议有）
#             actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \（建议有）
#             actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \（建议有）
#             actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \（建议有）
#             actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \（建议有）
#         兼容老版：
#             actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
#             actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
#             actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
#     actor_rollout_ref.actor.optim.lr_warmup_steps=10 \（暂时不要，学习率预热步数。训练开始时将 LR 从 lr_warmup_init（默认 0）线性升到设定的 lr，持续 n 个 step。若一开始出现 loss/KL/梯度“抖大、炸值”，把 warmup 步数适当加大）
#     actor_rollout_ref.actor.optim.weight_decay=0.1 \（暂时不要，权重衰减（AdamW 的 decoupled L2 正则），抑制过大的权值，缓解过拟合与“漂移”。在 LLM RL 微调里常用 0.01～0.1；文档示例给到 0.1。）
#     actor_rollout_ref.actor.grad_clip=1.0 \（与默认值相同，无影响）
#     algorithm.kl_ctrl.kl_coef=0.0 \(无kl_penalty，无影响)
#     actor_rollout_ref.rollout.enable_chunked_prefill=True \（vllm的提高速度开关，暂时不要）
#     actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \(与默认值相同，无影响)
#     reward_model.reward_manager=dapo \（暂时不要）
#     reward_model.overlong_buffer.enable=True \（暂时不要）
#     reward_model.overlong_buffer.len=${overlong_buffer_len} \（暂时不要）
#     reward_model.overlong_buffer.penalty_factor=1.0 \（暂时不要）
# 和dapo相比
# 值不同：
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
# 新增：
#     +actor_rollout_ref.actor.use_tokenwise_ratio_bounds=True \
#     +actor_rollout_ref.actor.tokenwise_ratio_bounds_method="trgppo" \
#     +actor_rollout_ref.actor.does_relax_high_p_bound=False \
#     +actor_rollout_ref.actor.rmax=10.0 \
#     +actor_rollout_ref.actor.trgppo_solve_method="bisect" \
#     +actor_rollout_ref.actor.trgppo_use_log_domain=False \
#     +actor_rollout_ref.actor.kl_delta=0.1 \
#     trainer.critic_warmup=0 \(与默认值相同，无影响)


ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "$BandPODir/RLtraining/verl" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="['$dapo_train_processed', '$math_500_l35_train']" \
    data.val_files="['$aime2024_test', '$aime2025_test', '$amc2023_test']" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_batch_size} \
    algorithm.adv_estimator='grpo' \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=acc \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.actor.use_tokenwise_ratio_bounds=True \
    +actor_rollout_ref.actor.tokenwise_ratio_bounds_method="trgppo" \
    +actor_rollout_ref.actor.does_relax_high_p_bound=False \
    +actor_rollout_ref.actor.rmax=10.0 \
    +actor_rollout_ref.actor.trgppo_solve_method="bisect" \
    +actor_rollout_ref.actor.trgppo_use_log_domain=False \
    +actor_rollout_ref.actor.kl_delta=0.1 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1  \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${validate_n} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
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
    +trainer.rollout_record_path="${rollout_record_path}" \
    +trainer.rollout_record_detail=${rollout_record_detail} \
    actor_rollout_ref.rollout.calculate_log_probs=${rollout_record_detail} \
    trainer.resume_mode=auto
