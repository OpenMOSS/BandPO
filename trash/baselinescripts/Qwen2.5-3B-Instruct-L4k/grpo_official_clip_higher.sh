set -x
: "${BandPODir:?未检测到 BandPODir。请先在 bash 中执行：source /path_to_BandPO/init.sh}"
: "${BandPODir_LargeData:?未检测到 BandPODir_LargeData。请先在 bash 中执行：source /path_to_BandPO/init.sh}"

set -xeuo pipefail
TS="${1:-$(date +'%Y-%m-%d_%H-%M-%S')}"

export RAY_ADDRESS=local    # ray.init() 将启动本地实例
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

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

# to modify
project_name="VERL_DapoMath_Qwen2.5Instruct3B"
exp_name="grpo_official_clip_higher"
model_path="$BandPODir/data/models/Qwen/Qwen2.5-3B-Instruct"

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
save_freq=10
total_epochs=10
val_before_train=True
clip_ratio_high=0.28
CKPTS_DIR="$BandPODir_LargeData/ckpts/${project_name}/${exp_name}"
loss_agg_mode="seq-mean-token-mean"
norm_adv_by_std_in_grpo=true
records_dir="$BandPODir_LargeData/records/${project_name}/${exp_name}"
mkdir -p "$records_dir"
rollout_record_path="${records_dir}/sample_traces+${TS}.jsonl"
: > "$rollout_record_path"   # 清空/创建空文件
rollout_record_detail=false  # 可改为 true
validate_record_path="${records_dir}/validate_samples+${TS}.jsonl"
: > "$validate_record_path"   # 清空/创建空文件（或者你也可以去掉这一行，改成只在 Python 里 append）
# resume_from_path="/remote-home1/bwang/workspace_yli/BandPO/data/ckpts/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo005_df/global_step_600"
# del_local_ckpt_after_load=false
resume_mode=auto # resume_path
    # trainer.resume_from_path="${resume_from_path}" \
    # trainer.del_local_ckpt_after_load="${del_local_ckpt_after_load}" \
export WANDB_MODE=offline
WANDB_DIR="$BandPODir_LargeData/wandb/${project_name}/${exp_name}/${TS}"
export WANDB_DIR

python3 -m verl.trainer.main_ppo \
    data.train_files="['$dapo_train_processed', '$math_500_l35_train']" \
    data.val_files="['$aime2024_test', '$aime2025_test', '$amc2023_test']" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    algorithm.adv_estimator='grpo' \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
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
    +trainer.validate_record_path="${validate_record_path}" \
    trainer.resume_mode="${resume_mode}" $@
