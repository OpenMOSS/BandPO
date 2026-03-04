#!/usr/bin/env bash
set -xeuo pipefail
# 不要在ray的脚本中定义环境变量，需要在env的yaml文件写入
# 默认prompt_key=prompt，dapo原始数据集使用的就是prompt
# 但是dapo-processed使用的是source_prompt！！！！！！！！！可以考虑prompt_key=source_prompt
# 又但是aime使用的是默认prompt_key=prompt，最终只能modify dapo-processed数据集为prompt_key=prompt
# 数据集的data_source决定了rule-based reward返回type是dict还是float，但是整个训练中只能使用其中一种type（可能包含多种data_source），否则会导致/verl/trainer/ppo/ray_trainer.py中的assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}断言报错
# dapo按照gen_prompt_bsz切分数据集，通常要多个gen_prompt_bsz才能达到train_batch_size大小（被dynamic filting了很多）
# 这导致很多问题都被浪费了，因此需要更多轮epoch来充分使用数据集训练，把那些曾经太难的数据集全错的题开始逐渐做对
# 多个slurm节点跑独立的ray实验，需要在每个节点起ray node
RUNTIME_ENV="/remote-home1/yli/Workspace/BandPO/RLtraining/verl/recipe/dapo/runtime_env.yaml"
project_name="test_DAPO"
exp_name="judge_policy_loss_func"
CKPTS_DIR="/remote-home1/yli/Workspace/BandPO/data/ckpts/${project_name}/${exp_name}"
# top_k 0 for HF rollout, -1 for vLLM rollout
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 5))
overlong_buffer_len=$((1024 * 1))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$((max_prompt_length + max_response_length))
train_batch_size=256
gen_prompt_bsz=$((train_batch_size * 3))
max_num_gen_batches=10

gsm8k_train=/remote-home1/yli/Workspace/BandPO/data/datasets/gsm8k/train.parquet
dapo_train=/remote-home1/yli/Workspace/BandPO/data/datasets/dapo/dapo-math-17k.parquet
dapo_train_processed=/remote-home1/yli/Workspace/BandPO/data/datasets/dapo_processed/modified/DAPO-Math-17k-Processed_modified.parquet
math_train=/remote-home1/yli/Workspace/BandPO/data/datasets/math/train/train.parquet
math_500_l35_train=/remote-home1/yli/Workspace/BandPO/data/datasets/math-500/train/train_L3-5.parquet
math_500_train=/remote-home1/yli/Workspace/BandPO/data/datasets/math-500/train/train.parquet
math_500_test=/remote-home1/yli/Workspace/BandPO/data/datasets/math-500/test/test.parquet
aime_2024_test=/remote-home1/yli/Workspace/BandPO/data/datasets/aime2024/aime-2024.parquet

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "/remote-home1/yli/Workspace/BandPO/RLtraining/verl" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="['$dapo_train_processed', '$math_500_train']" \
    data.val_files="['$aime_2024_test', '$math_500_test']" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_batch_size} \
    actor_rollout_ref.rollout.n=4 \
    algorithm.adv_estimator='grpo' \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=acc \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="/remote-home1/yli/Workspace/BandPO/data/models/qwen25/base/Qwen2.5-3B" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1  \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=16 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
