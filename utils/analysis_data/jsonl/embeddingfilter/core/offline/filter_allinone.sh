JSONL="/remote-home1/bwang/workspace_yli/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/sample_traces+2025-11-10_02-16-15.jsonl"
JSONL="/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/group/group24/filter_state=kept.jsonl"
JSONL="/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/group/group24.jsonl"
OUT_DIR="/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/filter_allinone_test"
OUT_DIR="/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/group/group24/filter_allinone_test"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 一个stage 1的worker至少对应500个，否则多进程开销会很大
# 一个stage 3的worker至少对应300个，否则多进程开销会很大
python filter_allinone.py \
  --jsonl "$JSONL" \
  --out-dir "$OUT_DIR" \
  --kept-or-dropped kept \
  --cls correct \
  --cpu-procs 8 \
  --chunk-lines 512 \
  --pattern-workers 16 \
  --correct-key "reward_extra_infos.acc" \
  --embed-model-name "/remote-home1/yli/Workspace/BandPO/data/models/BAAI/bge-m3" \
  --embed-backend hf \
  --hf-max-workers 8 \
  --hf-device cuda \
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
  --debug

  # --tensor-parallel-size 1 \
  # --dtype "bfloat16" \
  # --gpu-memory-utilization 0.90 \
  # --enforce-eager \

  # --step-min 240 \
  # --step-max 240 \