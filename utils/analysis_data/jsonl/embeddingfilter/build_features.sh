JSONL="/remote-home1/bwang/workspace_yli/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/sample_traces+2025-11-10_02-16-15.jsonl"
OUT="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/build_features"

python build_features.py \
  --jsonl "$JSONL" \
  --out-dir "$OUT" \
  --procs 8 \
  --chunk-lines 20000 \
  --correct-key "reward_extra_infos.acc" \
  --snapshot-keys "epoch,global_step,uid,filter_state,prompt_text,response_text,prompt_len_tokens,response_len_tokens,total_len_tokens,reward_seq_sum,reward_extra_infos.acc" \
  --embed-model "/remote-home1/yli/Workspace/BandPO/data/models/BAAI/bge-m3" \
  --embed-backend vllm \
  --embed-device "cuda:0" \
  --embed-batch-size 64 \
  --max-sentences-for-embedding 128
