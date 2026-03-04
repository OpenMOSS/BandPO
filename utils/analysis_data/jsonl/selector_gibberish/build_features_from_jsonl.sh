JSONL="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/sample_traces+2025-11-10_02-16-15.jsonl"
OUT="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/selector_gibberish_language"
python build_features_from_jsonl.py \
  --jsonl "$JSONL" \
  --out-dir "$OUT" \
  --procs 64 \
  --chunk-lines 20000 \
  --correct-key "reward_extra_infos.acc" \
  --snapshot-keys "epoch,global_step,uid,filter_state,prompt_text,response_text,prompt_len_tokens,response_len_tokens,total_len_tokens,reward_seq_sum,reward_extra_infos.acc"
