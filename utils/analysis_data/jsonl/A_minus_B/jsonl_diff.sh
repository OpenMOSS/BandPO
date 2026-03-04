AFile="/remote-home1/bwang/workspace_yli/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/selector_gibberish_language/repeat_word_run_max_25_kept_correct.jsonl"
BFile="/remote-home1/bwang/workspace_yli/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/selector_gibberish_language/repeat_word_run_max_30_kept_correct.jsonl"
CFile="/remote-home1/bwang/workspace_yli/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/selector_gibberish_language/repeat_word_run_max_25minux30_kept_correct.jsonl"
python jsonl_diff.py "$AFile" "$BFile" "$CFile" --key uid
