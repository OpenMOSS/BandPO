  python select_group.py \
    --jsonl "/remote-home1/bwang/workspace_yli/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/sample_traces+2025-11-10_02-16-15.jsonl" \
    --out-jsonl "/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/group/group24.jsonl" \
    --group-key "global_step" \
    --group-values "24"