# python sample_by_group.py \
#   --jsonl  /remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/split_by_filter_state/sample_traces.kept.jsonl \
#   --out-jsonl /remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/split_by_filter_state/sample_traces_kept_sampled_020percent.jsonl \
#   --group-key "global_step" \
#   --sample-frac 0.20 \
#   --round-mode round \
#   --min-per-group 1 \
#   --seed 42 \
#   --print-every 200000

python sample_by_group.py \
  --jsonl  "/remote-home1/bwang/workspace_yli/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/sample_traces+2025-11-10_02-16-15.jsonl" \
  --out-jsonl /remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/split_by_filter_state/sampled_0001percent.jsonl \
  --group-key "global_step" \
  --sample-frac 0.001 \
  --round-mode round \
  --min-per-group 1 \
  --seed 42 \
  --print-every 200000