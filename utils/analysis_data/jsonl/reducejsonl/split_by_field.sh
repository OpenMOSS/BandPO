JSONL="/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/group/group24.jsonl"
OUT_DIR="/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/group/group24"

python split_by_field.py \
  --jsonl "$JSONL" \
  --out-dir "$OUT_DIR" \
  --field-key filter_state
