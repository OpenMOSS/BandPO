FeaturesDir="/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/build_features_steps_sample010"
OUT="/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/build_features_steps_sample010/filter_test"

python filter.py \
  --features-dir "$FeaturesDir" \
  --out-jsonl "$OUT/filtered_samples.jsonl" \
  --plot-dir "$OUT/plot_dir" \
  --cls correct \
  --kept-or-dropped kept \
  --chinese-or-english all \
  --metrics all \
  --agg mean \
  --jitter 0.25 \
  --point-size 3 \
  --pattern-workers 120 \
  --alpha 0.25