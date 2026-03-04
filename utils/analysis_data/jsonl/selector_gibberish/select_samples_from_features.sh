JSONL="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/sample_traces+2025-11-10_02-16-15.jsonl"
OUT="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/selector_gibberish"
OUT_jsonl="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/selector_gibberish_language/control_private_unassigned_ratio_015_kept_correct.jsonl"
python select_samples_from_features.py \
  --out-dir "$OUT" \
  --out-jsonl "$OUT_jsonl" \
  --cls correct \
  --kept-or-dropped "kept" \
  --chinese-or-english all \
  --where "control_private_unassigned_ratio>=0.15" \
  --keys all

  # --keys "response_text"

#   --keys all
#   --cls any # correct wrong no_judge \
#   --step-min 280 --step-max 330 \
  # --cls correct \
  # --cls any \

# ce_ref_seq_mean