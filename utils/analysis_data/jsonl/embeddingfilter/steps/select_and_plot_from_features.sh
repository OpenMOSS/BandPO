FeaturesDir="/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/build_features_steps_sample010"
OUT="/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/build_features_steps_sample010/filter_num_self_reflect_sentences40"

python select_and_plot_from_features.py \
  --features-dir "$FeaturesDir" \
  --out-jsonl "$OUT/filtered_samples.jsonl" \
  --plot-dir "$OUT/plot_dir" \
  --cls correct \
  --kept-or-dropped kept \
  --chinese-or-english all \
  --where "num_self_reflect_sentences>40" \
  --metrics all \
  --agg mean \
  --jitter 0.25 \
  --point-size 3 \
  --alpha 0.25

  # --cls any \
  # --kept-or-dropped all \
# repeat_word_run_max>20
  # --where "num_self_reflect_sentences>=30 and num_self_reflect_sentences<40 and wait_cluster_size_max>=4" \
    # --where "num_self_reflect_sentences>30 and num_self_reflect_sentences<40 and wait_cluster_size_max>=4" \
    # marker_loopiness>20 and 

# marker_loopiness
# num_self_reflect_sentences
