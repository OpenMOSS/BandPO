#!/usr/bin/env bash
set -euo pipefail
: "${BandPODir:?source 你的 init.sh}"

# kept_or_dropped="all"
# JSONL="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/sample_traces+2025-11-10_02-16-15.jsonl"
# OUT="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/vis_gibberish_tmp_$kept_or_dropped"
# python analyze_gibberish_jsonl.py \
#   --jsonl "$JSONL" \
#   --out-dir "$OUT" \
#   --agg mean \
#   --procs 64 \
#   --chunk-lines 20000 \
#   --correct-key reward_extra_infos.acc \
#   --jitter 0.25 \
#   --point-size 3 \
#   --alpha 0.25 \
#   --agg-point-size 1.8 \
#   --kept-or-dropped "$kept_or_dropped"

# kept_or_dropped="kept"
# JSONL="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/sample_traces+2025-11-10_02-16-15.jsonl"
# OUT="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/vis_gibberish_$kept_or_dropped"
# python analyze_gibberish_jsonl.py \
#   --jsonl "$JSONL" \
#   --out-dir "$OUT" \
#   --agg mean \
#   --procs 64 \
#   --chunk-lines 20000 \
#   --correct-key reward_extra_infos.acc \
#   --jitter 0.25 \
#   --point-size 3 \
#   --alpha 0.25 \
#   --agg-point-size 1.8 \
#   --kept-or-dropped "$kept_or_dropped"

# kept_or_dropped="dropped"
# JSONL="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/sample_traces+2025-11-10_02-16-15.jsonl"
# OUT="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/vis_gibberish_$kept_or_dropped"
# python analyze_gibberish_jsonl.py \
#   --jsonl "$JSONL" \
#   --out-dir "$OUT" \
#   --agg mean \
#   --procs 64 \
#   --chunk-lines 20000 \
#   --correct-key reward_extra_infos.acc \
#   --jitter 0.25 \
#   --point-size 3 \
#   --alpha 0.25 \
#   --agg-point-size 1.8 \
#   --kept-or-dropped "$kept_or_dropped"




# kept_or_dropped="kept"
# JSONL="/remote-home1/bwang/workspace_yli/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/selector_gibberish/ce_ref_seq_mean13any.jsonl"
# OUT="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/vis_gibberish_ce_ref_seq_mean13any_$kept_or_dropped"
# python analyze_gibberish_jsonl.py \
#   --jsonl "$JSONL" \
#   --out-dir "$OUT" \
#   --agg mean \
#   --procs 64 \
#   --chunk-lines 20000 \
#   --correct-key reward_extra_infos.acc \
#   --jitter 0.25 \
#   --point-size 3 \
#   --alpha 0.25 \
#   --agg-point-size 1.8 \
#   --kept-or-dropped "$kept_or_dropped"