#!/usr/bin/env bash
set -euo pipefail

: "${BandPODir:?未检测到 BandPODir。请先 source 你的 init.sh}"

# === 目录与文件名 ===
Dir="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df"
jsonl_name="sample_traces+2025-11-10_02-16-15.jsonl"
png_name="vis.png"
jsonl_path="$Dir/$jsonl_name"
png_path="$Dir/$png_name"

jitter=0.3 # 同一step水平散开的宽度
alpha=0.3 # 越小越容易透出下层
  # --agg mean \

python visualize_ce_jsonl.py \
  --jsonl "$jsonl_path" \
  --agg median \
  --correct-key reward_extra_infos.acc \
  --no-reward-sum-fallback \
  --grid-out "$png_path" \
  --jitter $jitter \
  --point-size 3 \
  --alpha $alpha