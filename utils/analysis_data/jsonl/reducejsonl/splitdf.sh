#!/usr/bin/env bash
set -euo pipefail

JSONL="/remote-home1/bwang/workspace_yli/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/sample_traces+2025-11-10_02-16-15.jsonl"

OUT_DIR="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/split_by_filter_state"
mkdir -p "$OUT_DIR"

python splitdf.py \
  --jsonl "$JSONL" \
  --out-kept "$OUT_DIR/sample_traces.kept.jsonl" \
  --out-dropped "$OUT_DIR/sample_traces.dropped.jsonl" \
  --print-every 100000
