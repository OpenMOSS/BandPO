#!/usr/bin/env bash
set -euo pipefail
# JSONL
#   └─(Stage1: build_features_preprocess.py)
#       ├─ features.parquet      # 样本级 + 原有语言指标 + 分段统计
#       ├─ segments.parquet      # 句子级 (text + marker + lang)
#       └─ meta_preprocess.json

#   └─(Stage2: build_features_embedding.py)
#       ├─ sentence_embeddings.parquet  # 句子级 (sample_idx,section_idx,sent_idx,emb)
#       └─ meta_embedding.json

#   └─(Stage3: build_features_analysis.py)
#       ├─ features_analysis.parquet    # 在 features.parquet 基础上加 embedding 分析指标
#       └─ meta_analysis.json

JSONL="/remote-home1/yli/Workspace/BandPO/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/split_by_filter_state/sampled_010percent.jsonl"
OUT_DIR="$BandPODir/data/records/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_tokenmean_df/build_features_steps_sample010"

EmbeddingModelPath="/remote-home1/yli/Workspace/BandPO/data/models/BAAI/bge-m3"

# # Stage1: 预处理 + 基础特征
echo "[RUN] Stage1: preprocess from JSONL -> features.parquet + segments.parquet"
python build_features_preprocess.py \
  --jsonl "$JSONL" \
  --out-dir "$OUT_DIR" \
  --procs 128 \
  --chunk-lines 20000 \
  --correct-key "reward_extra_infos.acc" \
  --snapshot-keys "epoch,global_step,uid,filter_state,prompt_text,response_text,prompt_len_tokens,response_len_tokens,total_len_tokens,reward_seq_sum,reward_extra_infos.acc"

# Stage2: sentence embedding (vLLM)
echo "[RUN] Stage2: compute sentence embeddings with vLLM"
python build_features_embedding_vllm.py \
  --segments-parquet "$OUT_DIR/segments.parquet" \
  --out-dir "$OUT_DIR" \
  --model-name "$EmbeddingModelPath" \
  --tensor-parallel-size 1 \
  --dtype "bfloat16" \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --enforce-eager

# Stage3: embedding-based 分析
echo "[RUN] Stage3: fast analysis with embeddings"
python build_features_analysis_fast.py \
  --preprocess-dir "$OUT_DIR" \
  --embed-dir "$OUT_DIR" \
  --out-dir "$OUT_DIR" \
  --procs 120 \
  --dup-threshold 0.90 \
  --cluster-threshold 0.85 \
  --plateau-min-len 3 \
  --max-sent-per-sample 128 \
  --rambling-weight-self 1.0 \
  --rambling-weight-dup 1.0 \
  --rambling-weight-loop 1.0 \
  --rambling-weight-plateau 1.0 \
  --rambling-weight-lang 0.1 \
  --rambling-loopiness-cap 5.0 \
  --rambling-plateau-k 4

echo "[DONE] All stages finished. Outputs are under $OUT_DIR"
