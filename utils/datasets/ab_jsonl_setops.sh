python ab_jsonl_setops.py \
  "/remote-home1/yli/Workspace/DiagVerse/data/validate/qwen3-max/record+20251101_090528.jsonl" \
  "/remote-home1/yli/Workspace/DiagVerse/data/datasets/ntsb/parquet2datasets/select500/test500/test500.jsonl" \
  --out-dir "/remote-home1/yli/Workspace/DiagVerse/data/datasets/ntsb/parquet2datasets/select500/test500-remaining" \
  --key "index_id"
