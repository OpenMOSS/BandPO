# python convert_to_verl.py \
#   --input /remote-home1/yli/Workspace/BandPO/data/datasets/aime2024/test/aime2024_test.parquet \
#   --question_key problem \
#   --answer_key answer \
#   --solution_key solution \
#   --data_source "math_dapo" \
#   --data_source_extra_info "HuggingFaceH4/aime_2024" \
#   --ability MATH \
#   --reward_style rule-lighteval/MATH_v2 \
#   --keep_id --id_key id \
#   --split test

# python convert_to_verl.py \
#   --input /remote-home1/yli/Workspace/BandPO/data/datasets/aime2025/test/aime2025_test.parquet \
#   --question_key problem \
#   --answer_key answer \
#   --solution_key solution \
#   --data_source "math_dapo" \
#   --data_source_extra_info "yentinglin/aime_2025" \
#   --ability MATH \
#   --reward_style rule-lighteval/MATH_v2 \
#   --keep_id --id_key id \
#   --split test

python convert_to_verl.py \
  --input /remote-home1/yli/Workspace/BandPO/data/datasets/amc2023/test/amc2023.parquet \
  --question_key question \
  --answer_key answer \
  --solution_key "" \
  --data_source "math_dapo" \
  --data_source_extra_info "math-ai/amc23" \
  --ability MATH \
  --reward_style rule-lighteval/MATH_v2 \
  --keep_id --id_key id \
  --split test