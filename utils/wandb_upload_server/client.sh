#!/usr/bin/env bash
set -euo pipefail

# ====== 写死配置区 ======
HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
SERVER="http://{$HOST}:8000"
INTERVAL_MINUTES=15

# 要同步的 wandb offline runs（必须是 server 机器上存在的路径）
# 注意：这是 JSON 数组，双引号不能省；逗号分隔；最后一项不能多逗号
# "/inspire/hdd/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/dapo_official/2026-01-11_11-14-02/wandb/offline-run-20260111_112132-6unvjqye",
# "/inspire/hdd/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/grpo_plus_ray_trgppo005_df/2026-01-15_01-46-05/wandb/offline-run-20260115_015014-vrbebf92",
# "/inspire/hdd/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/grpo_plus_ray_trgppo005_df/2026-01-11_11-26-55/wandb/offline-run-20260111_113405-3eu8s1zv",
# "/inspire/hdd/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/grpo_plus_ray_trgppo005/2026-01-11_11-24-21/wandb/offline-run-20260111_113149-o1is7ld2",
# "/inspire/hdd/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/grpo_plus_ray_trgppo005/2026-01-12_12-29-10/wandb/offline-run-20260112_123321-gxxxx5ue",
# "/inspire/hdd/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/grpo_official/2026-01-11_11-28-15/wandb/offline-run-20260111_113405-dp4av0ms",
# "/inspire/hdd/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/grpo_official/2026-01-12_12-30-09/wandb/offline-run-20260112_123750-8cbg75zq",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/grpo_official_clip_higher/2026-01-20_06-25-47/wandb/offline-run-20260120_063034-jseot5lp",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/dapo_official_plus_trgppo005/2026-01-26_16-47-06/wandb/offline-run-20260126_165405-5pu69cm9"

# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_Qwen2.5Instruct3B/dapo_official/2026-01-15_13-53-09/wandb/offline-run-20260115_135709-n7fx5hwm",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_Qwen2.5Instruct3B/grpo_official/2026-01-15_14-12-04/wandb/offline-run-20260115_142304-qby5xycb",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_Qwen2.5Instruct3B/grpo_official_clip_higher/2026-01-20_06-39-34/wandb/offline-run-20260120_064926-spywgtek",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_Qwen2.5Instruct3B/grpo_plus_ray_trgppo005/2026-01-19_18-02-00/wandb/offline-run-20260119_181310-lb95bh8s",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_Qwen2.5Instruct3B/grpo_plus_ray_trgppo005_df_difficultybias_relaxkl/2026-01-22_03-25-25/wandb/offline-run-20260122_033033-2k8c38k1",

# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_MetaLlama3.2Instruct3B/dapo_official/2026-01-16_05-56-02/wandb/offline-run-20260116_060513-d6vgj0lo",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_MetaLlama3.2Instruct3B/dapo_official/2026-01-18_12-04-47/wandb/offline-run-20260118_121458-pymigiru",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_MetaLlama3.2Instruct3B/grpo_official/2026-01-16_06-14-12/wandb/offline-run-20260116_062245-olvhcfaq"
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_MetaLlama3.2Instruct3B/grpo_plus_ray_trgppo005/2026-01-19_18-05-06/wandb/offline-run-20260119_181451-0282l8ym",

# "/inspire/hdd/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/dapo_official/2026-01-11_15-39-29/wandb/offline-run-20260111_154834-njxw7r2j",
# "/inspire/hdd/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/dapo_official/2026-01-12_12-41-28/wandb/offline-run-20260112_124601-dale0d7s",
# "/inspire/hdd/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/grpo_plus_ray_trgppo005_df/2026-01-11_15-52-46/wandb/offline-run-20260111_160117-5kt1srer",
# "/inspire/hdd/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/grpo_plus_ray_trgppo005_df/2026-01-13_12-46-27/wandb/offline-run-20260113_125153-27doxn3j",
# "/inspire/hdd/project/exploration-topic/public/yli/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/grpo_plus_ray_trgppo005/2026-01-13_12-30-27/wandb/offline-run-20260113_123559-7b12yz42",
# "/inspire/hdd/project/exploration-topic/public/yli/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/grpo_plus_ray_trgppo005/2026-01-12_13-45-26/wandb/offline-run-20260112_135619-etb8bbjs",
# "/inspire/hdd/project/exploration-topic/public/yli/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/grpo_official/2026-01-12_14-01-56/wandb/offline-run-20260112_141255-b7ybinr1",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/grpo_plus_ray_trgppo003/2026-01-14_14-15-10/wandb/offline-run-20260114_142808-ne8bmkau",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/grpo_plus_ray_trgppo010/2026-01-14_13-50-06/wandb/offline-run-20260114_140324-2b4iqilu",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/grpo_plus_ray_trgppo005_df_difficultybias_relaxkl/2026-01-14_14-00-50/wandb/offline-run-20260114_141442-ksf0hf14",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/grpo_official_clip_higher/2026-01-20_07-11-20/wandb/offline-run-20260120_072145-zmq0y43g",

# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillLlama8B/grpo_official/2026-01-16_05-14-31/wandb/offline-run-20260116_052349-c6o9sz3l",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillLlama8B/dapo_official/2026-01-15_14-09-09/wandb/offline-run-20260115_142304-l63srave",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillLlama8B/grpo_plus_ray_trgppo005/2026-01-19_18-26-56/wandb/offline-run-20260119_184031-7wjzwepx",


# "/inspire/hdd/project/exploration-topic/public/yli/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillLlama8B/dapo_official/2026-01-13_02-16-09/wandb/offline-run-20260113_022905-v6u0cc9b",
# "/inspire/hdd/project/exploration-topic/public/yli/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillLlama8B/grpo_plus_ray_trgppo005_df/2026-01-13_02-44-59/wandb/offline-run-20260113_025850-u62z2uzk"

# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillLlama8B/grpo_plus_ray_trgppo005_df_difficultybias_relaxkl/2026-01-22_03-33-04/wandb/offline-run-20260122_034411-m8rww3zc",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillLlama8B/grpo_official_clip_higher/2026-01-22_04-34-35/wandb/offline-run-20260122_044537-o4anrkd3",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillLlama8B/grpo_plus_ray_trgppo005_relaxkl/2026-01-25_14-58-44/wandb/offline-run-20260125_150718-1n08d042",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen7B/grpo_plus_ray_trgppo005_relaxkl/2026-01-25_15-03-26/wandb/offline-run-20260125_151136-21y0ovfs",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_Qwen2.5Instruct3B/grpo_plus_ray_trgppo005_relaxkl/2026-01-25_15-09-17/wandb/offline-run-20260125_151644-7fytliui",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/grpo_plus_ray_trgppo005_relaxkl/2026-01-25_15-14-11/wandb/offline-run-20260125_152014-ms943dxf",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_Qwen2.5Instruct3B/grpo_plus_ray_trgppo010/2026-01-25_16-02-05/wandb/offline-run-20260125_160937-y5alqcin",
# "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_Qwen2.5Instruct3B/grpo_plus_ray_trgppo003/2026-01-25_16-24-39/wandb/offline-run-20260125_163210-75m8tfp2",
PATHS_JSON='[
  "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/bandpo__grpo_plus_ray_bandchi2005/2026-03-04_01-28-42/wandb/offline-run-20260304_013434-5djmkf3a",
  "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/bandpo__grpo_plus_ray_bandkl005/2026-03-04_01-28-21/wandb/offline-run-20260304_013440-spxasih9",
  "/inspire/qb-ilm/project/exploration-topic/liyuan-p-liyuan/workspace/BandPO/data/wandb/VERL_DapoMath_DeepSeekR1DistillQwen1.5B/bandpo__grpo_plus_ray_bandtv005/2026-03-04_01-29-00/wandb/offline-run-20260304_013440-jpjiot51"
]'
# ====== 写死配置区结束 ======

echo "[health]"
curl -sS "${SERVER}/health"
echo

echo "[jobs]"
curl -sS "${SERVER}/jobs"
echo

echo "[add]"
curl -sS -X POST "${SERVER}/add" \
  -H 'Content-Type: application/json' \
  -d "{\"paths\": ${PATHS_JSON}, \"interval_minutes\": ${INTERVAL_MINUTES}}"
echo
