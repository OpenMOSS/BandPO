: "${BandPODir:?未检测到 BandPODir。请先在 bash 中执行：source /path_to_BandPO/init.sh}"
TS=$(date +'%Y-%m-%d_%H-%M-%S')
LOG_DIR="$BandPODir/data/logs/clip_improve_v3/DapoMathDataset_DeepseekR1_1.5B_grpo_official"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$TS.log"

# 运行并把所有输出写入日志，同时在屏幕显示
bash GRPO_official.sh 2>&1 | tee -a "$LOG_FILE"
