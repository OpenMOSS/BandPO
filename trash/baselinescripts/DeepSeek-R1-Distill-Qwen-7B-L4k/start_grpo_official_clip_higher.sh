: "${BandPODir:?未检测到 BandPODir。请先在 bash 中执行：source /path_to_BandPO/init.sh}"
TS=$(date +'%Y-%m-%d_%H-%M-%S')
LOG_DIR="$BandPODir/data/logs/VERL_DapoMath_DeepSeekR1DistillQwen7B/grpo_official_clip_higher"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$TS.log"

# 运行并把所有输出写入日志，同时在屏幕显示
bash grpo_official_clip_higher.sh 2>&1 | tee -a "$LOG_FILE" >/dev/null 2>&1 # 控制台静音