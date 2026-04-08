: "${BandPODir:?BandPODir not detected. Please run 'source /path_to_BandPO/init.sh' in bash first.}"
TS=$(date +'%Y-%m-%d_%H-%M-%S')
LOG_DIR="$BandPODir/data/logs/VERL_commonsenseqa_Qwen2.5Instruct3B/grpo_official"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$TS.log"

bash grpo_official.sh 2>&1 | tee -a "$LOG_FILE" >/dev/null 2>&1