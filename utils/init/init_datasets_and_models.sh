#!/usr/bin/env bash
: "${BandPODir:?BandPODir not detected. Please run 'source /path_to_BandPO/init.sh' in bash first.}"

set -euo pipefail
IFS=$'\n\t'

command -v python3 >/dev/null 2>&1 || { echo "python3 not found. Please install it first."; exit 1; }

# ---------------- 1) Iterate and execute download.py in dataset directory ----------------
DATASET_DIR="$BandPODir/data/dataset"
if [[ -d "$DATASET_DIR" ]]; then
  echo "== Executing dataset download.py (recursive) =="
  # Find all download.py files, sort them, and execute one by one
  mapfile -t DL_SCRIPTS < <(find "$DATASET_DIR" -type f -name 'download.py' | sort || true)
  if ((${#DL_SCRIPTS[@]} == 0)); then
    echo "No download.py found in: $DATASET_DIR"
  else
    for s in "${DL_SCRIPTS[@]}"; do
      d="$(dirname "$s")"
      echo "▶ Running: $s"
      # Failure does not block subsequent execution
      ( cd "$d" && python3 download.py ) || { echo "❌ Execution failed: $s (Skipping to next)"; }
    done
  fi
else
  echo "Directory does not exist: $DATASET_DIR"
fi

# ---------------- 2) Download models to specified path ----------------
echo "== Downloading models to specified directories =="

# Model repos and target directories (based on BandPODir)
REPOS=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "Qwen/Qwen2.5-3B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
DESTS=(
  "$BandPODir/data/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "$BandPODir/data/models/Qwen/Qwen2.5-3B-Instruct"
  "$BandPODir/data/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "$BandPODir/data/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

for i in "${!REPOS[@]}"; do
  repo="${REPOS[$i]}"
  dest="${DESTS[$i]}"
  mkdir -p "$dest"
  echo "▼ $repo  ->  $dest"

  # Pass parameters to Python snippet via environment variables; failure does not block subsequent execution
  MODEL_REPO="$repo" LOCAL_DIR="$dest" python3 - <<'PY' || { echo "❌ Download failed: $repo (Skipping to next)"; continue; }
import os, sys
from huggingface_hub import snapshot_download

repo = os.environ["MODEL_REPO"]
dest = os.environ["LOCAL_DIR"]
token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

# Avoid using mirrors (force official endpoint)
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

try:
    out = snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=dest,
        local_dir_use_symlinks=False,  # Download actual files
        token=token,
    )
    print("✅ Downloaded to:", out)
except Exception as e:
    print("ERROR:", e)
    sys.exit(1)
PY
done

echo "== All models and datasets downloaded successfully =="