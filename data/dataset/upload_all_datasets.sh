#!/usr/bin/env bash
# Usage: bash upload_all_datasets.sh

# 1) Check for BandPODir
if [[ -z "${BandPODir:-}" ]]; then
  echo "Error: BandPODir not detected. Please run 'source /path_to_BandPO/init.sh' in bash first."
  exit 1
fi

# 2) Check for HUGGING_FACE_USERNAME (Required for constructing repo_id in upload scripts)
if [[ -z "${HUGGING_FACE_USERNAME:-}" ]]; then
  echo "Error: HUGGING_FACE_USERNAME not detected."
  echo "Please export it or run 'source /path_to_BandPO/init.sh' to set it up."
  exit 1
fi

set -euo pipefail
IFS=$'\n\t'

command -v python3 >/dev/null 2>&1 || { echo "python3 not found. Please install it first."; exit 1; }

# ---------------- Iterate and execute upload.py in dataset directory ----------------
DATASET_DIR="$BandPODir/data/dataset"

if [[ -d "$DATASET_DIR" ]]; then
  echo "== Executing dataset upload.py (recursive) =="
  echo "Target Directory: $DATASET_DIR"
  echo "Target Username: $HUGGING_FACE_USERNAME"

  # Find all upload.py files, sort them, and execute one by one
  mapfile -t UP_SCRIPTS < <(find "$DATASET_DIR" -type f -name 'upload.py' | sort || true)

  if ((${#UP_SCRIPTS[@]} == 0)); then
    echo "No upload.py found in: $DATASET_DIR"
  else
    for s in "${UP_SCRIPTS[@]}"; do
      d="$(dirname "$s")"
      echo "---------------------------------------------------"
      echo "▶ Found script: $s"
      echo "▶ Working directory: $d"
      
      # Execute in a subshell so directory change doesn't affect the loop
      # Failure does not block subsequent execution (|| true logic)
      ( 
        cd "$d" && python3 upload.py 
      ) || { 
        echo "❌ Upload failed for: $s (Skipping to next dataset)" 
      }
    done
  fi
else
  echo "Directory does not exist: $DATASET_DIR"
  exit 1
fi

echo "---------------------------------------------------"
echo "== All upload scripts executed =="