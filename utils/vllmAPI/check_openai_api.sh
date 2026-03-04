#!/usr/bin/env bash
set -euo pipefail

MODEL="/remote-home1/yli/Workspace/DiagVerse/data/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
ADDR="10.176.58.103:8000"
SCHEME="${3:-http}"
BASE_PATH="${4:-/v1}"

HOST="${ADDR%:*}"
PORT="${ADDR##*:}"
API_KEY="${OPENAI_API_KEY:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

set -x
if [[ -n "${API_KEY}" ]]; then
  "${PY}" "${SCRIPT_DIR}/check_openai_api.py" \
    --model "${MODEL}" --host "${HOST}" --port "${PORT}" \
    --scheme "${SCHEME}" --base-path "${BASE_PATH}" --api-key "${API_KEY}"
else
  "${PY}" "${SCRIPT_DIR}/check_openai_api.py" \
    --model "${MODEL}" --host "${HOST}" --port "${PORT}" \
    --scheme "${SCHEME}" --base-path "${BASE_PATH}"
fi
set +x