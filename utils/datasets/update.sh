#!/usr/bin/env bash
: "${BandPODir:?未检测到 BandPODir。请先在 bash 中执行：source /path_to_BandPO/init.sh}"

set -euo pipefail
IFS=$'\n\t'

command -v python3 >/dev/null 2>&1 || { echo "未找到 python3，请先安装。"; exit 1; }

# ---------------- 1) 遍历执行 dataset 下的 upload.py ----------------
DATASET_DIR="$BandPODir/data/dataset"
if [[ -d "$DATASET_DIR" ]]; then
  echo "== 执行数据集 upload.py（递归）=="
  # 找到所有 upload.py，排序后逐个执行
  mapfile -t DL_SCRIPTS < <(find "$DATASET_DIR" -type f -name 'upload.py' | sort || true)
  if ((${#DL_SCRIPTS[@]} == 0)); then
    echo "未找到任何 upload.py 于：$DATASET_DIR"
  else
    for s in "${DL_SCRIPTS[@]}"; do
      d="$(dirname "$s")"
      echo "▶ 运行：$s"
      # 失败不阻断后续
      ( cd "$d" && python3 upload.py ) || { echo "❌ 执行失败：$s（继续下一个）"; }
    done
  fi
else
  echo "目录不存在：$DATASET_DIR"
fi

echo "== 全部数据集上传完成 =="
