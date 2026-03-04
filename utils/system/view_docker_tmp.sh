set -euo pipefail

source_dir="/tmp"
target_dir="/inspire/hdd/global_user/liyuan-p-liyuan/tmp"

confirm_or_exit() {
  local msg="$1"
  local ans
  printf "%s\n" "$msg"
  read -r -p "Type 'yes' to proceed: " ans
  if [ "$ans" != "yes" ]; then
    echo "Aborted."
    exit 1
  fi
}

mkdir -p -- "$(dirname -- "$target_dir")"

if [ -L "$target_dir" ]; then
  echo "Detected: symlink"
  ls -ld -- "$target_dir" || true
  confirm_or_exit "About to remove symlink: $target_dir (will NOT delete the link target)"
  rm -- "$target_dir"

elif [ -d "$target_dir" ]; then
  echo "Detected: real directory"
  ls -ld -- "$target_dir" || true
  confirm_or_exit "About to remove directory recursively: $target_dir (ALL contents will be deleted)"
  rm -rf -- "$target_dir"

elif [ -e "$target_dir" ]; then
  echo "Detected: file or other type"
  ls -ld -- "$target_dir" || true
  confirm_or_exit "About to remove: $target_dir"
  rm -f -- "$target_dir"
fi

# 建立软链接：target_dir -> source_dir（加固：-n/-f 防止边界情况下当目录处理/残留导致失败）
ln -sfn -- "$source_dir" "$target_dir"

ls -ld -- "$target_dir"
readlink -- "$target_dir" || true

echo
read -r -p "Inspect what you need. Press ENTER to remove the symlink and exit: " _

# 删除软链接（不动源目录）
if [ -L "$target_dir" ]; then
  rm -- "$target_dir"
  echo "Symlink removed: $target_dir"
fi
