# 读取系统临时目录（优先 $TMPDIR，没有就用 /tmp）
TMP_DIR="${TMPDIR:-/tmp}"

# 安全护栏：变量为空或指向根目录时拒绝执行
if [[ -z "$TMP_DIR" || "$TMP_DIR" == "/" ]]; then
  echo "Refuse to operate on an empty path or root (/)." >&2
  exit 1
fi
# 展示将要删除的内容（演练）
# find "$TMP_DIR" -mindepth 1 -print
# 真正删除：递归删除 tmp_dir 下所有内容，但保留 tmp_dir 本身
find "$TMP_DIR" -mindepth 1 -delete





# [[ -d "$TMP_DIR" && "$TMP_DIR" != "/" ]] || { echo "危险路径，已取消"; exit 1; }
# # 覆盖普通 + 隐藏条目，但不会匹配 . 和 ..
# rm -rf -- "${TMP_DIR:?}/"{*,.[!.]*,..?*}