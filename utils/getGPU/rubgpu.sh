# 输入参数保留的那一个session的名字
keep="slurm3gpu8id0"   # 或者 keep="$1"

# 预览要删除哪一些
echo "Will keep: $keep"
echo "Sessions to delete:"
to_del=$(tmux ls 2>/dev/null | awk -F: -v k="$keep" '$1!=k{print $1}')
if [ -z "$to_del" ]; then
  echo "Nothing to delete."
  exit 0
fi
echo "$to_del"

# 输入yes就删除：
read -r -p "Type 'yes' to delete: " ans
if [ "$ans" = "yes" ]; then
  echo "$to_del" | xargs -r -n1 tmux kill-session -t
  echo "Done."
else
  echo "Aborted."
fi
