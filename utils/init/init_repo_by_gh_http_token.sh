#!/usr/bin/env bash
# 用法：source init_repo.sh

# 登陆方法二选一：
# 方法一：永久变量
# -------- GitHub token（最前面）--------
# 变量名同时兼容 GITHUB_TOKEN / GH_TOKEN，不打印明文
if [[ -n "${GITHUB_TOKEN:-}" || -n "${GH_TOKEN:-}" ]]; then
  echo "已检测到 GitHub token："
  [[ -n "${GITHUB_TOKEN:-}" ]] && echo " - GITHUB_TOKEN 已设置"
  [[ -n "${GH_TOKEN:-}" ]] && echo " - GH_TOKEN 已设置"
else
  read -rsp "未找到 GitHub token，请输入（输入不可见）： " gh_token
  echo
  if [[ -z "$gh_token" ]]; then
    echo "未输入 GitHub token，已退出。"
    exit 1
  fi
  export GITHUB_TOKEN="$gh_token"
  export GH_TOKEN="$gh_token"
  {
    echo
    echo "# added on $(date '+%Y-%m-%d %H:%M:%S')"
    echo "export GITHUB_TOKEN=\"$gh_token\""
    echo "export GH_TOKEN=\"$gh_token\""
  } >> "$HOME/.bashrc"
  echo "已写入 ~/.bashrc：GITHUB_TOKEN / GH_TOKEN"
fi

# 方法二：临时变量
# export GH_TOKEN='ghp_xxx_your_token'   # 或者用 GITHUB_TOKEN 变量名
# printf '%s\n' "$GH_TOKEN" | gh auth login --hostname github.com --with-token

# https克隆二选一：
# 方法一：gh
gh repo clone Yuan-Li-FNLP/BandPO
# 方法一：git
# git -c "http.https://github.com/.extraheader=Authorization: Bearer $GH_TOKEN" \
    # clone https://github.com/Yuan-Li-FNLP/BandPO.git
# git clone https://github.com/Yuan-Li-FNLP/BandPO.git

source ~/.bashrc
echo "完成。"