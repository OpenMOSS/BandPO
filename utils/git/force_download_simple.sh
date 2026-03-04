# 0) 先确认当前分支名
git branch --show-current
# 假设输出是 main

# 1) 保存本地改动（任选其一）
git add -A && git commit -m "WIP"
# 或者
# git stash -u

# 2) 拉取并在冲突时默认“远端优先”
git pull -X theirs origin main
# 等价分步写法：
# git fetch origin
# git merge -X theirs origin/main

# 2） plus 如果本地分支和远端分支「分叉」，Git 要求你明确选择 pull 的策略（merge / rebase / 只快进）
# 想要的是“有冲突以远端为主”，这更适合 merge（而不是 rebase）
# 仅这次：用 merge，并在冲突时偏向远端
git pull --no-rebase -X theirs origin main
