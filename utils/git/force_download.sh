git add -A
# 1) 把你当前 main 上的改动（含已 add 未提交的）拎到一个分支
git switch -c mywork
git commit -m "my local additions"   # 若还有未提交改动，建议先提交

# 2) 回到 main，对齐远端（以仓库为准）
git switch main                      # 如果主分支叫 master，就用 master
git fetch origin
git reset --hard origin/main         # 本地 main 完全等于远端

# 3) 把 mywork 合回 main，冲突一律选 main（=远端）胜
git merge mywork -X ours

# "update brand new git repo to gyf."