# 查看当前分支 & 远程
git branch -vv
git remote -v
git remote set-url origin git@github.com:Yuan-Li-FNLP/BandPO.git

git status

git add -A
git commit -m ""
git push origin main

# 测试
ssh -T -p 443 git@ssh.github.com
curl -v https://github.com
# 同一把公钥不能同时添加到两个 GitHub 账号；多账号必须用不同的公钥。

# 远端为准、直接覆盖
git fetch origin

git checkout -B main origin/main   # 用远端完全覆盖本地
git reset --hard origin/main # 如果已经在 main 上，也可以这样写

git clean -fd                      # 可选：删未跟踪文件；更干净用 -fdx

git rm --cached installing/*.whl

git branch --show-current

git add -A && git commit -m "update baseline" && git push origin main
git add -A && git commit -m "update baseline and rambling filter" && git push origin main