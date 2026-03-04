# 删除token
printf "protocol=https\nhost=github.com\n\n" | git credential reject

# 确认我们用的是可控的 helper：store
git config --global credential.helper store

cat ~/.git-credentials

printf "protocol=https\nhost=github.com\nusername=Yuan-Li-FNLP\npassword=ghp_<your_git_token>\n\n" | git credential approve