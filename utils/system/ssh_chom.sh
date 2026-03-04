# 仅本人可访问 .ssh 目录
chmod 700 ~/.ssh

# 私钥仅本人可读写；公钥可读
chmod 600 ~/.ssh/id_rsa
[ -f ~/.ssh/id_rsa.pub ] && chmod 644 ~/.ssh/id_rsa.pub

# (可选) 确保归属权正确
# chown "$USER:$USER" ~/.ssh ~/.ssh/id_rsa ~/.ssh/id_rsa.pub 2>/dev/null || true

# 核对
ls -ld ~/.ssh ~/.ssh/id_rsa ~/.ssh/id_rsa.pub
