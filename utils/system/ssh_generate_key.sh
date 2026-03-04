# ed25519 —— 首选，安全/短小/快。
# ed25519-sk —— FIDO2 安全密钥（YubiKey 等）版，私钥不落盘，更安全。
# ecdsa —— 椭圆曲线（NIST P-256/384/521），一般不如 ed25519 有优势。
# ecdsa-sk —— ECDSA 的 FIDO2 版本。
# rsa —— 为极老环境/设备提供广泛兼容（建议 3072/4096 位）。
# dsa —— 过时且默认禁用（基本不要用）。

# ssh-keygen -t ed25519 -C "你的邮箱或备注"

ssh-keygen -t rsa -b 4096 -C "你的邮箱或备注"

# # 曲线通过 -b 选择：256/384/521 → nistp256/p384/p521
# ssh-keygen -t ecdsa -b 256 -a 100 -C "ecdsa-p256" -f ~/.ssh/id_ecdsa
# ssh-keygen -t ecdsa -b 521 -a 100 -C "ecdsa-p521" -f ~/.ssh/id_ecdsa_521