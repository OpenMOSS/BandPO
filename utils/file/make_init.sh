# 只处理 theone 目录（推荐在项目根目录执行）
python make_init.py theone

# 同时处理 theone 与 service，且排除 data 目录
python make_init.py theone service --exclude data

# 预览将创建哪些文件（不实际写入）
python make_init.py theone --dry-run
