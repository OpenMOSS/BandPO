# 生成“显式规范”锁定文件（精确到构建号与下载 URL）
conda list --explicit > conda-linux-64.lock
pip freeze --exclude-editable > requirements-pip.txt
