conda create -n verl_911 --file conda-linux-64.lock
conda activate verl_911
python -m pip install -r requirements-pip.txt
# 最后安装 verl 源码（无依赖）
pip install --no-deps -e ./verl
# 跨平台可复现：conda-lock 生成多平台锁文件
