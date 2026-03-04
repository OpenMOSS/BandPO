# 直接用你给的目录（默认值），并发 8 线程
python sync_wandb_offline_parallel.py --workers 8

# 只扫描不上传
python sync_wandb_offline_parallel.py --dry-run
