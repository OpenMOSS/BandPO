# 1) 确保没有处于离线模式
wandb online
# 2) 查看当前配置（会显示 host/模式等）
wandb status --settings
# 3) 云端连通性（网络层面探测）
ping -c 3 api.wandb.ai
# 或：
curl -I https://api.wandb.ai

# 方法A：环境变量
export WANDB_BASE_URL=http://<your-host>:<port>
wandb login --verify

# 方法B：命令行参数
wandb login --host http://<your-host>:<port> --verify

wandb verify