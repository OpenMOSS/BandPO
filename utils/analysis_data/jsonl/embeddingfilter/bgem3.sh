# 自动识别 host
host=$(hostname -I | awk '{print $1}')
echo "host: $host"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /remote-home1/yli/Workspace/BandPO/data/models/BAAI/bge-m3 \
  --task embed \
  --served-model-name BAAI/bge-m3 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --host "$host" \
  --port 8888
