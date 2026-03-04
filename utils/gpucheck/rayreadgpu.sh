# 本机单机：
python - <<'PY'
import ray, os, torch
ray.init()  # 或 ray.init(address="auto") 看你的场景
print("driver CVD=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("driver cuda.is_available=", torch.cuda.is_available(), "count=", torch.cuda.device_count())
@ray.remote(num_gpus=1)
def gpu_probe():
    import os, torch
    return {
        "CVD": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda.is_available": torch.cuda.is_available(),
        "count": torch.cuda.device_count(),
        "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    }
print(ray.get(gpu_probe.remote()))
PY
