# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir  /remote-home1/yli/Workspace/BandPO/data/ckpts/clip_improve/DapoMathDataset_qwen25_3B_dapo/global_step_203/actor \
#     --target_dir /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo/global_step_203 \
#     --use_cpu_initialization

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir  /remote-home1/yli/Workspace/BandPO/data/ckpts/clip_improve/DapoMathDataset_qwen25_3B_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_128/actor \
    --target_dir /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo_rmax10_soft_clip_by_3seg_rho034/global_step_128 \
    --use_cpu_initialization

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir  /remote-home1/yli/Workspace/BandPO/data/ckpts/clip_improve/DapoMathDataset_qwen25_3B_dapo/global_step_128/actor \
#     --target_dir /remote-home1/yli/Workspace/BandPO/data/ckpt2hf/clip_improve/DapoMathDataset_qwen25_3B_dapo/global_step_128