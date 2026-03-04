rm -rf -- "/path/dir"

scp -r /path/src_dir user@host:/path/dest/
cp -a /remote-home1/yli/cudas/cuda-12.4 /remote-home1/yfgao/cudas/cuda-12.4
cp -a /remote-home1/yli/cudas/cuda-12.4 /remote-home1/bwang/cudas/cuda-12.4
cp -a /remote-home1/yli/Workspace/BandPO/data/dataset /remote-home1/bwang/workspace_yli/BandPO/data/dataset
cp -a /remote-home1/yfgao/workspace_yli/BandPO/data/models  /remote-home1/bwang/workspace_yli/BandPO/data/models
cp -a /remote-home1/yfgao/workspace_yli/BandPO/data/dataset  /remote-home1/bwang/workspace_yli/BandPO/data/dataset
cp -a /remote-home1/yfgao/workspace_yli/BandPO/data/ckpts/clip_improve_v3/DapoMathDataset_DeepseekR1_1.5B_grpo_310disturbed /remote-home1/yli/Workspace/BandPO/data/ckpts/clip_improve_v3/DapoMathDataset_DeepseekR1_1.5B_grpo_310disturbed

cp -a /remote-home1/yli/Workspace/BandPO/data/dataset /remote-home1/bwang/workspace_yli/BandPO/data/
cp -a /remote-home1/yli/Workspace/BandPO/data/dataset /remote-home1/yfgao/workspace_yli/BandPO/data

cp -a /remote-home1/yli/Workspace/BandPO/data/models/deepseek /remote-home1/bwang/workspace_yli/BandPO/data/models
cp -a /remote-home1/yli/Workspace/BandPO/data/models/deepseek /remote-home1/yfgao/workspace_yli/BandPO/data/models

mv /remote-home1/bwang/workspace_yli/BandPO/data/ckpts/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_L8k /remote-home1/yli/Workspace/BandPO/data/ckpts/RLtraining_DapoMathDataset_DeepseekR1_1.5B
cp -a /remote-home1/bwang/workspace_yli/BandPO/data/ckpts/RLtraining_DapoMathDataset_DeepseekR1_1.5B/grpo_plus_ray_trgppo01_L8k /remote-home1/yli/Workspace/BandPO/data/ckpts/RLtraining_DapoMathDataset_DeepseekR1_1.5B

# cuda switch
export CUDA_HOME=$HOME/cudas/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
# export CUTLASS_PATH=/remote-home1/yli/Workspace/cutlass
