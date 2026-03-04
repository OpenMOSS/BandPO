srun -p a800  --gres=gpu:8 --cpus-per-task=96 --mem-per-cpu=4G --pty bash
srun -p a800  --gres=gpu:1 --cpus-per-task=12 --mem-per-cpu=4G --pty bash
srun -p a800  --gres=gpu:1 --cpus-per-task=32 --mem-per-cpu=4G --pty bash

srun -p a800 --nodelist=slurmd-1 --gres=gpu:8 --cpus-per-task=96 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-2 --gres=gpu:8 --cpus-per-task=96 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-3 --gres=gpu:8 --cpus-per-task=96 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-4 --gres=gpu:8 --cpus-per-task=96 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-6 --gres=gpu:8 --cpus-per-task=96 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-7 --gres=gpu:8 --cpus-per-task=96 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-9 --gres=gpu:8 --cpus-per-task=96 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-10 --gres=gpu:8 --cpus-per-task=96 --mem-per-cpu=8G --pty bash

srun -p a800 --nodelist=slurmd-4 --gres=gpu:7 --cpus-per-task=84 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-4 --gres=gpu:1 --cpus-per-task=12 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-4 --gres=gpu:4 --cpus-per-task=48 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-1 --gres=gpu:1 --cpus-per-task=12 --mem-per-cpu=8G --pty bash

srun -p a800 --nodelist=slurmd-3 --gres=gpu:4 --cpus-per-task=48 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-9 --gres=gpu:4 --cpus-per-task=48 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-2 --gres=gpu:4 --cpus-per-task=48 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-2 --gres=gpu:4 --cpus-per-task=16 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-2 --gres=gpu:2 --cpus-per-task=24 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-2 --gres=gpu:1 --cpus-per-task=12 --mem-per-cpu=8G --pty bash

srun -p a800 --gres=gpu:0 --cpus-per-task=32 --mem-per-cpu=4G --pty bash
srun -p fnlp-4090 --gres=gpu:2 --cpus-per-task=32  --mem-per-cpu=4G --pty bash
srun -p fnlp-4090d --gres=gpu:1 --cpus-per-task=16  --mem-per-cpu=4G --pty bash
srun -p fnlp-4090d --gres=gpu:0 --cpus-per-task=32  --mem-per-cpu=4G --pty bash
srun -p fnlp-4090d --gres=gpu:0 --cpus-per-task=128  --mem-per-cpu=4G --pty bash
srun -p fnlp-4090d --gres=gpu:1 --cpus-per-task=32  --mem-per-cpu=4G --pty bash

srun -p fnlp-4090d --gres=gpu:8 --cpus-per-task=128  --mem-per-cpu=4G --pty bash
srun -p fnlp-4090d --gres=gpu:4 --cpus-per-task=116  --mem-per-cpu=4G --pty bash
srun -p fnlp-4090d --gres=gpu:4 --cpus-per-task=32  --mem-per-cpu=4G --pty bash
srun -p fnlp-3090 --gres=gpu:8 --cpus-per-task=80  --mem-per-cpu=4G --pty bash
srun -p fnlp-4090d --gres=gpu:1 --cpus-per-task=16  --mem-per-cpu=4G --pty bash

srun -p a800 --nodelist=slurmd-3 --gres=gpu:1 --cpus-per-task=8 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-3 --gres=gpu:4 --cpus-per-task=32 --mem-per-cpu=8G --pty bash
srun -p a800 --nodelist=slurmd-3 --gres=gpu:8 --cpus-per-task=64 --mem-per-cpu=8G --pty bash
