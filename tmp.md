# BandPO

## why fixed cliiping machenism 有问题

## 项目介绍
几句话说明我们做了什么

## 文件架构
init.sh文件用于一键初始化环境变量和下载模型数据集。
data文件夹存储数据文件，包括模型、数据集、ckpts、一些做实验用的records、logs等等，只同步到git文件夹结构，不同步文件内容。使用“.gitkeep”文件标记要保留的文件夹。
其中的数据集可以运行一键同步到指定的hugginface仓库。可以一件download数据集和base模型。
数据集：
每个数据集文件夹有“数据集名称.sh”和“数据集名称.py”文件用于一开始处理数据集为verl的标准格式。
每个数据集文件夹有“download.py”文件用于下载指定数据集到本地。
RLtraning文件夹为核心训练代码。
utils主要是需要安装的apex仓库代码，以及一些用于分析数据、画图、git、slurm相关操作的模板代码。

## 安装
### 下载代码文件
git clone https://github.com/OpenMOSS/BandPO.git
### CUDA安装
#### 版本选择与安装
verl need cuda >=12.4。本项目使用最小的cuda版本12.4.For users without root privileges, who can not use "sudo" command, select the Installer Type “runfile (local)” for corresponding system to install the appropriate CUDA.
官网下载链接及教程：https://developer.nvidia.com/cuda-12-4-0-download-archive
Available cuda list:
- the latest version: <https://developer.nvidia.com/cuda-downloads#>
- the previous version: <https://developer.nvidia.com/cuda-toolkit-archive>
一些查看自己电脑对应下载参数需要使用的指令：
```bash
# for ubuntu
cat /etc/os-release
lsb_release -a
uname -m
# for windows
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber
```

Note: 
- During installation, deselect the “Driver” component; otherwise, you’ll encounter an error. However, if you install a higher version's cuda, the original driver's version may not enough to install it.
- To conveniently install multiple CUDA versions, create a cudas directory and install each version there. To change the installation path, go to Options.
- If you’re installing multiple CUDA versions, in Options uncheck the highlighted item “Create symbolic link from /usr/local/cuda.”
- remember modify the installing path, such as "$HOME/cudas/cuda-12.4".

#### Modify the PATH of CUDA
Finally, modify your bash command by `vim ~/.bashrc`:
```bash
# cuda switch
export CUDA_HOME=$HOME/cudas/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
```

#### Check Installing success
```bash
which nvcc && nvcc --version
```

### Intsall Cudnn
Available cudnn list:
- the latest version: <https://developer.nvidia.com/cudnn-downloads>
- the previous version: <https://developer.nvidia.com/cudnn-archive>
- the previous version's archive: 
    - <https://developer.nvidia.com/rdp/cudnn-archive>
    - <https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/>
Note
- The official site’s default installation requires root privileges to run dpkg and install the .deb package.
- If you don’t have root access, use the archive method: download and extract the corresponding .tar package, then copy the files into the matching CUDA path:
    - Move cuDNN’s include/cudnn*.h to /xx/cudas/cuda-12.4/include/
    - Move cuDNN’s lib/libcudnn* to /xx/cudas/cuda-12.4/lib64/

#### Download Cudnn by archive
```bash
# an example for downloading the tar archive and extracting it.
mkdir -p ~/tmp/cudnn && cd ~/tmp/cudnn
tar -xf /path/to/cudnn-linux-*-archive.tar.xz
cd cudnn-linux-*-archive
```
#### INstall Cudnn by archive
```bash
mkdir -p "$CUDA_HOME/include" "$CUDA_HOME/lib64"
cp include/cudnn*.h "$CUDA_HOME/include/"
cp -P lib/libcudnn* "$CUDA_HOME/lib64/"
chmod a+r "$CUDA_HOME/include"/cudnn*.h "$CUDA_HOME/lib64"/libcudnn*
```
#### Check Installing success
```bash
grep -HE "CUDNN_(MAJOR|MINOR|PATCHLEVEL)" /xxx/cudas/cuda-12.4/include/cudnn_version.h
```
or
```bash
grep -HE "CUDNN_(MAJOR|MINOR|PATCHLEVEL)" "$CUDA_HOME/include/cudnn_version.h"
```

#### Check the python:
千万不能安装了GraalPy，要安装cpython。先检查一下：
```bash
python -c "import sys; print(sys.version); print(sys.implementation);"
```

### 搭建 env
#### create env
```bash
conda create -n bandpo python=3.10
conda activate bandpo
```
#### Install Torch
The highest available pytorch version using Cuda 12.4 is 2.6.0.
Install Apex need pytorch.

```python
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
#### Install Dependencies
```bash
pip install "distro<2,>=1.7.0"
pip install -U "pyyaml>=5.1" datasets huggingface-hub
```
#### Install Apex
To use Megatron-LM or FSDP as the training backend, NVIDIA’s Apex library need installing. These training backends invoke the optimizers and mixed-precision utilities in Apex to improve the performance and efficiency of large-model training.
我们的项目已经git clone https://github.com/NVIDIA/apex.git，所以无需再次clone apex仓库。进入到utils中国的apex即可直接安装
```bash
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```
#### install 训练和推理后端
本项目已经clone了verl仓库（git clone https://github.com/volcengine/verl.git）并做了大量修改。
verl中的提供了多种后端选择，训练后端有FSDP和 Megatron-LM，推理后端有 vllm 和 SGLang 和 huggingface TGI integration。
首先进入到RLtraining/verl文件夹,这里建议自己逐行运行其中的install_vllm_sglang_mcore.sh的文件代码如果直接运行下面的代码报错的话。
```bash
# Make sure you have activated verl conda env
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```
安装完成之后，还有两个安装包的wheel，文件过大会导致ray超过最大文件上限而报错，把它删除即可。
```bash
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl
```

#### fix opentelemetry dependencies conflict
```bash
pip install -U \
  "opentelemetry-api==1.26.0" \
  "opentelemetry-sdk==1.26.0" \
  "opentelemetry-semantic-conventions==0.47b0" \
  "opentelemetry-exporter-otlp==1.26.0" \
  "opentelemetry-exporter-otlp-proto-grpc==1.26.0" \
  "opentelemetry-exporter-otlp-proto-http==1.26.0" \
  "opentelemetry-exporter-prometheus==0.47b0" \
  --upgrade-strategy only-if-needed
```
pip install -U "opentelemetry-api>=1.34,<2" "opentelemetry-sdk>=1.34,<2" "opentelemetry-exporter-otlp>=1.34,<2"

#### fix megatron-core dependencies conflict
```bash
# 2) 补齐 megatron-core 0.12.2 声明的依赖（如果你确实要用 megatron-core）
python -m pip install flask-restful nltk tensorstore zarr pytest-cov pytest-mock pytest-random-order
# 仅当有 NVIDIA/CUDA 环境时再装（否则大概率失败）
python -m pip install nvidia-modelopt
```

#### fix cudnn version bug
```bash
pip install --no-cache-dir "nvidia-cudnn-cu12==9.1.0.70"
```

#### Install verl
```bash
cd verl
pip install --no-deps -e .
```







## 一些已知的issues
如果你发现wandb出现timeout问题，尝试打开vpn，如果无果，请将WANDB_MODE设置为"offline" in runtime_env.yaml。

## 爱与信仰
如果你是像我一样，从基于CPU的数学优化算法的AI领域，跨越到，基于GPU的深度学习算法的AI领域。我们仓库中的utils兴许可以快速帮助你上手相关系统，特别是基于slurm的GPu调度系统。
root目录下的init.sh文件可以帮助你快速迁移代码并部署，减少你的工程花费时间，把更多经历聚集在理论分析上。
愿我们都能在AI领域中做出自己的贡献，能够受益于后人。虽路途坎坷，但山高万仞 只登一步。

## 联系方式
liyuan24@m.fudan.edu.cn

## 引用
预留好arxive的位置