# Install CUDA
Recommand: CUDA 12.4 (verl need cuda >=12.4)
The GPU driver on SII supports up to CUDA 12.4.
## Check the system version:
千万不能安装了GraalPy，要安装cpython。先检查一下：
```bash
python -c "import sys; print(sys.version); print(sys.implementation);"
```

```bash
# for ubuntu
cat /etc/os-release
lsb_release -a
# for windows
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber
```
## check the system architecture
```bash
uname -m
```
## Download and install the coresponding-version cuda
Available cuda list:
- the latest version: <https://developer.nvidia.com/cuda-downloads#>
- the previous version: <https://developer.nvidia.com/cuda-toolkit-archive>

For users without root privileges, who can not use "sudo" command, select the Installer Type “runfile (local)” for corresponding system to install the appropriate CUDA.

```bash
# an example:
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sh cuda_12.4.0_550.54.14_linux.run

wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run
```

Note: 
- During installation, deselect the “Driver” component; otherwise, you’ll encounter an error. However, if you install a higher version's cuda, the original driver's version may not enough to install it.
- To conveniently install multiple CUDA versions, create a cudas directory and install each version there. To change the installation path, go to Options.
- If you’re installing multiple CUDA versions, in Options uncheck the highlighted item “Create symbolic link from /usr/local/cuda.”
- remember modify the installing path, such as "$HOME/cudas/cuda-12.4".

# Modify the PATH of CUDA
Finally, modify your bash command by `vim ~/.bashrc`:
```bash
# cuda switch
export CUDA_HOME=$HOME/cudas/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
```
## Check Installing success
```bash
which nvcc && nvcc --version
```
# Intsall Cudnn
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

## Download Cudnn by archive
```bash
# an example for downloading the tar archive and extracting it.
mkdir -p ~/tmp/cudnn && cd ~/tmp/cudnn
tar -xf /path/to/cudnn-linux-*-archive.tar.xz
cd cudnn-linux-*-archive
```
## INstall Cudnn by archive
```bash
mkdir -p "$CUDA_HOME/include" "$CUDA_HOME/lib64"
cp include/cudnn*.h "$CUDA_HOME/include/"
cp -P lib/libcudnn* "$CUDA_HOME/lib64/"
chmod a+r "$CUDA_HOME/include"/cudnn*.h "$CUDA_HOME/lib64"/libcudnn*
```
## Check Installing success
```bash
grep -HE "CUDNN_(MAJOR|MINOR|PATCHLEVEL)" /xxx/cudas/cuda-12.4/include/cudnn_version.h
```
or
```bash
grep -HE "CUDNN_(MAJOR|MINOR|PATCHLEVEL)" "$CUDA_HOME/include/cudnn_version.h"
```
# Create env
```bash
conda create -n verl_911_cpy -y python=3.10 pip

conda create -n verl_911 python=3.10
conda activate verl_911

conda create -n verl_1019 python=3.10
conda activate verl_1019
```
# Install Torch
The highest available pytorch version using Cuda 12.4 is 2.6.0.
Install Apex need pytorch.

```python
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
# Install Dependencies
```bash
pip install "distro<2,>=1.7.0"
pip install -U "pyyaml>=5.1" datasets huggingface-hub
```
# Install Apex
To use Megatron-LM or FSDP as the training backend, NVIDIA’s Apex library need installing. These training backends invoke the optimizers and mixed-precision utilities in Apex to improve the performance and efficiency of large-model training.
```bash
# change directory to anywher you like, in verl source code directory is not recommended
git clone https://github.com/NVIDIA/apex.git && \
cd apex && \
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

# Git Clone Verl
```bash
git clone https://github.com/volcengine/verl.git
cd verl
```
# Install Inference and Training Backend Engines
Training:
- FSDP
- Megatron-LM
Inference:
- vllm
- SGLang
- huggingface TGI integration

Before install apex, we have installed pytorch, which will be installed again in the script.
```bash
cd verl
# Make sure you have activated verl conda env
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```
安装完成之后，还有两个安装包的wheel，文件过大会导致ray超过最大文件上限而报错，把它移到installing文件夹或者删除即可。
```bash
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl
```

# fix dependencies conflict
## fix opentelemetry dependencies conflict
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

## fix megatron-core dependencies conflict
```bash
# 2) 补齐 megatron-core 0.12.2 声明的依赖（如果你确实要用 megatron-core）
python -m pip install flask-restful nltk tensorstore zarr pytest-cov pytest-mock pytest-random-order
# 仅当有 NVIDIA/CUDA 环境时再装（否则大概率失败）
python -m pip install nvidia-modelopt
```

## fix cudnn version bug
```bash
pip install --no-cache-dir "nvidia-cudnn-cu12==9.1.0.70"
```

# Install verl
```bash
cd verl
pip install --no-deps -e .
```
```bash
# 0) 清掉错误的 gh（如果之前 pip 安装过）
pip uninstall -y gh || true
# 1) 安装 GitHub 官方 CLI（conda-forge，免sudo）
conda install -c conda-forge -y gh
# 2) 确认版本与路径（应看到 "gh version x.y.z"）
gh --version
type -a gh
```

# 写在最后的后知后觉
安装带 +cu12x 的 PyTorch 轮子时，不需要另装系统 CUDA/cuDNN；只要显卡驱动版本够新，PyTorch 会使用它自己随包携带的运行库。因此，不装cuda和cudnn，直接装driver版本符合的pytorch版本，就可以直接使用对应的cuda和cudnn。
只有驱动必须系统级安装：驱动负责让 OS 与 GPU 通讯；而 CUDA/cuDNN 运行库由 PyTorch 轮子在 Python 环境内提供与加载。
所以要想满足verl的cuDNN: Version >= 9.8.0要求，需要使用pytorch大于2.6.0的版本，或者更高版本的cuda的pytorch，这样torch自己构建的wheel带的cudnn才能满足cuDNN: Version >= 9.8.0要求。建议是提高pytorch版本，因为如果自己构建的话，verl默认cuda12.4。
最后使用cuda12.4的pytorch为2.6.0，只能使用cudnn为9.1.0.70。

## 检查cudnn的自带版本
直接装一个cuda版本的pytoch，print其中的cudnn版本号
```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("is_available:", torch.cuda.is_available())
PY
```
## Note
从 pip 24.2 起，pip check 会额外检查“包是否声明支持当前平台”。一些老包（例如只发布了很早期 manylinux 的 wheel 或元数据写得不完整的包）即使能正常安装、使用，也会被标成“not supported”。这是已知现象，社区里像 ninja/xgboost/catboost 都被这样提示过。
所以decord会报decord 0.6.0 is not supported on this platform，但是可以忽略。

多个GPU作为worker，完成前向计算完成一次micro，直到完成的足够数量的size达到mini batch size 要求，就完成一次后向梯度更新，如此累计多次（梯度累计）完成一次train batch size。

git config --global user.email "you@example.com"
git config --global user.name "Your Name"