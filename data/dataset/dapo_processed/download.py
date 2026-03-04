# download_dataset.py
import os
import sys
from getpass import getpass
from huggingface_hub import snapshot_download

# 1) BandPODir must exist first
bandpo_dir = os.environ.get("BandPODir")
if not bandpo_dir:
    print("BandPODir not detected. Please run 'source /path_to_BandPO/init.sh' in bash first.")
    sys.exit(1)

# Avoid using mirrors (force official endpoint)
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

# 2) Read token: try two environment variables first, prompt input if not found
token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
if not token:
    print("Hugging Face token (HUGGING_FACE_HUB_TOKEN/HF_TOKEN) not detected.")
    print("It is recommended to run 'source /path_to_BandPO/init.sh' in bash first.")
    token = getpass("Please enter Hugging Face token (starting with hf_...): ")
    if not token:
        print("No token entered. Exiting.")
        sys.exit(1)

# 3) Assemble local directory path
repo_id = "Yuan-Li-FNLP/dapo_processed"
local_dir = os.path.join(bandpo_dir, "data", "dataset", "dapo_processed")
os.makedirs(local_dir, exist_ok=True)

# 4) Download
local_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # More stable in HPC/Containers
    token=token,
)
print("Downloaded to:", local_dir)