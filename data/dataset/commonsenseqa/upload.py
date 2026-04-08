# upload_dataset.py
import os
import sys
from getpass import getpass
from huggingface_hub import login, HfApi

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

# 3) Basic info and paths
# Logic Change: Read username from environment variable instead of hardcoding
username = os.environ.get("HUGGING_FACE_USERNAME")
if not username:
    print("HUGGING_FACE_USERNAME not detected. Please run 'source /path_to_BandPO/init.sh' in bash first.")
    sys.exit(1)

repo_id = f"{username}/commonsenseqa"
folder_path = os.path.join(bandpo_dir, "data", "dataset", "commonsenseqa")

if not os.path.isdir(folder_path):
    print(f"Local dataset directory does not exist: {folder_path}")
    sys.exit(1)

# 4) Login and Create/Upload
login(token=token)  # Caches locally
api = HfApi()

# Create repo (private=True by default for safety, exist_ok=True prevents errors if already exists)
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo=".",
    commit_message="Upload CommonsenseQA dataset folder",
)
print("Upload finished:", repo_id, "from", folder_path)
