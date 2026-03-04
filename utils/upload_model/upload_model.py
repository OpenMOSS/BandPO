#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import inspect
from pathlib import Path
from getpass import getpass

def require_bandpo_dir() -> str:
    bandpo_dir = os.environ.get("BandPODir")
    if not bandpo_dir:
        print("未检测到 BandPODir。请先在 bash 中执行：source /path_to_BandPO/init.sh")
        sys.exit(1)
    return bandpo_dir

def get_hf_token() -> str:
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("未检测到 Hugging Face token（HUGGING_FACE_HUB_TOKEN/HF_TOKEN）。"
              "建议先在 bash 中执行：source /path_to_BandPO/init.sh")
        token = getpass("请输入 Hugging Face token（形如 hf_...）：").strip()
    if not token:
        print("未输入 token，退出。")
        sys.exit(1)
    return token

def sanity_check_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        print(f"本地模型目录不存在：{model_dir}")
        sys.exit(1)

    # 仅做轻量检查（不强制），避免误传空目录
    key_files = ["config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
    present = [f for f in key_files if (model_dir / f).exists()]
    if not present:
        print(f"警告：在 {model_dir} 未检测到常见配置/分词器文件（{', '.join(key_files)}）。"
              "如果这是特殊结构目录，请确认无误后继续。")

def hf_api_client(token: str):
    # 强制不用镜像：endpoint 设为官方
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

    # 注意：HfApi 支持 endpoint/token 参数。 :contentReference[oaicite:1]{index=1}
    from huggingface_hub import HfApi
    return HfApi(token=token, endpoint=os.environ["HF_ENDPOINT"])

def create_repo_if_needed(api, repo_id: str, private: bool):
    # 优先用 api.create_repo（版本差异更小）；失败再 fallback 到 root create_repo
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    except TypeError:
        from huggingface_hub import create_repo
        create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=api.token)

def upload_folder_robust(api, folder_path: Path, repo_id: str, commit_message: str):
    # 基本参数（upload_folder 的具体可选参数见官方上传指南：allow/ignore/delete_patterns 等） :contentReference[oaicite:2]{index=2}
    kwargs = dict(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        # 视需要过滤；默认不过滤，避免漏文件
        # ignore_patterns=["**/.cache/**", "**/__pycache__/**", "**/.DS_Store"],
    )

    # 大目录上传：如果当前 huggingface_hub 支持 multi_commits，就打开，提升容错
    sig = inspect.signature(api.upload_folder)
    if "multi_commits" in sig.parameters:
        kwargs["multi_commits"] = True
        if "multi_commits_verbose" in sig.parameters:
            kwargs["multi_commits_verbose"] = True

    return api.upload_folder(**kwargs)

def main():
    require_bandpo_dir()
    token = get_hf_token()

    username = "Yuan-Li-FNLP"
    PRIVATE = True  # 默认私有更稳妥；确认合规后可改 False

    models = {
        "Llama-3.2-3B-Instruct": Path("/remote-home1/share/models/meta-llama/Llama-3.2-3B-Instruct"),
        "Meta-Llama-3-8B-Instruct": Path("/remote-home1/share/models/meta-llama/Meta-Llama-3-8B-Instruct/"),
    }

    api = hf_api_client(token)

    for repo_name, local_dir in models.items():
        local_dir = local_dir.resolve()
        sanity_check_model_dir(local_dir)

        repo_id = f"{username}/{repo_name}"
        print(f"\n=== 准备上传 ===")
        print(f"Local: {local_dir}")
        print(f"Repo : {repo_id} (private={PRIVATE})")

        print("1) 创建/确认仓库存在 ...")
        create_repo_if_needed(api, repo_id=repo_id, private=PRIVATE)

        print("2) 开始上传（可能耗时较久） ...")
        upload_folder_robust(
            api,
            folder_path=local_dir,
            repo_id=repo_id,
            commit_message=f"Upload {repo_name} from local folder",
        )

        print(f"完成：{repo_id}")

    print("\n全部上传完成。")

if __name__ == "__main__":
    main()
