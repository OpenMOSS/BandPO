#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_online_rambling_filter.py

用途：
- 模拟 Ray 侧对 OnlineRamblingFilterServer 的调用；
- 验证：
    1) keep_mask 是否返回成功；
    2) keep_mask 长度是否等于 JSONL 行数；
    3) keep_mask 的顺序是否与 JSONL 行顺序一一对应。

用法示例：

1) 用你真实训练产生的 JSONL 测试：
    python test_online_rambling_filter.py \
        --base-url http://10.176.59.108:8008 \
        --jsonl /path/to/rambling_filter_step000100_epoch0_ts20251122.jsonl

2) 让脚本自己造一个测试 JSONL：
    python test_online_rambling_filter.py \
        --base-url http://10.176.59.108:8008
"""

import argparse
import json
import os
import time
from typing import Optional, List, Dict, Any

import numpy as np
import requests


class OnlineRamblingFilterClient:
    """与你训练代码里的版本保持一致的精简版客户端"""

    def __init__(
        self,
        base_url: str,
        timeout: float = 600.0,
        project_name: Optional[str] = None,
        exp_name: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

        # 如果只传了一个，直接报错，避免 server 侧目录结构混乱
        if bool(project_name) ^ bool(exp_name):
            raise ValueError(
                "OnlineRamblingFilterClient: project_name 和 exp_name 必须同时提供或同时为空。"
            )

        self.project_name = project_name
        self.exp_name = exp_name

    def filter_jsonl(
        self,
        jsonl_path: str,
        project_name: Optional[str] = None,
        exp_name: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        # 允许在调用时临时覆盖；如果不传就用实例上的默认值
        if project_name is None:
            project_name = self.project_name
        if exp_name is None:
            exp_name = self.exp_name

        # 两个要么都给，要么都不给
        if bool(project_name) ^ bool(exp_name):
            raise ValueError(
                "OnlineRamblingFilterClient.filter_jsonl: project_name 和 exp_name 必须同时提供或同时为空。"
            )

        payload = {"jsonl_path": jsonl_path}
        if project_name and exp_name:
            payload["project_name"] = project_name
            payload["exp_name"] = exp_name

        try:
            resp = requests.post(
                self.base_url + "/filter",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[WARN] Online filter request failed: {e}")
            return None

        if not data.get("ok", False):
            print(f"[WARN] Online filter service returned error: {data}")
            return None

        mask = data.get("keep_mask", None)
        if mask is None:
            print("[WARN] Online filter response missing keep_mask")
            return None

        try:
            arr = np.array(mask, dtype=bool)
        except Exception as e:
            print(f"[WARN] Online filter keep_mask cast failed: {e}")
            return None

        return arr


def count_lines(path: str) -> int:
    """统计 JSONL 文件的行数"""
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def build_dummy_jsonl(path: str, num_samples: int = 8) -> None:
    """
    构造一个“长得像 Ray 写出来”的 JSONL：
    - 有 uid / prompt_text / response_text / reward_extra_infos.acc 等字段；
    - 其中有几条 response 特意弄得比较“啰嗦”，方便触发部分 pattern（不保证一定会触发）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[INFO] Building dummy JSONL at: {path}")

    records: List[Dict[str, Any]] = []
    for i in range(num_samples):
        uid = f"dummy-{i:03d}"

        if i in (2, 5):
            # 故意啰嗦一点的样本
            response = (
                "This is a very very very very very long and repetitive response. " * 5
                + "LOOP LOOP LOOP. I am thinking, thinking, thinking..."
            )
        else:
            response = f"Short answer {i}. 42."

        rec = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epoch": 0,
            "global_step": 0,
            "uid": uid,
            "filter_state": "kept",  # 与 offline pipeline 一致
            "prompt_text": f"Dummy prompt {i}",
            "response_text": response,
            "reward_seq_sum": 1.0,
            "reward_extra_infos": {
                "score": 1.0,
                "acc": True,  # 让它们都算在 'correct' 里（匹配 --cls correct）
            },
            "ce_ref_seq_mean": 1.0,  # 随便填个值
        }
        records.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_uids(path: str) -> List[Any]:
    """把 JSONL 里的 uid 按行顺序读出来，方便和 keep_mask 对比"""
    uids: List[Any] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                j = json.loads(line)
            except Exception:
                uids.append(None)
                continue
            uids.append(j.get("uid"))
    return uids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8008",
        help="Online rambling filter server 的 base URL",
    )
    ap.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="用于测试的 JSONL 路径；不指定时会自动生成一个 dummy JSONL",
    )
    ap.add_argument(
        "--tmp-dir",
        type=str,
        default="./tmp_online_filter_test",
        help="自动生成 JSONL 时使用的临时目录",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="自动生成 JSONL 时的样本数",
    )
    ap.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="可选：本次测试的 project_name（需要和 exp_name 一起提供）",
    )
    ap.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="可选：本次测试的 exp_name（需要和 project_name 一起提供）",
    )
    args = ap.parse_args()

    # 0) 检查 project_name / exp_name 是否成对出现
    if bool(args.project_name) ^ bool(args.exp_name):
        print(
            "[ERROR] --project-name 和 --exp-name 必须同时提供或同时省略，"
            f"当前 project_name={args.project_name!r}, exp_name={args.exp_name!r}"
        )
        return

    # 1) 准备 JSONL
    if args.jsonl is not None:
        jsonl_path = args.jsonl
        print(f"[INFO] Using existing JSONL: {jsonl_path}")
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        jsonl_path = os.path.join(args.tmp_dir, f"dummy_{ts}.jsonl")
        build_dummy_jsonl(jsonl_path, num_samples=args.num_samples)

    if not os.path.exists(jsonl_path):
        print(f"[ERROR] JSONL not found: {jsonl_path}")
        return

    # 2) 数一下行数
    num_lines = count_lines(jsonl_path)
    print(f"[INFO] JSONL line count = {num_lines}")

    # 3) 调用 online filter
    client = OnlineRamblingFilterClient(
        base_url=args.base_url,
        timeout=600.0,
        project_name=args.project_name,
        exp_name=args.exp_name,
    )
    keep_mask = client.filter_jsonl(
        jsonl_path=jsonl_path,
        # 这里也可以不传，让它用构造函数里的默认 project/exp
        # project_name=args.project_name,
        # exp_name=args.exp_name,
    )

    if keep_mask is None:
        print("[ERROR] keep_mask is None, please check server logs.")
        return

    print(f"[INFO] keep_mask shape = {keep_mask.shape}")

    # 4) 检查长度是否匹配
    if len(keep_mask) != num_lines:
        print(
            f"[WARN] keep_mask length ({len(keep_mask)}) != JSONL line count ({num_lines})"
        )
    else:
        print("[INFO] keep_mask length matches JSONL line count ✅")

    # 5) 打印逐行的 (idx, uid, keep)
    uids = load_uids(jsonl_path)
    if len(uids) != num_lines:
        print(
            f"[WARN] uid count ({len(uids)}) != JSONL line count ({num_lines}); "
            f"可能有空行或解析失败。"
        )

    print("\n[RESULT] Per-line keep flags:")
    print("idx\tuid\tkeep")
    for i in range(min(len(keep_mask), len(uids))):
        print(f"{i}\t{uids[i]}\t{bool(keep_mask[i])}")

    # 6) 额外统计一下 bad 样本的位置
    bad_idx = np.where(~keep_mask)[0]
    print(f"\n[SUMMARY] bad_idx = {bad_idx.tolist()} ({len(bad_idx)}/{num_lines})")


if __name__ == "__main__":
    main()
