#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

def get_first_record_keys(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # 如果有注释行（可选）
            if line.startswith("#"):
                continue

            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"第一条非空记录不是一个 JSON 对象，而是：{type(obj)}")

            keys = list(obj.keys())
            print(f"文件: {jsonl_path}")
            print("第一条记录的 key 列表：")
            for k in keys:
                print(f"- {k}")
            return

    raise ValueError("文件中没有找到任何非空的 JSON 行")

def main():
    parser = argparse.ArgumentParser(description="获取 jsonl 文件第一条记录的 key 列表")
    parser.add_argument("jsonl_path", help="jsonl 文件路径")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"找不到文件：{jsonl_path}")

    get_first_record_keys(jsonl_path)

if __name__ == "__main__":
    main()
