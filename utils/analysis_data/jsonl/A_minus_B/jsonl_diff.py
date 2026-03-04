#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 jsonl 文件 A 中剔除与 jsonl 文件 B 在某个 key 上相同的样本，输出 A - B。

用法示例：
    python jsonl_diff.py A.jsonl B.jsonl out_A_minus_B.jsonl --key uid

说明：
- 默认使用 key="uid" 来判断交集。
- B 中缺少该 key 的行会被忽略（不参与交集）。
- A 中缺少该 key 的行会被保留（因为无法判断是否在交集中）。
"""

import argparse
import json
from pathlib import Path
from typing import Set, Any


def load_key_set(path: Path, key: str) -> Set[Any]:
    """从 jsonl 文件中收集所有包含指定 key 的值，返回一个 set。"""
    values: Set[Any] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{e}") from e
            if key in obj:
                values.add(obj[key])
    return values


def filter_a_minus_b(file_a: Path, file_b: Path, out_path: Path, key: str) -> None:
    """实现 A - B：从 A 中剔除在 B 中具有相同 key 值的行。"""
    # 1. 先读取 B，收集所有 key 的取值
    key_set_b = load_key_set(file_b, key)
    print(f"[INFO] 从 {file_b} 中共收集到 {len(key_set_b)} 个唯一的 `{key}` 值。")

    kept = 0
    dropped = 0

    # 2. 遍历 A，判断是否在 B 的集合中
    with file_a.open("r", encoding="utf-8") as fa, \
            out_path.open("w", encoding="utf-8") as fo:

        for line_no, line in enumerate(fa, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{file_a} 第 {line_no} 行不是合法 JSON：{e}") from e

            if key not in obj:
                # 如果 A 中这一行没有 key，则无法参与交集，保留
                fo.write(raw + "\n")
                kept += 1
                continue

            value = obj[key]
            if value in key_set_b:
                # 在 B 中出现过 -> 认为是交集的一部分，被剔除
                dropped += 1
            else:
                fo.write(raw + "\n")
                kept += 1

    print(f"[INFO] 处理完成：保留 {kept} 行，剔除 {dropped} 行。")
    print(f"[INFO] 结果已写入：{out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="从 jsonl 文件 A 中剔除与 B 在某个 key 上相同的样本，输出 A - B。"
    )
    parser.add_argument("file_a", type=Path, help="输入 jsonl 文件 A 的路径")
    parser.add_argument("file_b", type=Path, help="输入 jsonl 文件 B 的路径")
    parser.add_argument("out", type=Path, help="输出 jsonl 文件路径（A-B 的结果）")
    parser.add_argument(
        "--key",
        type=str,
        default="uid",
        help='用于判定交集的字段名（默认："uid"）'
    )

    args = parser.parse_args()
    filter_a_minus_b(args.file_a, args.file_b, args.out, args.key)


if __name__ == "__main__":
    main()
