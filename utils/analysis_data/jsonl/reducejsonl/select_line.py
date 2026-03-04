#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_jsonl_line.py

从一个（可能很大的）JSONL 文件中，按行号抽取某一行，原样写到输出路径。

用法示例：
    python extract_jsonl_line.py \
        --input /path/to/huge.jsonl \
        --line-no 12345 \
        --output /path/to/single_line.jsonl
"""

import argparse
import gzip
import os
import sys
from typing import TextIO


def smart_open(path: str, mode: str = "rt", encoding: str = "utf-8") -> TextIO:
    """
    根据后缀自动选择普通 open 或 gzip.open。
    mode: "rt" 读文本, "wt"/"at" 写文本
    """
    if path.endswith(".gz"):
        return gzip.open(path, mode=mode, encoding=encoding)  # type: ignore[arg-type]
    return open(path, mode=mode, encoding=encoding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 JSONL 中按行号抽取某一行，保存到指定位置。"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入 JSONL 路径（支持 .gz）",
    )
    parser.add_argument(
        "--line-no",
        "-n",
        type=int,
        required=True,
        help="要抽取的行号（1-based，第一行为 1）",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="输出文件路径（已有则覆盖，同样可为 .gz）",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="若指定，则以追加模式写入输出文件（默认覆盖）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.line_no <= 0:
        print(f"[ERROR] --line-no 必须为正整数（当前: {args.line_no}）", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"[ERROR] 输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    target_line = None

    # 读取指定行
    with smart_open(args.input, "rt") as fin:
        for idx, line in enumerate(fin, start=1):
            if idx == args.line_no:
                target_line = line
                break

    if target_line is None:
        print(
            f"[ERROR] 文件总行数 < 目标行号：未找到第 {args.line_no} 行。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 写出到指定位置
    write_mode = "at" if args.append else "wt"
    with smart_open(args.output, write_mode) as fout:
        # 确保以换行结束（如果原来没有）
        if not target_line.endswith("\n"):
            target_line = target_line + "\n"
        fout.write(target_line)

    print(
        f"[OK] 已从 {args.input} 抽取第 {args.line_no} 行，写入 {args.output} "
        f"（模式: {'append' if args.append else 'overwrite'}）。"
    )


if __name__ == "__main__":
    main()
