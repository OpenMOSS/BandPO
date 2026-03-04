#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 jsonl 文件中随机抽取 n 条记录（不一次性全读入内存，用蓄水池抽样）。

用法示例：
    python sample_jsonl.py input.jsonl -n 100 -o sampled.jsonl --seed 42
    # 如果不加 -o，就直接打印到 stdout
"""

import argparse
import random
from pathlib import Path


def sample_jsonl(
    input_path: Path,
    n: int,
    output_path: Path | None = None,
    seed: int | None = None,
    encoding: str = "utf-8",
) -> None:
    rng = random.Random(seed)

    reservoir: list[str] = []
    seen = 0  # 只统计非空行

    with input_path.open("r", encoding=encoding) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue  # 跳过空行
            seen += 1
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                # 蓄水池抽样：以 n/seen 的概率保留新样本
                j = rng.randint(0, seen - 1)
                if j < n:
                    reservoir[j] = line

    if seen == 0:
        raise ValueError(f"文件 {input_path} 中没有有效行（全是空行？）")

    if output_path is None:
        # 打印到 stdout
        for line in reservoir:
            print(line)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding=encoding) as f_out:
            for line in reservoir:
                f_out.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 jsonl 文件中随机抽取 n 条记录"
    )
    parser.add_argument("input", type=Path, help="输入 jsonl 文件路径")
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        required=True,
        help="要抽取的记录条数 n",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 jsonl 文件路径（不指定则输出到 stdout）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（可选，便于复现实验）",
    )
    args = parser.parse_args()

    sample_jsonl(
        input_path=args.input,
        n=args.num,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
