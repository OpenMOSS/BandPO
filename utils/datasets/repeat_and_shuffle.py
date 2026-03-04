#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path
import sys

import pandas as pd


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("true", "t", "1", "yes", "y"):
        return True
    if v in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("布尔参数只能是 true/false 。")


def get_root_and_ext(p: Path):
    # 保留多重后缀（例如 ".snappy.parquet"）
    ext = "".join(p.suffixes)
    root = p.name[: -len(ext)] if ext else p.name
    return root, ext


def main():
    parser = argparse.ArgumentParser(description="重复并可选打乱 Parquet 数据。")
    parser.add_argument("parquet_path", type=str, help="待读取的 Parquet 文件路径")
    parser.add_argument("-n", "--repeat", type=int, required=True, help="重复遍数 n（正整数）")
    parser.add_argument("--shuffle", type=str2bool, required=True, help="是否随机打乱：true 或 false")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（未提供则使用当前时间戳）")

    args = parser.parse_args()

    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists() or not parquet_path.is_file():
        print(f"[错误] 文件不存在或不可读：{parquet_path}", file=sys.stderr)
        sys.exit(1)

    n = args.repeat
    if n <= 0:
        print("[错误] 重复遍数 n 必须为正整数。", file=sys.stderr)
        sys.exit(1)

    shuffle = args.shuffle
    seed = args.seed

    # 读取
    try:
        df = pd.read_parquet(str(parquet_path))
    except Exception as e:
        print(f"[错误] 读取 Parquet 失败：{e}", file=sys.stderr)
        sys.exit(1)

    original_rows = len(df)

    # 重复
    if n == 1:
        out_df = df.copy()
    else:
        out_df = pd.concat([df] * n, ignore_index=True)

    # 打乱（如需）
    used_seed = None
    if shuffle:
        used_seed = seed if seed is not None else int(time.time())
        out_df = out_df.sample(frac=1, random_state=used_seed).reset_index(drop=True)

    # 生成输出路径（同目录，保留原扩展名）
    root, ext = get_root_and_ext(parquet_path)
    if not ext:
        ext = ".parquet"  # 兜底：若原文件无扩展名，则使用 .parquet

    if shuffle:
        out_name = f"{root}_{n}_shuffle{used_seed}{ext}"
    else:
        out_name = f"{root}_{n}{ext}"

    out_path = parquet_path.with_name(out_name)

    # 写出
    try:
        out_df.to_parquet(str(out_path), index=False)
    except Exception as e:
        print(f"[错误] 写出 Parquet 失败：{e}", file=sys.stderr)
        sys.exit(1)

    # 汇总信息
    print("==== 完成 ====")
    print(f"输入文件：{parquet_path}")
    print(f"原始行数：{original_rows}")
    print(f"重复倍数：{n}")
    if shuffle:
        print(f"是否打乱：是（seed={used_seed}）")
    else:
        print("是否打乱：否")
    print(f"输出文件：{out_path}")
    print(f"输出行数：{len(out_df)}")


if __name__ == "__main__":
    main()
