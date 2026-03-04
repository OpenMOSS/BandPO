#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 filter_state 将大 JSONL 文件拆分为 kept.jsonl 和 dropped.jsonl。

用法示例：
  python split_kept_dropped.py \
    --jsonl /path/to/huge.jsonl \
    --out-kept /path/to/huge.kept.jsonl \
    --out-dropped /path/to/huge.dropped.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# 优先使用 orjson，加速解析
try:
    import orjson

    def json_loads(s: str):
        return orjson.loads(s)

    def json_dumps(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:
    def json_loads(s: str):
        return json.loads(s)

    def json_dumps(obj) -> str:
        return json.dumps(obj, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="输入的大 JSONL 文件路径",
    )
    ap.add_argument(
        "--out-kept",
        type=Path,
        required=True,
        help="输出 kept 记录的 JSONL 文件路径",
    )
    ap.add_argument(
        "--out-dropped",
        type=Path,
        required=True,
        help="输出 dropped 记录的 JSONL 文件路径",
    )
    ap.add_argument(
        "--print-every",
        type=int,
        default=100000,
        help="每处理多少行打印一次进度",
    )
    args = ap.parse_args()

    in_path: Path = args.jsonl
    out_kept: Path = args.out_kept
    out_dropped: Path = args.out_dropped

    if not in_path.is_file():
        print(f"[ERROR] 输入文件不存在: {in_path}", file=sys.stderr)
        sys.exit(1)

    # 确保输出目录存在
    out_kept.parent.mkdir(parents=True, exist_ok=True)
    out_dropped.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input JSONL : {in_path}")
    print(f"[INFO] Out kept    : {out_kept}")
    print(f"[INFO] Out dropped : {out_dropped}")

    total = 0
    n_kept = 0
    n_dropped = 0
    n_other = 0
    n_bad_json = 0

    # 文本模式 + 大 buffer
    with in_path.open("r", encoding="utf-8", buffering=1024 * 1024) as fin, \
         out_kept.open("w", encoding="utf-8", buffering=1024 * 1024) as fkept, \
         out_dropped.open("w", encoding="utf-8", buffering=1024 * 1024) as fdrop:

        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            total += 1

            try:
                obj = json_loads(line)
            except Exception:
                n_bad_json += 1
                if total % args.print_every == 0:
                    print(
                        f"[WARN] line {total:,}: JSON parse error (累计 {n_bad_json} 条)",
                        file=sys.stderr,
                    )
                continue

            if not isinstance(obj, dict):
                n_other += 1
                continue

            fs = obj.get("filter_state", None)

            if fs == "kept":
                fkept.write(json_dumps(obj) + "\n")
                n_kept += 1
            elif fs == "dropped":
                fdrop.write(json_dumps(obj) + "\n")
                n_dropped += 1
            else:
                # 既不是 kept 也不是 dropped 的记录（比如缺失字段），统计一下
                n_other += 1

            if total % args.print_every == 0:
                print(
                    f"[PROGRESS] lines={total:,}  kept={n_kept:,}  dropped={n_dropped:,}  "
                    f"other={n_other:,}  bad_json={n_bad_json:,}"
                )

    print("[DONE] Split finished.")
    print(f"  total lines     : {total:,}")
    print(f"  kept            : {n_kept:,}")
    print(f"  dropped         : {n_dropped:,}")
    print(f"  other filter    : {n_other:,}  (filter_state 不是 kept/dropped 或缺失)")
    print(f"  bad json lines  : {n_bad_json:,}")
    print(f"  out kept file   : {out_kept}")
    print(f"  out dropped file: {out_dropped}")


if __name__ == "__main__":
    main()
