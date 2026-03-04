#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_by_field.py

根据某个字段（默认: filter_state）把一个巨大的 JSONL 拆分成多个 JSONL 文件，
每个字段取值一个文件，例如：
  filter_state=kept.jsonl
  filter_state=dropped.jsonl
  filter_state=__MISSING__.jsonl
  ...

特点：
- 纯 streaming，一行一行读，适合超大 JSONL。
- 支持 dot-path 字段名，例如 "meta.filter_state"。
- 自动按字段取值创建输出文件，命名为 "{field_key}={value_sanitized}.jsonl"。

用法示例（按 filter_state 拆分）：
  python split_by_field.py \
    --jsonl huge.records.jsonl \
    --out-dir split_by_filter_state \
    --field-key filter_state

用法示例（按嵌套字段 meta.status 拆分）：
  python split_by_field.py \
    --jsonl huge.records.jsonl \
    --out-dir split_by_status \
    --field-key meta.status
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, IO

# 优先用 orjson，加速 JSON 解析
try:
    import orjson

    def json_loads(b: bytes):
        return orjson.loads(b)

except Exception:
    def json_loads(b: bytes):
        return json.loads(b.decode("utf-8", errors="replace"))


def get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """支持 dot path 的取值，比如 path='meta.filter_state'。"""
    cur: Any = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def sanitize_for_filename(value: str, max_len: int = 80) -> str:
    """
    把任意字符串变成适合作为文件名的一部分：
    - 非 [0-9a-zA-Z._-] 的字符全部替换为 '_'
    - 截断到 max_len
    - 避免为空
    """
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    out = []
    for ch in value:
        if ch in safe_chars:
            out.append(ch)
        else:
            out.append("_")
        if len(out) >= max_len:
            break
    s = "".join(out)
    return s if s else "EMPTY"


def split_by_field(jsonl_path: Path,
                   out_dir: Path,
                   field_key: str,
                   print_every: int = 100000) -> None:
    """
    按 field_key 拆分 JSONL，输出到 out_dir 下的多个文件。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input JSONL : {jsonl_path}")
    print(f"[INFO] Output dir  : {out_dir}")
    print(f"[INFO] field-key   : {field_key}")

    total_lines = 0
    bad_json = 0
    missing_field = 0

    # 每个字段取值 -> 输出文件句柄
    writers: Dict[str, IO[bytes]] = {}
    # 统计各字段值写了多少行
    counts: Dict[str, int] = {}

    try:
        with jsonl_path.open("rb", buffering=1024 * 1024) as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                if not line.strip():
                    continue

                total_lines += 1

                try:
                    obj = json_loads(line)
                except Exception:
                    bad_json += 1
                    if total_lines % print_every == 0:
                        print(f"[WARN] line {total_lines:,}: JSON parse error (累积 {bad_json})",
                              file=sys.stderr)
                    continue

                if not isinstance(obj, dict):
                    continue

                v = get_nested(obj, field_key, None)
                if v is None:
                    missing_field += 1
                    v_str = "__MISSING__"
                else:
                    v_str = str(v)

                # 懒创建对应的输出文件
                if v_str not in writers:
                    fname = f"{field_key}={sanitize_for_filename(v_str)}.jsonl"
                    out_path = out_dir / fname
                    # 'ab' 以防你后续想继续 append；这里正常流程是新建
                    writers[v_str] = out_path.open("ab", buffering=1024 * 1024)
                    counts[v_str] = 0
                    print(f"[INFO] New bucket '{v_str}' -> {out_path}")

                writers[v_str].write(line)
                counts[v_str] += 1

                if total_lines % print_every == 0:
                    buckets = len(writers)
                    print(
                        f"[INFO] lines={total_lines:,}, buckets={buckets:,}, "
                        f"bad_json={bad_json:,}, missing_field={missing_field:,}"
                    )

    finally:
        # 关闭所有打开的文件
        for v_str, f in writers.items():
            try:
                f.close()
            except Exception:
                pass

    print("[SUMMARY]")
    print(f"  total_lines   = {total_lines:,}")
    print(f"  bad_json      = {bad_json:,}")
    print(f"  missing_field = {missing_field:,}")
    print(f"  buckets       = {len(counts):,}")
    for v_str, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    value={v_str!r:20s} -> {c:,} lines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True,
                    help="输入的大 JSONL 文件")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="输出目录，每个字段取值一个 JSONL 文件")
    ap.add_argument("--field-key", type=str, default="filter_state",
                    help="按哪个字段拆分，支持 dot-path，例如 'filter_state' 或 'meta.filter_state'")
    ap.add_argument("--print-every", type=int, default=100000,
                    help="每处理多少行打印一次进度")

    args = ap.parse_args()

    if not args.jsonl.is_file():
        print(f"[ERROR] 输入文件不存在: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    split_by_field(
        jsonl_path=args.jsonl,
        out_dir=args.out_dir,
        field_key=args.field_key,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    main()
