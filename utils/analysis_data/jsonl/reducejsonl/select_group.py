#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_one_group.py

从一个巨大的 JSONL 中，按 group-key 取值筛选，只保留特定 group 的样本，
并写到新的 JSONL 文件中。

特点：
- 只按行 streaming，不把所有数据载入内存。
- 支持 group-key 为 dot path，例如 "meta.group_id"。
- 支持一次筛选一个或多个 group 值（逗号分隔）。

用法示例（只保留 uid == 'cc883230-...' 这一组）：
  python filter_one_group.py \
    --jsonl huge.kept.jsonl \
    --out-jsonl huge.kept.uid_cc883230.jsonl \
    --group-key "uid" \
    --group-values "cc883230-eb75-4207-85b4-4fc3059b86e4"

用法示例（保留多个组）：
  python filter_one_group.py \
    --jsonl huge.kept.jsonl \
    --out-jsonl huge.kept.some_uids.jsonl \
    --group-key "uid" \
    --group-values "uid1,uid2,uid3"
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Set

# 优先用 orjson，加速 JSON 解析
try:
    import orjson

    def json_loads(b: bytes):
        return orjson.loads(b)

except Exception:
    def json_loads(b: bytes):
        return json.loads(b.decode("utf-8", errors="replace"))


def get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """支持 dot path 的取值，比如 path='meta.group_id'。"""
    cur: Any = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def load_target_groups(group_values: str) -> Set[str]:
    """
    把命令行传入的 group-values 字符串解析成一个集合。
    例如 "a,b,c" -> {"a", "b", "c"}；会自动 strip 空白并忽略空值。
    """
    items = []
    for part in group_values.split(","):
        p = part.strip()
        if p:
            items.append(p)
    return set(items)


def filter_by_group(jsonl_path: Path,
                    out_path: Path,
                    group_key: str,
                    target_groups: Iterable[str],
                    print_every: int = 100000) -> None:
    """
    按 group_key 过滤，只写入 group 值在 target_groups 里的样本。
    """
    target_groups_set: Set[str] = set(str(x) for x in target_groups)
    if not target_groups_set:
        print("[ERROR] target_groups 为空，什么也不会被写出。", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Filtering from {jsonl_path} -> {out_path}")
    print(f"[INFO] group-key      : {group_key}")
    print(f"[INFO] target-groups  : {sorted(target_groups_set)}")

    total_lines = 0
    written_lines = 0
    bad_json = 0
    missing_group = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("rb", buffering=1024 * 1024) as fin, \
         out_path.open("wb", buffering=1024 * 1024) as fout:

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

            g = get_nested(obj, group_key, None)
            if g is None:
                # 可以按需要决定是否把 None 当作某个特殊组，比如 "__MISSING__"
                missing_group += 1
                g_str = "__MISSING__"
            else:
                g_str = str(g)

            if g_str in target_groups_set:
                # 原始行里已有 '\n'，直接写
                fout.write(line)
                written_lines += 1

            if total_lines % print_every == 0:
                print(
                    f"[INFO] lines={total_lines:,}, written={written_lines:,}, "
                    f"bad_json={bad_json:,}, missing_group={missing_group:,}"
                )

    print("[SUMMARY]")
    print(f"  total_lines   = {total_lines:,}")
    print(f"  written_lines = {written_lines:,}")
    print(f"  bad_json      = {bad_json:,}")
    print(f"  missing_group = {missing_group:,}")
    print(f"  target_groups = {sorted(target_groups_set)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True,
                    help="输入的大 JSONL 文件")
    ap.add_argument("--out-jsonl", type=Path, required=True,
                    help="输出 JSONL 文件，只包含指定 group 的样本")
    ap.add_argument("--group-key", type=str, required=True,
                    help="按哪个字段分组，支持 dot-path，比如 'uid' 或 'meta.group_id'")
    ap.add_argument("--group-values", type=str, required=True,
                    help="要保留的 group 值，逗号分隔，比如 'g1,g2,g3'；"
                         "如果只取一个 group，就写成 'g1'")
    ap.add_argument("--print-every", type=int, default=100000,
                    help="每处理多少行打印一次进度")

    args = ap.parse_args()

    if not args.jsonl.is_file():
        print(f"[ERROR] 输入文件不存在: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    target_groups = load_target_groups(args.group_values)
    if not target_groups:
        print("[ERROR] --group-values 解析后为空，请检查输入。", file=sys.stderr)
        sys.exit(1)

    filter_by_group(
        jsonl_path=args.jsonl,
        out_path=args.out_jsonl,
        group_key=args.group_key,
        target_groups=target_groups,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    main()
