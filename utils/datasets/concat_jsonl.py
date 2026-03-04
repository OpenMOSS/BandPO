#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
concat_jsonl.py

将两个 jsonl 文件按顺序合并为一个。
用法：
python concat_jsonl.py in1.jsonl in2.jsonl -o merged.jsonl
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Iterable

def _expand(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p))).resolve()

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {path} 行 {ln}: {e}")

def _write_jsonl(objs: Iterable[Dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in1")
    ap.add_argument("in2")
    ap.add_argument("-o", "--output", required=True)
    # 如需按 index_id 去重，可放开下面两行并在生成器里加判断
    # ap.add_argument("--dedupe-by-index", action="store_true")
    # ap.add_argument("--key", default="index_id")
    args = ap.parse_args()

    p1 = _expand(args.in1)
    p2 = _expand(args.in2)
    po = _expand(args.output)

    if not p1.exists():
        raise FileNotFoundError(p1)
    if not p2.exists():
        raise FileNotFoundError(p2)

    def _gen_all():
        for obj in _iter_jsonl(p1):
            yield obj
        for obj in _iter_jsonl(p2):
            yield obj

        # 如需去重（按 index_id）：
        # seen = set()
        # for obj in _iter_jsonl(p1):
        #     iid = obj.get(args.key)
        #     if iid not in seen:
        #         seen.add(iid); yield obj
        # for obj in _iter_jsonl(p2):
        #     iid = obj.get(args.key)
        #     if iid not in seen:
        #         seen.add(iid); yield obj

    n = _write_jsonl(_gen_all(), po)
    print(f"✅ merged -> {po}  rows={n}")

if __name__ == "__main__":
    main()
