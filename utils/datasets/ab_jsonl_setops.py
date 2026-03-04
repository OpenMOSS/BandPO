#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ab_jsonl_setops.py

从 A.jsonl 与 B.jsonl 进行集合运算（按 index_id）：
- A_cap_B.jsonl: 交集的样本（以 A 为准输出）
- B_sub_A.jsonl: B - A 的样本（以 B 为准输出）

用法：
python ab_jsonl_setops.py A.jsonl B.jsonl --out-dir out_dir [--key index_id]

备注：
- 流式读取，不把整文件载入内存
- 保持样本原样输出（不改动字段）
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Iterable, Set, Tuple

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
                else:
                    # 非对象类型也跳过（可按需改为 raise）
                    continue
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

def _scan_ids(path: Path, key: str) -> Tuple[Set[int], int, int]:
    """
    返回: (ids, n_total, n_missing_key)
    """
    ids: Set[int] = set()
    n_total = 0
    n_missing = 0
    for obj in _iter_jsonl(path):
        n_total += 1
        if key not in obj:
            n_missing += 1
            continue
        try:
            ids.add(int(obj[key]))
        except Exception:
            # 不能转 int 的直接跳过
            n_missing += 1
    return ids, n_total, n_missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_jsonl", help="数据集 A 的 jsonl 文件")
    ap.add_argument("b_jsonl", help="数据集 B 的 jsonl 文件")
    ap.add_argument("--out-dir", default="./out_ab_sets", help="输出目录")
    ap.add_argument("--key", default="index_id", help="主键字段名（默认 index_id）")
    args = ap.parse_args()

    a_path = _expand(args.a_jsonl)
    b_path = _expand(args.b_jsonl)
    out_dir = _expand(args.out_dir)
    key = args.key

    if not a_path.exists():
        raise FileNotFoundError(a_path)
    if not b_path.exists():
        raise FileNotFoundError(b_path)

    # 先各自扫描 id 集合
    b_ids, b_total, b_miss = _scan_ids(b_path, key)
    a_ids, a_total, a_miss = _scan_ids(a_path, key)

    inter_ids = a_ids & b_ids        # A∩B
    only_b_ids = b_ids - a_ids       # B - A

    print(f"A: total={a_total}, missing_key={a_miss}, uniq_ids={len(a_ids)}")
    print(f"B: total={b_total}, missing_key={b_miss}, uniq_ids={len(b_ids)}")
    print(f"∩: {len(inter_ids)};  B - A: {len(only_b_ids)}")

    # 生成 A_cap_B.jsonl（以 A 的顺序遍历）
    a_cap_b_path = out_dir / "A_cap_B.jsonl"
    def _gen_a_cap_b():
        for obj in _iter_jsonl(a_path):
            if key in obj:
                try:
                    if int(obj[key]) in inter_ids:
                        yield obj
                except Exception:
                    pass
    n_cap = _write_jsonl(_gen_a_cap_b(), a_cap_b_path)
    print(f"✅ write {a_cap_b_path}  rows={n_cap}")

    # 生成 B_sub_A.jsonl（以 B 的顺序遍历）
    b_sub_a_path = out_dir / "B_sub_A.jsonl"
    def _gen_b_sub_a():
        for obj in _iter_jsonl(b_path):
            if key in obj:
                try:
                    iid = int(obj[key])
                    if iid in only_b_ids:
                        yield obj
                except Exception:
                    pass
    n_sub = _write_jsonl(_gen_b_sub_a(), b_sub_a_path)
    print(f"✅ write {b_sub_a_path}  rows={n_sub}")

if __name__ == "__main__":
    main()
