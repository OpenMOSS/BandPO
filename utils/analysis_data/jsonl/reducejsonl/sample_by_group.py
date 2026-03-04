#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 group 分层随机抽样（每个 group 抽取一定百分比），适用于超大 JSONL。

特点：
- 只按行 streaming，不把所有数据载入内存。
- 两遍扫描：
  1) 第一遍统计每个 group 的样本数，算出每个 group 的目标抽样数量 k_g。
  2) 第二遍对每个 group 做 reservoir sampling，确定要抽的行对应的 (offset, length)。
  最后根据这些 offset 再顺序读一次文件，把抽中的行写到输出 JSONL。
- 每个 group 抽取比例由 --sample-frac 指定，可配合 --min-per-group / --max-per-group。
- 支持 group-key 为 dot path，例如 "meta.group_id"。

用法示例：
  python sample_by_group.py \
    --jsonl huge.kept.jsonl \
    --out-jsonl huge.kept.sampled.jsonl \
    --group-key "uid" \
    --sample-frac 0.1 \
    --round-mode round \
    --min-per-group 1 \
    --seed 42
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List

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


def scan_group_counts(jsonl_path: Path,
                      group_key: str,
                      print_every: int = 100000) -> Tuple[Dict[str, int], int, int]:
    """
    第一遍扫描：统计每个 group 的样本数。

    返回:
      - group_counts: {group_value_str: count}
      - total_lines: 总行数
      - bad_json: JSON 解析失败的行数
    """
    group_counts: Dict[str, int] = {}
    total = 0
    bad_json = 0

    print(f"[PASS1] Scanning group counts from {jsonl_path} ...")

    with jsonl_path.open("rb", buffering=1024 * 1024) as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            total += 1

            try:
                obj = json_loads(line)
            except Exception:
                bad_json += 1
                if total % print_every == 0:
                    print(f"[PASS1][WARN] line {total:,}: JSON parse error (累积 {bad_json})",
                          file=sys.stderr)
                continue

            if not isinstance(obj, dict):
                continue

            g = get_nested(obj, group_key, None)
            if g is None:
                # 把缺失 group 的样本划到一个特殊 bucket
                g_str = "__MISSING__"
            else:
                g_str = str(g)

            group_counts[g_str] = group_counts.get(g_str, 0) + 1

            if total % print_every == 0:
                print(f"[PASS1] lines={total:,}, groups={len(group_counts):,}, bad_json={bad_json:,}")

    print(f"[PASS1] Done. total_lines={total:,}, groups={len(group_counts):,}, bad_json={bad_json:,}")
    return group_counts, total, bad_json


def compute_target_sizes(group_counts: Dict[str, int],
                         sample_frac: float,
                         round_mode: str = "round",
                         min_per_group: int = 0,
                         max_per_group: int = -1) -> Dict[str, int]:
    """
    根据每个 group 的大小和 sample_frac 计算目标抽样数 k_g。

    round_mode: "floor" | "round" | "ceil"
    """
    target: Dict[str, int] = {}
    for g, n in group_counts.items():
        if n <= 0:
            target[g] = 0
            continue

        raw = n * sample_frac
        if round_mode == "floor":
            k = int(math.floor(raw))
        elif round_mode == "ceil":
            k = int(math.ceil(raw))
        else:
            # 默认 round
            k = int(round(raw))

        if min_per_group > 0 and n > 0:
            k = max(k, min_per_group)
        if max_per_group > 0:
            k = min(k, max_per_group)

        # 不要超过 group 本身大小
        if k > n:
            k = n

        target[g] = max(0, k)

    return target


def reservoir_sample_by_group(jsonl_path: Path,
                              group_key: str,
                              target_sizes: Dict[str, int],
                              seed: int,
                              print_every: int = 100000) -> Dict[str, List[Tuple[int, int]]]:
    """
    第二遍扫描：对每个 group 做 reservoir sampling，决定哪些行被抽中。
    返回：
      reservoirs: {group_value_str: [(offset, length), ...]}
    """
    random.seed(seed)
    reservoirs: Dict[str, List[Tuple[int, int]]] = {g: [] for g in target_sizes}
    seen_counts: Dict[str, int] = {g: 0 for g in target_sizes}

    total = 0
    bad_json = 0

    print(f"[PASS2] Reservoir sampling per group from {jsonl_path} ...")

    with jsonl_path.open("rb", buffering=1024 * 1024) as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            ln = len(line)
            if not line.strip():
                continue
            total += 1

            try:
                obj = json_loads(line)
            except Exception:
                bad_json += 1
                if total % print_every == 0:
                    print(f"[PASS2][WARN] line {total:,}: JSON parse error (累积 {bad_json})",
                          file=sys.stderr)
                continue

            if not isinstance(obj, dict):
                continue

            g = get_nested(obj, group_key, None)
            if g is None:
                g_str = "__MISSING__"
            else:
                g_str = str(g)

            if g_str not in target_sizes:
                # 理论上不应该出现（第一遍已经统计过所有 group），安全起见还是跳过
                continue

            k = target_sizes[g_str]
            if k <= 0:
                # 这个 group 不抽样
                continue

            seen_counts[g_str] = seen_counts.get(g_str, 0) + 1
            i = seen_counts[g_str]  # 第 i 个落在该 group 的样本

            if i <= k:
                reservoirs[g_str].append((off, ln))
            else:
                # reservoir sampling: 以概率 k/i 替换
                j = random.randint(1, i)
                if j <= k:
                    replace_idx = random.randint(0, k - 1)
                    reservoirs[g_str][replace_idx] = (off, ln)

            if total % print_every == 0:
                picked = sum(len(v) for v in reservoirs.values())
                print(
                    f"[PASS2] lines={total:,}, seen_groups={len([g for g,c in seen_counts.items() if c>0]):,}, "
                    f"picked_total={picked:,}, bad_json={bad_json:,}"
                )

    picked_total = sum(len(v) for v in reservoirs.values())
    print(f"[PASS2] Done. total_lines={total:,}, picked_total={picked_total:,}, bad_json={bad_json:,}")
    return reservoirs


def materialize_sample(jsonl_path: Path,
                       out_path: Path,
                       reservoirs: Dict[str, List[Tuple[int, int]]]) -> int:
    """
    第三步：根据 reservoirs 中记录的 (offset, length) 从原文件读取行，写入 out_path。

    为保持输出大致是原文件顺序：
      - 把所有 (group, offset, length) flatten 成一个 list
      - 按 offset 升序排序
      - 顺序 seek+read 把这些行写出
    返回：写出的行数
    """
    # flatten
    all_entries: List[Tuple[int, int]] = []
    for g, lst in reservoirs.items():
        all_entries.extend(lst)

    if not all_entries:
        print("[PASS3] No sampled entries, output will be empty.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return 0

    # 按 offset 排序，便于顺序读
    all_entries.sort(key=lambda x: x[0])

    print(f"[PASS3] Materializing {len(all_entries):,} sampled lines to {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with jsonl_path.open("rb", buffering=1024 * 1024) as fin, \
         out_path.open("wb", buffering=1024 * 1024) as fout:
        for off, ln in all_entries:
            fin.seek(off)
            data = fin.read(ln)
            # 原始行里已有 '\n'，不再额外添加
            fout.write(data)
            written += 1

    print(f"[PASS3] Done. written={written:,} lines.")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True,
                    help="输入的大 JSONL 文件")
    ap.add_argument("--out-jsonl", type=Path, required=True,
                    help="输出抽样后的 JSONL 文件")
    ap.add_argument("--group-key", type=str, required=True,
                    help="按哪个字段分组，支持 dot-path，比如 'uid' 或 'meta.group_id'")
    ap.add_argument("--sample-frac", type=float, required=True,
                    help="每个 group 抽取的比例 (0 < sample-frac <= 1)")
    ap.add_argument("--round-mode", type=str, default="round",
                    choices=["floor", "round", "ceil"],
                    help="计算 k_g = n_g * frac 时的取整方式")
    ap.add_argument("--min-per-group", type=int, default=0,
                    help="每个 group 至少抽多少个样本 (0 表示不限制)")
    ap.add_argument("--max-per-group", type=int, default=-1,
                    help="每个 group 抽样上限，-1 表示不限制")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子，保证可复现")
    ap.add_argument("--print-every", type=int, default=100000,
                    help="每处理多少行打印一次进度")
    args = ap.parse_args()

    jsonl_path: Path = args.jsonl
    out_path: Path = args.out_jsonl

    if not jsonl_path.is_file():
        print(f"[ERROR] 输入文件不存在: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    if not (0.0 < args.sample_frac <= 1.0):
        print("[ERROR] --sample-frac 必须在 (0, 1] 区间内", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Input JSONL  : {jsonl_path}")
    print(f"[INFO] Output JSONL : {out_path}")
    print(f"[INFO] group-key    : {args.group_key}")
    print(f"[INFO] sample-frac  : {args.sample_frac}")
    print(f"[INFO] round-mode   : {args.round_mode}")
    print(f"[INFO] min-per-group: {args.min_per_group}")
    print(f"[INFO] max-per-group: {args.max_per_group}")
    print(f"[INFO] seed         : {args.seed}")

    t0 = time.time()
    group_counts, total_lines, bad_json = scan_group_counts(
        jsonl_path, args.group_key, print_every=args.print_every
    )

    t1 = time.time()
    target_sizes = compute_target_sizes(
        group_counts,
        sample_frac=args.sample_frac,
        round_mode=args.round_mode,
        min_per_group=args.min_per_group,
        max_per_group=args.max_per_group,
    )

    # 打印一下总体抽样规模
    total_in = sum(group_counts.values())
    total_out = sum(target_sizes.values())
    print(f"[INFO] Total samples={total_in:,}, target sampled={total_out:,} (~{total_out / max(total_in,1):.3f} overall)")
    nonzero_groups = sum(1 for k in target_sizes.values() if k > 0)
    print(f"[INFO] Groups with k_g>0: {nonzero_groups:,} / {len(target_sizes):,}")
    print(f"[INFO] Pass1+compute_target_sizes took {t1 - t0:.1f}s")

    t2 = time.time()
    reservoirs = reservoir_sample_by_group(
        jsonl_path,
        args.group_key,
        target_sizes,
        seed=args.seed,
        print_every=args.print_every,
    )

    t3 = time.time()
    written = materialize_sample(jsonl_path, out_path, reservoirs)
    t4 = time.time()

    print("[SUMMARY]")
    print(f"  total_lines    = {total_lines:,}")
    print(f"  bad_json       = {bad_json:,}")
    print(f"  sampled_lines  = {written:,}")
    print(f"  pass1+target   = {t1 - t0:.1f}s")
    print(f"  pass2_reservoir= {t3 - t2:.1f}s")
    print(f"  pass3_output   = {t4 - t3:.1f}s")
    print(f"  total_time     = {t4 - t0:.1f}s")


if __name__ == "__main__":
    main()
