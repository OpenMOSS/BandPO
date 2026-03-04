#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描指定目录下所有名为 offline-run-* 的 W&B 离线 run 文件夹，
把完整路径保存到一个文本文件，然后多线程并发执行 `wandb sync` 上传。
"""

import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import os
import sys

def scan_offline_runs(root: Path):
    # 递归找 offline-run-* 目录
    runs = sorted({p.resolve() for p in root.rglob("offline-run-*") if p.is_dir()})
    return runs

def sync_one(run_dir: Path):
    # 调用 wandb CLI 上传
    proc = subprocess.run(
        ["wandb", "sync", str(run_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return run_dir, proc.returncode, proc.stdout

def main():
    parser = argparse.ArgumentParser(description="并行上传 W&B 离线 runs")
    parser.add_argument(
        "--root",
        default="/remote-home1/yli/Workspace/BandPO/data/validate/clip_improve",
        help="要扫描的根目录（默认即为你提供的路径）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, (os.cpu_count() or 4) * 2),
        help="并发线程数（默认 2×CPU，最多 8）",
    )
    parser.add_argument(
        "--list-file",
        default="offline_run_dirs.txt",
        help="保存扫描到的目录列表的文件名",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描并保存列表，不执行上传",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"根目录不存在：{root}", file=sys.stderr)
        sys.exit(2)

    runs = scan_offline_runs(root)
    if not runs:
        print("未找到任何 offline-run-* 目录。")
        return

    # 保存地址列表
    list_path = Path(args.list_file).resolve()
    with open(list_path, "w", encoding="utf-8") as f:
        for p in runs:
            f.write(str(p) + "\n")

    print(f"共找到 {len(runs)} 个 offline-run-* 目录，已保存到：{list_path}")

    if args.dry_run:
        return

    # 多线程上传
    ok, fail = 0, 0
    print(f"\n开始并发上传（workers={args.workers}）...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(sync_one, p) for p in runs]
        for fut in as_completed(futures):
            run_dir, code, out = fut.result()
            if code == 0:
                ok += 1
                print(f"[OK ] {run_dir}")
            else:
                fail += 1
                print(f"[ERR] {run_dir}\n{out}\n")

    print(f"\n上传完成：成功 {ok}，失败 {fail}，总计 {len(runs)}")

if __name__ == "__main__":
    main()
