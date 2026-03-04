#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_init.py
递归为指定目录创建缺失的 __init__.py 文件；已存在则保持不变。

用法示例：
  python make_init.py theone
  python make_init.py theone service --exclude data --exclude node_modules
  python make_init.py . --dry-run

参数说明：
  paths                  要处理的一个或多个目录
  -x, --exclude NAME     追加排除的目录名（可重复）
  --dry-run              只打印将要创建的文件，不真正写入
  -q, --quiet            安静模式：只输出汇总
"""

import argparse
import os
import sys
from typing import List, Set

DEFAULT_EXCLUDES: Set[str] = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", "env", "build", "dist", "node_modules", ".idea", ".vscode"
}

HEADER = "# -*- coding: utf-8 -*-\n# Auto-generated: make this directory a Python package.\n"

def create_init(dirpath: str, dry_run: bool) -> bool:
    init_path = os.path.join(dirpath, "__init__.py")
    if os.path.exists(init_path):
        return False
    if not dry_run:
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(HEADER)
    return True

def process(root: str, excludes: Set[str], dry_run: bool, quiet: bool) -> int:
    created = 0
    # topdown=True 便于在遍历时剪枝排除目录
    for dirpath, dirnames, _filenames in os.walk(root, topdown=True):
        # 只按目录名排除（不含路径），并跳过以 "." 开头的隐藏目录
        dirnames[:] = [
            d for d in dirnames
            if d not in excludes and d not in DEFAULT_EXCLUDES and not d.startswith(".ipynb_checkpoints")
        ]
        # 为当前目录确保 __init__.py
        if create_init(dirpath, dry_run):
            created += 1
            if not quiet:
                print(f"Created: {os.path.join(dirpath, '__init__.py')}")
        else:
            if not quiet:
                print(f"Exists : {os.path.join(dirpath, '__init__.py')}")
    return created

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Create missing __init__.py files under given paths.")
    ap.add_argument("paths", nargs="+", help="Directories to process")
    ap.add_argument("-x", "--exclude", action="append", default=[], help="Directory name to exclude (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be created without writing files")
    ap.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    args = ap.parse_args(argv)

    excludes = set(args.exclude or [])
    total_created = 0
    for p in args.paths:
        root = os.path.abspath(p)
        if not os.path.isdir(root):
            print(f"Skip (not a directory): {p}", file=sys.stderr)
            continue
        if not args.quiet:
            print(f"[Processing] {root}")
        total_created += process(root, excludes, args.dry_run, args.quiet)

    if args.dry_run:
        print(f"\nDry-run summary: would create {total_created} new __init__.py file(s).")
    else:
        print(f"\nDone: created {total_created} new __init__.py file(s).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
