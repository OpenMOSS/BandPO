#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 JSONL 中抽取指定行，并额外保存结构描述：
- 行号默认 1 基（第 1 行是 1），可用 --zero-based 改为 0 基
- 输出 1：<stem>+L<行号>.jsonl
- 输出 2：<stem>+L<行号>+structure.json  （UTF-8, indent=4）

结构规则：
- 若某 key 的值为 null => 视为“没取到”（got=false）；否则 got=true
- list（1维）=> length、元素类型是否一致；若元素是 list，检查子 list 是否等长
- array（二维 list-of-lists）=> shape=[rows, cols]、size；若行不等长则 ragged=true 并列出 row_lengths（最多前100行）
"""

import argparse
import sys
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------- 基础类型识别 ----------
def _json_type(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


# ---------- list/array 形状与一致性 ----------
def _is_list_of_lists(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, list) for x in v)

def _list_element_types(v: List[Any], cap: int = 1000) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for elem in v[:cap]:
        t = _json_type(elem)
        counts[t] = counts.get(t, 0) + 1
    if len(v) > cap:
        counts["_truncated_after"] = cap
    return counts

def _is_uniform_element_type(v: List[Any]) -> Tuple[bool, str]:
    """判断 list 的元素类型是否一致；若一致返回 (True, 类型名)；否则 (False, "")"""
    if not v:
        return True, ""  # 空列表视作一致但无类型
    first_t = _json_type(v[0])
    for x in v[1:]:
        if _json_type(x) != first_t:
            return False, ""
    return True, first_t

def _matrix_shape(v: List[List[Any]]) -> Tuple[bool, Tuple[int, int], List[int]]:
    """
    检查二维 list-of-lists 是否等长。
    返回 (is_matrix, (rows, cols), row_lengths)
    - is_matrix=True 表示每行长度相等
    - row_lengths 给出每行长度（最多前100行）
    """
    rows = len(v)
    if rows == 0:
        return True, (0, 0), []
    row_lengths = [len(r) for r in v]
    is_matrix = len(set(row_lengths)) == 1
    cols = row_lengths[0] if rows > 0 else 0
    return is_matrix, (rows, cols), row_lengths[:100]


# ---------- 字段摘要 ----------
def _summarize_value(v: Any) -> Dict[str, Any]:
    """
    统一的值摘要：
    - null: {"type":"null","got":False}
    - 其他: {"type":..., "got":True, ... 额外属性 ...}
    """
    t = _json_type(v)
    if v is None:
        return {"type": "null", "got": False}

    info: Dict[str, Any] = {"type": t, "got": True}

    if t == "string":
        info["length"] = len(v)

    elif t == "number":
        if isinstance(v, float) and math.isnan(v):
            # NaN 仍视为 got=True（不是 null），但注明 is_nan
            info["is_nan"] = True

    elif t == "array":
        # Python/JSON 里的 array 就是 list；先做 1 维统计
        info["length"] = len(v)
        uniform, etype = _is_uniform_element_type(v)
        info["uniform_element_type"] = uniform
        if uniform and etype:
            info["element_type"] = etype
        info["element_types"] = _list_element_types(v)

        # 若元素本身是 list，进一步检查是否构成二维“矩阵”
        if _is_list_of_lists(v):
            is_matrix, shape, row_lengths = _matrix_shape(v)
            if is_matrix:
                info["is_matrix"] = True
                rows, cols = shape
                info["shape"] = [rows, cols]
                info["size"] = rows * cols
            else:
                info["is_matrix"] = False
                info["ragged"] = True
                info["row_lengths"] = row_lengths  # 最多前 100 行以免太大

    elif t == "object":
        # 顶层/内层对象：仅给出 key 信息
        info["keys_count"] = len(v)
        keys: List[str] = list(v.keys())
        info["keys"] = keys[:100]
        if len(keys) > 100:
            info["keys_truncated_after"] = 100

    # boolean 等其他无需补充
    return info


def _summarize_top_level(obj: Any, source: Path, line_1based: int, idx_0based: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "_meta": {
            "source": str(source),
            "line_number_1_based": line_1based,
            "index_0_based": idx_0based,
            "top_level_type": _json_type(obj),
        }
    }

    if isinstance(obj, dict):
        fields: Dict[str, Any] = {}
        for k, v in obj.items():
            fields[k] = _summarize_value(v)
        summary["fields"] = fields
    else:
        # 非对象顶层也做整体摘要
        summary["value_summary"] = _summarize_value(obj)

        # 若是顶层 list：按 array/矩阵规则附加更具体信息（便于你看“多长多宽”）
        if isinstance(obj, list):
            # 在 value_summary 里已经给了 length/uniform 等
            pass

    return summary


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="Extract one line from a JSONL and save the line + a structure summary JSON.")
    ap.add_argument("-p", "--path", required=True, help="Path to the .jsonl file")
    ap.add_argument("-n", "--line", required=True, type=int, help="Line number (1-based by default)")
    ap.add_argument("--zero-based", action="store_true", help="Treat --line as 0-based index instead of 1-based")
    ap.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    ap.add_argument("--out-dir", default=None, help="Optional output directory (default: same as source)")
    args = ap.parse_args()

    src = Path(args.path).expanduser().resolve()
    if not src.is_file():
        print(f"[ERROR] File not found: {src}", file=sys.stderr)
        sys.exit(1)

    # 目标行索引
    if args.zero_based:
        if args.line < 0:
            print("[ERROR] --zero-based 模式下 --line 必须 >= 0", file=sys.stderr)
            sys.exit(1)
        target_idx = args.line
        display_line = args.line
    else:
        if args.line <= 0:
            print("[ERROR] 行号必须是正整数（1 基）", file=sys.stderr)
            sys.exit(1)
        target_idx = args.line - 1
        display_line = args.line

    # 读取该行
    picked_line = None
    with src.open("r", encoding=args.encoding, newline="") as f:
        for i, line in enumerate(f):
            if i == target_idx:
                picked_line = line.rstrip("\n")
                break

    if picked_line is None:
        print(f("[ERROR] 超出行数：文件没有第 {display_line} 行"), file=sys.stderr)
        sys.exit(1)

    # 解析 JSON
    try:
        obj = json.loads(picked_line)
    except json.JSONDecodeError as e:
        print(f"[ERROR] 第 {display_line} 行不是合法 JSON：{e}", file=sys.stderr)
        sys.exit(1)

    # 输出路径
    stem = src.stem
    suffix = src.suffix or ".jsonl"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_line_path = out_dir / f"{stem}+L{display_line}{suffix}"
    out_struct_path = out_dir / f"{stem}+L{display_line}+structure.json"

    # 写出该行
    with out_line_path.open("w", encoding=args.encoding, newline="\n") as f:
        f.write(picked_line + "\n")

    # 结构摘要（UTF-8, indent=4）
    structure = _summarize_top_level(obj, source=src, line_1based=display_line, idx_0based=target_idx)
    with out_struct_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(structure, f, ensure_ascii=False, indent=4)

    print(str(out_line_path))
    print(str(out_struct_path))


if __name__ == "__main__":
    main()
