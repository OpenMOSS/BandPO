#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a Parquet dataset to VERL format (with a Qwen-style prompt) and save as a new Parquet.
HF datasets compatibility: avoid Arrow map<>; use list<struct<key,value>> instead.

Output columns:
    data_source: string
    prompt: list<struct<content: string, role: string>>
    ability: string
    reward_model: struct<ground_truth: string, style: string>
    extra_info: struct<
        index: string (optional),
        raw_problem: string,
        source_fields: list<struct<key: string, value: string>>,  # HF-friendly
        solution: string,
        data_source: string  # from CLI --data_source_extra_info
    >
    split: string  ("train" or "test")
Changes:
- If solution is not provided (no column or empty/NaN/""), extra_info.solution defaults to the answer.
- Removed the top-level 'data_source_extra_info' column; CLI arg is stored inside extra_info['data_source'].
- Added switch to control whether to keep remaining columns into extra_info.source_fields:
    * --keep_extra_fields (default)
    * --no_keep_extra_fields
Usage:
    python convert_to_verl.py \
        --input /path/to/data.parquet \
        --question_key problem \
        --answer_key answer \
        --solution_key solution \
        --data_source "Datac-Source-Name" \
        --data_source_extra_info "some string" \
        --ability MATH \
        --reward_style rule-lighteval/MATH_v2 \
        --keep_id --id_key id \
        --split train \
        --keep_extra_fields
"""
import argparse
import json
import os
from typing import Any, Dict, Optional, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _json_dumps_or_str(x: Any) -> str:
    """Best-effort stringify: JSON-serialize dict/list; otherwise str()."""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return "" if x is None else str(x)


def _is_empty_like(x: Any) -> bool:
    """Check if value is effectively empty (None/NaN/empty string)."""
    if x is None:
        return True
    try:
        import pandas as _pd
        if _pd.isna(x):
            return True
    except Exception:
        pass
    return str(x).strip() == ""


def convert_parquet_to_verl(
    input_path: str,
    question_key: str,
    answer_key: str,
    solution_key: Optional[str],
    data_source: str,
    data_source_extra_info: Optional[str] = None,
    ability: str = "MATH",
    reward_style: str = "rule-lighteval/MATH_v2",
    keep_id: bool = False,
    id_key: Optional[str] = None,
    split: str = "train",
    keep_extra_fields: bool = True,
) -> str:
    """
    Convert the input Parquet to VERL format and write a sibling file with suffix "_verl.parquet".
    Returns the output path.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input Parquet not found: {input_path}")

    # Read source data
    df = pd.read_parquet(input_path)
    if df.empty:
        raise ValueError("Input Parquet has no rows.")

    # Validate keys
    for k, name in [
        (question_key, "question_key"),
        (answer_key, "answer_key"),
    ]:
        if k not in df.columns:
            raise KeyError(f"{name}='{k}' not in columns: {list(df.columns)}")

    if keep_id and (not id_key or id_key not in df.columns):
        raise KeyError(f"keep_id=True but id_key missing or not found: {id_key}")

    if solution_key is not None and solution_key != "" and solution_key not in df.columns:
        raise KeyError(f"solution_key='{solution_key}' not in columns")

    # Build output arrays
    data_source_col = []
    prompt_col = []
    ability_col = []
    reward_model_col = []
    extra_info_col = []
    split_col = []

    # Known keys to exclude from 'source_fields'
    exclude_keys = {question_key, answer_key}
    if solution_key:
        exclude_keys.add(solution_key)
    if keep_id and id_key:
        exclude_keys.add(id_key)

    # Build rows
    for _, row in df.iterrows():
        q_raw = row[question_key]
        a_raw = row[answer_key]

        # Basic filtering: skip rows missing core fields
        if pd.isna(q_raw) or pd.isna(a_raw):
            continue

        question = _json_dumps_or_str(q_raw)
        answer = _json_dumps_or_str(a_raw)

        # Build prompt (Qwen-style roles)
        prompt_val = [
            {"content": "Please reason step by step, and put your final answer within \\boxed{}.", "role": "system"},
            {"content": question, "role": "user"},
        ]

        # Reward model
        reward_model_val = {"ground_truth": answer, "style": reward_style}

        # Extra info: index
        idx_val = None
        if keep_id and id_key:
            idx_val = _json_dumps_or_str(row[id_key])

        # Collect other fields: as list<struct<key,value>> for HF compatibility
        if keep_extra_fields:
            sf_list: List[dict] = []
            for col in df.columns:
                if col in exclude_keys:
                    continue
                v = row[col]
                if pd.isna(v):
                    continue
                sf_list.append({"key": str(col), "value": _json_dumps_or_str(v)})
        else:
            sf_list = []

        # Determine solution with default to answer
        if solution_key:
            sol_raw = row[solution_key]
            if _is_empty_like(sol_raw):
                solution_val = answer
            else:
                solution_val = _json_dumps_or_str(sol_raw)
        else:
            solution_val = answer

        # Put CLI string into extra_info['data_source']
        extra_info_entry = {
            "index": idx_val,
            "raw_problem": question,
            "source_fields": sf_list,
            "solution": solution_val,
            "data_source": ("" if data_source_extra_info is None else str(data_source_extra_info)),
        }

        # Append columns
        data_source_col.append(str(data_source))
        prompt_col.append(prompt_val)
        ability_col.append(str(ability))
        reward_model_col.append(reward_model_val)
        extra_info_col.append(extra_info_entry)
        split_col.append(str(split))

    if not prompt_col:
        raise ValueError("No valid rows after filtering (missing question/answer).")

    # Define Arrow schema with nested types (HF-friendly)
    prompt_type = pa.list_(pa.struct([
        ("content", pa.string()),
        ("role", pa.string()),
    ]))

    reward_model_type = pa.struct([
        ("ground_truth", pa.string()),
        ("style", pa.string()),
    ])

    source_fields_type = pa.list_(pa.struct([
        ("key", pa.string()),
        ("value", pa.string()),
    ]))

    extra_info_type = pa.struct([
        ("index", pa.string()),
        ("raw_problem", pa.string()),
        ("source_fields", source_fields_type),
        ("solution", pa.string()),
        ("data_source", pa.string()),
    ])

    table = pa.table({
        "data_source": pa.array(data_source_col, type=pa.string()),
        "prompt": pa.array(prompt_col, type=prompt_type),
        "ability": pa.array(ability_col, type=pa.string()),
        "reward_model": pa.array(reward_model_col, type=reward_model_type),
        "extra_info": pa.array(extra_info_col, type=extra_info_type),
        "split": pa.array(split_col, type=pa.string()),
    })

    # Output path with _verl suffix
    base, ext = os.path.splitext(input_path)
    if ext.lower() != ".parquet":
        out_path = f"{base}_verl{ext or '.parquet'}"
    else:
        out_path = f"{base}_verl.parquet"

    pq.write_table(table, out_path, compression="zstd")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert Parquet to VERL format with Qwen prompt.")
    parser.add_argument("--input", required=True, help="Path to input Parquet file")
    parser.add_argument("--question_key", required=True, help="Column name for the question")
    parser.add_argument("--answer_key", required=True, help="Column name for the final answer")
    parser.add_argument("--solution_key", default="", help="Column name for solution ('' means none)")
    parser.add_argument("--data_source", required=True, help="Value for top-level 'data_source'")
    parser.add_argument("--data_source_extra_info", default="", help="String to store in extra_info['data_source']")
    parser.add_argument("--ability", default="MATH", help="Ability tag (default: MATH)")
    parser.add_argument("--reward_style", default="rule-lighteval/MATH_v2", help="Reward style (default: rule-lighteval/MATH_v2)")
    parser.add_argument("--keep_id", action="store_true", help="Whether to keep an id as extra_info.index")
    parser.add_argument("--id_key", default="", help="Column name for id (required if --keep_id)")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Split value (train/test)")
    # Mutually exclusive flags for keeping remaining columns
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--keep_extra_fields", dest="keep_extra_fields", action="store_true", help="Keep remaining columns in extra_info.source_fields (default)")
    group.add_argument("--no_keep_extra_fields", dest="keep_extra_fields", action="store_false", help="Do NOT keep remaining columns in extra_info.source_fields")
    parser.set_defaults(keep_extra_fields=True)

    args = parser.parse_args()

    out = convert_parquet_to_verl(
        input_path=args.input,
        question_key=args.question_key,
        answer_key=args.answer_key,
        solution_key=(args.solution_key if args.solution_key != "" else None),
        data_source=args.data_source,
        data_source_extra_info=args.data_source_extra_info,
        ability=args.ability,
        reward_style=args.reward_style,
        keep_id=args.keep_id,
        id_key=(args.id_key if args.id_key != "" else None),
        split=args.split,
        keep_extra_fields=args.keep_extra_fields,
    )
    print(out)


if __name__ == "__main__":
    main()