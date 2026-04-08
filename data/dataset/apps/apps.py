# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess APPS dataset to parquet format
"""

import argparse
import json
import os
import random
import pandas as pd
import sys

# Python 3.10+ 需要增加整数转换限制
sys.set_int_max_str_digits(10000)


def validate_input_output(input_output):
    """验证input_output是否包含有效的测试用例"""
    if not input_output:
        return False

    # 如果是字符串，尝试解析为JSON
    if isinstance(input_output, str):
        if input_output.strip() == '' or input_output.strip() == '{}':
            return False
        try:
            # 临时增加大整数转换限制
            import sys
            original_limit = sys.get_int_max_str_digits()
            sys.set_int_max_str_digits(0)  # 0 means unlimited
            try:
                input_output = json.loads(input_output)
            finally:
                sys.set_int_max_str_digits(original_limit)
        except (json.JSONDecodeError, ValueError):
            return False

    # 检查是否包含有效的inputs和outputs
    if not isinstance(input_output, dict):
        return False

    inputs = input_output.get('inputs')
    outputs = input_output.get('outputs')

    if not inputs or not outputs:
        return False
    if len(inputs) == 0 or len(outputs) == 0:
        return False
    if len(inputs) != len(outputs):
        return False

    return True


def process_example(example, idx, split):
    """处理单个APPS样本"""
    problem_id = example.get('problem_id', example.get('id', idx))
    question = example['question']
    input_output = example.get('input_output', '{}')
    difficulty = example.get('difficulty', 'unknown')
    url = example.get('url', '')
    starter_code = example.get('starter_code', '')

    # 验证input_output是否有效
    if not validate_input_output(input_output):
        return None  # 返回None表示跳过此样本

    # 构建prompt
    prefix = "Please solve the following programming problem. Write Python code to solve it.\n\n"
    suffix = "\n\nYour code should read from standard input and write to standard output.\n\nPlease provide your solution in Python."

    if starter_code:
        suffix += f"\n\nStarter code:\n```python\n{starter_code}\n```"

    full_question = prefix + question + suffix

    # 解析input_output - 保持为字符串避免序列化问题
    if isinstance(input_output, str):
        io_str = input_output
    elif isinstance(input_output, dict):
        io_str = json.dumps(input_output)
    else:
        return None  # 无效格式，跳过

    return {
        "data_source": "apps",
        "prompt": [{"role": "user", "content": full_question}],  # Python列表，不序列化
        "ability": "code",
        "reward_model": {
            "style": "prime_code",
            "ground_truth": io_str
        },
        "extra_info": {
            "split": split,
            "problem_id": str(problem_id),
            "difficulty": str(difficulty),
            "url": str(url),
            "has_starter_code": bool(starter_code),
            "index": int(idx),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=".")
    parser.add_argument("--val_size", type=int, default=1024, help="Number of validation samples from test")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling validation")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # ==================== APPS Train ====================
    print("Loading APPS train dataset...", flush=True)
    with open(os.path.join(local_dir, 'train.jsonl'), 'r') as f:
        train_data = [json.loads(line) for line in f]

    print(f"Loaded {len(train_data)} train samples")

    # 处理训练集
    train_records = []
    for idx, example in enumerate(train_data):
        record = process_example(example, idx, "train")
        if record is not None:  # 只添加有效样本
            train_records.append(record)

    train_df = pd.DataFrame(train_records)
    train_path = os.path.join(local_dir, "train.parquet")
    train_df.to_parquet(train_path, index=False)
    print(f"Saved training set to: {train_path} ({len(train_df)} samples)")

    # 保存示例
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(train_records[0], f, indent=2, ensure_ascii=False)

    # ==================== APPS Test (split into val and test) ====================
    print("\nLoading APPS test dataset...", flush=True)
    with open(os.path.join(local_dir, 'test.jsonl'), 'r') as f:
        test_data = [json.loads(line) for line in f]

    print(f"Loaded {len(test_data)} test samples")

    # 随机抽取val_size条作为validation
    random.seed(args.seed)
    indices = list(range(len(test_data)))
    random.shuffle(indices)

    val_indices = indices[:args.val_size]
    test_indices = indices[args.val_size:]

    print(f"Split test into: val={len(val_indices)}, test_remaining={len(test_indices)}")

    # 处理validation集
    val_records = []
    for i, idx in enumerate(val_indices):
        record = process_example(test_data[idx], i, "val")
        val_records.append(record)

    val_df = pd.DataFrame(val_records)
    val_path = os.path.join(local_dir, "val_1k.parquet")
    val_df.to_parquet(val_path, index=False)
    print(f"Saved validation set to: {val_path} ({len(val_df)} samples)")

    # 保存val示例
    with open(os.path.join(local_dir, "val_example.json"), "w") as f:
        json.dump(val_records[0], f, indent=2, ensure_ascii=False)

    # 处理剩余test集
    test_records = []
    for i, idx in enumerate(test_indices):
        record = process_example(test_data[idx], i, "test")
        test_records.append(record)

    test_df = pd.DataFrame(test_records)
    test_path = os.path.join(local_dir, "test.parquet")
    test_df.to_parquet(test_path, index=False)
    print(f"Saved test set to: {test_path} ({len(test_df)} samples)")

    # 保存test示例
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(test_records[0], f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("Dataset preparation completed!")
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples (seed={args.seed})")
    print(f"Test: {len(test_df)} samples")
    print("="*60)
