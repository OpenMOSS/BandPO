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
Preprocess CommonsenseQA and ARC-Challenge datasets to parquet format
"""

import argparse
import json
import os

import datasets
from verl.utils.hdfs_io import copy, makedirs


def format_choices(choices):
    """将choices格式化为字符串"""
    labels = choices['label']
    texts = choices['text']
    lines = []
    for label, text in zip(labels, texts):
        lines.append(f"({label}) {text}")
    return '\n'.join(lines)


def make_csqa_map_fn(split):
    """CommonsenseQA数据处理函数"""
    def process_fn(example, idx):
        question = example['question']
        choices = example['choices']
        answer_key = example['answerKey']
        question_concept = example.get('question_concept', '')

        # 构建prompt - 严格匹配reward解析逻辑
        prefix = "Answer the following multiple choice question. "
        prefix += "The last line of your response should be of the form 'Answer: X' where X is the letter of the correct choice (A, B, C, D, or E).\n\n"

        question_text = f"Question: {question}\n\nChoices:\n{format_choices(choices)}"
        suffix = "\n\nPlease provide your answer:"

        full_prompt = prefix + question_text + suffix

        data = {
            "data_source": "commonsense_qa",
            "prompt": [{"role": "user", "content": full_prompt}],
            "ability": "commonsense",
            "reward_model": {"style": "multiple_choice", "ground_truth": answer_key},
            "extra_info": {
                "split": split,
                "question": question,
                "choices": choices,
                "answerKey": answer_key,
                "question_concept": question_concept,
                "index": idx,
                "id": example.get('id', f'{split}_{idx}'),
            },
        }
        return data
    return process_fn


def make_arc_map_fn(split):
    """ARC-Challenge数据处理函数"""
    def process_fn(example, idx):
        question = example['question']
        choices = example['choices']
        answer_key = example['answerKey']

        # 构建prompt - 严格匹配reward解析逻辑 (与CSQA一致)
        prefix = "Answer the following multiple choice question. "
        prefix += "The last line of your response should be of the form 'Answer: X' where X is the letter of the correct choice (A, B, C, D, or E).\n\n"

        question_text = f"Question: {question}\n\nChoices:\n{format_choices(choices)}"
        suffix = "\n\nPlease provide your answer:"

        full_prompt = prefix + question_text + suffix

        data = {
            "data_source": "commonsense_qa",  # 使用相同的data_source和reward机制
            "prompt": [{"role": "user", "content": full_prompt}],
            "ability": "commonsense",
            "reward_model": {"style": "multiple_choice", "ground_truth": answer_key},
            "extra_info": {
                "split": split,
                "question": question,
                "choices": choices,
                "answerKey": answer_key,
                "index": idx,
                "id": example.get('id', f'{split}_{idx}'),
            },
        }
        return data
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/commonsenseqa")
    parser.add_argument("--train_size", type=int, default=4096, help="Number of training samples to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # ==================== CommonsenseQA ====================
    print("Loading CommonsenseQA dataset...", flush=True)
    csqa = datasets.load_dataset("tau/commonsense_qa")

    # 从train中随机抽取指定数量
    csqa_train = csqa['train'].shuffle(seed=args.seed).select(range(min(args.train_size, len(csqa['train']))))
    print(f"Selected {len(csqa_train)} samples from CommonsenseQA train (seed={args.seed})")

    # 处理数据
    csqa_train = csqa_train.map(function=make_csqa_map_fn("train"), with_indices=True)

    # 保存训练集
    train_path = os.path.join(local_dir, "train_4k.parquet")
    csqa_train.to_parquet(train_path)
    print(f"Saved training set to: {train_path}")

    # 保存示例
    example = csqa_train[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2, ensure_ascii=False)

    # ==================== ARC-Challenge ====================
    print("\nLoading ARC-Challenge dataset...", flush=True)
    arc = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge")

    # 使用validation作为测试集
    arc_val = arc['validation']
    print(f"ARC-Challenge validation size: {len(arc_val)}")

    # 处理数据
    arc_val = arc_val.map(function=make_arc_map_fn("val"), with_indices=True)

    # 保存测试集
    val_path = os.path.join(local_dir, "arc_challenge_val.parquet")
    arc_val.to_parquet(val_path)
    print(f"Saved validation set to: {val_path}")

    # 保存示例
    example = arc_val[0]
    with open(os.path.join(local_dir, "val_example.json"), "w") as f:
        json.dump(example, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("Dataset preparation completed!")
    print(f"Train: {len(csqa_train)} samples")
    print(f"Val: {len(arc_val)} samples")
    print("="*60)
