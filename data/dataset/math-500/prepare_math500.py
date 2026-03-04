import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

import numpy as np  # Import numpy to use int64

import pdb
def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math")
    args = parser.parse_args()

    # 由于官方 MATH 数据集已下架，现有常用的社区备份包括：
    # 1. `ricdomolm/MATH-500`
    # 2. `nlile/hendrycks-MATH-benchmark`
    # 这些备份数据集都包含 12,500 条样本，其中：
    # - 训练集包含 12,000 条数据
    # - 测试集包含 500 条数据
    # 这些数据集与原版 MATH 数据集的分布一致，适用于数学推理任务的训练与评测。

    # 选择备份数据集
    # data_source = "ricdomolm/MATH-500"
    data_source = "nlile/hendrycks-MATH-benchmark"
    print(f"Loading {data_source} dataset from HuggingFace...", flush=True)

    dataset = datasets.load_dataset(data_source) # , trust_remote_code=True

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
    suffix = "\n\nRemember to put your answer on its own line after \"Answer:\"."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            question_raw = question
            question = prefix + " " + question_raw + " " + suffix
            answer = example.pop("answer")

            idx_str = np.int64(idx)

            data = {
                # "data_source": "HuggingFaceH4/MATH-500",
                "solution": answer,
                "data_source": "math_dapo",
                "prompt": [{"role": "user", "content": question}],
                "ability": "MATH",
                "reward_model": {"style": "rule-lighteval\/MATH_v2", "ground_truth": answer},
                "extra_info": {
                    "split": split, 
                    # "index": idx_str,
                    "data_source": data_source, 
                    "question_raw": question_raw,
                    "answer": answer
                },
            }
            return data

        return process_fn

    def make_map_fn_test(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            question_raw = question
            question = prefix + " " + question_raw + " " + suffix
            answer = example.pop("answer")

            idx_str = np.int64(idx)

            data = {
                # "data_source": "HuggingFaceH4/MATH-500",
                "solution": answer,
                # "data_source": "math_dapo",
                "data_source": "aime_math500",
                "prompt": [{"role": "user", "content": question}],
                "ability": "MATH",
                "reward_model": {"style": "rule-lighteval\/MATH_v2", "ground_truth": answer},
                "extra_info": {
                    "split": split, 
                    # "index": idx_str,
                    "data_source": data_source, 
                    "question_raw": question_raw,
                    "answer": answer
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn_test("test"), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train/train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test/test.parquet"))
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train/train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test/test_example.json"), "w") as f:
        json.dump(example, f, indent=2)

    import pandas as pd
    # 读取原始数据
    df = pd.read_parquet(os.path.join(local_dir, "train/train.parquet"))

    # 过滤 Level 3 到 Level 5 的数据
    df_filtered = df[(df["level"] >= 3) & (df["level"] <= 5)]
    # df_filtered = df[df["level"].isin([3, 4, 5])]

    # 保存新的数据集
    df_filtered.to_parquet(os.path.join(local_dir, "train/train_L3-5.parquet"))
