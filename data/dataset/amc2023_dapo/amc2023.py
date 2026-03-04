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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os

import datasets
import pdb
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math")
    args = parser.parse_args()

    data_source = "math-ai/amc23"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source)
    test_dataset = dataset["test"]

    prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
    suffix = "\n\nRemember to put your answer on its own line after \"Answer:\"."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            id = example.pop("id")
            question_raw = example.pop("question")
            question = prefix + " " + question_raw + " " + suffix
            answer = example.pop("answer")
            url = example.pop("url")
            data = {
                "data_source": "lighteval/MATH",
                "prompt": [{"role": "user", "content": question}],
                "ability": "MATH",
                "reward_model": {"style": "rule-lighteval\/MATH_v2", "ground_truth": answer},
                "extra_info": {
                    "split": split, 
                    "answer": answer,
                    "question_raw": question_raw,
                    "data_source": data_source,
                    "url": url,
                    "index": idx,
                    "id": id,
                },
            }
            return data
        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    local_dir = os.path.expanduser(args.local_dir)
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)