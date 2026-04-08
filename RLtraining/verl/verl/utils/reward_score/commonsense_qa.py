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
Reward function for CommonsenseQA multiple choice questions.
Extracts answer in format "Answer: X" or standalone letter from model response.
"""

import re
from typing import Optional


def extract_answer(solution_str: str) -> Optional[str]:
    """
    Extract the answer letter (A-E) from model response.

    Priority 1: Match "Answer: X" pattern (case insensitive)
    Priority 2: Find standalone capital letter A-E in the last line

    Args:
        solution_str: Model's response text

    Returns:
        Extracted letter A-E or None if not found
    """
    if not solution_str:
        return None

    text = solution_str.strip()

    # Priority 1: Match "Answer: X" pattern (case insensitive)
    # Matches: "Answer: A", "answer: B", "ANSWER: C", "The answer is: D", "answer is D"
    # Try "Answer: X" first, then "answer is: X", then "answer is X"
    patterns = [
        r"(?i)Answer\s*:\s*([A-E])",        # Answer: A
        r"(?i)answer is\s*:\s*([A-E])",      # answer is: A
        r"(?i)answer is\s+([A-E])\b",         # answer is A (word boundary to avoid partial matches)
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    # Priority 2: Find standalone capital letter A-E in the last non-empty line
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and len(line) == 1 and line in ['A', 'B', 'C', 'D', 'E']:
            return line

    return None


def compute_score(solution_str: str, ground_truth: str) -> dict:
    """
    Compute reward score for CommonsenseQA.

    Args:
        solution_str: Model's response text
        ground_truth: Correct answer letter (e.g., "A", "B", "C", "D", "E")

    Returns:
        Dictionary with:
        - score: 1.0 if correct, -1.0 if incorrect or invalid
        - acc: bool indicating correctness
        - pred: extracted prediction or "[INVALID]"
    """
    pred = extract_answer(solution_str)
    gold = str(ground_truth).strip().upper()

    if pred is None:
        return {
            "score": -1.0,
            "acc": False,
            "pred": "[INVALID]"
        }

    correct = (pred == gold)
    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": pred
    }
