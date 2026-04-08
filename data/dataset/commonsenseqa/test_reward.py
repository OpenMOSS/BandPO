#!/usr/bin/env python3
"""
Test script for CommonsenseQA reward function.
Must pass all tests before using in training.
"""
import sys
import json

sys.path.insert(0, '/inspire/hdd/global_user/liyuan-p-liyuan/workspace/BandPO/RLtraining/verl')

from verl.utils.reward_score import default_compute_score


def test_reward(solution_str, ground_truth, expected_score, expected_pred, description):
    """Run a single test case."""
    result = default_compute_score(
        data_source='commonsense_qa',
        solution_str=solution_str,
        ground_truth=ground_truth
    )

    passed = True
    if isinstance(result, dict):
        score = result['score']
        pred = result['pred']
    else:
        score = result
        pred = "N/A"

    if score != expected_score:
        print(f"  ❌ SCORE MISMATCH: expected {expected_score}, got {score}")
        passed = False
    if expected_pred and pred != expected_pred:
        print(f"  ❌ PRED MISMATCH: expected {expected_pred}, got {pred}")
        passed = False

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {description}")
    if not passed:
        print(f"  Input: {repr(solution_str[:80])}")
        print(f"  Gold: {ground_truth}, Pred: {pred}, Score: {score}")
    return passed


def main():
    print("=" * 80)
    print("CommonsenseQA Reward Function Test Suite")
    print("=" * 80)
    print()

    gold = "B"  # Ground truth for all tests

    tests = [
        # (solution_str, expected_score, expected_pred, description)
        ("The answer is Answer: B", 1.0, "B", "Standard 'Answer: B' format"),
        ("After thinking, I conclude Answer: B", 1.0, "B", "Answer: B with prefix text"),
        ("ANSWER: B", 1.0, "B", "All caps ANSWER"),
        ("The final answer is: B", 1.0, "B", "Colon after 'is'"),
        ("Answer:B (no space)", 1.0, "B", "No space after colon"),
        ("My reasoning...\nAnswer: B", 1.0, "B", "Answer on new line"),
        ("I think\nA\nis wrong, so\nB", 1.0, "B", "Standalone B in last line"),
        ("Thinking...\nC\nD\nB", 1.0, "B", "Last standalone letter is B"),

        # Incorrect answers (score should be -1.0)
        ("The answer is Answer: A", -1.0, "A", "Wrong answer A"),
        ("Answer: C", -1.0, "C", "Wrong answer C"),
        ("Answer: D is my choice", -1.0, "D", "Wrong answer D"),
        ("I choose Answer: E", -1.0, "E", "Wrong answer E"),

        # Invalid answers (score should be -1.0, pred should be [INVALID])
        ("I don't know the answer", -1.0, "[INVALID]", "No answer found"),
        ("The answer is probably F", -1.0, "[INVALID]", "Invalid letter F"),
        ("", -1.0, "[INVALID]", "Empty string"),
        ("ABC", -1.0, "[INVALID]", "Multiple letters not valid"),
        ("My answer is definitely X", -1.0, "[INVALID]", "Letter X not A-E, invalid"),

        # Edge cases
        ("Answer: b", 1.0, "B", "Lowercase b should be uppercased"),
        ("   Answer: B   ", 1.0, "B", "Extra whitespace"),
        ("Line 1\nLine 2\nB", 1.0, "B", "B alone on last line"),
    ]

    passed = 0
    failed = 0

    for solution_str, expected_score, expected_pred, description in tests:
        if test_reward(solution_str, gold, expected_score, expected_pred, description):
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 80)

    if failed > 0:
        print("\n❌ TEST SUITE FAILED - Fix issues before using in training")
        return 1
    else:
        print("\n✅ ALL TESTS PASSED - Reward function is ready for training")
        return 0


if __name__ == "__main__":
    exit(main())
