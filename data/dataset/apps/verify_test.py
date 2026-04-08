#!/usr/bin/env python3
"""
全面测试APPS Reward Verify链条
生成不同类型的模型输出，测试verify逻辑
"""
import sys
import json
import pandas as pd
from datetime import datetime

sys.path.insert(0, '/inspire/hdd/global_user/liyuan-p-liyuan/workspace/BandPO/RLtraining/verl')

from verl.utils.reward_score import default_compute_score


def load_val_samples(n=5):
    """加载val数据集的前n个样本"""
    val_df = pd.read_parquet('val_1k.parquet')
    return val_df.head(n).to_dict('records')


def generate_test_cases(sample):
    """为单个样本生成不同类型的测试输出"""
    test_cases = []

    # 解析ground_truth
    reward_model = sample['reward_model']
    ground_truth = reward_model['ground_truth']

    # Test 1: 正确的解法（简单的echo测试）
    # 假设输出等于输入（仅适用于特定问题）
    test_cases.append({
        'name': 'correct_echo',
        'solution': '''
import sys
data = sys.stdin.read()
print(data, end='')
''',
        'expected': '可能正确（如果是echo问题）'
    })

    # Test 2: 语法错误的代码
    test_cases.append({
        'name': 'syntax_error',
        'solution': '''
import sys
print(int(input())  # 缺少右括号
''',
        'expected': '失败（语法错误）'
    })

    # Test 3: 运行时错误（除以零）
    test_cases.append({
        'name': 'runtime_error',
        'solution': '''
print(1/0)
''',
        'expected': '失败（运行时错误）'
    })

    # Test 4: 错误的输出
    test_cases.append({
        'name': 'wrong_output',
        'solution': '''
print("999999")
''',
        'expected': '失败（输出不匹配）'
    })

    # Test 5: 空输出
    test_cases.append({
        'name': 'empty_output',
        'solution': '''
pass
''',
        'expected': '失败（无输出）'
    })

    # Test 6: 使用```python代码块格式
    test_cases.append({
        'name': 'code_block_format',
        'solution': '''```python
import sys
data = sys.stdin.read()
print(data, end='')
```''',
        'expected': '可能正确（代码块格式）'
    })

    # Test 7: 超时代码（无限循环）
    test_cases.append({
        'name': 'timeout',
        'solution': '''
while True:
    pass
''',
        'expected': '失败（超时）'
    })

    return test_cases


def run_verify_test(sample, test_case):
    """运行单个verify测试"""
    try:
        reward_model = sample['reward_model']
        ground_truth = reward_model['ground_truth']

        result = default_compute_score(
            data_source='apps',
            solution_str=test_case['solution'],
            ground_truth=ground_truth,
            extra_info=sample.get('extra_info')
        )

        return {
            'success': True,
            'result': result,
            'result_type': type(result).__name__
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


def main():
    print("="*80)
    print("APPS Reward Verify 全面测试")
    print("="*80)
    print(f"测试时间: {datetime.now()}")
    print()

    # 加载样本
    samples = load_val_samples(n=3)
    print(f"加载了 {len(samples)} 个val样本")
    print()

    all_results = []

    for idx, sample in enumerate(samples):
        print(f"\n{'='*80}")
        print(f"样本 {idx+1}/{len(samples)}")
        print(f"{'='*80}")

        # 样本信息
        extra_info = sample.get('extra_info', {})
        print(f"Problem ID: {extra_info.get('problem_id', 'N/A')}")
        print(f"Difficulty: {extra_info.get('difficulty', 'N/A')}")
        print(f"URL: {extra_info.get('url', 'N/A')}")

        # 解析prompt查看问题描述
        prompt = json.loads(sample['prompt'])
        question_preview = prompt[0]['content'][:200] if prompt else "N/A"
        print(f"Question Preview: {question_preview}...")

        # Ground truth info
        reward_model = sample['reward_model']
        try:
            gt = json.loads(reward_model['ground_truth'])
            print(f"Test Cases: {len(gt.get('inputs', []))} input(s)")
            if gt.get('inputs'):
                print(f"First Input Preview: {repr(gt['inputs'][0][:100])}...")
        except:
            print("Ground Truth: 解析失败")

        print()

        # 生成测试用例并运行
        test_cases = generate_test_cases(sample)
        sample_results = []

        for test_case in test_cases:
            print(f"  测试: {test_case['name']}")
            print(f"  预期: {test_case['expected']}")

            result = run_verify_test(sample, test_case)

            if result['success']:
                print(f"  结果: {result['result']} (类型: {result['result_type']})")
            else:
                print(f"  错误: {result['error']} (类型: {result['error_type']})")

            sample_results.append({
                'sample_idx': idx,
                'problem_id': extra_info.get('problem_id'),
                'test_name': test_case['name'],
                'expected': test_case['expected'],
                'solution_preview': test_case['solution'][:50].replace('\n', '\\n'),
                **result
            })
            print()

        all_results.extend(sample_results)

    # 保存结果
    output_file = f'verify_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'test_time': str(datetime.now()),
            'num_samples': len(samples),
            'results': all_results
        }, f, indent=2, default=str)

    print("\n" + "="*80)
    print(f"测试结果已保存到: {output_file}")
    print("="*80)

    # 汇总统计
    print("\n汇总统计:")
    total_tests = len(all_results)
    success_tests = sum(1 for r in all_results if r.get('success'))
    print(f"  总测试数: {total_tests}")
    print(f"  成功执行: {success_tests}")
    print(f"  执行失败: {total_tests - success_tests}")

    # 按test_name统计
    from collections import defaultdict
    by_test_name = defaultdict(list)
    for r in all_results:
        by_test_name[r['test_name']].append(r)

    print("\n按测试类型统计:")
    for test_name, results in by_test_name.items():
        avg_score = sum(r['result'] for r in results if r.get('success')) / len(results) if results else 0
        print(f"  {test_name}: 平均得分 {avg_score:.2f}")


if __name__ == "__main__":
    main()
