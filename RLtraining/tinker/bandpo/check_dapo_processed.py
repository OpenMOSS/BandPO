#!/usr/bin/env python3
"""
检查 DAPO Processed 数据集的完整性
对应 LocalDapoProcessedTrainDataset 的字段提取逻辑
"""

import pandas as pd
from pathlib import Path
from typing import Any
from collections import defaultdict

def check_dapo_processed_dataset(parquet_path: str) -> dict[str, Any]:
    """
    检查 DAPO Processed 数据集
    
    字段提取逻辑（来自 LocalDapoProcessedTrainDataset._make_env_group_builder）:
        problem = (x.get("prompt") or "").strip()
        answer = (x.get("solution") or "").strip()
    """
    print("=" * 80)
    print(f"检查数据集: {parquet_path}")
    print("=" * 80)
    
    # 加载数据
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"❌ 无法加载文件: {e}")
        return {"error": str(e)}
    
    total_samples = len(df)
    print(f"\n📊 总样本数: {total_samples}")
    print(f"📋 列名: {df.columns.tolist()}")
    
    # 统计问题
    issues = defaultdict(list)  # issue_type -> [row_indices]
    valid_count = 0
    
    for idx, row in df.iterrows():
        row_issues = []
        
        # 检查 1: prompt 字段
        prompt = row.get("prompt")
        if prompt is None:
            row_issues.append("prompt 字段不存在或为 None")
            issues["missing_prompt"].append(idx)
        elif not isinstance(prompt, str):
            row_issues.append(f"prompt 类型错误: {type(prompt).__name__}")
            issues["invalid_prompt_type"].append(idx)
        elif prompt.strip() == "":
            row_issues.append("prompt 为空字符串")
            issues["empty_prompt"].append(idx)
        
        # 检查 2: solution 字段
        solution = row.get("solution")
        if solution is None:
            row_issues.append("solution 字段不存在或为 None")
            issues["missing_solution"].append(idx)
        elif not isinstance(solution, str):
            row_issues.append(f"solution 类型错误: {type(solution).__name__}")
            issues["invalid_solution_type"].append(idx)
        elif solution.strip() == "":
            row_issues.append("solution 为空字符串")
            issues["empty_solution"].append(idx)
        
        # 检查 3: 额外的质量检查
        if prompt and isinstance(prompt, str) and solution and isinstance(solution, str):
            # 检查 prompt 是否过短（可能是无效问题）
            if len(prompt.strip()) < 10:
                row_issues.append(f"prompt 过短 ({len(prompt.strip())} 字符)")
                issues["too_short_prompt"].append(idx)
            
            # 检查 solution 是否过短
            if len(solution.strip()) < 1:
                row_issues.append(f"solution 过短 ({len(solution.strip())} 字符)")
                issues["too_short_solution"].append(idx)
        
        # 记录结果
        if row_issues:
            if len(issues["detailed_issues"]) < 20:  # 只保存前20个详细问题
                issues["detailed_issues"].append({
                    "index": idx,
                    "issues": row_issues,
                    "raw_data": {
                        "prompt": str(prompt)[:200] if prompt else None,
                        "solution": str(solution)[:200] if solution else None
                    }
                })
        else:
            valid_count += 1
    
    # 打印统计结果
    print("\n" + "=" * 80)
    print("📈 检查结果统计")
    print("=" * 80)
    
    invalid_count = total_samples - valid_count
    print(f"\n✅ 有效样本数: {valid_count} ({valid_count/total_samples*100:.2f}%)")
    print(f"❌ 无效样本数: {invalid_count} ({invalid_count/total_samples*100:.2f}%)")
    
    if invalid_count > 0:
        print("\n📋 问题类型分布:")
        print("-" * 60)
        for issue_type, indices in sorted(issues.items()):
            if issue_type != "detailed_issues":
                print(f"  - {issue_type}: {len(indices)} 个样本")
                if len(indices) <= 5:
                    print(f"    索引: {indices}")
                else:
                    print(f"    索引 (前5个): {indices[:5]} ...")
        
        print("\n📝 问题样本详情 (前10个):")
        print("-" * 60)
        for detail in issues.get("detailed_issues", [])[:10]:
            print(f"\n  索引 {detail['index']}:")
            for issue in detail['issues']:
                print(f"    ⚠️  {issue}")
            if detail['raw_data']['prompt']:
                print(f"    📄 prompt (截断): {detail['raw_data']['prompt'][:80]}...")
            if detail['raw_data']['solution']:
                print(f"    📄 solution (截断): {detail['raw_data']['solution'][:80]}...")
    
    # 模拟训练时的行为
    print("\n" + "=" * 80)
    print("🔮 模拟训练影响分析")
    print("=" * 80)
    
    batch_size = 256
    expected_batches = (total_samples + batch_size - 1) // batch_size
    actual_builders_per_batch = []
    
    for batch_idx in range(expected_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_samples)
        batch_valid = 0
        
        for idx in range(start, end):
            row = df.iloc[idx]
            prompt = (row.get("prompt") or "").strip() if isinstance(row.get("prompt"), str) else ""
            solution = (row.get("solution") or "").strip() if isinstance(row.get("solution"), str) else ""
            if prompt and solution:
                batch_valid += 1
        
        actual_builders_per_batch.append(batch_valid)
    
    print(f"\n假设 batch_size = {batch_size}:")
    print(f"  - 预期 batch 数: {expected_batches}")
    print(f"  - 每个 batch 实际有效 builders:")
    
    problem_batches = []
    for i, count in enumerate(actual_builders_per_batch):
        expected = min(batch_size, total_samples - i * batch_size)
        if count < expected:
            problem_batches.append((i, count, expected))
    
    if problem_batches:
        print(f"\n  ⚠️  有 {len(problem_batches)} 个 batch 的有效 builders 少于预期:")
        for batch_idx, actual, expected in problem_batches[:10]:
            deficit = expected - actual
            print(f"    - Batch {batch_idx}: 有效 {actual}/{expected} (缺少 {deficit})")
        if len(problem_batches) > 10:
            print(f"    ... 还有 {len(problem_batches) - 10} 个有问题的 batch")
    else:
        print("  ✅ 所有 batch 的有效 builders 数量正常")
    
    # 额外统计：字段值分布
    print("\n" + "=" * 80)
    print("📊 字段值分布统计")
    print("=" * 80)
    
    if "prompt" in df.columns:
        prompt_lengths = df["prompt"].apply(lambda x: len(str(x).strip()) if x else 0)
        print(f"\nprompt 长度统计:")
        print(f"  - 最小: {prompt_lengths.min()}")
        print(f"  - 最大: {prompt_lengths.max()}")
        print(f"  - 平均: {prompt_lengths.mean():.1f}")
        print(f"  - 中位数: {prompt_lengths.median():.1f}")
    
    if "solution" in df.columns:
        solution_lengths = df["solution"].apply(lambda x: len(str(x).strip()) if x else 0)
        print(f"\nsolution 长度统计:")
        print(f"  - 最小: {solution_lengths.min()}")
        print(f"  - 最大: {solution_lengths.max()}")
        print(f"  - 平均: {solution_lengths.mean():.1f}")
        print(f"  - 中位数: {solution_lengths.median():.1f}")
    
    return {
        "total_samples": total_samples,
        "valid_samples": valid_count,
        "invalid_samples": invalid_count,
        "issues": {k: len(v) if k != "detailed_issues" else v for k, v in issues.items()},
        "problem_batches": problem_batches
    }


def check_sample_content(parquet_path: str, sample_indices: list[int] = None):
    """查看指定样本的详细内容"""
    df = pd.read_parquet(parquet_path)
    
    if sample_indices is None:
        sample_indices = [0, 1, 2]
    
    print("\n" + "=" * 80)
    print("🔍 样本内容详情")
    print("=" * 80)
    
    for idx in sample_indices:
        if idx >= len(df):
            print(f"\n索引 {idx} 超出范围")
            continue
            
        row = df.iloc[idx]
        print(f"\n{'='*40}")
        print(f"样本索引: {idx}")
        print(f"{'='*40}")
        
        for col in df.columns:
            value = row[col]
            print(f"\n📌 {col}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    v_str = str(v)[:200] + "..." if len(str(v)) > 200 else str(v)
                    print(f"    - {k}: {v_str}")
            else:
                v_str = str(value)[:300] + "..." if len(str(value)) > 300 else str(value)
                print(f"    {v_str}")


def export_invalid_indices(parquet_path: str, output_path: str = None):
    """导出所有无效样本的索引，方便后续清洗"""
    df = pd.read_parquet(parquet_path)
    
    invalid_indices = []
    for idx, row in df.iterrows():
        prompt = row.get("prompt")
        solution = row.get("solution")
        
        is_valid = (
            isinstance(prompt, str) and prompt.strip() and
            isinstance(solution, str) and solution.strip()
        )
        
        if not is_valid:
            invalid_indices.append(idx)
    
    if output_path is None:
        output_path = parquet_path.replace(".parquet", "_invalid_indices.txt")
    
    with open(output_path, "w") as f:
        for idx in invalid_indices:
            f.write(f"{idx}\n")
    
    print(f"\n📁 无效索引已导出到: {output_path}")
    print(f"   共 {len(invalid_indices)} 个无效样本")
    
    return invalid_indices


if __name__ == "__main__":
    # 修改为你的实际路径
    PARQUET_PATH = "/remote-home1/yli/Workspace/BandPO/data/dataset/dapo_processed/all/DAPO-Math-17k-Processed.parquet"
    
    # 运行检查
    result = check_dapo_processed_dataset(PARQUET_PATH)
    
    # 查看几个样本的详细内容
    check_sample_content(PARQUET_PATH, [0, 1, 2])
    
    # 如果有问题样本，查看第一个问题样本
    if result["invalid_samples"] > 0 and result["issues"].get("detailed_issues"):
        first_invalid_idx = result["issues"]["detailed_issues"][0]["index"]
        print(f"\n\n🔍 查看第一个问题样本 (索引 {first_invalid_idx}):")
        check_sample_content(PARQUET_PATH, [first_invalid_idx])
    
    # 可选：导出无效索引
    # export_invalid_indices(PARQUET_PATH)