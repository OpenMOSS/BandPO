#!/usr/bin/env python3
"""
检查 Math500 L3-5 Train 数据集的完整性
对应 LocalMath500L35TrainDataset 的字段提取逻辑
"""

import pandas as pd
from pathlib import Path
from typing import Any
from collections import defaultdict

def check_math500_l35_dataset(parquet_path: str) -> dict[str, Any]:
    """
    检查 Math500 L3-5 数据集
    
    字段提取逻辑（来自 LocalMath500L35TrainDataset._make_env_group_builder）:
        extra = x.get("extra_info") or {}
        problem = (extra.get("question_raw") or "").strip()
        answer = (extra.get("answer") or "").strip()
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
        
        # 检查 1: extra_info 字段是否存在
        extra_info = row.get("extra_info")
        if extra_info is None:
            row_issues.append("extra_info 字段不存在或为 None")
            issues["missing_extra_info"].append(idx)
        elif not isinstance(extra_info, dict):
            row_issues.append(f"extra_info 类型错误: {type(extra_info).__name__}")
            issues["invalid_extra_info_type"].append(idx)
        else:
            # 检查 2: question_raw 字段
            question_raw = extra_info.get("question_raw")
            if question_raw is None:
                row_issues.append("extra_info.question_raw 不存在或为 None")
                issues["missing_question_raw"].append(idx)
            elif not isinstance(question_raw, str):
                row_issues.append(f"question_raw 类型错误: {type(question_raw).__name__}")
                issues["invalid_question_raw_type"].append(idx)
            elif question_raw.strip() == "":
                row_issues.append("question_raw 为空字符串")
                issues["empty_question_raw"].append(idx)
            
            # 检查 3: answer 字段
            answer = extra_info.get("answer")
            if answer is None:
                row_issues.append("extra_info.answer 不存在或为 None")
                issues["missing_answer"].append(idx)
            elif not isinstance(answer, str):
                row_issues.append(f"answer 类型错误: {type(answer).__name__}")
                issues["invalid_answer_type"].append(idx)
            elif answer.strip() == "":
                row_issues.append("answer 为空字符串")
                issues["empty_answer"].append(idx)
        
        # 记录结果
        if row_issues:
            if len(issues["detailed_issues"]) < 20:  # 只保存前20个详细问题
                issues["detailed_issues"].append({
                    "index": idx,
                    "issues": row_issues,
                    "raw_data": {
                        "extra_info": str(extra_info)[:200] if extra_info else None
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
            if detail['raw_data']['extra_info']:
                print(f"    📄 extra_info (截断): {detail['raw_data']['extra_info'][:100]}...")
    
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
            extra_info = row.get("extra_info")
            if isinstance(extra_info, dict):
                question = (extra_info.get("question_raw") or "").strip()
                answer = (extra_info.get("answer") or "").strip()
                if question and answer:
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


if __name__ == "__main__":
    # 修改为你的实际路径
    PARQUET_PATH = "/remote-home1/yli/Workspace/BandPO/data/dataset/math-500/train/train_L3-5.parquet"
    
    # 运行检查
    result = check_math500_l35_dataset(PARQUET_PATH)
    
    # 查看几个样本的详细内容
    check_sample_content(PARQUET_PATH, [0, 1, 2])
    
    # 如果有问题样本，查看第一个问题样本
    if result["invalid_samples"] > 0 and result["issues"].get("detailed_issues"):
        first_invalid_idx = result["issues"]["detailed_issues"][0]["index"]
        print(f"\n\n🔍 查看第一个问题样本 (索引 {first_invalid_idx}):")
        check_sample_content(PARQUET_PATH, [first_invalid_idx])