#!/usr/bin/env python3
"""
Ray Job Runtime Environment Checker

This script retrieves runtime environment information for a Ray job
given the Ray cluster address and submission ID.

Usage:
    python ray_runtime_checker.py <ray_address> <submission_id> [options]

Example:
    python ray_runtime_checker.py http://127.0.0.1:8265 raysubmit_12345
    python ray_runtime_checker.py http://10.0.0.100:8265 my_job_id --verbose
"""

import argparse
import json
import sys
from typing import Optional

try:
    import ray
    from ray.job_submission import JobSubmissionClient
except ImportError:
    print("Error: Ray is not installed. Please install it with: pip install ray")
    sys.exit(1)


def get_job_runtime_env(ray_address: str, submission_id: str, verbose: bool = False) -> Optional[dict]:
    """
    获取指定Ray job的runtime environment信息
    
    Args:
        ray_address: Ray集群地址 (例如: http://127.0.0.1:8265)
        submission_id: Job的submission ID
        verbose: 是否显示详细信息
    
    Returns:
        runtime_env字典或None
    """
    try:
        # 连接到Ray集群
        client = JobSubmissionClient(ray_address)
        
        # 尝试不同的API调用方式
        job_details = None
        
        # 方法1: 直接使用submission_id作为job_id
        try:
            job_details = client.get_job_info(submission_id)
        except (TypeError, AttributeError):
            # 方法2: 尝试使用job_id参数
            try:
                job_details = client.get_job_info(job_id=submission_id)
            except (TypeError, AttributeError):
                # 方法3: 列出所有jobs并查找匹配的submission_id
                jobs = client.list_jobs()
                for job_id, job_info in jobs.items():
                    if hasattr(job_info, 'submission_id') and job_info.submission_id == submission_id:
                        job_details = job_info
                        break
                    elif job_id == submission_id:
                        job_details = client.get_job_info(job_id)
                        break
        
        if job_details is None:
            print(f"Error: Job with submission ID '{submission_id}' not found")
            return None
        
        if verbose:
            print(f"Job ID: {getattr(job_details, 'job_id', 'N/A')}")
            print(f"Submission ID: {getattr(job_details, 'submission_id', submission_id)}")
            print(f"Job Status: {getattr(job_details, 'status', 'N/A')}")
            print(f"Start Time: {getattr(job_details, 'start_time', 'N/A')}")
            print(f"End Time: {getattr(job_details, 'end_time', 'N/A')}")
            print("-" * 50)
        
        return getattr(job_details, 'runtime_env', None)
        
    except Exception as e:
        print(f"Error: Failed to get job information - {str(e)}")
        return None


def format_runtime_env(runtime_env: dict) -> str:
    """
    格式化runtime_env输出
    
    Args:
        runtime_env: runtime environment字典
    
    Returns:
        格式化的字符串
    """
    if not runtime_env:
        return "No runtime environment configured"
    
    output_lines = ["Runtime Environment Configuration:"]
    output_lines.append("=" * 40)
    
    # 常见的runtime_env字段
    common_fields = {
        'working_dir': 'Working Directory',
        'py_modules': 'Python Modules',
        'pip': 'Pip Dependencies',
        'conda': 'Conda Environment',
        'env_vars': 'Environment Variables',
        'excludes': 'Excluded Files/Patterns',
        'container': 'Container Config',
        'plugin_config': 'Plugin Configuration'
    }
    
    for key, display_name in common_fields.items():
        if key in runtime_env:
            value = runtime_env[key]
            output_lines.append(f"{display_name}:")
            if isinstance(value, (dict, list)):
                output_lines.append(json.dumps(value, indent=2, ensure_ascii=False))
            else:
                output_lines.append(f"  {value}")
            output_lines.append("")
    
    # 显示其他字段
    other_fields = set(runtime_env.keys()) - set(common_fields.keys())
    if other_fields:
        output_lines.append("Other Configuration:")
        for key in sorted(other_fields):
            value = runtime_env[key]
            output_lines.append(f"{key}:")
            if isinstance(value, (dict, list)):
                output_lines.append(json.dumps(value, indent=2, ensure_ascii=False))
            else:
                output_lines.append(f"  {value}")
            output_lines.append("")
    
    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Get runtime environment information for a Ray job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s http://127.0.0.1:8265 raysubmit_12345
  %(prog)s http://10.0.0.100:8265 my_job_id --verbose
  %(prog)s http://127.0.0.1:8265 job_abc123 --json
        """
    )
    
    parser.add_argument(
        "ray_address",
        help="Ray cluster dashboard address (e.g., http://127.0.0.1:8265)"
    )
    
    parser.add_argument(
        "submission_id",
        help="Ray job submission ID"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show additional job information"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output runtime_env as raw JSON"
    )
    
    args = parser.parse_args()
    
    # 确保ray_address格式正确
    if not args.ray_address.startswith(('http://', 'https://')):
        args.ray_address = f"http://{args.ray_address}"
    
    print(f"Connecting to Ray cluster at: {args.ray_address}")
    print(f"Looking up submission ID: {args.submission_id}")
    print()
    
    # 获取runtime_env信息
    runtime_env = get_job_runtime_env(args.ray_address, args.submission_id, args.verbose)
    
    if runtime_env is None:
        sys.exit(1)
    
    # 输出结果
    if args.json:
        print("Runtime Environment (JSON):")
        print(json.dumps(runtime_env, indent=2, ensure_ascii=False))
    else:
        print(format_runtime_env(runtime_env))


if __name__ == "__main__":
    main()