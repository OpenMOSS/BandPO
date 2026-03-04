import pandas as pd
import json
import random

def parquet_to_json(parquet_file_path):
    # 读取 Parquet 文件
    df = pd.read_parquet(parquet_file_path)
    
    # # 将 DataFrame 转换为 JSON 格式并保存
    json_file_path = parquet_file_path.replace('.parquet', '.jsonl')
    df.to_json(json_file_path, orient='records', lines=True)
    
    # 随机抽取3条数据
    random_sample = df.sample(n=3)
    
    # 将随机抽取的数据保存为 _simple 后缀的 JSON 文件
    simple_json_file_path = parquet_file_path.replace('.parquet', '_simple.jsonl')
    random_sample.to_json(simple_json_file_path, orient='records', lines=True)
    
    print(f"文件已保存为: {json_file_path} 和 {simple_json_file_path}")

# 示例：调用函数进行转换
parquet_file_path = "/remote-home1/yli/Workspace/BandPO/data/dataset/math-500/train/train_L3-5.parquet"  # 替换为你要读取的 parquet 文件路径
parquet_to_json(parquet_file_path)
