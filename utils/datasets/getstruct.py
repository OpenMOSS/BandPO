import pandas as pd

def parquet_info(parquet_file_path):
    # 读取 Parquet 文件
    df = pd.read_parquet(parquet_file_path)
    
    # 输出数据集的大小
    print(f"数据集的大小: {df.shape[0]} 条数据, {df.shape[1]} 列")
    
    # 输出第一个数据项的键（列名）
    first_row = df.iloc[0]  # 获取第一行数据
    keys = first_row.index.tolist()  # 获取第一行的列名，即键
    print(f"第一个数据项的键: {keys}")

# 示例：调用函数查看 Parquet 文件的信息
parquet_file_path = "/remote-home1/yli/Workspace/BandPO/data/dataset/dapo_processed/all/DAPO-Math-17k-Processed.parquet"
parquet_file_path = "/remote-home1/yli/Workspace/BandPO/data/dataset/math-500/train/train_L3-5.parquet"
parquet_file_path = "/remote-home1/yli/Workspace/BandPO/data/dataset/minerva_dapo/test.parquet"
# parquet_file_path = "/remote-home1/yli/Workspace/BandPO/data/dataset/olympiad_dapo/OE_TO_maths_en_COMP/test.parquet"

# parquet_file_path = "/remote-home1/yli/Workspace/BandPO/data/dataset/math-500/train/train_L3-5.parquet"  # 替换为你要读取的 parquet 文件路径
parquet_info(parquet_file_path)
