import pandas as pd
import os
def modify_parquet(parquet_file_path, rename_key=None, delete_key=None):
    # 读取 Parquet 文件
    df = pd.read_parquet(parquet_file_path)

    # 如果提供了要删除的键名称，执行删除
    if delete_key:
        if delete_key in df.columns:
            df.drop(columns=[delete_key], inplace=True)  # 删除指定的列    
    # 如果提供了要修改的键名称，执行修改
    if rename_key:
        df.rename(columns=rename_key, inplace=True)  # rename_key是一个字典, 如{'old_key': 'new_key'}
    
    # 保存修改后的 DataFrame 为新的 Parquet 文件（你可以选择覆盖原文件，或保存为新的文件）
    modified_parquet_file_path = parquet_file_path.replace('.parquet', '_modified.parquet')
    df.to_parquet(modified_parquet_file_path)
    
    print(f"修改后的数据已保存为: {modified_parquet_file_path}")

# 1) 必须先有 BandPODir
bandpo_dir = os.environ.get("BandPODir")
if not bandpo_dir:
    print("未检测到 BandPODir。请先在 bash 中执行：source /path_to_BandPO/init.sh")
    sys.exit(1)

# 示例：调用函数进行修改
parquet_file_path = f"{bandpo_dir}/data/dataset/dapo_processed/modified/DAPO-Math-17k-Processed_en.parquet"  # 替换为你要读取的 parquet 文件路径
delete_key = 'prompt'  # 指定要删除的键名
rename_key = {'source_prompt': 'prompt'}  # 指定要修改的键名称，格式：{'原键名': '新键名'}

modify_parquet(parquet_file_path, rename_key=rename_key, delete_key=delete_key)
