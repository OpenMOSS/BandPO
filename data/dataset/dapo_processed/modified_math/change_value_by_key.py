import os
import sys
import pandas as pd

def modify_parquet(parquet_file_path, key_name=None, new_value=None):
    # 读取 Parquet 文件
    try:
        df = pd.read_parquet(parquet_file_path)
    except Exception as e:
        print(f"读取 parquet 失败：{e}")
        return

    if not key_name:
        print("键名为空")
        print("可用键名如下：", list(df.columns))
        return

    # 检查是否存在该列
    if key_name in df.columns:
        if new_value is not None:
            # 将该 key 对应的内容替换为 new_value
            df[key_name] = new_value
            # 生成输出路径
            root, ext = os.path.splitext(parquet_file_path)
            modified_parquet_file_path = f"{root}_modified{ext or '.parquet'}"
            try:
                df.to_parquet(modified_parquet_file_path, index=False)
            except Exception as e:
                print(f"写出 parquet 失败：{e}")
                return
            print(f"修改后的数据已保存为: {modified_parquet_file_path}")
        else:
            print("value为空")
    else:
        print("键不存在，键名如下：")
        print(list(df.columns))


# 1) 必须先有 BandPODir
bandpo_dir = os.environ.get("BandPODir")
if not bandpo_dir:
    print("未检测到 BandPODir。请先在 bash 中执行：source /path_to_BandPO/init.sh")
    sys.exit(1)

# 示例：调用函数进行修改
parquet_file_path = os.path.join(
    bandpo_dir,
    "data/dataset/dapo_processed/modified_math",
    "DAPO-Math-17k-Processed_modified.parquet"
)  # 替换为你要读取的 parquet 文件路径
key_name = "data_source"
new_value = "HuggingFaceH4/MATH-500"
modify_parquet(parquet_file_path, key_name=key_name, new_value=new_value)
