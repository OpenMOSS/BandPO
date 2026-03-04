import pandas as pd
import pdb

system_message = {
    "role": "system", 
    "content": "Please reason step by step, and put your final answer within \\boxed{}."
}

def parquet_info(parquet_file_path):
    df = pd.read_parquet(parquet_file_path)
    if 'prompt' in df.columns:
        def modify_prompt(prompt_data):           
            if isinstance(prompt_data, list):
                # 检查是否已经有system消息（避免重复添加）
                has_system = any(msg.get('role') == 'system' for msg in prompt_data)
                
                if not has_system:
                    # 在列表开头插入system消息
                    new_prompt = [system_message] + prompt_data
                    return new_prompt
                else:
                    print("已存在system消息，跳过修改")
                    return prompt_data
            
            # 处理其他格式（如果有）
            return prompt_data

        # 批量修改所有prompt
        df['prompt'] = df['prompt'].apply(modify_prompt)
        output_parquet_path = parquet_file_path.replace('.parquet', '_qwen_template.parquet')
        df.to_parquet(output_parquet_path, index=False)
        print(f"修改后的数据已保存到: {output_parquet_path}")
        pdb.set_trace()
    
    # 输出数据集的大小
    print(f"数据集的大小: {df.shape[0]} 条数据, {df.shape[1]} 列")
    # 输出第一个数据项的键（列名）
    first_row = df.iloc[0]  # 获取第一行数据
    keys = first_row.index.tolist()  # 获取第一行的列名，即键
    print(f"第一个数据项的键: {keys}")
    pdb.set_trace()

# 示例：调用函数查看 Parquet 文件的信息
parquet_file_path = "/remote-home1/yli/Workspace/BandPO/data/dataset/gsm8k/original/test.parquet"  # 替换为你要读取的 parquet 文件路径
parquet_info(parquet_file_path)
