import re
from datasets import load_dataset, Dataset

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))


ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
ds = ds.rename_columns({"prompt" : "source_prompt"})
ds = ds.map(lambda x : {"prompt": x["source_prompt"][0]["content"], "solution": x["reward_model"]["ground_truth"]}, num_proc=24)
# Convert to DataFrame for dedup
df = ds.to_pandas()
df = df.drop_duplicates("prompt")
ds_dedup = Dataset.from_pandas(df, preserve_index=False)
ds_dedup = ds_dedup.select_columns(['prompt', 'solution', 'data_source', 'source_prompt', 'ability', 'reward_model', 'extra_info'])

# Filter for language
en = ds_dedup.filter(lambda x : not contains_chinese_extended(x["prompt"]))
cn = ds_dedup.filter(lambda x : contains_chinese_extended(x["prompt"]))

# Push to Hub
ds_dedup.push_to_hub("open-r1/DAPO-Math-17k", config_name="all")
en.push_to_hub("open-r1/DAPO-Math-17k", config_name="en")
cn.push_to_hub("open-r1/DAPO-Math-17k", config_name="cn")