import asyncio
from tinker_cookbook.recipes.math_rl.math_env import CombinedMathDatasetBuilder
from tinker_cookbook import model_info

async def test_combined_dataset():
    model_name = "meta-llama/Llama-3.1-8B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    
    builder = CombinedMathDatasetBuilder(
        batch_size=256,  # 小批量测试
        group_size=16,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
        seed=42,
        sampling_strategy="interleave",
        total_epochs=1,
        dataset_configs=[
            {
                "type": "math500l35train",
                "parquet_path": "/remote-home1/yli/Workspace/BandPO/data/dataset/math-500/train/train_L3-5.parquet",
            },
            # {
            #     "type": "dapoprocessedtrain",
            #     "parquet_path": "/remote-home1/yli/Workspace/BandPO/data/dataset/dapo_processed/en/train-00000-of-00001.parquet",
            # },
            {
                "type": "dapoprocessedtrain",
                "parquet_path": "/remote-home1/yli/Workspace/BandPO/data/dataset/dapo_processed/all/DAPO-Math-17k-Processed.parquet",
            },
        ],
    )
    
    train_ds, test_ds = await builder()
    
    print(f"训练集总批次数: {len(train_ds)}")
    print(f"测试集总批次数: {len(test_ds) if test_ds else 'None'}")
    
    # 测试获取第一个批次
    batch = train_ds.get_batch(0)
    print(f"第一个批次包含 {len(batch)} 个 env group builders")

if __name__ == "__main__":
    asyncio.run(test_combined_dataset())