import asyncio
import chz
import sys
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train
from functools import partial
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_env import (
    LocalDapoProcessedTrainDatasetBuilder,
    CombinedMathDatasetBuilder,
    OlympiadTestDataset,
    Math500TestDataset,
    MinervaTestDataset,
    Aime2024TestDataset,
    Aime2025TestDataset,
    Amc2023TestDataset,
)
def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    namespace, repo = model_name.split("/", 1)
    repo = repo.strip()
    # tinker render：数据集最后是通过model_name和renderer_name获取tokenizer，最终获取renderer，因此，test集更直接的去接收renderer
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    # ====== 训练参数 ======
    # 重点（关于上下文长度）：1. Tinker 文档在“Sequence Extension”里明确把 context length limits 当作需要工程化处理的约束，并给出“Periodic Compaction（周期性压缩/归并历史）”这类策略，目的就是让长轨迹/长对话在不超过单次上下文上限的前提下继续跑。2. 文档/代码层面没有看到“Tinker 服务端对 prompt 自动截断后继续执行”的明确承诺；相反，cookbook 把“长度管理”当成调用方/训练循环需要负责的事情。如果把历史越堆越长而不做 compaction / truncation，就迟早会触顶（请求无法继续、或者不得不提前停止）。因此，建议自己检查。但是对于单轮math而言一般不会超过。推理侧你一般需要自己在调用前做：获取模型卡中的上下文上限（自己找），计算 prompt_len = model_input.length()，保证 prompt_len + max_tokens <= 模型上下文上限。超了就做 截断（丢弃最老消息） 或 compaction（总结/压缩历史）（文档推荐思路）。
    max_prompt_length = 1024 # 该参数无效，Tinker 的采样 / 训练底层输入是 ModelInput，长度统计接口：ModelInput.length()
    max_tokens = 4096 # 等于原来的max_response_length_real，实际控制的是“生成时最多生成多少个新 token”

    learning_rate = 4e-5
    temperature = 1.0
    # 这里的batch都是按照prompt数来计算，而不是乘以了group后的rollout数目
    # PPO minibatch：256 / 64 = 4 个 minibatches
    train_batch_size = 256            # groups per batch
    ppo_mini_batch_size = 64          # groups per minibatch
    num_substeps = train_batch_size // ppo_mini_batch_size # 有多少个mini batch
    ppo_micro_batch_size=8 # 数据更新角度来看，等同于verl中的ppo_micro_batch_size，但是不完全是，因为采样和更新逻辑不一样
    num_minibatches = ppo_mini_batch_size // ppo_micro_batch_size # 不是mini_batch的数量,是mini batch被切分成多少个micro batch的数量，仅对streaming有效
    stream_minibatch_config = train.StreamMinibatchConfig(
        groups_per_batch=train_batch_size,
        num_minibatches=num_minibatches,
    )
    # stream_minibatch_config = None
    pad_to_full_batch = stream_minibatch_config is not None
    rollout_n = 16                    # group_size
    total_epochs = 2                 # 这里建议你用“重复 dataset”的方式实现（见下方说明）
    save_every = 10
    # 训练方式： (2)(3) 都是默认关闭，需要显式开启
    # 1. 默认同步 on-policy（不加任何高级配置）：不设置 StreamMinibatchConfig，也不设置 AsyncConfig，就是默认模式。
    # 2. Streaming minibatch training（开启流水：采样与训练重叠）：加 StreamMinibatchConfig(...) 就是在开启第二种。把一次训练 step 的数据拆成多个 minibatch，并且让采样与训练流水化重叠。on-policy 不变，只是把“采样等待”变成“凑够一小份就先训练”，用于提升吞吐的流水化优化
    # 3. Async off-policy training（bounded off-policy：允许训练用“落后 K 步”的策略数据）：加 AsyncConfig(off_by=K) 开启。off_by=K 的语义是“训练可能用到最多落后 K step 的策略生成的数据”，它能提高吞吐，但会带来更强的 off-policy 风险，需要关注 KL 等稳定性指标。
    # ====== 评测参数 ======
    sampling_temperature = 1.0 # 该参数无效，RLTestSetEvaluator调用TinkerTokenCompleter时没有传参温度的参数，默认是temperature: float = 1.0
    eval_batch_size = 32
    eval_every = 5
    # ====== 特殊参数 ======
    seed=42
    project_str = "tinker_testv2" # 万一和其他使用的人重合了，这里改复杂一点
    name_str = f"{repo}_dapo"
    if stream_minibatch_config:
        name_str = name_str + "_stream"
    log_path = f"/tmp/{project_str}/{name_str}"
    wandb_project = project_str
    wandb_name = name_str
    check_dataset_only = True
    check_dataset_before_train = True
    # sample都没有Top p的设置， sample的参数只有三个 temperature=1.0 max_tokens=self.max_tokens stop=self.stop_condition
    # ====== 训练集 ======
    train_builder = CombinedMathDatasetBuilder(
        batch_size=train_batch_size,
        group_size=rollout_n,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
        seed=seed,
        sampling_strategy="interleave",  # 可选: "interleave", "concatenate", "proportional"
        total_epochs=total_epochs,
        pad_to_full_batch=pad_to_full_batch,  # ✅ 新增
        dataset_configs=[
            {
                "type": "math500l35train",
                "parquet_path": "/remote-home1/yli/Workspace/BandPO/data/dataset/math-500/train/train_L3-5.parquet",
                "timeout": 3.0,
            },
            {
                "type": "dapoprocessedtrain",
                "parquet_path": "/remote-home1/yli/Workspace/BandPO/data/dataset/dapo_processed/all/DAPO-Math-17k-Processed.parquet",
                "timeout": 3.0,
            },
        ],
    )
    # train_builder = LocalDapoProcessedTrainDatasetBuilder(
    #     parquet_path="/remote-home1/yli/Workspace/BandPO/data/dataset/dapo_processed/all/DAPO-Math-17k-Processed.parquet",
    #     batch_size=train_batch_size,
    #     model_name_for_tokenizer=model_name,
    #     renderer_name=renderer_name,
    #     group_size=rollout_n,
    #     seed=seed,
    #     timeout=3.0,
    # )
    # ====== 测试集 ======
    olympiad_ds = OlympiadTestDataset(
        parquet_path="/remote-home1/yli/Workspace/BandPO/data/dataset/olympiad_dapo/OE_TO_maths_en_COMP/test.parquet",
        batch_size=eval_batch_size,
        group_size=2,
        renderer=renderer,
        timeout=3.0,
    )
    math500_ds = Math500TestDataset(
        parquet_path="/remote-home1/yli/Workspace/BandPO/data/dataset/math-500/test/test.parquet",
        batch_size=eval_batch_size,
        group_size=2,
        renderer=renderer,
        timeout=3.0,
    )
    minerva_ds = MinervaTestDataset(
        parquet_path="/remote-home1/yli/Workspace/BandPO/data/dataset/minerva_dapo/test.parquet",
        batch_size=eval_batch_size,
        group_size=2,
        renderer=renderer,
        timeout=3.0,
    )
    aime2024_ds = Aime2024TestDataset(
        parquet_path="/remote-home1/yli/Workspace/BandPO/data/dataset/aime2024_dapo/test.parquet",
        batch_size=eval_batch_size,
        group_size=32,
        renderer=renderer,
        timeout=3.0,
    )
    aime2025_ds = Aime2025TestDataset(
        parquet_path="/remote-home1/yli/Workspace/BandPO/data/dataset/aime2025_dapo/test.parquet",
        batch_size=eval_batch_size,
        group_size=32,
        renderer=renderer,
        timeout=3.0,
    )
    amc2023_ds = Amc2023TestDataset(
        parquet_path="/remote-home1/yli/Workspace/BandPO/data/dataset/amc2023_dapo/test.parquet",
        batch_size=eval_batch_size,
        group_size=32,
        renderer=renderer,
        timeout=3.0,
    )

    # ====== 6 个 evaluator builder（最正统：直接 partial 构造 RLTestSetEvaluator）======
    evaluator_builders = [
        partial(RLTestSetEvaluator, olympiad_ds, max_tokens, name="olympiadtest"),
        partial(RLTestSetEvaluator, math500_ds,  max_tokens, name="math500test"),
        partial(RLTestSetEvaluator, minerva_ds,  max_tokens, name="minervatest"),
        partial(RLTestSetEvaluator, aime2024_ds, max_tokens, name="aime2024test"),
        partial(RLTestSetEvaluator, aime2025_ds, max_tokens, name="aime2025test"),
        partial(RLTestSetEvaluator, amc2023_ds,  max_tokens, name="amc2023test"),
    ]
    return chz.Blueprint(train.Config).apply(
        {
            "model_name": model_name,
            "log_path": log_path,
            "dataset_builder": train_builder,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "loss_fn": "ppo",
            "num_substeps": num_substeps,
            "stream_minibatch_config": stream_minibatch_config,
            "check_dataset_only": check_dataset_only,
            "check_dataset_before_train": check_dataset_before_train,
            "save_every": save_every,
            "eval_every": eval_every,
            "evaluator_builders": evaluator_builders,
            "wandb_project": wandb_project,
            "wandb_name": wandb_name,
        }
    )


def main(config: train.Config):
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
