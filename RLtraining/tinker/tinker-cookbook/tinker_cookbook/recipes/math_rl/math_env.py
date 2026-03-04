import math
import re
from functools import partial
from typing import Any, List, Literal, Sequence, cast
from itertools import cycle

import chz
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

def _pad_hf_dataset_to_multiple(ds: Dataset, multiple: int) -> Dataset:
    """
    把 ds 的长度补齐到 multiple 的整数倍：把“最前面”的 pad_count 条 append 到末尾。
    例如 len=4353, multiple=256 -> pad_count=255, append ds[:255]
    """
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")
    n = len(ds)
    pad_count = (-n) % multiple
    if pad_count == 0:
        return ds
    pad_part = ds.select(range(pad_count))
    return concatenate_datasets([ds, pad_part])


class MathEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
    ):
        super().__init__(renderer, convo_prefix)
        self.problem = problem
        self.answer = answer
        self.grader = grader
        self.timeout = timeout

    @classmethod
    def question_suffix(cls) -> str:
        return " Write your answer in \\boxed{} format."

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return False
        return safe_grade(answer, self.answer, self.grader, self.timeout)

    def get_reference_answer(self) -> str:
        return self.answer

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": "How many r's are in strawberry?" + MathEnv.question_suffix(),
            },
            {
                "role": "assistant",
                "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
            },
        ]


def safe_grade(given_answer: str, ground_truth: str, grader: str = "sympy", timeout: float = 1.0):
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    out = run_with_timeout_signal(
        grader_func, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        return False
    return out


def extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final numeric/string answer from a GSM8K solution field.

    GSM8K format typically places the final answer on a line starting with
    '####'. We take the substring following '####' on the last such line.
    """
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            content = content.replace(",", "").strip()
            return content
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise ValueError("No GSM8K final answer found")


def _get_hendrycks_math_test() -> Dataset:
    test_dataset = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    return cast(Dataset, test_dataset)


def _get_hendrycks_math_train() -> Dataset:
    # For Hendrycks MATH, the standard is to use both the "train" and "test" splits for
    # training. The "test" split here is NOT the same as the MATH-500 test split above,
    # which is a commonly-held-out subset of 500 of the below 12.5k problems. To construct
    # a clean training set, we filter out problems that exist in the MATH-500 test set,
    # resulting in 12000 train and 500 test problems.

    test_problems: set[str] = {
        problem["problem"]  # pyright: ignore[reportArgumentType, reportCallIssue]
        for problem in _get_hendrycks_math_test()
    }

    dataset_name = "EleutherAI/hendrycks_math"
    configs = get_dataset_config_names(dataset_name)
    pieces = []
    for cfg in configs:
        for split in ("train", "test"):
            ds = load_dataset(dataset_name, name=cfg, split=split)
            ds = ds.filter(lambda example: example["problem"] not in test_problems)
            pieces.append(ds)
    full_dataset = concatenate_datasets(pieces)

    return full_dataset


class MathDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        if split == "train":
            self.ds = _get_hendrycks_math_train().shuffle(seed=seed)
        elif split == "test":
            self.ds = _get_hendrycks_math_test()
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            answer = extract_boxed(x["solution"])
        except ValueError:  # not sure if this happens
            logger.warning(f"No answer found for {x['solution']}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, x["problem"], answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
        )


@chz.chz
class MathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0

    async def __call__(self) -> tuple[MathDataset, MathDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            MathDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


class PolarisDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
    ):
        # Don't call super().__init__ since we're overriding the dataset loading
        self.ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train").shuffle(
            seed=seed
        )
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="polaris",
        )


@chz.chz
class PolarisDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0

    async def __call__(self) -> tuple[PolarisDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return PolarisDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            seed=self.seed,
        ), None


class DeepMathDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
    ):
        # Don't call super().__init__ since we're overriding the dataset loading
        self.ds = load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("question", "")
        answer = x.get("final_answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="deepmath",
        )


@chz.chz
class DeepMathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0

    async def __call__(self) -> tuple[DeepMathDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return DeepMathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            seed=self.seed,
        ), None


class Gsm8kDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split=split))
        if split == "train":
            self.ds = self.ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    @classmethod
    def question_suffix(cls) -> str:
        return " Provide a numerical answer without units, written inside \\boxed{}."

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            problem = x["question"]
            answer = extract_gsm8k_final_answer(x["answer"])
        except Exception as e:
            logger.warning(f"Failed to parse GSM8K row: {e}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
        )


@chz.chz
class Gsm8kDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0

    async def __call__(self) -> tuple[Gsm8kDataset, Gsm8kDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            Gsm8kDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


class LocalMath500L35TrainDataset(RLDataset):
    def __init__(
        self,
        parquet_path: str,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
        pad_to_full_batch: bool = False,
    ):
        ds = load_dataset("parquet", data_files=parquet_path, split="train")
        self.ds = cast(Dataset, ds).shuffle(seed=seed)

        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.grader = grader
        self.timeout = timeout
        self.pad_to_full_batch = pad_to_full_batch

        if self.pad_to_full_batch:
            self.ds = _pad_hf_dataset_to_multiple(self.ds, self.batch_size)

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    # def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
    #     bs = index * self.batch_size
    #     be = min((index + 1) * self.batch_size, len(self.ds))
    #     assert bs < be
    #     out = []
    #     for row in self.ds.select(range(bs, be)):
    #         b = self._make_env_group_builder(row, self.group_size)
    #         if b is not None:
    #             out.append(b)
    #     return out
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        bs = index * self.batch_size

        if self.pad_to_full_batch:
            be = bs + self.batch_size
            if be > len(self.ds):
                raise IndexError(
                    f"[LocalMath500L35TrainDataset] index={index} out of range: "
                    f"bs={bs}, be={be}, len(ds)={len(self.ds)}"
                )

            out: list[EnvGroupBuilder] = []
            bad_rows: list[int] = []

            # 固定窗口，严格不允许 None
            window = self.ds.select(range(bs, be))
            for offset, row in enumerate(window):
                b = self._make_env_group_builder(row, self.group_size)
                if b is None:
                    bad_rows.append(bs + offset)
                else:
                    out.append(b)

            if bad_rows:
                # 打印少量样例帮助定位数据问题
                sample_bad = bad_rows[:5]
                raise ValueError(
                    f"[LocalMath500L35TrainDataset] pad_to_full_batch=True requires a full valid batch, "
                    f"but found {len(bad_rows)} invalid rows in batch index={index}. "
                    f"Example row indices: {sample_bad}. "
                    f"Please check missing/empty fields in parquet (extra_info.question_raw / extra_info.answer)."
                )

            # 这里应该严格等于 batch_size
            if len(out) != self.batch_size:
                raise ValueError(
                    f"[LocalMath500L35TrainDataset] Internal error: expected {self.batch_size} builders, got {len(out)}"
                )
            return out

        # -------- 非 pad：允许尾 batch 不满，允许跳过坏行，但要 warning --------
        be = min((index + 1) * self.batch_size, len(self.ds))
        assert bs < be

        out = []
        num_total = be - bs
        num_bad = 0
        for row in self.ds.select(range(bs, be)):
            b = self._make_env_group_builder(row, self.group_size)
            if b is None:
                num_bad += 1
                continue
            out.append(b)

        if num_bad > 0:
            logger.warning(
                f"[LocalMath500L35TrainDataset] get_batch(index={index}) skipped {num_bad}/{num_total} invalid rows "
                f"(pad_to_full_batch=False)."
            )
        return out

    def _make_env_group_builder(self, x: dict[str, Any], group_size: int) -> ProblemGroupBuilder | None:
        extra = x.get("extra_info") or {}
        problem = (extra.get("question_raw") or "").strip()
        answer = (extra.get("answer") or "").strip()

        if not (problem and answer):
            return None

        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                grader=self.grader,
                timeout=self.timeout,
            ),
            num_envs=group_size,
            dataset_name="local_math500_l3_5_train",
        )


@chz.chz
class LocalMath500L35TrainDatasetBuilder(RLDatasetBuilder):
    parquet_path: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    grader: Literal["sympy", "math_verify"] = "sympy"
    timeout: float = 1.0

    async def __call__(self) -> tuple[LocalMath500L35TrainDataset, None]:
        prefix = MathEnv.standard_fewshot_prefix() if self.convo_prefix == "standard" else self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        train = LocalMath500L35TrainDataset(
            parquet_path=self.parquet_path,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=prefix,
            seed=self.seed,
            grader=self.grader,
            timeout=self.timeout,
        )
        return train, None


class LocalDapoProcessedTrainDataset(RLDataset):
    def __init__(
        self,
        parquet_path: str,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
        pad_to_full_batch: bool = False,
    ):
        ds = load_dataset("parquet", data_files=parquet_path, split="train")
        self.ds = cast(Dataset, ds).shuffle(seed=seed)

        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.grader = grader
        self.timeout = timeout
        self.pad_to_full_batch = pad_to_full_batch

        if self.pad_to_full_batch:
            self.ds = _pad_hf_dataset_to_multiple(self.ds, self.batch_size)

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    # def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
    #     bs = index * self.batch_size
    #     be = min((index + 1) * self.batch_size, len(self.ds))
    #     assert bs < be
    #     out = []
    #     for row in self.ds.select(range(bs, be)):
    #         b = self._make_env_group_builder(row, self.group_size)
    #         if b is not None:
    #             out.append(b)
    #     return out
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        bs = index * self.batch_size

        if self.pad_to_full_batch:
            be = bs + self.batch_size
            if be > len(self.ds):
                raise IndexError(
                    f"[LocalDapoProcessedTrainDataset] index={index} out of range: "
                    f"bs={bs}, be={be}, len(ds)={len(self.ds)}"
                )

            out: list[EnvGroupBuilder] = []
            bad_rows: list[int] = []

            window = self.ds.select(range(bs, be))
            for offset, row in enumerate(window):
                b = self._make_env_group_builder(row, self.group_size)
                if b is None:
                    bad_rows.append(bs + offset)
                else:
                    out.append(b)

            if bad_rows:
                sample_bad = bad_rows[:5]
                raise ValueError(
                    f"[LocalDapoProcessedTrainDataset] pad_to_full_batch=True requires a full valid batch, "
                    f"but found {len(bad_rows)} invalid rows in batch index={index}. "
                    f"Example row indices: {sample_bad}. "
                    f"Please check missing/empty fields in parquet (prompt / solution)."
                )

            if len(out) != self.batch_size:
                raise ValueError(
                    f"[LocalDapoProcessedTrainDataset] Internal error: expected {self.batch_size} builders, got {len(out)}"
                )
            return out

        # -------- 非 pad：允许尾 batch 不满，允许跳过坏行，但要 warning --------
        be = min((index + 1) * self.batch_size, len(self.ds))
        assert bs < be

        out = []
        num_total = be - bs
        num_bad = 0
        for row in self.ds.select(range(bs, be)):
            b = self._make_env_group_builder(row, self.group_size)
            if b is None:
                num_bad += 1
                continue
            out.append(b)

        if num_bad > 0:
            logger.warning(
                f"[LocalDapoProcessedTrainDataset] get_batch(index={index}) skipped {num_bad}/{num_total} invalid rows "
                f"(pad_to_full_batch=False)."
            )
        return out

    def _make_env_group_builder(self, x: dict[str, Any], group_size: int) -> ProblemGroupBuilder | None:
        # 纯题面就是 prompt 字符串
        problem = (x.get("prompt") or "").strip()
        answer = (x.get("solution") or "").strip()

        if not (problem and answer):
            return None

        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                grader=self.grader,
                timeout=self.timeout,
            ),
            num_envs=group_size,
            dataset_name="local_dapo_processed_train",
        )


@chz.chz
class LocalDapoProcessedTrainDatasetBuilder(RLDatasetBuilder):
    parquet_path: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    grader: Literal["sympy", "math_verify"] = "sympy"
    timeout: float = 1.0

    async def __call__(self) -> tuple[LocalDapoProcessedTrainDataset, None]:
        prefix = MathEnv.standard_fewshot_prefix() if self.convo_prefix == "standard" else self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        train = LocalDapoProcessedTrainDataset(
            parquet_path=self.parquet_path,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=prefix,
            seed=self.seed,
            grader=self.grader,
            timeout=self.timeout,
        )
        return train, None


class CombinedMathDataset(RLDataset):
    """组合多个数学数据集，支持多种采样策略"""
    def __init__(
        self,
        datasets: List[RLDataset],
        sampling_strategy: Literal["interleave", "concatenate", "proportional"] = "interleave",
        weights: List[float] | None = None,
        dataset_names: List[str] | None = None,
        num_epochs: int = 1,  # ✅ 新增：repeat epochs
    ):
        self.datasets = datasets
        self.sampling_strategy = sampling_strategy
        self.weights = weights or [1.0] * len(datasets)
        self.dataset_names = dataset_names or [f"dataset_{i}" for i in range(len(datasets))]
        self.num_epochs = max(1, int(num_epochs))

        # 预计算每个数据集的长度和累积长度（长度都是“各 dataset 的 batch 数”）
        self._dataset_lengths = [len(ds) for ds in datasets]
        self._cumulative_lengths = []
        cumsum = 0
        for length in self._dataset_lengths:
            cumsum += length
            self._cumulative_lengths.append(cumsum)

        # 为 interleave 策略创建索引映射
        if sampling_strategy == "interleave":
            self._build_interleave_mapping()

        self._print_dataset_info()
        # epoch 进度打印控制：每个 epoch 只打印一次
        self.log_epoch_progress = True
        self._last_logged_epoch = -1

    def _base_len(self) -> int:
        """单个 epoch 的 batch 数"""
        if self.sampling_strategy == "concatenate":
            return sum(self._dataset_lengths)
        elif self.sampling_strategy == "interleave":
            return len(self._interleave_map)
        else:  # proportional（你目前简化成 concatenate）
            return sum(self._dataset_lengths)

    def _epoch_info(self, global_batch_index: int) -> tuple[int, int]:
        """
        给定全局 batch index，返回：
          - epoch0: 0-based epoch index
          - local_idx: 在当前 epoch 内的 batch index（0-based）
        """
        base_len = self._base_len()
        if base_len <= 0:
            raise ValueError("Empty CombinedMathDataset")
        return global_batch_index // base_len, global_batch_index % base_len

    def _print_dataset_info(self):
        """打印数据集加载的详细信息"""
        print("\n" + "=" * 60)
        print("📊 CombinedMathDataset 加载完成")
        print("=" * 60)
        print(f"\n{'序号':<4} {'数据集名称':<25} {'样本数':<10} {'批次数':<10}")
        print("-" * 60)

        total_samples = 0
        for i, (ds, name, batch_count) in enumerate(
            zip(self.datasets, self.dataset_names, self._dataset_lengths)
        ):
            sample_count = len(ds.ds) if hasattr(ds, "ds") else "N/A"
            print(f"{i+1:<4} {name:<25} {str(sample_count):<10} {batch_count:<10}")
            if isinstance(sample_count, int):
                total_samples += sample_count

        print("-" * 60)
        print(f"{'合计':<4} {'':<25} {total_samples:<10} {sum(self._dataset_lengths):<10}")
        print(f"\n⚙️  采样策略: {self.sampling_strategy}")
        print(f"🔁 num_epochs: {self.num_epochs}")
        print(f"🧱 base_batches_per_epoch: {self._base_len()}")
        print(f"🚀 total_batches: {len(self)}")
        print(f"📌 epoch 推导: epoch_idx = global_batch // {self._base_len()} (0-based), 显示时 +1")
        print("=" * 60 + "\n")

    def _build_interleave_mapping(self):
        """构建交错采样的索引映射"""
        self._interleave_map = []  # (dataset_idx, batch_idx)
        if not self._dataset_lengths:
            return
        max_len = max(self._dataset_lengths)
        for batch_idx in range(max_len):
            for ds_idx, ds_len in enumerate(self._dataset_lengths):
                if batch_idx < ds_len:
                    self._interleave_map.append((ds_idx, batch_idx))

    def __len__(self) -> int:
        # ✅ 关键：repeat epochs
        return self._base_len() * self.num_epochs

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        global_index = index  # 训练循环传进来的全局 batch idx（0..len(dataset)-1）
        epoch0, local_index = self._epoch_info(global_index)
        # 只在每个 epoch 的第一个 batch 打印一次
        if (
            self.log_epoch_progress
            and local_index == 0
            and epoch0 != self._last_logged_epoch
        ):
            self._last_logged_epoch = epoch0
            print(f"🔁 [CombinedMathDataset] epoch {epoch0 + 1}/{self.num_epochs} (global_batch={global_index})")

        # epoch repeat 的索引映射：把全局 index 映射回单 epoch 内的 index
        index = local_index

        # ====== 下面保持你原来的 sampling_strategy 路由逻辑不变 ======
        if self.sampling_strategy == "concatenate":
            for ds_idx, cumlen in enumerate(self._cumulative_lengths):
                if index < cumlen:
                    local_idx = index if ds_idx == 0 else index - self._cumulative_lengths[ds_idx - 1]
                    return self.datasets[ds_idx].get_batch(local_idx)
            raise IndexError(f"Batch index {index} out of range")

        elif self.sampling_strategy == "interleave":
            ds_idx, local_idx = self._interleave_map[index]
            return self.datasets[ds_idx].get_batch(local_idx)

        else:  # proportional - 这里简化为concatenate行为
            for ds_idx, cumlen in enumerate(self._cumulative_lengths):
                if index < cumlen:
                    local_idx = index if ds_idx == 0 else index - self._cumulative_lengths[ds_idx - 1]
                    return self.datasets[ds_idx].get_batch(local_idx)
            raise IndexError(f"Batch index {index} out of range")

@chz.chz
class CombinedMathDatasetBuilder(RLDatasetBuilder):
    """组合多个数学数据集的Builder"""
    
    # 通用参数
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    sampling_strategy: Literal["interleave", "concatenate", "proportional"] = "interleave"
    
    # 数据集配置 - 使用列表形式
    dataset_configs: List[dict]  # 每个dict包含数据集的配置
    total_epochs: int = 1
    pad_to_full_batch: bool = False

    async def __call__(self) -> tuple[CombinedMathDataset, RLDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        convo_prefix = MathEnv.standard_fewshot_prefix()
        
        train_datasets = []
        test_datasets = []
        train_dataset_names = []
        test_dataset_names = []
        for config in self.dataset_configs:
            ds_type = config["type"]
            
            if ds_type == "gsm8k":
                train_ds = Gsm8kDataset(
                    batch_size=self.batch_size,
                    group_size=self.group_size,
                    renderer=renderer,
                    convo_prefix=convo_prefix,
                    split="train",
                    seed=self.seed,
                )
                test_ds = Gsm8kDataset(
                    batch_size=self.batch_size,
                    group_size=1,
                    renderer=renderer,
                    convo_prefix=convo_prefix,
                    split="test",
                    seed=self.seed,
                )
                train_datasets.append(train_ds)
                test_datasets.append(test_ds)
                test_dataset_names.append(ds_type)
                train_dataset_names.append(ds_type)
            elif ds_type == "math":
                train_ds = MathDataset(
                    batch_size=self.batch_size,
                    group_size=self.group_size,
                    renderer=renderer,
                    convo_prefix=convo_prefix,
                    split="train",
                    seed=self.seed,
                )
                test_ds = MathDataset(
                    batch_size=self.batch_size,
                    group_size=1,
                    renderer=renderer,
                    convo_prefix=convo_prefix,
                    split="test",
                    seed=self.seed,
                )
                train_datasets.append(train_ds)
                test_datasets.append(test_ds)
                test_dataset_names.append(ds_type)
                train_dataset_names.append(ds_type)
            elif ds_type == "math500l35train":
                parquet_path = config["parquet_path"]
                grader = config.get("grader", "sympy")
                timeout = config.get("timeout", 1.0)
                train_ds = LocalMath500L35TrainDataset(
                    parquet_path=parquet_path,
                    batch_size=self.batch_size,
                    group_size=self.group_size,
                    renderer=renderer,
                    convo_prefix=convo_prefix,
                    seed=self.seed,
                    grader=grader,
                    timeout=timeout,
                    pad_to_full_batch=self.pad_to_full_batch,
                )
                train_datasets.append(train_ds)
                train_dataset_names.append(ds_type)
            elif ds_type == "dapoprocessedtrain":
                parquet_path = config["parquet_path"]
                grader = config.get("grader", "sympy")
                timeout = config.get("timeout", 1.0)
                train_ds = LocalDapoProcessedTrainDataset(
                    parquet_path=parquet_path,
                    batch_size=self.batch_size,
                    group_size=self.group_size,
                    renderer=renderer,
                    convo_prefix=convo_prefix,
                    seed=self.seed,
                    grader=grader,
                    timeout=timeout,
                    pad_to_full_batch=self.pad_to_full_batch,
                )
                train_datasets.append(train_ds)
                train_dataset_names.append(ds_type)
            elif ds_type == "polaris":
                train_ds = PolarisDataset(
                    batch_size=self.batch_size,
                    group_size=self.group_size,
                    renderer=renderer,
                    convo_prefix=convo_prefix,
                    seed=self.seed,
                )
                train_datasets.append(train_ds)
                train_dataset_names.append(ds_type)
            elif ds_type == "deepmath":
                train_ds = DeepMathDataset(
                    batch_size=self.batch_size,
                    group_size=self.group_size,
                    renderer=renderer,
                    convo_prefix=convo_prefix,
                    seed=self.seed,
                )
                train_datasets.append(train_ds)
                train_dataset_names.append(ds_type)
        combined_train = CombinedMathDataset(
            datasets=train_datasets,
            sampling_strategy=self.sampling_strategy,
            dataset_names=train_dataset_names,
            num_epochs=self.total_epochs,  # ✅ 新增
        )
        
        # 如果有test数据集，也组合它们
        combined_test = None
        if test_datasets:
            combined_test = CombinedMathDataset(
                datasets=test_datasets,
                sampling_strategy="concatenate",  # test通常用concatenate
                dataset_names=test_dataset_names,
            )
        
        return combined_train, combined_test


# =========================
# Local RLVR Validation Datasets (Parquet)
# 每个数据集单独写读取逻辑（不要统一函数），便于后续每个数据集各自演化。
#
# 当前样例字段约定（你后面要改就改这里）：
#   - 问题：row["extra_info"]["question_raw"]
#   - 答案：row["reward_model"]["ground_truth"]
# =========================

class _LocalParquetTestDatasetBase(RLDataset):
    """
    仅作为最小复用：batch/get_batch 模板相同。
    但每个子类自己实现 _extract_problem_answer（后续各自改读取逻辑）。
    """

    DATASET_NAME: str = "local_test"

    def __init__(
        self,
        parquet_path: str,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard",
        seed: int = 0,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
        shuffle: bool = False,  # test 默认不 shuffle，便于可复现
    ):
        ds = load_dataset("parquet", data_files=parquet_path, split="train")
        self.ds = cast(Dataset, ds)
        if shuffle:
            self.ds = self.ds.shuffle(seed=seed)
        if convo_prefix == "standard":
            self.convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            self.convo_prefix = convo_prefix

        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.grader = grader
        self.timeout = timeout

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        bs = index * self.batch_size
        be = min((index + 1) * self.batch_size, len(self.ds))
        assert bs < be
        out = []
        for row in self.ds.select(range(bs, be)):
            b = self._make_env_group_builder(row, self.group_size)
            if b is not None:
                out.append(b)
        return out

    # ---- 子类必须覆盖：后续你要改 key 就改这里 ----
    def _extract_problem_answer(self, x: dict[str, Any]) -> tuple[str, str]:
        raise NotImplementedError

    def _make_env_group_builder(self, x: dict[str, Any], group_size: int) -> ProblemGroupBuilder | None:
        problem, answer = self._extract_problem_answer(x)
        problem = (problem or "").strip()
        answer = (answer or "").strip()
        if not (problem and answer):
            return None

        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                grader=self.grader,
                timeout=self.timeout,
            ),
            num_envs=group_size,
            dataset_name=self.DATASET_NAME,
        )


# -------- Olympiad --------
class OlympiadTestDataset(_LocalParquetTestDatasetBase):
    DATASET_NAME = "olympiad"

    def _extract_problem_answer(self, x: dict[str, Any]) -> tuple[str, str]:
        extra = x.get("extra_info") or {}
        rm = x.get("reward_model") or {}
        return (extra.get("question_raw") or "", rm.get("ground_truth") or "")


@chz.chz
class OlympiadTestDatasetBuilder(RLDatasetBuilder):
    parquet_path: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    grader: Literal["sympy", "math_verify"] = "sympy"
    timeout: float = 1.0

    async def __call__(self) -> tuple[OlympiadTestDataset, None]:
        prefix = MathEnv.standard_fewshot_prefix() if self.convo_prefix == "standard" else self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        ds = OlympiadTestDataset(
            parquet_path=self.parquet_path,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=prefix,
            seed=self.seed,
            grader=self.grader,
            timeout=self.timeout,
            shuffle=False,
        )
        return ds, None


# -------- Math500 --------
class Math500TestDataset(_LocalParquetTestDatasetBase):
    DATASET_NAME = "math500"

    def _extract_problem_answer(self, x: dict[str, Any]) -> tuple[str, str]:
        extra = x.get("extra_info") or {}
        rm = x.get("reward_model") or {}
        return (extra.get("question_raw") or "", rm.get("ground_truth") or "")


@chz.chz
class Math500TestDatasetBuilder(RLDatasetBuilder):
    parquet_path: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    grader: Literal["sympy", "math_verify"] = "sympy"
    timeout: float = 1.0

    async def __call__(self) -> tuple[Math500TestDataset, None]:
        prefix = MathEnv.standard_fewshot_prefix() if self.convo_prefix == "standard" else self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        ds = Math500TestDataset(
            parquet_path=self.parquet_path,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=prefix,
            seed=self.seed,
            grader=self.grader,
            timeout=self.timeout,
            shuffle=False,
        )
        return ds, None


# -------- Minerva --------
class MinervaTestDataset(_LocalParquetTestDatasetBase):
    DATASET_NAME = "minerva"

    def _extract_problem_answer(self, x: dict[str, Any]) -> tuple[str, str]:
        extra = x.get("extra_info") or {}
        rm = x.get("reward_model") or {}
        return (extra.get("question_raw") or "", rm.get("ground_truth") or "")


@chz.chz
class MinervaTestDatasetBuilder(RLDatasetBuilder):
    parquet_path: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    grader: Literal["sympy", "math_verify"] = "sympy"
    timeout: float = 1.0

    async def __call__(self) -> tuple[MinervaTestDataset, None]:
        prefix = MathEnv.standard_fewshot_prefix() if self.convo_prefix == "standard" else self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        ds = MinervaTestDataset(
            parquet_path=self.parquet_path,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=prefix,
            seed=self.seed,
            grader=self.grader,
            timeout=self.timeout,
            shuffle=False,
        )
        return ds, None


# -------- AIME2024 --------
class Aime2024TestDataset(_LocalParquetTestDatasetBase):
    DATASET_NAME = "aime2024"

    def _extract_problem_answer(self, x: dict[str, Any]) -> tuple[str, str]:
        extra = x.get("extra_info") or {}
        rm = x.get("reward_model") or {}
        return (extra.get("question_raw") or "", rm.get("ground_truth") or "")


@chz.chz
class Aime2024TestDatasetBuilder(RLDatasetBuilder):
    parquet_path: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    grader: Literal["sympy", "math_verify"] = "sympy"
    timeout: float = 1.0

    async def __call__(self) -> tuple[Aime2024TestDataset, None]:
        prefix = MathEnv.standard_fewshot_prefix() if self.convo_prefix == "standard" else self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        ds = Aime2024TestDataset(
            parquet_path=self.parquet_path,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=prefix,
            seed=self.seed,
            grader=self.grader,
            timeout=self.timeout,
            shuffle=False,
        )
        return ds, None


# -------- AIME2025 --------
class Aime2025TestDataset(_LocalParquetTestDatasetBase):
    DATASET_NAME = "aime2025"

    def _extract_problem_answer(self, x: dict[str, Any]) -> tuple[str, str]:
        extra = x.get("extra_info") or {}
        rm = x.get("reward_model") or {}
        return (extra.get("question_raw") or "", rm.get("ground_truth") or "")


@chz.chz
class Aime2025TestDatasetBuilder(RLDatasetBuilder):
    parquet_path: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    grader: Literal["sympy", "math_verify"] = "sympy"
    timeout: float = 1.0

    async def __call__(self) -> tuple[Aime2025TestDataset, None]:
        prefix = MathEnv.standard_fewshot_prefix() if self.convo_prefix == "standard" else self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        ds = Aime2025TestDataset(
            parquet_path=self.parquet_path,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=prefix,
            seed=self.seed,
            grader=self.grader,
            timeout=self.timeout,
            shuffle=False,
        )
        return ds, None


# -------- AMC2023 --------
class Amc2023TestDataset(_LocalParquetTestDatasetBase):
    DATASET_NAME = "amc2023"

    def _extract_problem_answer(self, x: dict[str, Any]) -> tuple[str, str]:
        extra = x.get("extra_info") or {}
        rm = x.get("reward_model") or {}
        return (extra.get("question_raw") or "", rm.get("ground_truth") or "")


@chz.chz
class Amc2023TestDatasetBuilder(RLDatasetBuilder):
    parquet_path: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    grader: Literal["sympy", "math_verify"] = "sympy"
    timeout: float = 1.0

    async def __call__(self) -> tuple[Amc2023TestDataset, None]:
        prefix = MathEnv.standard_fewshot_prefix() if self.convo_prefix == "standard" else self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        ds = Amc2023TestDataset(
            parquet_path=self.parquet_path,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=prefix,
            seed=self.seed,
            grader=self.grader,
            timeout=self.timeout,
            shuffle=False,
        )
        return ds, None


# Populate the dataset builder map after all classes are defined
DATASET_BUILDER_MAP = {
    "math": MathDatasetBuilder,
    "polaris": PolarisDatasetBuilder,
    "deepmath": DeepMathDatasetBuilder,
    "gsm8k": Gsm8kDatasetBuilder,
    "math500l35train": LocalMath500L35TrainDatasetBuilder,
    "dapoprocessedtrain": LocalDapoProcessedTrainDatasetBuilder,
    "combined": CombinedMathDatasetBuilder,
    # "olympiad_test": OlympiadTestDatasetBuilder,
    # "math500_test": Math500TestDatasetBuilder,
    # "minerva_test": MinervaTestDatasetBuilder,
    # "aime2024_test": Aime2024TestDatasetBuilder,
    # "aime2025_test": Aime2025TestDatasetBuilder,
    # "amc2023_test": Amc2023TestDatasetBuilder,
}


def get_math_dataset_builder(
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
) -> RLDatasetBuilder:
    """
    Unified function to get any math dataset builder.
    Args:
        dataset_name: One of "math", "polaris", "deepmath", or "gsm8k"
        batch_size: Number of groups per batch
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
        seed: Random seed for data shuffling (default: 0)
    Returns:
        The appropriate dataset builder instance
    """
    if dataset_name not in DATASET_BUILDER_MAP:
        raise ValueError(
            f"Unknown math dataset: {dataset_name}. Available: {list(DATASET_BUILDER_MAP.keys())}"
        )

    builder_class = DATASET_BUILDER_MAP[dataset_name]

    return builder_class(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        group_size=group_size,
        seed=seed,
    )
