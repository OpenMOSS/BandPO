import asyncio
import itertools
from collections import defaultdict
from typing import Dict, List

import numpy as np
import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, TrajectoryGroup
from tinker_cookbook.utils.misc_utils import all_same, dict_mean
from tinker_cookbook.utils import logtree
from tinker_cookbook.completers import TokenCompleter

def _traj_correct(traj) -> float:
    """
    Return correctness for one trajectory.
    For ProblemEnv it is stored in the *last* transition's metrics["correct"].
    Default to 0.0 if missing.
    """
    if not traj.transitions:
        return 0.0
    # scan from last transition backwards to be robust
    for t in reversed(traj.transitions):
        if t.metrics and "correct" in t.metrics:
            try:
                return float(t.metrics["correct"])
            except Exception:
                return 0.0
    return 0.0


def _traj_lengths_tokens(traj) -> tuple[int, int, int]:
    """
    (prompt_tokens, response_tokens, total_tokens) for one trajectory.
    - prompt_tokens: use the first transition's observation length (initial prompt).
    - response_tokens: sum of generated tokens across turns (safe for multi-turn).
    """
    if not traj.transitions:
        return (0, 0, 0)
    prompt_tokens = int(getattr(traj.transitions[0].ob, "length", 0))
    response_tokens = int(sum(len(t.ac.tokens) for t in traj.transitions))
    total_tokens = prompt_tokens + response_tokens
    return (prompt_tokens, response_tokens, total_tokens)


def _compute_validate_core_metrics(
    trajectory_groups_P: List[TrajectoryGroup],
    dataset_name: str,
) -> Dict[str, float]:
    """
    Compute validate_core/<dataset_name>/... metrics:
      - pass@k (k = group_size)
      - mean@k
      - avg_prompt_tokens / avg_response_tokens / avg_total_tokens
    Also emit validate_core/<dataset_name>/_raw/... additive stats for later global aggregation.
    """
    n_problems = len(trajectory_groups_P)
    if n_problems == 0:
        # edge case: empty evaluator (shouldn't happen, but keep it safe)
        return {
            f"validate_core/{dataset_name}/pass@0": 0.0,
            f"validate_core/{dataset_name}/mean@0": 0.0,
            f"validate_core/{dataset_name}/avg_prompt_tokens": 0.0,
            f"validate_core/{dataset_name}/avg_response_tokens": 0.0,
            f"validate_core/{dataset_name}/avg_total_tokens": 0.0,
            f"validate_core/{dataset_name}/_raw/num_problems": 0.0,
            f"validate_core/{dataset_name}/_raw/num_samples": 0.0,
            f"validate_core/{dataset_name}/_raw/k": 0.0,
            f"validate_core/{dataset_name}/_raw/passed_problems": 0.0,
            f"validate_core/{dataset_name}/_raw/correct_samples": 0.0,
            f"validate_core/{dataset_name}/_raw/sum_prompt_tokens": 0.0,
            f"validate_core/{dataset_name}/_raw/sum_response_tokens": 0.0,
            f"validate_core/{dataset_name}/_raw/sum_total_tokens": 0.0,
        }

    k_list = [len(tg.trajectories_G) for tg in trajectory_groups_P]
    k_all_same = all_same(k_list)
    k = int(k_list[0]) if k_all_same and k_list else -1

    passed_problems = 0
    correct_samples = 0

    sum_prompt_tokens = 0
    sum_response_tokens = 0
    sum_total_tokens = 0
    num_samples = 0

    for tg in trajectory_groups_P:
        # per-sample correctness/lengths
        group_corrects = []
        for traj in tg.trajectories_G:
            c = _traj_correct(traj)
            group_corrects.append(c)
            correct_samples += int(c >= 0.5)  # c is 0/1 in your env, but keep robust
            p, r, tot = _traj_lengths_tokens(traj)
            sum_prompt_tokens += p
            sum_response_tokens += r
            sum_total_tokens += tot
            num_samples += 1

        # per-problem pass@k: any correct in group
        if len(group_corrects) > 0 and max(group_corrects) >= 0.5:
            passed_problems += 1

    pass_val = passed_problems / n_problems if n_problems > 0 else 0.0
    mean_val = correct_samples / num_samples if num_samples > 0 else 0.0
    avg_prompt = sum_prompt_tokens / num_samples if num_samples > 0 else 0.0
    avg_resp = sum_response_tokens / num_samples if num_samples > 0 else 0.0
    avg_total = sum_total_tokens / num_samples if num_samples > 0 else 0.0

    # IMPORTANT: metric names show the actual k
    if k_all_same and k > 0:
        pass_key = f"validate_core/{dataset_name}/pass@{k}"
        mean_key = f"validate_core/{dataset_name}/mean@{k}"
    else:
        # in case some dataset accidentally uses varying group sizes
        pass_key = f"validate_core/{dataset_name}/pass@var"
        mean_key = f"validate_core/{dataset_name}/mean@var"

    out: Dict[str, float] = {
        pass_key: float(pass_val),
        mean_key: float(mean_val),
        f"validate_core/{dataset_name}/avg_prompt_tokens": float(avg_prompt),
        f"validate_core/{dataset_name}/avg_response_tokens": float(avg_resp),
        f"validate_core/{dataset_name}/avg_total_tokens": float(avg_total),
        # _raw additive stats for aggregation
        f"validate_core/{dataset_name}/_raw/num_problems": float(n_problems),
        f"validate_core/{dataset_name}/_raw/num_samples": float(num_samples),
        f"validate_core/{dataset_name}/_raw/passed_problems": float(passed_problems),
        f"validate_core/{dataset_name}/_raw/correct_samples": float(correct_samples),
        f"validate_core/{dataset_name}/_raw/sum_prompt_tokens": float(sum_prompt_tokens),
        f"validate_core/{dataset_name}/_raw/sum_response_tokens": float(sum_response_tokens),
        f"validate_core/{dataset_name}/_raw/sum_total_tokens": float(sum_total_tokens),
    }

    if k_all_same and k > 0:
        out[f"validate_core/{dataset_name}/_raw/k"] = float(k)
    else:
        # store k stats if varying
        out[f"validate_core/{dataset_name}/_raw/k"] = -1.0
        out[f"validate_core/{dataset_name}/_raw/k_min"] = float(min(k_list))
        out[f"validate_core/{dataset_name}/_raw/k_max"] = float(max(k_list))
        out[f"validate_core/{dataset_name}/_raw/k_avg"] = float(sum(k_list) / len(k_list))

    return out


def _compute_by_group_metrics(trajectory_groups_P: List[TrajectoryGroup], good_thresh: float = 0.5):
    n_groups = len(trajectory_groups_P)
    n_mixed = n_good = n_bad = 0
    for tg in trajectory_groups_P:
        grp_rewards = tg.get_total_rewards()
        if all_same(grp_rewards):
            if grp_rewards[0] >= good_thresh:
                n_good += 1
            else:
                n_bad += 1
        else:
            n_mixed += 1
    return {
        "by_group/frac_mixed": n_mixed / n_groups,
        "by_group/frac_all_good": n_good / n_groups,
        "by_group/frac_all_bad": n_bad / n_groups,
    }


def compute_trajectory_metrics(
    trajectory_groups_P: List[TrajectoryGroup], taglist_P: List[list[str]]
) -> Dict[str, float]:
    tag2trajgroups = defaultdict(list)
    for taglist, trajectory_group in zip(taglist_P, trajectory_groups_P):
        for tag in taglist:
            tag2trajgroups[tag].append(trajectory_group)
    out = {}
    have_nontrivial_tags = any(
        len(trajgroups) < len(trajectory_groups_P) for trajgroups in tag2trajgroups.values()
    )  # check if any tag gives us a strict subset of the full trajectory groups
    if have_nontrivial_tags:
        for tag, trajectory_groups in tag2trajgroups.items():
            prefixed_metrics = {
                f"env/{tag}/{k}": v
                for k, v in _compute_trajectory_metrics(trajectory_groups).items()
            }
            out.update(prefixed_metrics)
    out.update(
        {f"env/all/{k}": v for k, v in _compute_trajectory_metrics(trajectory_groups_P).items()}
    )
    return out


def _compute_trajectory_metrics(trajectory_groups_P: List[TrajectoryGroup]) -> Dict[str, float]:
    """Compute metrics for the trajectory groups."""
    flat_trajs_PG = [traj for tg in trajectory_groups_P for traj in tg.trajectories_G]
    ac_tokens_by_turn = [
        len(transition.ac.tokens) for traj in flat_trajs_PG for transition in traj.transitions
    ]
    ob_tokens_by_turn = [
        transition.ob.length for traj in flat_trajs_PG for transition in traj.transitions
    ]
    turns_by_trajectory = [len(traj.transitions) for traj in flat_trajs_PG]
    # Compute metrics
    metrics = {
        "ac_tokens_per_turn": sum(ac_tokens_by_turn) / sum(turns_by_trajectory),
        "ob_tokens_per_turn": sum(ob_tokens_by_turn) / sum(turns_by_trajectory),
        "turns_per_episode": sum(turns_by_trajectory) / len(flat_trajs_PG),
        "total_episodes": len(flat_trajs_PG),
        "total_turns": sum(turns_by_trajectory),
        "total_ac_tokens": sum(ac_tokens_by_turn),
        "total_ob_tokens": sum(ob_tokens_by_turn),
    }
    metrics["reward/total"] = np.mean(
        [reward for tg in trajectory_groups_P for reward in tg.get_total_rewards()]
    ).item()
    # Per-transition metrics
    transition_metrics = [
        transition.metrics
        for tg in trajectory_groups_P
        for traj in tg.trajectories_G
        for transition in traj.transitions
    ]
    traj_metrics = [metrics for tg in trajectory_groups_P for metrics in tg.metrics_G]
    metrics.update(dict_mean(transition_metrics + traj_metrics))
    # combine traj_metrics and transition_metrics in case there's some key
    # (like format error) that appears in the per-step metrics for some envs
    # but the compute_group_rewards metric for other envs.
    metrics.update(_compute_by_group_metrics(trajectory_groups_P))
    return metrics


def dataset_to_env_group_builders(dataset: RLDataset) -> list[EnvGroupBuilder]:
    """
    Get the whole dataset as a list of env group builders.
    """
    return list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))


class RLTestSetEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        dataset: RLDataset,
        max_tokens: int,
        name: str = "test",
        num_groups_to_log: int = 4,
    ):
        self.env_group_builders_P = dataset_to_env_group_builders(dataset)
        self.max_tokens = max_tokens
        self.name = name
        self.num_groups_to_log = num_groups_to_log

    async def eval_token_completer(self, policy: TokenCompleter) -> dict[str, float]:
        async def run_group_rollout(builder, i):
            enable_logging = i < self.num_groups_to_log
            with logtree.optional_enable_logging(enable=enable_logging):
                return await do_group_rollout(builder, policy)

        trajectory_groups_P = await asyncio.gather(
            *[run_group_rollout(builder, i) for i, builder in enumerate(self.env_group_builders_P)]
        )
        taglist_P = [builder.logging_tags() for builder in self.env_group_builders_P]
        traj_metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)
        traj_metrics = {f"{self.name}/{k}": v for k, v in traj_metrics.items()}

        core_metrics = _compute_validate_core_metrics(trajectory_groups_P, dataset_name=self.name)

        # return both: existing rich metrics + compact validate_core metrics
        out = {}
        out.update(traj_metrics)
        out.update(core_metrics)
        return out



    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)
        return await self.eval_token_completer(policy)
