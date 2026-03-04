# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

import json, time
from pathlib import Path
import requests  # 用于 HTTP 调用 filter server
from typing import Optional

class OnlineRamblingFilterClient:
    """
    简单 HTTP 客户端：
    - base_url 形如 "http://filter-host:8000"
    - filter_jsonl(jsonl_path, cls, debug) 返回 np.ndarray[bool] 的 keep_mask；
      出现错误时返回 None（终止继续，便于找到错误，不让问题隐藏）。
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 600.0,
        project_name: Optional[str] = None,
        exp_name: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        # 如果只传了一个，直接报错，避免目录结构混乱
        if bool(project_name) ^ bool(exp_name):
            raise ValueError(
                "OnlineRamblingFilterClient: project_name 和 exp_name 必须同时提供或同时为空。"
            )
        self.project_name = project_name
        self.exp_name = exp_name

    def filter_jsonl(
        self,
        jsonl_path: str,
        project_name: Optional[str] = None,
        exp_name: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        发送一个只包含 jsonl_path 的请求。
        哪个 cls、kept/dropped 被选入“子集”，由 server 启动参数 --kept-or-dropped / --cls 决定。

        project_name / exp_name 如果不显式传入，则使用构造函数里配置的默认值。
        """
        # 允许 per-call 覆盖 project/exp；为空则回退到实例默认值
        if project_name is None:
            project_name = self.project_name
        if exp_name is None:
            exp_name = self.exp_name

        if bool(project_name) ^ bool(exp_name):
            raise ValueError(
                "OnlineRamblingFilterClient.filter_jsonl: project_name 和 exp_name 必须同时提供或同时为空。"
            )

        payload = {
            "jsonl_path": jsonl_path,
        }
        if project_name and exp_name:
            payload["project_name"] = project_name
            payload["exp_name"] = exp_name

        try:
            resp = requests.post(
                self.base_url + "/filter",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[WARN] Online filter request failed for jsonl={jsonl_path}: {e}")
            return None

        if not data.get("ok", False):
            print(f"[WARN] Online filter service returned error: {data}")
            return None

        mask = data.get("keep_mask", None)
        if mask is None:
            print("[WARN] Online filter response missing keep_mask")
            return None

        try:
            arr = np.array(mask, dtype=bool)
        except Exception as e:
            print(f"[WARN] Online filter keep_mask cast failed: {e}")
            return None

        return arr

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    # ====== Rollout record helpers (driver-side JSONL, single-writer) ======

    @staticmethod
    def _to_py(obj):
        import numpy as np
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.ndim == 0:
                return obj.item()
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [RayDAPOTrainer._to_py(x) for x in obj]
        if isinstance(obj, dict):
            return {k: RayDAPOTrainer._to_py(v) for k, v in obj.items()}
        return obj

    # === PATCH 1: 新增严格版构造函数（紧跟 _to_py 后面） ===
    @staticmethod
    def _build_per_sample_reward_extra_strict(
        reward_extra_info: dict,
        *,
        B: int,
        prompt_uids: list | None,
        repeated_uids: list | None,
        n_repeat: int,
        strict: bool = True,
    ):
        """
        将 reward_extra_info（dict[str, Any]）转换为长度 B 的 List[dict]（每样本一个 dict）。
        规则：
        - 标量：广播到 B
        - 长度 == B：直接按样本使用
        - 长度 == P（按 prompt 返回）：用 prompt 顺序把每个值重复 n_repeat 次扩展到 B
        - 其它：strict=True 抛错（推荐）；否则做截断/填 None 的容错
        """
        import numpy as np

        per_sample = [dict() for _ in range(B)]
        if not reward_extra_info:
            return per_sample

        # 推断 P（prompt 个数）
        P = None
        if prompt_uids is not None:
            P = len(dict.fromkeys(prompt_uids))  # 保序去重

        def as_list(val):
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (list, tuple)):
                return list(val)
            return None  # 非序列，按标量处理

        for k, v in reward_extra_info.items():
            lst = as_list(v)

            # 1) 标量：广播
            if lst is None:
                for i in range(B):
                    per_sample[i][k] = v
                continue

            # 2) 长度 == B：完美对齐
            if len(lst) == B:
                for i in range(B):
                    per_sample[i][k] = lst[i]
                continue

            # 3) 长度 == P：按 prompt 扩展到 B
            if P is not None and len(lst) == P:
                if repeated_uids is None or n_repeat <= 0:
                    if strict:
                        raise ValueError(f"Key {k}: len=P but missing uids/n_repeat for expansion.")
                    else:
                        # 容错：循环广播
                        for i in range(B):
                            per_sample[i][k] = lst[i % len(lst)]
                        continue

                # 用 prompt 的出现顺序稳定化（与 batch 次序一致）
                ordered_prompts = []
                seen = set()
                for uid in prompt_uids:
                    if uid not in seen:
                        ordered_prompts.append(uid)
                        seen.add(uid)
                if len(ordered_prompts) != P and strict:
                    raise ValueError("prompt_uids unique count mismatch.")

                expanded = []
                for val in lst:
                    expanded.extend([val] * n_repeat)

                if len(expanded) != B:
                    if strict:
                        raise ValueError(f"Key {k}: expanded len {len(expanded)} != B {B}.")
                    else:
                        # 容错兜底
                        expanded = (expanded * ((B + len(expanded) - 1)//len(expanded)))[:B]

                for i in range(B):
                    per_sample[i][k] = expanded[i]
                continue

            # 4) 其它异常长度
            if strict:
                raise ValueError(f"Key {k}: unexpected length {len(lst)} (B={B}, P={P}).")
            else:
                # 容错：截断/填 None
                if len(lst) < B:
                    lst = lst + [None] * (B - len(lst))
                else:
                    lst = lst[:B]
                for i in range(B):
                    per_sample[i][k] = lst[i]

        return per_sample

    def _compute_ce_ref_seq_mean_for_filter(self, batch: DataProto) -> Optional[np.ndarray]:
        """
        仅用于 online filter 的 ce_ref_mean_too_large pattern：
        在一个临时拷贝上计算 response_mask 与 ref_log_prob，
        得到 sequence-level ce_ref_seq_mean = -mean(log p_ref(y_t|...))。

        注意：不修改传入的 batch，也不复用训练阶段那一套 log_prob。
        """
        if not self.use_reference_policy:
            return None

        tmp = deepcopy(batch)
        tmp.batch["response_mask"] = compute_response_mask(tmp)

        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(tmp)
        tmp = tmp.union(ref_log_prob)

        ref_lp = tmp.batch["ref_log_prob"]        # [B, L]
        resp_mask = tmp.batch["response_mask"]    # [B, L]

        with torch.no_grad():
            x = ref_lp.float()
            m = resp_mask.float()
            s = (x * m).sum(dim=-1)               # [B]
            cnt = m.sum(dim=-1).clamp(min=1.0)    # [B]
            mean_logp = s / cnt                   # [B]
            ce_ref = (-mean_logp).detach().cpu().numpy()

        return ce_ref

    def _run_online_rambling_filter(
        self,
        *,
        new_batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: dict | None,
        epoch: int,
    ) -> None:
        """
        - new_batch: reward_fn 之后、DAPO filter_groups 之前的 batch（已包含 repeated 响应）。
        - reward_tensor: 与 new_batch.batch['token_level_scores'] 指向同一 Tensor（按引用传入）。
        - reward_extra_infos_dict: reward_fn 返回的 reward_extra_info dict（list 形式）。
        """
        if self._rambling_filter_client is None:
            return
        if not reward_extra_infos_dict:
            return

        per_sample_extras = new_batch.non_tensor_batch.get("reward_extra_infos_dict", None)
        if per_sample_extras is None:
            # 正常情况下 PATCH 2 已经构造完 per-sample dict，这里只是兜底
            return

        B = len(new_batch.batch["attention_mask"])
        if B == 0:
            return

        # 只在 use_reference_policy=True 且配置允许时计算 ce_ref
        ce_ref_seq_mean: Optional[np.ndarray] = None
        use_ce_ref = bool(self.config.trainer.get("rambling_filter_use_ce_ref", True))
        if use_ce_ref and self.use_reference_policy:
            try:
                ce_ref_seq_mean = self._compute_ce_ref_seq_mean_for_filter(new_batch)
                if ce_ref_seq_mean is not None and len(ce_ref_seq_mean) != B:
                    print(
                        f"[WARN] ce_ref_seq_mean len={len(ce_ref_seq_mean)} "
                        f"!= batch size {B}, ignore ce_ref."
                    )
                    ce_ref_seq_mean = None
            except Exception as e:
                print(f"[WARN] compute ce_ref_seq_mean for filter failed: {e}")
                ce_ref_seq_mean = None

        # 解码 prompt/response 文本
        prompts = new_batch.batch.get("prompts", None)
        responses = new_batch.batch.get("responses", None)
        prompt_texts = self._decode_texts(prompts) if prompts is not None else [None] * B
        response_texts = self._decode_texts(responses) if responses is not None else [None] * B

        uids = new_batch.non_tensor_batch.get("uid", [None] * B)

        # 写临时 JSONL 文件
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"rambling_filter_step{self.global_steps:06d}_epoch{epoch}_ts{ts}.jsonl"
        jsonl_path = os.path.join(self._rambling_filter_tmp_dir, fname)
        os.makedirs(self._rambling_filter_tmp_dir, exist_ok=True)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i in range(B):
                rec = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "epoch": int(epoch),
                    "global_step": int(self.global_steps),
                    "uid": None if uids is None else str(uids[i]),
                    "filter_state": "kept",
                    "prompt_text": prompt_texts[i],
                    "response_text": response_texts[i],
                    # 方便 offline/online 对齐: reward_seq_sum
                    "reward_seq_sum": float(reward_tensor[i].sum().item()),
                    "reward_extra_infos": self._to_py(per_sample_extras[i]),
                }
                if ce_ref_seq_mean is not None:
                    rec["ce_ref_seq_mean"] = float(ce_ref_seq_mean[i])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # keep_mask = self._rambling_filter_client.filter_jsonl(
        #     jsonl_path=jsonl_path,
        # )

        # if keep_mask is None:
        #     return
        # if len(keep_mask) != B:
        #     print(
        #         f"[WARN] Online filter keep_mask len={len(keep_mask)} != batch size {B}, skip."
        #     )
        #     return
        # 调用 online filter server；具体在哪个 cls / kept_or_dropped 上做子集，
        # 由 server 启动参数 --kept-or-dropped / --cls 决定
        keep_mask = self._rambling_filter_client.filter_jsonl(
            jsonl_path=jsonl_path,
        )

        # === 强制严格：一旦 online filter 配置了，就要求返回必须正常 ===
        if keep_mask is None:
            msg = (
                f"[ERROR] Online rambling filter returned None "
                f"(url={getattr(self._rambling_filter_client, 'base_url', '?')}, "
                f"jsonl={jsonl_path}). Abort training for debugging."
            )
            print(msg, flush=True)
            # 不删除 jsonl，留下现场
            raise RuntimeError(msg)

        if len(keep_mask) != B:
            msg = (
                f"[ERROR] Online filter keep_mask len={len(keep_mask)} != batch size {B} "
                f"(jsonl={jsonl_path}). Abort training for debugging."
            )
            print(msg, flush=True)
            # 同样保留 jsonl
            raise RuntimeError(msg)

        # === 正常情况：成功拿到与 batch 对齐的 keep_mask，如果不是 debug 模式就把临时 JSONL 删掉 ===
        if not getattr(self, "_rambling_filter_debug", False):
            try:
                os.remove(jsonl_path)
            except OSError as e:
                # 删除失败不是致命错误，只打个 warning
                print(f"[WARN] Failed to remove temporary filter JSONL '{jsonl_path}': {e}", flush=True)

        bad_idx = np.where(~keep_mask)[0]
        if len(bad_idx) == 0:
            return

        print(
            f"[INFO] Online rambling filter: bad={len(bad_idx)}/{B} "
            f"({len(bad_idx)/B:.2%}) at global_step={self.global_steps}."
        )

        # 修改 reward_tensor & per-sample reward_extra_infos_dict：
        #   - 把这些样本视为“被降权的正确样本”：score: +1 -> 0, acc: True -> False
        #   - 同时保留原始 score/acc 方便 debug。
        for i in bad_idx:
            extras = per_sample_extras[i]
            if not isinstance(extras, dict):
                continue
            old_score = extras.get("score", None)
            old_acc = extras.get("acc", None)
            extras["rambling_filtered"] = True
            extras["score_before_rambling_filter"] = old_score
            extras["acc_before_rambling_filter"] = old_acc

            # 这里选择“中性化”：score=0, acc=False，不惩罚也不奖励
            extras["score"] = 0.0
            extras["acc"] = False

            # 简单做法：把该 trajectory 所有 token 的 reward 清零
            reward_tensor[i, :] = 0.0

    # === PATCH 1: 将标准答案等字段展开为“每样本(B)”并挂回 non_tensor_batch ===
    def _attach_ground_truth_fields(self, batch: DataProto) -> None:
        """
        读取 batch.non_tensor_batch 里可能存在的 ground truth 相关信息，并展开到每条轨迹（B）。
        支持：
          - reward_model: List[dict]（含 ground_truth/style），可能是按 P（prompt）或按 B 返回
          - ground_truth: List[str] / 标量 / 按 P / 按 B
          - data_source / ability: 同上（可选）
        最终写回：
          - non_tensor_batch["ground_truth"]           -> np.array(len=B, dtype=object)
          - non_tensor_batch["ground_truth_style"]     -> np.array(len=B, dtype=object)（若有）
          - non_tensor_batch["data_source"]            -> np.array(len=B, dtype=object)（若有）
          - non_tensor_batch["ability"]                -> np.array(len=B, dtype=object)（若有）
        """
        import numpy as np

        B = len(batch.batch["attention_mask"])
        n_repeat = int(self.config.actor_rollout_ref.rollout.n)
        uids = list(map(str, batch.non_tensor_batch.get("uid", [None] * B)))

        # 估算 P（prompt 个数）：B = P * n
        P = None
        if n_repeat > 0 and B % n_repeat == 0:
            P = B // n_repeat

        def _to_list(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (list, tuple)):
                return list(x)
            return x  # 标量或其他

        def _expand_to_B(seq_or_scalar):
            # 标量 -> 广播
            if not isinstance(seq_or_scalar, (list, tuple)):
                return [seq_or_scalar] * B
            seq = list(seq_or_scalar)
            if len(seq) == B:
                return seq
            if P is not None and len(seq) == P and n_repeat > 0:
                # 按 prompt 值，重复 n 次，展成 B
                expanded = []
                for v in seq:
                    expanded.extend([v] * n_repeat)
                if len(expanded) == B:
                    return expanded
            # 严格：长度异常，直接报错（你也可改成容错填充）
            raise ValueError(f"Unexpected length {len(seq)} (expect B={B} or P={P}).")

        n2b = batch.non_tensor_batch

        # -------- 1) 先尝试从 reward_model 抽 ground_truth/style --------
        ground_truth_B = None
        ground_style_B = None
        rm = n2b.get("reward_model", None)
        if rm is not None:
            rm_list = _to_list(rm)
            # rm_list 可能是长度 P 或 B；每个元素通常是 dict（含 ground_truth/style）
            if isinstance(rm_list, list) and len(rm_list) in {B, P}:
                # 先按“当前长度”抽出 list，再展开成 B（若当前为 P）
                def _extract_from_rm_list(key):
                    vals = []
                    for item in rm_list:
                        if isinstance(item, dict):
                            vals.append(item.get(key))
                        else:
                            vals.append(None)
                    return _expand_to_B(vals)

                try:
                    ground_truth_B = _extract_from_rm_list("ground_truth")
                except Exception:
                    ground_truth_B = None
                try:
                    ground_style_B = _extract_from_rm_list("style")
                except Exception:
                    ground_style_B = None

        # -------- 2) 如果没有从 reward_model 抽到，再看顶层 ground_truth --------
        if ground_truth_B is None and "ground_truth" in n2b:
            gt_any = _to_list(n2b["ground_truth"])
            try:
                ground_truth_B = _expand_to_B(gt_any)
            except Exception:
                ground_truth_B = None  # 保持 None，不强行容错

        # -------- 3) 可选字段：data_source / ability（如果存在就展开） --------
        data_source_B = None
        if "data_source" in n2b:
            ds_any = _to_list(n2b["data_source"])
            try:
                data_source_B = _expand_to_B(ds_any)
            except Exception:
                data_source_B = None

        ability_B = None
        if "ability" in n2b:
            ab_any = _to_list(n2b["ability"])
            try:
                ability_B = _expand_to_B(ab_any)
            except Exception:
                ability_B = None

        # -------- 4) 写回 non_tensor_batch（保持 B 对齐） --------
        if ground_truth_B is not None:
            n2b["ground_truth"] = np.array(ground_truth_B, dtype=object)
        if ground_style_B is not None:
            n2b["ground_truth_style"] = np.array(ground_style_B, dtype=object)
        if data_source_B is not None:
            n2b["data_source"] = np.array(data_source_B, dtype=object)
        if ability_B is not None:
            n2b["ability"] = np.array(ability_B, dtype=object)

    def _rr_open_if_needed(self):
        """Open JSONL file lazily if trainer.rollout_record_path is set."""
        if not getattr(self, "_rr_path", ""):
            return
        if getattr(self, "_rr_fh", None) is None:
            os.makedirs(os.path.dirname(self._rr_path), exist_ok=True)
            self._rr_fh = open(self._rr_path, "a", encoding="utf-8")
            self._rr_buf = []

    def _rr_flush(self, force=False):
        if getattr(self, "_rr_fh", None) is None:
            return
        if self._rr_buf and (force or len(self._rr_buf) >= 64):
            self._rr_fh.write("".join(self._rr_buf))
            self._rr_fh.flush()
            self._rr_buf.clear()

    def _rr_close(self):
        if getattr(self, "_rr_fh", None) is not None:
            self._rr_flush(force=True)
            try:
                self._rr_fh.close()
            finally:
                self._rr_fh = None

    def _decode_texts(self, token_ids, skip_special_tokens=True):
        # token_ids: Tensor[B, L]
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _make_rr_lines_from_batch(
        self,
        batch,
        *,
        filter_state: str,
        filter_reason: str | None,
        epoch: int,
        global_step: int,
        actor_metrics: dict | None = None,
    ):
        """
        Build JSONL lines (strings) for each sample in a DataProto batch.
        Requires batch has: attention_mask, response_mask (我们会确保 dropped 也补上)
        """
        import torch

        if batch is None:
            return []

        B = len(batch.batch["attention_mask"])
        if B == 0:
            return []

        prompts   = batch.batch.get("prompts", None)
        responses = batch.batch.get("responses", None)
        attn      = batch.batch["attention_mask"]          # [B, L]
        resp_mask = batch.batch["response_mask"]           # [B, L]
        # === PATCH 3a: 取每样本字典列表 ===
        reward_extra_list = batch.non_tensor_batch.get("reward_extra_infos_dict", None)

        old_log_probs     = batch.batch.get("old_log_probs", None)
        rollout_log_probs = batch.batch.get("rollout_log_probs", None)
        ref_log_prob      = batch.batch.get("ref_log_prob", None)

        token_scores  = batch.batch.get("token_level_scores", None)
        token_rewards = batch.batch.get("token_level_rewards", None)
        advantages    = batch.batch.get("advantages", None)
        values        = batch.batch.get("values", None)

        # lengths
        total_len  = torch.sum(attn, dim=-1)           # [B]
        resp_len   = torch.sum(resp_mask, dim=-1)      # [B]
        prompt_len = total_len - resp_len              # [B]

        # texts (can be large; you control truncation if needed)
        prompt_texts   = self._decode_texts(prompts)   if prompts   is not None else [None]*B
        response_texts = self._decode_texts(responses) if responses is not None else [None]*B

        def masked_mean_sum(x):
            if x is None:
                return None, None
            x = x.float()
            m = resp_mask.float()
            s = torch.sum(x * m, dim=-1)                  # [B]
            cnt = torch.clamp(m.sum(dim=-1), min=1.0)     # [B]
            mean = s / cnt
            return mean, s

        old_lp_mean, old_lp_sum   = masked_mean_sum(old_log_probs)
        roll_lp_mean, roll_lp_sum = masked_mean_sum(rollout_log_probs)
        ref_lp_mean,  ref_lp_sum  = masked_mean_sum(ref_log_prob)
        rew_mean,      rew_sum    = masked_mean_sum(token_rewards)
        adv_mean,      _          = masked_mean_sum(advantages)
        val_mean,      _          = masked_mean_sum(values)
        # === 额外：KL（用 log-ratio 近似，每 token: logpi - logpref）===
        kl_token_log_ratio = None
        if (old_log_probs is not None) and (ref_log_prob is not None):
            kl_token_log_ratio = (old_log_probs - ref_log_prob)  # [B, L]
        kl_mean, kl_sum = masked_mean_sum(kl_token_log_ratio)
        # === 额外：Entropy（token 级），如果可用 ===
        token_entropys = batch.batch.get("entropys_tok", None)
        ent_mean, ent_sum = masked_mean_sum(token_entropys)

        # pull some actor metrics if available (entropy/kl etc.)
        kl_value_mean = None
        entropy_mean  = None
        if actor_metrics:
            kl_value_mean = actor_metrics.get("actor/kl_value_mean", None)
            entropy_mean  = actor_metrics.get("actor/entropy", actor_metrics.get("actor/entropy_mean", None))

        rollout_cfg = self.config.actor_rollout_ref.rollout
        gen_cfg = dict(
            temperature = rollout_cfg.get("temperature", None),
            top_p       = rollout_cfg.get("top_p", None),
            top_k       = rollout_cfg.get("top_k", None),
            max_new_tokens = self.config.data.get("max_response_length", None),
            n           = rollout_cfg.get("n", None),
            do_sample   = rollout_cfg.get("val_kwargs", {}).get("do_sample", None) if "val_kwargs" in rollout_cfg else None,
        )

        uids = batch.non_tensor_batch.get("uid", [None]*B)
        # === PATCH 3a: 取每样本的 ground truth / meta（均为长度 B 或 None） ===
        gt_list        = batch.non_tensor_batch.get("ground_truth", None)
        gt_style_list  = batch.non_tensor_batch.get("ground_truth_style", None)
        ds_list        = batch.non_tensor_batch.get("data_source", None)
        ability_list   = batch.non_tensor_batch.get("ability", None)

        lines = []
        for i in range(B):
            rec = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "uid": None if uids is None else str(uids[i]),
                "filter_state": filter_state,
                "filter_reason": filter_reason,

                "prompt_text":   prompt_texts[i],
                "response_text": response_texts[i],

                "prompt_len_tokens":  int(prompt_len[i].item()),
                "response_len_tokens": int(resp_len[i].item()),
                "total_len_tokens":    int(total_len[i].item()),

                "gen_cfg": gen_cfg,

                "old_logprob_mean":      None if old_lp_mean is None else float(old_lp_mean[i].item()),
                "old_logprob_sum":       None if old_lp_sum  is None else float(old_lp_sum[i].item()),
                "rollout_logprob_mean":  None if roll_lp_mean is None else float(roll_lp_mean[i].item()),
                "rollout_logprob_sum":   None if roll_lp_sum  is None else float(roll_lp_sum[i].item()),
                "ref_logprob_mean":      None if ref_lp_mean  is None else float(ref_lp_mean[i].item()),
                "ref_logprob_sum":       None if ref_lp_sum   is None else float(ref_lp_sum[i].item()),

                "reward_seq_mean":       None if rew_mean is None else float(rew_mean[i].item()),
                "reward_seq_sum":        None if rew_sum  is None else float(rew_sum[i].item()),
                "adv_seq_mean":          None if adv_mean is None else float(adv_mean[i].item()),
                "value_seq_mean":        None if val_mean is None else float(val_mean[i].item()),

                "kl_value_mean":         None if kl_value_mean is None else float(kl_value_mean),
                "entropy_mean":          None if entropy_mean  is None else float(entropy_mean),
                # KL 的序列级统计（基于 log-ratio 的蒙特卡洛近似）
                "kl_seq_mean_log_ratio": None if kl_mean is None else float(kl_mean[i].item()),
                "kl_seq_sum_log_ratio":  None if kl_sum  is None else float(kl_sum[i].item()),
                # 熵的序列级统计（可用于 sanity check）
                "entropy_seq_mean":      None if ent_mean is None else float(ent_mean[i].item()),
                "entropy_seq_sum":       None if ent_sum  is None else float(ent_sum[i].item()),
            }
            # === PATCH 3b: 写入每样本 reward extra 字典 ===
            if reward_extra_list is not None:
                rec["reward_extra_infos"] = self._to_py(reward_extra_list[i])
            # === PATCH 3b: 把标准答案及数据源等写入 JSON 行 ===
            if gt_list is not None:
                rec["ground_truth"] = self._to_py(gt_list[i])
            if gt_style_list is not None:
                rec["ground_truth_style"] = self._to_py(gt_style_list[i])
            if ds_list is not None:
                rec["data_source"] = self._to_py(ds_list[i])
            if ability_list is not None:
                rec["ability"] = self._to_py(ability_list[i])
            if getattr(self, "_rr_detail", False):
                rec.update({
                    "token_scores":      None if token_scores   is None else self._to_py(token_scores[i]),
                    "token_rewards":     None if token_rewards  is None else self._to_py(token_rewards[i]),
                    "token_advantage":   None if advantages     is None else self._to_py(advantages[i]),
                    "token_values":      None if values         is None else self._to_py(values[i]),
                    "old_log_probs":     None if old_log_probs  is None else self._to_py(old_log_probs[i]),
                    "ref_log_prob":      None if ref_log_prob   is None else self._to_py(ref_log_prob[i]),
                    "rollout_log_probs": None if rollout_log_probs is None else self._to_py(rollout_log_probs[i]),
                    # KL 的 token 级明细（可选，文件会很大）
                    "kl_token_log_ratio": None if kl_token_log_ratio is None else self._to_py(kl_token_log_ratio[i]),
                    # token 级 entropy 明细（可选）
                    "token_entropy":      None if token_entropys is None else self._to_py(token_entropys[i]),
                })

            lines.append(json.dumps(rec, ensure_ascii=False) + "\n")

        return lines

    def _write_rollout_records(
        self,
        kept_batch,
        *,
        dropped_batch=None,
        epoch: int,
        global_step: int,
        actor_output_metrics: dict | None = None,
    ):
        """Append records for kept and dropped samples (if any)."""
        if not getattr(self, "_rr_path", ""):
            return
        self._rr_open_if_needed()

        kept_lines = self._make_rr_lines_from_batch(
            kept_batch,
            filter_state="kept",
            filter_reason=None,
            epoch=epoch,
            global_step=global_step,
            actor_metrics=actor_output_metrics or {},
        )
        self._rr_buf.extend(kept_lines)

        if dropped_batch is not None:
            drop_lines = self._make_rr_lines_from_batch(
                dropped_batch,
                filter_state="dropped",
                filter_reason="filtered_by_trainer",
                epoch=epoch,
                global_step=global_step,
                actor_metrics=actor_output_metrics or {},
            )
            self._rr_buf.extend(drop_lines)

        self._rr_flush(force=False)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # === online rambling filter client（可选） ===
        rambling_filter_url = (self.config.trainer.get("rambling_filter_url", "") or "").strip()
        self._rambling_filter_client = None
        self._rambling_filter_tmp_dir = str(self.config.trainer.get("rambling_filter_tmp", "") or "")
        # 新增：是否保留 JSONL 调试文件
        self._rambling_filter_debug = bool(self.config.trainer.get("rambling_filter_debug", False))
        if rambling_filter_url:
            if not self._rambling_filter_tmp_dir:
                raise ValueError(
                    "trainer.rambling_filter_url is non-empty, but trainer.rambling_filter_tmp is not set. "
                    "Please configure trainer.rambling_filter_tmp to a writable directory."
                )
            os.makedirs(self._rambling_filter_tmp_dir, exist_ok=True)
            self._rambling_filter_client = OnlineRamblingFilterClient(
                base_url=rambling_filter_url,
                timeout=600.0,
                project_name=str(self.config.trainer.project_name),
                exp_name=str(self.config.trainer.experiment_name),
            )
            print(
                f"[INFO] Online rambling filter enabled: url={rambling_filter_url}, "
                f"tmp_dir={self._rambling_filter_tmp_dir}, debug={self._rambling_filter_debug}, "
                f"project={self.config.trainer.project_name}, exp={self.config.trainer.experiment_name}"
            )
        else:
            print(f"[INFO] Online rambling filter disabled (trainer.rambling_filter_url is empty).")

        self.global_steps = 0
        self.gen_steps = 0

        # === rollout record config ===
        self._rr_path = str(self.config.trainer.get("rollout_record_path", "") or "")
        self._rr_detail = bool(self.config.trainer.get("rollout_record_detail", False))
        self._rr_fh = None
        self._rr_buf = []

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._rr_close()
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        dropped_batches_accum = []  # 收集当前 step 全部被过滤的子批次
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                actor_output_metrics = None  # NEW：即便 warmup 不更新 actor，也能安全写 JSON
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    # === PATCH 2: 将标准答案/数据源等字段展开并挂到每个样本 ===
                    try:
                        self._attach_ground_truth_fields(new_batch)
                    except Exception as e:
                        print(f"[WARN] _attach_ground_truth_fields failed: {e}")
                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # === PATCH 2: 构造“每样本字典”，严格模式 ===
                        if reward_extra_infos_dict:
                            B = len(new_batch.batch["attention_mask"])
                            # 当前 batch 的 uid 已在 repeat 之后，长度为 B
                            repeated_uids = list(map(str, new_batch.non_tensor_batch["uid"]))
                            n_repeat = int(self.config.actor_rollout_ref.rollout.n)
                            prompt_uids = repeated_uids[::n_repeat] if n_repeat > 0 else None

                            try:
                                per_sample_extras = self._build_per_sample_reward_extra_strict(
                                    reward_extra_infos_dict,
                                    B=B,
                                    prompt_uids=prompt_uids,
                                    repeated_uids=repeated_uids,
                                    n_repeat=n_repeat,
                                    strict=True,  # 建议严格：异常长度直接报错，便于及早发现上游问题
                                )
                            except Exception as ex:
                                # 如果希望不中断训练，可降级为 strict=False：
                                # per_sample_extras = self._build_per_sample_reward_extra_strict(
                                #     reward_extra_infos_dict, B=B, prompt_uids=prompt_uids,
                                #     repeated_uids=repeated_uids, n_repeat=n_repeat, strict=False
                                # )
                                raise
                            new_batch.non_tensor_batch["reward_extra_infos_dict"] = np.array(per_sample_extras, dtype=object)
                            new_batch.meta_info["reward_extra_keys"] = list(reward_extra_infos_dict.keys())

                        # === Online rambling filter（在 score/acc 已经展开为 per-sample dict 之后） ===
                        if self._rambling_filter_client is not None and reward_extra_infos_dict:
                            try:
                                self._run_online_rambling_filter(
                                    new_batch=new_batch,
                                    reward_tensor=new_batch.batch["token_level_scores"],
                                    reward_extra_infos_dict=reward_extra_infos_dict,
                                    epoch=epoch,
                                )
                            except Exception as e:
                                # print(f"[WARN] online rambling filter failed at step {self.global_steps}: {e}")
                                # 打印清晰的错误，再抛出去终止训练
                                print(
                                    f"[ERROR] online rambling filter failed at step {self.global_steps}: {e}",
                                    flush=True,
                                )
                                raise
                            
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        # === [NEW] 过滤前快照，用于提取被丢弃的样本 ===
                        candidate_batch_before_filter = new_batch

                        # === [NEW] 计算 dropped 索引并累加子批 ===
                        all_idxs = list(range(len(candidate_batch_before_filter.batch["attention_mask"])))
                        kept_set = set(kept_traj_idxs)
                        dropped_traj_idxs = [i for i in all_idxs if i not in kept_set]
                        if len(dropped_traj_idxs) > 0:
                            dropped_sub = candidate_batch_before_filter[dropped_traj_idxs]
                            # 为 dropped 子批补上 response_mask，便于长度/统计
                            try:
                                dropped_sub.batch["response_mask"] = compute_response_mask(dropped_sub)
                            except Exception as _:
                                # 如果失败，忽略长度的更精细统计，后续字段会为 None 或按 attention_mask 兜底
                                pass
                            dropped_batches_accum.append(dropped_sub)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys_tok = old_log_prob.batch["entropys"].detach()           # <-- 新增：备份 token 级熵
                        # entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        # entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        entropy_agg = agg_loss(loss_mat=entropys_tok, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        batch.batch["entropys_tok"] = entropys_tok              # <-- 新增：把熵写回 batch，供 JSON 使用

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        # DataProto: ray_trainer->fsdp_workers.update_actor->dp_actor.update_policy
                        batch.meta_info["global_steps"]  = int(self.global_steps)
                        batch.meta_info["total_training_steps"] = int(self.total_training_steps)
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # === [NEW] 统一写 kept + dropped 样本的 JSONL 记录 ===
                    try:
                        dropped_all = None
                        if len(dropped_batches_accum) > 0:
                            dropped_all = dropped_batches_accum[0] if len(dropped_batches_accum) == 1 \
                                else DataProto.concat(dropped_batches_accum)

                            # 再次兜底 response_mask（如果之前没成功）
                            if "response_mask" not in dropped_all.batch:
                                try:
                                    dropped_all.batch["response_mask"] = compute_response_mask(dropped_all)
                                except Exception:
                                    pass

                        # 确保 kept batch 有 response_mask（正常逻辑已有，这里兜底）
                        if "response_mask" not in batch.batch:
                            batch.batch["response_mask"] = compute_response_mask(batch)

                        self._write_rollout_records(
                            kept_batch=batch,
                            dropped_batch=dropped_all,
                            epoch=epoch,
                            global_step=self.global_steps,
                            actor_output_metrics=actor_output_metrics,
                        )
                    except Exception as e:
                        print(f"[WARN] _write_rollout_records failed at step {self.global_steps}: {e}")

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                dropped_batches_accum = []  # 清空，为下一 step 准备

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self._rr_close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
        self._rr_close()
        return
