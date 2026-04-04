# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os
import time

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
from omegaconf import open_dict, OmegaConf, DictConfig
from dataclasses import is_dataclass, asdict
from copy import deepcopy
if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

# ====== Stream Aggregator for clipping visualization (put near file top) ======
import math

class _Stats:
    __slots__ = ("sum", "cnt", "minv", "maxv")
    def __init__(self):
        self.sum  = 0.0
        self.cnt  = 0.0
        self.minv = float("inf")
        self.maxv = float("-inf")
    def update(self, mean_val, min_val, max_val, n):
        if n is None or n <= 0:
            return
        mv = float(mean_val); mn = float(min_val); mx = float(max_val)
        self.sum += mv * float(n)
        self.cnt += float(n)
        if mn < self.minv: self.minv = mn
        if mx > self.maxv: self.maxv = mx
    
    def finish(self):
        if self.cnt > 0:
            mean = self.sum / self.cnt
            mn   = self.minv
            mx   = self.maxv
        else:
            mean = mn = mx = float("nan")
        # 返回顺序(mean, max, min)便于批量写回
        return mean, mx, mn

class _DirStats:
    """对某一前缀的 up/down/total 三个子集维护 mean/max/min 的流式聚合"""
    SCOPES = ("up", "down", "total")
    def __init__(self):
        self.stats = {s: _Stats() for s in self.SCOPES}
    def update(self, prefix: str, m: dict, counts: dict):
        for s in self.SCOPES:
            n = float(counts.get(s, 0.0))
            mean_v = m.get(f"{prefix}_{s}_mean", -1.0)
            max_v  = m.get(f"{prefix}_{s}_max",  -1.0)
            min_v  = m.get(f"{prefix}_{s}_min",  -1.0)
            self.stats[s].update(mean_v, min_v, max_v, n)
    def writeback(self, prefix: str, out: dict):
        for s, st in self.stats.items():
            mean, mx, mn = st.finish()
            out[f"{prefix}_{s}_mean"] = float(mean)
            out[f"{prefix}_{s}_max"]  = float(mx)
            out[f"{prefix}_{s}_min"]  = float(mn)

class _ClipAgg:
    """
    聚合 compute_policy_loss_vanilla 返回的 extra_clip_metrics（micro 级），
    在 mini 结束时产出一次“已加权”的标量结果。
    """
    VAR_PREFIXES = {
        "actor/clip_old_probs":   "old_probs",
        "actor/clip_probs":       "probs",
        "actor/clipping_ratio":   "clipping_ratio",
        "actor/clipped_ratio":    "clipped_ratio",
        "actor/clip_bound_low":   "bound_low",
        "actor/clip_bound_high":  "bound_high",
    }
    def __init__(self):
        # 总体计数
        self.total_tokens = 0.0
        self.clip_total   = 0.0
        self.clip_up      = 0.0
        self.clip_down    = 0.0
        # 总体三向
        self.dir_overall = {p: _DirStats() for p in self.VAR_PREFIXES.keys()}
        # 分桶
        self.B = 5
        self.bin_total = [0.0] * self.B
        self.bin_up    = [0.0] * self.B
        self.bin_down  = [0.0] * self.B
        self.dir_bins  = [{p: _DirStats() for p in self.VAR_PREFIXES.keys()} for _ in range(self.B)]

    @staticmethod
    def _to_float_dict(tensor_dict: dict) -> dict:
        out = {}
        for k, v in tensor_dict.items():
            try:
                out[k] = float(v.detach().item())
            except Exception:
                out[k] = float(v)
        return out

    def update(self, m_raw: dict):
        m = self._to_float_dict(m_raw)

        # 1) 总体计数累加
        self.total_tokens += m.get("actor/total_tokens_count", 0.0)
        up   = m.get("actor/clipped_up_count",    0.0)
        down = m.get("actor/clipped_down_count",  0.0)
        tot  = m.get("actor/clipped_total_count", 0.0)
        self.clip_up    += up
        self.clip_down  += down
        self.clip_total += tot

        counts_overall = {"up": up, "down": down, "total": tot}

        # 2) 总体三向聚合（权重=对应子集的计数）
        for p in self.VAR_PREFIXES.keys():
            self.dir_overall[p].update(p, m, counts_overall)

        # 3) 分桶
        for b in range(self.B):
            bc  = m.get(f"actor/clip_bin{b}_count", 0.0)
            bup = m.get(f"actor/clip_bin{b}_up_count", 0.0)
            bdn = m.get(f"actor/clip_bin{b}_down_count", 0.0)
            self.bin_total[b] += bc
            self.bin_up[b]    += bup
            self.bin_down[b]  += bdn
            counts_bin = {"up": bup, "down": bdn, "total": bc}
            for p, suf in self.VAR_PREFIXES.items():
                # 分桶前缀与 loss 中一致：actor/clip_bin{b}_{suf}
                self.dir_bins[b][p].update(f"actor/clip_bin{b}_{suf}", m, counts_bin)

    def finalize(self) -> dict:
        out = {}
        # 计数
        out["actor/total_tokens_count"]  = float(self.total_tokens)
        out["actor/clipped_total_count"] = float(self.clip_total)
        out["actor/clipped_up_count"]    = float(self.clip_up)
        out["actor/clipped_down_count"]  = float(self.clip_down)

        # 你要求的三类比例（全局统一计算，避免 mean-of-means）
        denom_total = max(self.total_tokens, 1.0)
        denom_clip  = max(self.clip_total,   1.0)
        out["actor/clip_clipped_over_total"]   = float(self.clip_total / denom_total)
        out["actor/clip_up_frac_in_clipped"]   = float(self.clip_up   / denom_clip)
        out["actor/clip_down_frac_in_clipped"] = float(self.clip_down / denom_clip)

        # 总体三向
        for p in self.VAR_PREFIXES.keys():
            self.dir_overall[p].writeback(p, out)

        # 分桶：计数、方向占比、在 clipped 中占比 + 三向统计
        for b in range(self.B):
            bc  = self.bin_total[b]
            bup = self.bin_up[b]
            bdn = self.bin_down[b]
            out[f"actor/clip_bin{b}_count"]      = float(bc)
            out[f"actor/clip_bin{b}_up_count"]   = float(bup)
            out[f"actor/clip_bin{b}_down_count"] = float(bdn)

            bden = max(bc, 1.0)
            out[f"actor/clip_bin{b}_up_frac"]         = float(bup / bden)
            out[f"actor/clip_bin{b}_down_frac"]       = float(bdn / bden)   # bdn / bc
            out[f"actor/clip_bin{b}_frac_in_clipped"] = float(bc / max(self.clip_total, 1.0))

            for p, suf in self.VAR_PREFIXES.items():
                self.dir_bins[b][p].writeback(f"actor/clip_bin{b}_{suf}", out)

        # 把 ±inf 置为 NaN（NaN 保持原样）
        for k, v in out.items():
            if isinstance(v, float) and math.isinf(v):
                out[k] = float("nan")
        return out

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"
        self.is_debug = self.config.get("is_debug", False)
        self.debug_record_path = self.config.get("debug_record_path", "")
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self._last_time_comparison_summary = None
        
    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        _sync_if_cuda()
        update_policy_t0 = time.perf_counter()
        
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        global_steps  = int(data.meta_info.get("global_steps", -1))
        total_training_steps = int(data.meta_info.get("total_training_steps", -1))

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        time_method = (
            self.config.get("tokenwise_ratio_bounds_method", "clip")
            if self.config.get("use_tokenwise_ratio_bounds", False)
            else "clip"
        )

        time_summary_total = {
            "step": int(global_steps),
            "method": time_method,
            "num_mini_batches": 0,
            "num_policy_loss_calls": 0,
            "num_optimizer_steps": 0,
            "dense_tokens_sum": 0,
            "valid_tokens_sum": 0,
            "bound_total_ms_sum": 0.0,
            "bound_core_ms_sum": 0.0,
            "clip_apply_ms_sum": 0.0,
            "clip_related_ms_sum": 0.0,
            "mini_batch_total_ms_sum": 0.0,
            "update_policy_total_ms": 0.0,
        }
        # for _ in range(self.config.ppo_epochs):
        for epoch_idx in range(self.config.ppo_epochs):
            if self.is_debug and self.debug_record_path:
                with open(self.debug_record_path, "a") as f:
                    f.write(f"_________one_epoch_start_________\n")
            _clip_agg = _ClipAgg()
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                
                _sync_if_cuda()
                mini_t0 = time.perf_counter()

                mini_time = {
                    "num_policy_loss_calls": 0,
                    "dense_tokens_sum": 0,
                    "valid_tokens_sum": 0,
                    "bound_total_ms_sum": 0.0,
                    "bound_core_ms_sum": 0.0,
                    "clip_apply_ms_sum": 0.0,
                }
                
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    # 添加参数：self.config
                    # addon = OmegaConf.create({
                    #     "global_steps": int(global_steps),
                    #     "total_training_steps": int(total_training_steps),
                    # })
                    # cfg = OmegaConf.merge(addon, self.config)

                    # # 1) 把 self.config 转成“非结构化”的 DictConfig 副本（不会携带 dataclass 的严格类型/struct 校验）
                    # cfg_for_loss: DictConfig = OmegaConf.create(
                    #     OmegaConf.to_container(self.config, resolve=False, enum_to_str=True)
                    # )

                    # # 2) 在“副本”里加入运行期键（untyped 配置默认就允许新增键，不需要 open_dict）
                    # cfg_for_loss.global_steps = int(global_steps)
                    # cfg_for_loss.total_training_steps = int(total_training_steps)

                    def _to_untyped_dictconfig(cfg) -> DictConfig:
                        # 1) 已经是 DictConfig：先去容器，再重建为“非结构化” DictConfig
                        if isinstance(cfg, DictConfig):
                            container = OmegaConf.to_container(cfg, resolve=False, enum_to_str=True)
                            return OmegaConf.create(container)
                        # 2) dataclass 实例：用 asdict 拿到纯 Python dict（不会触发 OmegaConf 的类型校验）
                        if is_dataclass(cfg):
                            container = asdict(cfg)                 # 纯 dict/list/标量
                            return OmegaConf.create(container)      # 非结构化 DictConfig（允许自由加键）
                        # 3) 普通 dict：深拷贝后创建
                        if isinstance(cfg, dict):
                            return OmegaConf.create(deepcopy(cfg))
                        # 4) 兜底：尝试用对象的 __dict__
                        return OmegaConf.create(deepcopy(vars(cfg)))

                    # —— 在 update_policy 里 —— #
                    cfg_for_loss: DictConfig = _to_untyped_dictconfig(self.config)
                    cfg_for_loss.global_steps = int(global_steps)
                    cfg_for_loss.total_training_steps = int(total_training_steps)

                    outputs = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=cfg_for_loss,
                        # config=cfg,
                        rollout_log_probs=rollout_log_probs,
                    )
                    # if isinstance(outputs, tuple) and len(outputs) == 5:
                    #     pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, extra_clip_metrics = outputs
                    # else:
                    #     pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = outputs
                    #     extra_clip_metrics = {}
                    if isinstance(outputs, tuple) and len(outputs) == 5:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, extra_clip_metrics = outputs
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = outputs
                        extra_clip_metrics = {}

                    mini_time["num_policy_loss_calls"] += 1

                    time_info = None
                    if isinstance(extra_clip_metrics, dict):
                        time_info = extra_clip_metrics.pop("__time_comparison__", None)

                    if time_info is not None:
                        mini_time["dense_tokens_sum"] += int(time_info.get("dense_tokens", 0))
                        mini_time["valid_tokens_sum"] += int(time_info.get("valid_tokens", 0))
                        mini_time["bound_total_ms_sum"] += float(time_info.get("bound_total_ms", 0.0))
                        mini_time["bound_core_ms_sum"] += float(time_info.get("bound_core_ms", 0.0))
                        mini_time["clip_apply_ms_sum"] += float(time_info.get("clip_apply_ms", 0.0))
                    
                    
                    
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    if extra_clip_metrics:
                        _clip_agg.update(extra_clip_metrics)
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
                
                _sync_if_cuda()
                mini_batch_total_ms = (time.perf_counter() - mini_t0) * 1000.0
                clip_related_ms_sum = mini_time["bound_total_ms_sum"] + mini_time["clip_apply_ms_sum"]

                time_summary_total["num_mini_batches"] += 1
                time_summary_total["num_policy_loss_calls"] += mini_time["num_policy_loss_calls"]
                time_summary_total["num_optimizer_steps"] += 1
                time_summary_total["dense_tokens_sum"] += mini_time["dense_tokens_sum"]
                time_summary_total["valid_tokens_sum"] += mini_time["valid_tokens_sum"]
                time_summary_total["bound_total_ms_sum"] += mini_time["bound_total_ms_sum"]
                time_summary_total["bound_core_ms_sum"] += mini_time["bound_core_ms_sum"]
                time_summary_total["clip_apply_ms_sum"] += mini_time["clip_apply_ms_sum"]
                time_summary_total["clip_related_ms_sum"] += clip_related_ms_sum
                time_summary_total["mini_batch_total_ms_sum"] += mini_batch_total_ms

                if torch.distributed.get_rank() == 0:
                    clip_related_ratio = clip_related_ms_sum / mini_batch_total_ms if mini_batch_total_ms > 0 else 0.0
                    clip_related_ms_per_1k_dense_tokens = (
                        clip_related_ms_sum * 1000.0 / max(mini_time["dense_tokens_sum"], 1)
                    )
                    print(
                        f"[time comparison] "
                        f"step={global_steps} "
                        f"method={time_method} "
                        f"epoch={epoch_idx} "
                        f"mini_batch={batch_idx + 1}/{len(mini_batches)} "
                        f"mini_batch_total_ms={mini_batch_total_ms:.3f} "
                        f"num_micro_batches={len(micro_batches)} "
                        f"policy_loss_calls={mini_time['num_policy_loss_calls']} "
                        f"optimizer_steps=1 "
                        f"dense_tokens={mini_time['dense_tokens_sum']} "
                        f"valid_tokens={mini_time['valid_tokens_sum']} "
                        f"bound_total_ms={mini_time['bound_total_ms_sum']:.3f} "
                        f"bound_core_ms={mini_time['bound_core_ms_sum']:.3f} "
                        f"clip_apply_ms={mini_time['clip_apply_ms_sum']:.3f} "
                        f"clip_related_ms={clip_related_ms_sum:.3f} "
                        f"clip_related_ratio={clip_related_ratio:.6f} "
                        f"clip_related_ms_per_1k_dense_tokens={clip_related_ms_per_1k_dense_tokens:.6f}"
                    )
                    
            final_clip_metrics = _clip_agg.finalize()
            append_to_dict(metrics, final_clip_metrics)
            if self.is_debug and self.debug_record_path:
                with open(self.debug_record_path, "a") as f:
                    f.write(f"_________one_epoch_end__________\n")
                    f.write(f"str(metrics): {str(metrics)}\n")
        self.actor_optimizer.zero_grad()
        _sync_if_cuda()
        time_summary_total["update_policy_total_ms"] = (time.perf_counter() - update_policy_t0) * 1000.0
        self._last_time_comparison_summary = time_summary_total
        return metrics
