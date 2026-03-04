#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_and_plot_from_features.py

基于 build_features_preprocess.py / build_features_analysis.py 生成的 Parquet：

1）按各种条件筛选 sample：
    --cls {any,correct,wrong,no_judge}
    --kept-or-dropped {all,kept,dropped}
    --chinese-or-english {all,chinese,english}
    --step-min / --step-max
    --where 'pandas 表达式'
    --ge/--le/--eq 'col:val'
    --dedup-uid
    --limit / --per-step-limit

2）把筛选结果导出成 JSONL（从原始大 JSONL 按 byte_offset 精确拷贝）。

3）可视化（可关掉：--no-plot）：
    - 横轴 global_step
    - 纵轴各种 metrics
    - scatter + jitter + 颜色区分 correct / wrong / no_judge
    - 每 step 一个聚合值（mean/median/sum）画黑色折线
    - 按类别分三个子目录：
        <plot_dir>/training/*.png
        <plot_dir>/text/*.png
        <plot_dir>/lang_emb/*.png

如果筛选条件导致 0 行：
    - JSONL 保持空（符合筛选逻辑）
    - Plot 自动用全体样本做可视化（打印 WARNING）
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, os, sys, time, math, hashlib
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import re
import multiprocessing as mp
from collections import Counter

# ------------------ 模式匹配过滤 ------------------ #
# 文本模式匹配使用的默认进程数（会在 main() 里根据参数重新设置）
PATTERN_WORKERS: int = 1

# ===== 连续重复片段规则的一些可调阈值 =====
# 最小需要多少个 token 才进行“重复片段”检测（太短的回答直接跳过，省时间也减少误报）
MIN_TOKENS_FOR_REPEAT_CHECK = 30

# 片段长度用 token 的 n-gram 来扫：从 1 到 MAX
REPEAT_NGRAM_MIN = 1
REPEAT_NGRAM_MAX = 16   # 你可以改成 8 / 12 之类的

# 判定“严重重复”的阈值：
# - 同一个片段（n-gram）连续重复至少 MIN_REPEAT_BLOCKS 次
# - 且这些重复一共覆盖的 token 数 >= MIN_REPEAT_TOTAL_TOKENS
MIN_REPEAT_BLOCKS = 16           # 8 段连在一起，已经非常怪了
MIN_REPEAT_TOTAL_TOKENS = max(20, MIN_REPEAT_BLOCKS)    # 总长度 ≥ 60 个 token，太短了会把一些数学的数字list误删

# ------------- 重复啰嗦 -----------------------------
def _tokens_from_text(resp: str) -> List[str]:
    """
    一个非常简单的分词：按空白切分。
    对我们关心的重复片段（no,no,no... / Answer: 39916800 / 8 = 4989600 ...）足够用。
    """
    return re.findall(r"\S+", resp)
def _tokens_to_ids(tokens: List[str]) -> List[int]:
    """
    把 token 映射到稳定的整数 ID，方便做 rolling hash。
    每条样本单独一个小词表，开销很低。
    """
    mapping: Dict[str, int] = {}
    ids: List[int] = []
    next_id = 1
    for t in tokens:
        if t not in mapping:
            mapping[t] = next_id
            next_id += 1
        ids.append(mapping[t])
    return ids
def _max_tandem_repeat_blocks(
    seq: List[int],
    min_block_len: int = REPEAT_NGRAM_MIN,
    max_block_len: int = REPEAT_NGRAM_MAX,
) -> Tuple[int, int]:
    """
    在整数序列 seq 上寻找“同一片段连续重复”的最大次数。

    返回: (max_blocks, max_total_len)
        max_blocks     : 最多重复了多少“段”（>=1）
        max_total_len  : 对应重复片段覆盖的 token 总数 = max_blocks * block_len

    我们只关心连续的 A A A A A... 或 AB AB AB AB... 这种“串联重复”，
    而不是全局的出现次数。
    """
    n = len(seq)
    if n < 2 * min_block_len:
        return 1, 0

    max_block_len = min(max_block_len, n // 2)
    if max_block_len < min_block_len:
        return 1, 0

    # 64-bit rolling hash
    MOD_MASK = (1 << 64) - 1
    BASE = 911382323

    prefix = [0] * (n + 1)
    pow_base = [1] * (n + 1)
    for i, x in enumerate(seq):
        prefix[i + 1] = (prefix[i] * BASE + (x + 1)) & MOD_MASK
        pow_base[i + 1] = (pow_base[i] * BASE) & MOD_MASK

    def _hash(l: int, r: int) -> int:
        """哈希 seq[l:r]"""
        return (prefix[r] - (prefix[l] * pow_base[r - l] & MOD_MASK)) & MOD_MASK

    best_blocks = 1
    best_total_len = 0

    # 按块长 k（n-gram 长度）枚举
    for k in range(min_block_len, max_block_len + 1):
        i = 0
        # 至少要有两段才能称为“重复”
        while i + 2 * k <= n:
            h1 = _hash(i, i + k)
            h2 = _hash(i + k, i + 2 * k)
            if h1 != h2:
                i += 1
                continue

            # 找到了至少 2 段完全相同的片段，从这里往后延伸
            blocks = 2
            j = i + 2 * k
            while j + k <= n and _hash(j, j + k) == h1:
                blocks += 1
                j += k

            total_len = blocks * k
            if blocks > best_blocks or (blocks == best_blocks and total_len > best_total_len):
                best_blocks = blocks
                best_total_len = total_len

            # 跳过这一整串重复片段，避免 O(n^2) 行为
            i = j

    return best_blocks, best_total_len
def _rule_tandem_repetition(resp: str) -> bool:
    """
    规则 1：检测“连续重复的语言片段”（单词 / 短语 / 句子片段）。

    思路：
      1）把 response 按空白切成 token 序列。
      2）用 rolling hash 在 token 序列上找：
           - 某个 n-gram（1~REPEAT_NGRAM_MAX）连续重复了很多次
           - 且覆盖足够长的 token 数。
      3）如果满足 (重复段数 >= MIN_REPEAT_BLOCKS 且 总长度 >= MIN_REPEAT_TOTAL_TOKENS)
         则认为这是“严重重复啰嗦”，返回 True。
    """
    if not isinstance(resp, str):
        return False

    s = resp.strip()
    if not s:
        return False

    tokens = _tokens_from_text(s)
    if len(tokens) < MIN_TOKENS_FOR_REPEAT_CHECK:
        # 太短的不做重复片段检测，既省算力也很少会是我们想过滤的垃圾
        return False

    ids = _tokens_to_ids(tokens)
    max_blocks, max_total_len = _max_tandem_repeat_blocks(ids)

    # 你可以调下面两个阈值：
    # - max_blocks：同一个片段连续重复了多少“段”
    # - max_total_len：这些重复一共覆盖了多少 token
    if max_blocks >= MIN_REPEAT_BLOCKS and max_total_len >= MIN_REPEAT_TOTAL_TOKENS:
        return True

    return False

def _is_redundant_noisy_response(resp: str) -> bool:
    """
    顶层的“啰嗦/垃圾响应”判定入口。

    以后可以把各种规则写成 _rule_xxx，然后加到 RULE_FNS 里即可。
    通过“注释/取消注释”来选择开启哪几条规则。
    """
    if not isinstance(resp, str) or resp == "":
        return False

    # 需要启用的规则列表：
    RULE_FNS = [
        _rule_tandem_repetition,  # 规则 1：连续重复片段（单词/短语/句子片段）
        # 以后你可以在这里继续加：
        # _rule_answer_tail_garbage,
        # _rule_brace_noise,
        # _rule_wait_spam,
    ]

    for fn in RULE_FNS:
        if fn(resp):
            return True
    return False
def _check_redundant_chunk(responses: List[str]) -> List[bool]:
    """
    多进程用：对一块 response 列表跑 _is_redundant_noisy_response。
    """
    out: List[bool] = []
    for resp in responses:
        out.append(_is_redundant_noisy_response(resp))
    return out
def pattern_redundant_answer_noise(df: pd.DataFrame) -> pd.Series:
    """
    模式 X：文本层面的“重复啰嗦 / 垃圾尾巴”检测。

    完全基于 response_text：
      - 不依赖任何 text_metrics
      - 不区分 prompt 类型
    """
    print("[INFO] Filter pattern_redundant_answer_noise...")
    if "response_text" not in df.columns:
        raise KeyError("pattern_redundant_answer_noise 需要 df 中存在 'response_text' 列")

    responses = df["response_text"].astype(str).tolist()
    n = len(responses)
    if n == 0:
        return pd.Series(False, index=df.index)

    workers = max(1, int(PATTERN_WORKERS or 1))

    # 小数据或只用一个进程：直接单线程
    if workers == 1 or n < 1000:
        flags = [_is_redundant_noisy_response(resp) for resp in responses]
    else:
        # 大数据：按块切，降低进程间通信开销，充分利用多核
        chunk_size = max(1, n // (workers * 4))
        chunks: List[List[str]] = []
        for i in range(0, n, chunk_size):
            chunks.append(responses[i : i + chunk_size])

        flags: List[bool] = []
        with mp.Pool(processes=workers) as pool:
            for part in pool.map(_check_redundant_chunk, chunks):
                flags.extend(part)

    return pd.Series(flags, index=df.index)

# ------------ 其他过滤项 ------------------------
def pattern_ce_ref_mean_too_large(df: pd.DataFrame) -> pd.Series:
    """
    模式 1：
    CE ref mean (列名: ce_ref_seq_mean) > 1.8 的样本视为“命中模式”，需要被过滤掉。
    """
    print("[INFO] Filter pattern_ce_ref_mean_too_large...")
    if "ce_ref_seq_mean" not in df.columns:
        # 没有这个列就当没命中
        return pd.Series(False, index=df.index)
    return df["ce_ref_seq_mean"].astype(float) > 1.8

def pattern_repeat_word_run_max_too_large(df: pd.DataFrame) -> pd.Series:
    """
    模式 2：
    repeat_word_run_max > 25 的样本视为“命中模式”，需要被过滤掉。
    """
    print("[INFO] Filter pattern_repeat_word_run_max_too_large...")
    if "repeat_word_run_max" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["repeat_word_run_max"].astype(float) > 25

def pattern_num_self_reflect_sentences_too_more(df: pd.DataFrame) -> pd.Series:
    print("[INFO] Filter pattern_num_self_reflect_sentences_too_more...")
    if "num_self_reflect_sentences" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["num_self_reflect_sentences"].astype(float) > 40

def pattern_marker_loopiness_too_more(df: pd.DataFrame) -> pd.Series:
    print("[INFO] Filter pattern_marker_loopiness_too_more...")
    if "marker_loopiness" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["marker_loopiness"].astype(float) > 50

# -----------过滤器核心 ------------------------
def build_filter_mask_by_patterns(
    df: pd.DataFrame,
    target_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于一组“模式函数”做过滤：
        - 每个模式函数返回一个 bool Series，True 表示“命中该模式”。
        - 多个模式之间是 OR 关系。
        - 只有 target_mask 为 True 的样本才会被真正过滤；
          target_mask 为 False 的样本永远不会被标记为命中。

    返回:
        keep_mask: np.ndarray[bool]，True = 保留
        bad_mask : np.ndarray[bool]，True = 被模式命中（会被过滤）
    """
    n = len(df)
    if n == 0:
        return np.ones(0, dtype=bool), np.zeros(0, dtype=bool)

    if target_mask is None:
        target_mask_series = pd.Series(True, index=df.index)
    else:
        if len(target_mask) != n:
            raise ValueError("target_mask 长度与 df 不一致")
        target_mask_series = pd.Series(target_mask, index=df.index)

    print("[INFO] Start filtering:")
    # 在这里注册所有模式
    pattern_fns = [
        pattern_ce_ref_mean_too_large,
        pattern_repeat_word_run_max_too_large,
        pattern_num_self_reflect_sentences_too_more,
        pattern_marker_loopiness_too_more,
        pattern_redundant_answer_noise,  # NEW: 文本级重复/垃圾检测
        # 以后新增模式继续往下加
    ]

    bad_mask = pd.Series(False, index=df.index)

    for fn in pattern_fns:
        m = fn(df)
        if m is None:
            continue
        if len(m) != n:
            raise ValueError(f"Pattern {fn.__name__} 返回的 mask 长度不匹配")
        # 只在 target_mask 为 True 的位置上应用模式
        bad_mask |= (m & target_mask_series)

    keep_mask = ~bad_mask
    return keep_mask.to_numpy(), bad_mask.to_numpy()

# ------------------ 小工具函数 ------------------ #

def parse_cmp_list(items: List[str]) -> List[Tuple[str, float]]:
    out = []
    for it in items or []:
        if ":" not in it:
            raise ValueError(f"Bad 'col:val' item: {it}")
        col, val = it.split(":", 1)
        out.append((col.strip(), float(val)))
    return out

def salt64_from_uid(uid: Any) -> int:
    s = str(uid)
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)

def jitter_from_salt(salts: np.ndarray, scale: float) -> np.ndarray:
    # 把 salt 映射到 [0,1) 再线性映射到 [-scale, scale]
    r = (salts % 10_000_019) / 10_000_019.0
    return (r - 0.5) * 2.0 * scale


# ------------------ 聚合 & 画图 ------------------ #

COLOR_CORRECT = "blue"
COLOR_WRONG   = "red"
COLOR_UNKNOWN = "purple"
COLOR_AGG     = "black"

def aggregate_per_step(values: np.ndarray,
                       steps: np.ndarray,
                       agg: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 (steps, values) 做 per-step 聚合。
    values 中 NaN 会被忽略；若某 step 全是 NaN，则跳过。
    """
    df = pd.DataFrame({"step": steps, "val": values})
    df = df[np.isfinite(df["val"].values)]
    if df.empty:
        return np.array([], dtype=np.int64), np.array([], dtype=float)

    if agg == "median":
        gb = df.groupby("step")["val"].median()
    elif agg == "sum":
        gb = df.groupby("step")["val"].sum()
    else:
        gb = df.groupby("step")["val"].mean()

    xs = gb.index.to_numpy()
    ys = gb.values
    return xs, ys


def scatter_one_metric(
    out_png: Path,
    metric_name: str,
    values: np.ndarray,     # shape [N]
    steps: np.ndarray,      # shape [N]
    cls_id: np.ndarray,     # 1=correct, 0=wrong, 2=no_judge
    jitter_seeds: np.ndarray,
    agg: str,
    jitter: float,
    point_size: int,
    alpha: float,
    y_label: str,
    title: str,
):
    """
    沙箱图：scatter + per-step 聚合折线
    """
    t0 = time.time()
    fig, ax = plt.subplots(figsize=(14, 6))

    ok = np.isfinite(values)
    if not np.any(ok):
        print(f"[WARN] metric={metric_name} 全是 NaN，跳过绘图")
        plt.close(fig)
        return

    steps_ok = steps[ok].astype(float)
    vals_ok = values[ok]
    cls_ok = cls_id[ok]
    jitter_seed_ok = jitter_seeds[ok]

    js = jitter_from_salt(jitter_seed_ok, jitter)

    mask_wrong   = (cls_ok == 0)
    mask_correct = (cls_ok == 1)
    mask_unknown = (cls_ok == 2)

    def _scatter(mask, marker, color, label):
        if not np.any(mask):
            return
        xs = steps_ok[mask] + js[mask]
        ys = vals_ok[mask]
        ax.scatter(xs, ys, s=point_size, alpha=alpha,
                   marker=marker, facecolors="none",
                   edgecolors=color, linewidths=0.8,
                   label=label)

    _scatter(mask_wrong,   "^", COLOR_WRONG,   "wrong")
    _scatter(mask_correct, "o", COLOR_CORRECT, "correct")
    _scatter(mask_unknown, "s", COLOR_UNKNOWN, "no_judge")

    # per-step 聚合线
    agg_x, agg_y = aggregate_per_step(vals_ok, steps_ok.astype(int), agg)
    if agg_x.size > 0:
        ax.plot(agg_x, agg_y, marker="o", markersize=3, linewidth=1.5,
                color=COLOR_AGG, label=f"{agg} per step")

    ax.set_xlabel("global_step")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(),
              loc="best", fontsize="small", frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    dt = time.time() - t0
    print(f"[PLOT] {metric_name} -> {out_png}  (took {dt:.1f}s)")


# ------------------ 指标分类逻辑 ------------------ #

TRAINING_BASE = {
    # 你在 builder 里一定有的
    "ce_self_seq_mean", "ce_self_seq_sum",
    "ce_ref_seq_mean", "ce_ref_seq_sum",
    "reward_seq_sum",
    "kl_seq_mean_log_ratio",
    "entropy_seq_mean",
    "response_len_tokens",
    # 可能扩展出来的
    "reward_seq_mean", "reward_seq_max",
    "adv_seq_mean", "value_seq_mean",
    "kl_value_mean", "kl_seq_sum_log_ratio",
    "entropy_mean", "entropy_seq_sum",
}

TEXT_BASE = {
    # text_metrics 全家桶
    "resp_char_len", "resp_byte_len",
    "non_ascii_ratio", "replacement_char_ratio",
    "control_private_unassigned_ratio",
    "repeat_char_run_max", "repeat_word_run_max",
    "unique_word_ratio", "top1_word_prop",
    "trigram_repeat_ratio",
    "compress_ratio",
    "char_entropy_bits_per_char",
    # 结构指标（section/sentence）
    "num_sections", "num_sentences",
    "num_marker_sentences",
    "num_self_reflect_sentences",
    "num_contrast_sentences",
    "num_conclude_sentences",
    "num_marker_sections",
    # 语言是否含中文（0/1/-1）也可以看成文本分布
    "contain_chinese",
}

LANG_EMB_BASE = {
    # 语言统计
    "H_lang", "num_lang_major", "code_switch_count",
    # embedding + cluster + loopiness
    "all_r_dup", "all_cluster_count", "all_cluster_size_max",
    "all_rho_max", "all_loopiness", "all_plateau_len_max",
    "marker_r_dup", "marker_cluster_count", "marker_cluster_size_max",
    "marker_rho_max", "marker_loopiness", "marker_plateau_len_max",
    "wait_r_dup", "wait_cluster_count", "wait_cluster_size_max",
    "wait_rho_max", "wait_loopiness", "wait_plateau_len_max",
    "embedding_truncated",
    # 综合指标
    "rambling_score",
}

def categorize_metric(name: str) -> str:
    """
    返回类别: 'training' / 'text' / 'lang_emb'
    尽量全面：先查基础集合，再用名字 heuristic。
    """
    if name in TRAINING_BASE:
        return "training"
    if name in TEXT_BASE:
        return "text"
    if name in LANG_EMB_BASE:
        return "lang_emb"

    # heuristics
    lower = name.lower()

    # RL / 训练相关
    if lower.startswith("ce_") or "logprob" in lower or "reward" in lower:
        return "training"
    if "kl_" in lower or lower.startswith("entropy_") or lower.endswith("_tokens"):
        return "training"

    # 文本结构 / 质量
    if lower.startswith("resp_") or "repeat_" in lower or "compress" in lower:
        return "text"
    if "entropy_bits_per_char" in lower or "byte_len" in lower or "char_len" in lower:
        return "text"
    if lower.startswith("num_") and ("section" in lower or "sentence" in lower or "marker" in lower):
        return "text"
    if "non_ascii" in lower or "replacement_char" in lower or "control_private_unassigned" in lower:
        return "text"

    # 语言/embedding/discourse
    if "lang" in lower or "code_switch" in lower:
        return "lang_emb"
    if lower.startswith("all_") or lower.startswith("marker_") or lower.startswith("wait_"):
        return "lang_emb"
    if "loopiness" in lower or "rho_max" in lower or "cluster_" in lower:
        return "lang_emb"
    if "rambling" in lower or "embedding" in lower:
        return "lang_emb"

    # 默认：当作 training（纯数值指标）
    return "training"


def pretty_metric_name(name: str) -> str:
    """
    为常见指标给一个更友好的标题；没列出的就原样返回。
    """
    mapping = {
        # training
        "ce_self_seq_mean": "CE self (mean, nats)",
        "ce_self_seq_sum":  "CE self (sum, nats)",
        "ce_ref_seq_mean":  "CE ref (mean, nats)",
        "ce_ref_seq_sum":   "CE ref (sum, nats)",
        "reward_seq_sum":   "Reward (sum)",
        "kl_seq_mean_log_ratio": "KL approx (mean log-ratio)",
        "entropy_seq_mean": "Token entropy (mean)",
        "response_len_tokens": "Response length (tokens)",

        # text
        "resp_char_len": "Response length (chars)",
        "resp_byte_len": "Response length (bytes)",
        "non_ascii_ratio": "Non-ASCII char ratio",
        "replacement_char_ratio": "Replacement char (�) ratio",
        "control_private_unassigned_ratio": "Ctrl/Private/Unassigned cat ratio",
        "repeat_char_run_max": "Max repeated char run",
        "repeat_word_run_max": "Max repeated word run",
        "unique_word_ratio": "Unique word ratio",
        "top1_word_prop": "Top-1 word proportion",
        "trigram_repeat_ratio": "Repeated trigram ratio",
        "compress_ratio": "Zlib compression ratio (↓ => more repetitive)",
        "char_entropy_bits_per_char": "Char entropy (bits/char)",
        "num_sections": "Num sections (\\n-based)",
        "num_sentences": "Num sentences (pysbd)",
        "num_marker_sentences": "Num sentences w/ markers",
        "num_self_reflect_sentences": "Num SELF_REFLECT sentences",
        "num_contrast_sentences": "Num CONTRAST sentences",
        "num_conclude_sentences": "Num CONCLUDE sentences",
        "num_marker_sections": "Num sections w/ markers",
        "contain_chinese": "Contains Chinese (0/1)",

        # lang / embedding
        "H_lang": "Language entropy H_lang",
        "num_lang_major": "Num major languages (>10%)",
        "code_switch_count": "Language code-switch count",
        "all_r_dup": "ALL: near-duplicate ratio",
        "all_cluster_count": "ALL: cluster count",
        "all_cluster_size_max": "ALL: largest cluster size",
        "all_rho_max": "ALL: largest cluster ratio",
        "all_loopiness": "ALL: loopiness (L/D)",
        "all_plateau_len_max": "ALL: max plateau length",
        "marker_r_dup": "MARKER: near-duplicate ratio",
        "marker_cluster_count": "MARKER: cluster count",
        "marker_cluster_size_max": "MARKER: largest cluster size",
        "marker_rho_max": "MARKER: largest cluster ratio",
        "marker_loopiness": "MARKER: loopiness (L/D)",
        "marker_plateau_len_max": "MARKER: max plateau length",
        "wait_r_dup": "WAIT: near-duplicate ratio",
        "wait_cluster_count": "WAIT: cluster count",
        "wait_cluster_size_max": "WAIT: largest cluster size",
        "wait_rho_max": "WAIT: largest cluster ratio",
        "wait_loopiness": "WAIT: loopiness (L/D)",
        "wait_plateau_len_max": "WAIT: max plateau length",
        "embedding_truncated": "Embedding truncated flag",
        "rambling_score": "Rambling score",
    }
    return mapping.get(name, name)


# ------------------ 主函数 ------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, required=True,
                    help="build_features_preprocess / analysis 输出目录 (含 features*.parquet 和 meta_preprocess.json)")
    ap.add_argument("--features-file", type=str, default=None,
                    help="默认优先使用 features_analysis.parquet；若不存在则用 features.parquet；也可以手动指定文件名。")
    ap.add_argument("--out-jsonl", type=Path, required=True,
                    help="筛选后的样本 JSONL 输出路径（原始行的 byte copy）")
    ap.add_argument("--plot-dir", type=Path, default=None,
                    help="可视化输出目录（默认 = --features-dir）")

    # 过滤条件
    ap.add_argument("--cls", type=str, default="any",
                    choices=["any","correct","wrong","no_judge"],
                    help="按 cls_id 过滤：1=correct,0=wrong,2=no_judge")
    ap.add_argument("--kept-or-dropped", type=str, default="all",
                    choices=["all","kept","dropped"],
                    help="按 kept_flag 过滤")
    ap.add_argument("--chinese-or-english", type=str, default="all",
                    choices=["all","chinese","english"],
                    help="基于 contain_chinese 列 (1/0/-1)")
    ap.add_argument("--step-min", type=int, default=None)
    ap.add_argument("--step-max", type=int, default=None)
    ap.add_argument("--where", type=str, default=None,
                    help="pandas query 表达式，例如 'non_ascii_ratio>0.15 and repeat_word_run_max>10'")
    ap.add_argument("--ge", action="append", default=[],
                    help="阈值 'col:val' 表示 col >= val，可重复")
    ap.add_argument("--le", action="append", default=[],
                    help="阈值 'col:val' 表示 col <= val，可重复")
    ap.add_argument("--eq", action="append", default=[],
                    help="阈值 'col:val' 表示 col == val，可重复")
    ap.add_argument("--logic", type=str, default="and", choices=["and","or"],
                    help="--ge/--le/--eq 之间用 AND 还是 OR 组合")
    ap.add_argument("--dedup-uid", action="store_true",
                    help="按 uid 去重：同 uid 保留 (global_step, byte_offset) 最小的一条")
    ap.add_argument("--limit", type=int, default=None,
                    help="全局最多保留多少条样本（排序之后截断）")
    ap.add_argument("--per-step-limit", type=int, default=None,
                    help="每个 global_step 最多保留多少条样本（排序之后 groupby head）")
    ap.add_argument("--pattern-workers", type=int, default=0,
                    help="文本模式匹配（重复啰嗦检测）使用的进程数；0 表示自动选择")

    # 可视化控制
    ap.add_argument("--no-plot", action="store_true",
                    help="只做筛选 + JSONL 导出，不画图")
    ap.add_argument("--metrics", type=str, default="all",
                    help=(
                        "要画哪些 metrics：\n"
                        "  - 'all'        : 所有数值列按类别分三类画图\n"
                        "  - 'training'   : 只画训练/RL 类指标\n"
                        "  - 'text'       : 只画文本结构/质量指标\n"
                        "  - 'lang_emb'   : 只画语言 & embedding 类指标\n"
                        "  - 'm1,m2,...'  : 只画这些列名（会自动分到对应子目录）"
                    ))
    ap.add_argument("--agg", type=str, default="mean",
                    choices=["mean","median","sum"],
                    help="per-step 聚合方式")
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--point-size", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.25)

    args = ap.parse_args()
    # 设置全局的文本模式匹配并行度
    global PATTERN_WORKERS
    cpu_total = os.cpu_count() or 1
    if args.pattern_workers is None or args.pattern_workers <= 0:
        # 默认：不把机器跑满，给 IO / 系统留一点空间
        PATTERN_WORKERS = max(1, min(cpu_total, 32))
    else:
        # 不超过实际 CPU 数
        PATTERN_WORKERS = max(1, min(int(args.pattern_workers), cpu_total))
    print(f"[INFO] pattern_workers = {PATTERN_WORKERS} (available CPUs = {cpu_total})")

    features_dir = args.features_dir
    plot_dir = args.plot_dir or features_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # 1) 选择 features parquet + meta
    if args.features_file is not None:
        features_path = features_dir / args.features_file
    else:
        c1 = features_dir / "features_analysis.parquet"
        c2 = features_dir / "features.parquet"
        if c1.exists():
            features_path = c1
        elif c2.exists():
            features_path = c2
        else:
            raise FileNotFoundError("在 --features-dir 下找不到 features_analysis.parquet 或 features.parquet；"
                                    "请用 --features-file 指定。")

    meta_pre = features_dir / "meta_preprocess.json"
    if not meta_pre.exists():
        meta_pre = features_dir / "meta.json"
    if not meta_pre.exists():
        raise FileNotFoundError("未找到 meta_preprocess.json 或 meta.json，用于解析 source_jsonl 和 offset 信息。")

    print(f"[INFO] Loading meta from {meta_pre}")
    meta = json.loads(meta_pre.read_text(encoding="utf-8"))
    source_jsonl = Path(meta["source_jsonl"])
    if not source_jsonl.exists():
        raise FileNotFoundError(f"source_jsonl 不存在: {source_jsonl}")

    print(f"[INFO] Loading features from {features_path}")
    t0 = time.time()
    df = pd.read_parquet(features_path)
    print(f"[INFO] features loaded: rows={len(df):,}, cols={len(df.columns)} in {time.time()-t0:.1f}s")

    required_cols = ["global_step","byte_offset","byte_len","cls_id","kept_flag","contain_chinese"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"features 中缺少必要列: {c}")

    # 基本统计信息
    total_rows = len(df)
    num_kept = int((df["kept_flag"] == 1).sum())
    num_dropped = int((df["kept_flag"] == 0).sum())

    is_kept = (df["kept_flag"] == 1)
    is_correct = (df["cls_id"] == 1)
    is_wrong = (df["cls_id"] == 0)

    num_kept_correct = int((is_kept & is_correct).sum())
    num_kept_wrong   = int((is_kept & is_wrong).sum())

    print(f"[STATS] Loaded samples (rows / jsonl lines): {total_rows:,}")
    print(f"[STATS] kept    = {num_kept:,}")
    print(f"[STATS] dropped = {num_dropped:,}")
    print(f"[STATS] kept & correct = {num_kept_correct:,}")
    print(f"[STATS] kept & wrong   = {num_kept_wrong:,}")

    # ---------- 从原始 JSONL 补充 response_text 列，用于文本模式检测 ---------- #
    if "response_text" not in df.columns:
        print("[INFO] 'response_text' not in features; will load from source_jsonl for ALL rows...")
        t_txt0 = time.time()
        resp_list: List[str] = []

        # 注意：这里假定每一行 JSON 里都有 'response_text' 字段
        with source_jsonl.open("rb", buffering=1024 * 1024) as fsrc:
            fileno = fsrc.fileno()
            for _, row in df.iterrows():
                off = int(row["byte_offset"])
                ln  = int(row["byte_len"])
                b   = os.pread(fileno, ln, off)
                try:
                    j = json.loads(b)
                except Exception as e:
                    # 出问题就给个占位，避免模式函数崩
                    # 也方便后面你手动检查这几行
                    print(f"error: e.")
                    resp_list.append("")
                    continue
                resp_list.append(j.get("response_text", ""))

        df["response_text"] = resp_list
        print(f"[INFO] response_text column attached, took {time.time() - t_txt0:.1f}s")

    # jitter 的种子：优先使用 sample_idx，其次 uid，最后用行号
    if "sample_idx" in df.columns:
        jitter_seed_full = df["sample_idx"].astype(int).to_numpy()
    elif "uid" in df.columns:
        jitter_seed_full = np.array([salt64_from_uid(u) for u in df["uid"].astype(str)], dtype=np.int64)
    else:
        jitter_seed_full = np.arange(len(df), dtype=np.int64)

    # # 2) 构建筛选 mask
    # n = len(df)
    # mask = np.ones(n, dtype=bool)
    # filter_applied = False

    # # kept/dropped
    # if args.kept_or_dropped == "kept":
    #     mask &= (df["kept_flag"] == 1)
    #     filter_applied = True
    # elif args.kept_or_dropped == "dropped":
    #     mask &= (df["kept_flag"] == 0)
    #     filter_applied = True

    # # cls
    # if args.cls != "any":
    #     cls_map = {"correct":1, "wrong":0, "no_judge":2}
    #     mask &= (df["cls_id"] == cls_map[args.cls])
    #     filter_applied = True

    # # 中文/英文
    # if args.chinese_or_english != "all":
    #     if args.chinese_or_english == "chinese":
    #         mask &= (df["contain_chinese"] == 1)
    #     else:
    #         mask &= (df["contain_chinese"] == 0)
    #     filter_applied = True

    # # global_step 范围
    # if args.step_min is not None:
    #     mask &= (df["global_step"] >= int(args.step_min))
    #     filter_applied = True
    # if args.step_max is not None:
    #     mask &= (df["global_step"] <= int(args.step_max))
    #     filter_applied = True

    # # ge/le/eq
    # ge_list = parse_cmp_list(args.ge)
    # le_list = parse_cmp_list(args.le)
    # eq_list = parse_cmp_list(args.eq)
    # if ge_list or le_list or eq_list:
    #     filter_applied = True
    #     cur = np.zeros(n, dtype=bool) if args.logic == "or" else np.ones(n, dtype=bool)
    #     def _comb(c1, c2):
    #         return (c1 | c2) if args.logic == "or" else (c1 & c2)

    #     for col, val in ge_list:
    #         if col not in df.columns:
    #             raise KeyError(f"--ge: unknown column {col}")
    #         cond = (df[col].astype(float) >= float(val)).to_numpy()
    #         cur = _comb(cur, cond)
    #     for col, val in le_list:
    #         if col not in df.columns:
    #             raise KeyError(f"--le: unknown column {col}")
    #         cond = (df[col].astype(float) <= float(val)).to_numpy()
    #         cur = _comb(cur, cond)
    #     for col, val in eq_list:
    #         if col not in df.columns:
    #             raise KeyError(f"--eq: unknown column {col}")
    #         cond = (df[col].astype(float) == float(val)).to_numpy()
    #         cur = _comb(cur, cond)
    #     mask &= cur

    # # where 表达式
    # if args.where:
    #     filter_applied = True
    #     try:
    #         idx_where = df.query(args.where, engine="python").index
    #         cond = df.index.to_series().isin(idx_where).to_numpy()
    #         mask &= cond
    #     except Exception as e:
    #         print(f"[ERROR] 解析 --where 失败: {e}", file=sys.stderr)
    #         sys.exit(2)

    # df_sel = df[mask].copy()
    # print(f"[INFO] After filtering: rows={len(df_sel):,} (filter_applied={filter_applied})")
    # # 2) 基于“模式匹配”做过滤
    # keep_mask = build_filter_mask_by_patterns(df)
    # df_sel = df[keep_mask].copy()

    # filtered_cnt = int((~keep_mask).sum())
    # filter_applied = filtered_cnt > 0

    # print(f"[INFO] After pattern filtering: rows={len(df_sel):,}, "
    #       f"filtered_out={filtered_cnt:,}")
    # 2) 构建“过滤目标子集”：只在这个子集上应用模式过滤
    #    你现在的用法一般是: --kept-or-dropped kept --cls correct
    #    这样 target_mask 就刚好是 "kept & correct"
    target_mask = np.ones(len(df), dtype=bool)

    # 按 kept_or_dropped 子集
    if args.kept_or_dropped == "kept":
        target_mask &= (df["kept_flag"] == 1)
    elif args.kept_or_dropped == "dropped":
        target_mask &= (df["kept_flag"] == 0)
    # "all" 的时候不收缩子集

    # 按 cls 子集
    if args.cls != "any":
        cls_map = {"correct": 1, "wrong": 0, "no_judge": 2}
        target_mask &= (df["cls_id"] == cls_map[args.cls])

    # # 3) 基于模式过滤（只作用在 target_mask=True 的样本上）
    # keep_mask, bad_mask = build_filter_mask_by_patterns(df, target_mask=target_mask)

    # df_sel = df[keep_mask].copy()
    # filtered_cnt = int(bad_mask.sum())
    # filter_applied = filtered_cnt > 0

    # # 在 kept 集合上的命中数量和比例
    # bad_in_kept = int((bad_mask & is_kept.to_numpy()).sum())
    # ratio_bad_in_kept = (bad_in_kept / num_kept) if num_kept > 0 else 0.0

    # print(f"[INFO] After pattern filtering: rows={len(df_sel):,}")
    # print(f"[STATS] pattern-matched samples (overall)  = {filtered_cnt:,}")
    # print(f"[STATS] pattern-matched within kept       = {bad_in_kept:,} "
    #       f"({ratio_bad_in_kept:.4%} of kept)")
    # 3) 基于模式过滤（只作用在 target_mask=True 的样本上）
    keep_mask, bad_mask = build_filter_mask_by_patterns(df, target_mask=target_mask)

    # ⚠️ 这里“符合过滤条件的样本” = 被模式命中的样本 = bad_mask == True
    df_sel = df[bad_mask].copy()
    selected_cnt = int(bad_mask.sum())
    filter_applied = True  # 确实应用了筛选逻辑

    # 在 kept 集合上的命中数量和比例（占 kept 的比例）
    bad_in_kept = int((bad_mask & is_kept.to_numpy()).sum())
    ratio_bad_in_kept = (bad_in_kept / num_kept) if num_kept > 0 else 0.0

    print(f"[INFO] Selected samples (pattern-matched): rows={len(df_sel):,}")
    print(f"[STATS] pattern-matched samples (overall)  = {selected_cnt:,}")
    print(f"[STATS] pattern-matched within kept       = {bad_in_kept:,} "
          f"({ratio_bad_in_kept:.4%} of kept)")

    # 去重 uid
    if args.dedup_uid and "uid" in df_sel.columns:
        df_sel = (df_sel.sort_values(["global_step","byte_offset"], kind="mergesort")
                        .drop_duplicates("uid", keep="first"))
        print(f"[INFO] After dedup uid: rows={len(df_sel):,}")

    # per-step limit
    if args.per_step_limit is not None and args.per_step_limit > 0:
        df_sel = (df_sel.sort_values(["global_step","byte_offset"], kind="mergesort")
                        .groupby("global_step", sort=False)
                        .head(int(args.per_step_limit))
                        .reset_index(drop=True))
        print(f"[INFO] After per-step-limit={args.per_step_limit}: rows={len(df_sel):,}")

    # global limit
    if args.limit is not None and args.limit > 0:
        df_sel = df_sel.sort_values(["global_step","byte_offset"], kind="mergesort")
        df_sel = df_sel.head(int(args.limit))
        print(f"[INFO] After limit={args.limit}: rows={len(df_sel):,}")
    else:
        df_sel = df_sel.sort_values(["global_step","byte_offset"], kind="mergesort")

    # 3) 导出 JSONL（筛选结果）
    print(f"[INFO] Export selected samples to JSONL: {args.out_jsonl}")
    t1 = time.time()
    with source_jsonl.open("rb", buffering=1024*1024) as fsrc, \
         args.out_jsonl.open("wb") as fdst:
        for _, row in df_sel.iterrows():
            off = int(row["byte_offset"])
            ln = int(row["byte_len"])
            b = os.pread(fsrc.fileno(), ln, off)
            fdst.write(b)
    print(f"[INFO] JSONL export done, rows={len(df_sel):,}, took {time.time()-t1:.1f}s")

    # 4) 可视化
    if args.no_plot:
        print("[INFO] --no-plot set, skip plotting.")
        return

    # 如果筛选结果为空 & 确实应用了过滤条件 -> plot 用全体样本
    if df_sel.empty:
        if filter_applied:
            print("[WARN] 过滤条件太严格，选出 0 行；Plot 将回退使用全体样本。")
            df_plot = df.copy()
            jitter_seed_plot = jitter_seed_full
        else:
            print("[WARN] 没有任何样本（features 本身为空？），跳过 Plot。")
            return
    else:
        df_plot = df_sel
        jitter_seed_plot = jitter_seed_full[df_sel.index.to_numpy()]

    print(f"[INFO] Plotting on rows={len(df_plot):,}")

    steps = df_plot["global_step"].astype(int).to_numpy()
    cls_id = df_plot["cls_id"].astype(int).to_numpy()

    # 数值列（候选 metrics）
    numeric_cols = [c for c in df_plot.columns
                    if pd.api.types.is_numeric_dtype(df_plot[c])]
    base_exclude = {"global_step", "line_no", "byte_offset", "byte_len",
                    "valid", "kept_flag", "cls_id", "sample_idx"}
    metric_candidates = [c for c in numeric_cols if c not in base_exclude]

    # 根据 --metrics 决定要画哪些
    metrics_arg = args.metrics.strip().lower()
    metrics_training: List[str] = []
    metrics_text: List[str] = []
    metrics_lang_emb: List[str] = []

    # 先分类所有候选
    for name in metric_candidates:
        cat = categorize_metric(name)
        if cat == "training":
            metrics_training.append(name)
        elif cat == "text":
            metrics_text.append(name)
        else:
            metrics_lang_emb.append(name)

    if metrics_arg == "all":
        pass  # 三类都用
    elif metrics_arg in {"training","text","lang_emb"}:
        if metrics_arg != "training":
            metrics_training = []
        if metrics_arg != "text":
            metrics_text = []
        if metrics_arg != "lang_emb":
            metrics_lang_emb = []
    else:
        # 显式 metrics 列表
        wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
        wanted_set = set(wanted)
        metrics_training = [m for m in metrics_training if m in wanted_set]
        metrics_text     = [m for m in metrics_text     if m in wanted_set]
        metrics_lang_emb = [m for m in metrics_lang_emb if m in wanted_set]

    if not (metrics_training or metrics_text or metrics_lang_emb):
        print("[WARN] 没有任何可画的 metrics（可能列名错误或者没有数值列），跳过 Plot。")
        return

    # 创建三个子文件夹
    training_dir = plot_dir / "training"
    text_dir     = plot_dir / "text"
    lang_dir     = plot_dir / "lang_emb"
    training_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    lang_dir.mkdir(parents=True, exist_ok=True)

    # 真正绘图
    def _plot_group(metric_list: List[str], subdir: Path, group_name: str):
        if not metric_list:
            print(f"[INFO] No metrics for group '{group_name}', skip.")
            return
        print(f"[INFO] Plotting group '{group_name}' ({len(metric_list)} metrics) -> {subdir}")
        for m in metric_list:
            vals = df_plot[m].astype("float64").to_numpy()
            out_png = subdir / f"{m}.png"
            scatter_one_metric(
                out_png=out_png,
                metric_name=m,
                values=vals,
                steps=steps,
                cls_id=cls_id,
                jitter_seeds=jitter_seed_plot,
                agg=args.agg,
                jitter=args.jitter,
                point_size=args.point_size,
                alpha=args.alpha,
                y_label=pretty_metric_name(m),
                title=pretty_metric_name(m),
            )

    _plot_group(metrics_training, training_dir, "training")
    _plot_group(metrics_text,     text_dir,     "text")
    _plot_group(metrics_lang_emb, lang_dir,     "lang_emb")

    print(f"[DONE] Selection + plotting finished.\n"
          f"  JSONL saved to: {args.out_jsonl}\n"
          f"  Plots under:    {plot_dir} (subfolders: training/, text/, lang_emb/)")

if __name__ == "__main__":
    main()
