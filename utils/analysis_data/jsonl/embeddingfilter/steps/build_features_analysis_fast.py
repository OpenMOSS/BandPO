#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features_analysis.py  (Stage 3, 多进程版)

从:
  - preprocess_dir/features.parquet          (sample-level)
  - preprocess_dir/segments.parquet          (sentence-level, 含 marker/lang)
  - embed_dir/sentence_embeddings.parquet    (sentence-level, 向量)

做 per-sample 的 embedding 分析:
  - 语言分布: H_lang, num_lang_major, code_switch_count
  - 冗余/聚类: R_dup, cluster_count, rho_max, plateau_len_max, loopiness
  - marker 子集 (SELF_REFLECT/对比/结论) 和 wait-like 子集的相同指标
  - RamblingScore (车轱辘话打分)

并把结果 merge 回 features.parquet，输出:
  - out_dir/features_analysis.parquet
"""

from __future__ import annotations
from pathlib import Path
import argparse, os, sys, time, math, json
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp

# ------- 与 Stage1 对齐的 marker bitmask -------
MARKER_SELF_REFLECT = 1
MARKER_CONTRAST     = 2
MARKER_CONCLUDE     = 4
# worker 输出的列名统一在这里维护
WORKER_METRIC_COLUMNS = [
    "sample_idx",
    "H_lang", "num_lang_major", "code_switch_count",
    "all_r_dup", "all_cluster_count", "all_cluster_size_max", "all_rho_max", "all_loopiness", "all_plateau_len_max",
    "marker_r_dup", "marker_cluster_count", "marker_cluster_size_max", "marker_rho_max", "marker_loopiness", "marker_plateau_len_max",
    "wait_r_dup", "wait_cluster_count", "wait_cluster_size_max", "wait_rho_max", "wait_loopiness", "wait_plateau_len_max",
    "embedding_truncated",
]

# ------- 全局变量 (在 fork 的子进程中共享，只读) -------
GLOBAL_SEGMENTS_PATH: str | None = None
GLOBAL_EMB_PATH: str | None = None
GLOBAL_VALID_SAMPLE_SET: set[int] | None = None
GLOBAL_PARAMS: Dict[str, Any] | None = None

# 进程内缓存的 ParquetFile 对象（lazy init，一次打开，多次读）
PF_SEG: pq.ParquetFile | None = None
PF_EMB: pq.ParquetFile | None = None


# ===================== 辅助函数 =====================

def compute_lang_stats(sub: pd.DataFrame) -> Tuple[float, int, int]:
    """
    对某个 sample 的 sentence 子集 (sub) 计算:
      - H_lang: 语言熵 (按字符数加权)
      - num_lang_major: 占比 >0.1 的语言种类数
      - code_switch_count: 相邻句子 lang_main 变化次数
    依赖列:
      - lang_main (str)
      - sent_char_len (float)
    """
    if sub.empty:
        return (float("nan"), 0, 0)

    langs = sub["lang_main"].fillna("unk").astype(str).values
    char_lens = sub["sent_char_len"].fillna(0.0).values.astype(float)
    char_lens[char_lens <= 0] = 1.0

    total_chars = float(char_lens.sum())
    if total_chars <= 0:
        return (float("nan"), 0, 0)

    lang2chars: Dict[str, float] = {}
    for lang, c in zip(langs, char_lens):
        lang2chars[lang] = lang2chars.get(lang, 0.0) + float(c)
    probs = np.array([v / total_chars for v in lang2chars.values()], dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        H_lang = float("nan")
    else:
        H_lang = float(-(probs * np.log(probs)).sum())

    num_lang_major = int((np.array(list(lang2chars.values())) / total_chars > 0.1).sum())

    code_switch = 0
    if len(langs) > 1:
        for i in range(1, len(langs)):
            if langs[i] != langs[i - 1]:
                code_switch += 1

    return (float(H_lang), int(num_lang_major), int(code_switch))


def compute_subset_from_gram(
    G: np.ndarray,
    emb: np.ndarray,
    sec_idx: np.ndarray,
    sent_idx: np.ndarray,
    mask: np.ndarray,
    dup_threshold: float,
    cluster_threshold: float,
    plateau_min_len: int,
    loop_eps: float,
) -> Dict[str, float]:
    """
    在一个 sample 内，对 emb / Gram 矩阵的某个子集 mask 计算：
      - r_dup: 近重复比例
      - cluster_count, cluster_size_max, rho_max
      - loopiness
      - plateau_len_max
    """
    result = {
        "r_dup": float("nan"),
        "cluster_count": float("nan"),
        "cluster_size_max": float("nan"),
        "rho_max": float("nan"),
        "loopiness": float("nan"),
        "plateau_len_max": float("nan"),
    }
    idx = np.nonzero(mask)[0]
    m = int(idx.size)
    if m == 0:
        return result

    G_sub = G[np.ix_(idx, idx)]
    emb_sub = emb[idx]
    sec_sub = sec_idx[idx]
    sent_sub = sent_idx[idx]

    if m >= 2:
        # ---- R_dup ----
        G_tmp = G_sub.copy()
        np.fill_diagonal(G_tmp, -1.0)
        max_sim = G_tmp.max(axis=1)
        r_dup = float((max_sim >= dup_threshold).mean())

        # ---- cluster (基于相似度阈值的连通分量) ----
        adj = (G_sub >= cluster_threshold)
        np.fill_diagonal(adj, False)
        visited = np.zeros(m, dtype=bool)
        cluster_labels = -np.ones(m, dtype=int)
        cluster_sizes: List[int] = []
        c_id = 0
        for i in range(m):
            if not visited[i]:
                stack = [i]
                visited[i] = True
                cluster_labels[i] = c_id
                size = 1
                while stack:
                    u = stack.pop()
                    neighbors = np.nonzero(adj[u] & ~visited)[0]
                    if neighbors.size > 0:
                        visited[neighbors] = True
                        cluster_labels[neighbors] = c_id
                        size += int(neighbors.size)
                        stack.extend(neighbors.tolist())
                cluster_sizes.append(size)
                c_id += 1
        if cluster_sizes:
            cluster_count = len(cluster_sizes)
            cluster_size_max = max(cluster_sizes)
            rho_max = cluster_size_max / m
        else:
            cluster_count = m
            cluster_size_max = 1
            rho_max = 1.0 / m

        # ---- loopiness (语义路径弧长 / 端点弧长) ----
        order = np.lexsort((sent_sub, sec_sub))  # 先 section_idx, 再 sent_idx
        seq = emb_sub[order]
        # 邻接弧长
        dot_next = (seq[:-1] * seq[1:]).sum(axis=1)
        dot_next = np.clip(dot_next, -1.0, 1.0)
        d = np.arccos(dot_next)
        L = float(d.sum())
        # 端点位移
        dot_end = float(np.clip((seq[0] * seq[-1]).sum(), -1.0, 1.0))
        D = float(np.arccos(dot_end))
        loopiness = L / (D + loop_eps)

        # ---- plateau (cluster 序列中最长连续节点数) ----
        cl_seq = cluster_labels[order]
        plateau_max = 1
        cur_len = 1
        for i in range(1, m):
            if cl_seq[i] == cl_seq[i - 1]:
                cur_len += 1
                if cur_len > plateau_max:
                    plateau_max = cur_len
            else:
                cur_len = 1

        result.update({
            "r_dup": float(r_dup),
            "cluster_count": float(cluster_count),
            "cluster_size_max": float(cluster_size_max),
            "rho_max": float(rho_max),
            "loopiness": float(loopiness),
            "plateau_len_max": float(plateau_max),
        })
    else:
        # m == 1
        result.update({
            "r_dup": 0.0,
            "cluster_count": 1.0,
            "cluster_size_max": 1.0,
            "rho_max": 1.0,
            "loopiness": 1.0,
            "plateau_len_max": 1.0,
        })
    return result


def compute_embedding_metrics_for_sample(
    sub_emb: pd.DataFrame,
    dup_threshold: float,
    cluster_threshold: float,
    plateau_min_len: int,
    loop_eps: float,
    max_sent_per_sample: int,
) -> Dict[str, float]:
    """
    对一个 sample 的句子 + embedding 子集做 embedding 指标统计。
    输入子 DataFrame 要包含列:
      - emb: list[float] or np.ndarray
      - section_idx, sent_idx, marker_mask, marker_wait_like
    """
    result = {
        # all
        "all_r_dup": float("nan"),
        "all_cluster_count": float("nan"),
        "all_cluster_size_max": float("nan"),
        "all_rho_max": float("nan"),
        "all_loopiness": float("nan"),
        "all_plateau_len_max": float("nan"),

        # marker
        "marker_r_dup": float("nan"),
        "marker_cluster_count": float("nan"),
        "marker_cluster_size_max": float("nan"),
        "marker_rho_max": float("nan"),
        "marker_loopiness": float("nan"),
        "marker_plateau_len_max": float("nan"),

        # wait-like
        "wait_r_dup": float("nan"),
        "wait_cluster_count": float("nan"),
        "wait_cluster_size_max": float("nan"),
        "wait_rho_max": float("nan"),
        "wait_loopiness": float("nan"),
        "wait_plateau_len_max": float("nan"),

        "embedding_truncated": 0.0,
    }

    if sub_emb.empty:
        return result

    # 转成 emb 矩阵（过滤掉 None）
    emb_list = []
    valid_idx = []
    for i, e in enumerate(sub_emb["emb"].tolist()):
        if isinstance(e, (list, tuple, np.ndarray)) and len(e) > 0:
            arr = np.asarray(e, dtype=np.float32)
            emb_list.append(arr)
            valid_idx.append(i)
    if not emb_list:
        return result

    emb = np.stack(emb_list, axis=0)  # [N_valid, d]
    N_valid = emb.shape[0]
    sec_idx = sub_emb["section_idx"].to_numpy()[valid_idx].astype(np.int32)
    sent_idx = sub_emb["sent_idx"].to_numpy()[valid_idx].astype(np.int32)
    marker_mask = sub_emb["marker_mask"].to_numpy()[valid_idx].astype(np.int32)
    wait_like = sub_emb["marker_wait_like"].to_numpy()[valid_idx].astype(np.int32)

    # 超过上限则子采样，保证 O(N^2) 可控
    if max_sent_per_sample > 0 and N_valid > max_sent_per_sample:
        idx_sub = np.linspace(0, N_valid - 1, num=max_sent_per_sample, dtype=int)
        emb = emb[idx_sub]
        sec_idx = sec_idx[idx_sub]
        sent_idx = sent_idx[idx_sub]
        marker_mask = marker_mask[idx_sub]
        wait_like = wait_like[idx_sub]
        N_valid = emb.shape[0]
        result["embedding_truncated"] = 1.0

    # Gram 矩阵（cos，相当于内积）
    G = emb @ emb.T

    # ---- 全部句子 ----
    mask_all = np.ones(N_valid, dtype=bool)
    stats_all = compute_subset_from_gram(
        G, emb, sec_idx, sent_idx,
        mask_all,
        dup_threshold, cluster_threshold, plateau_min_len, loop_eps
    )
    result["all_r_dup"] = stats_all["r_dup"]
    result["all_cluster_count"] = stats_all["cluster_count"]
    result["all_cluster_size_max"] = stats_all["cluster_size_max"]
    result["all_rho_max"] = stats_all["rho_max"]
    result["all_loopiness"] = stats_all["loopiness"]
    result["all_plateau_len_max"] = stats_all["plateau_len_max"]

    # ---- marker 句子 ----
    mask_marker = marker_mask != 0
    if mask_marker.any():
        stats_marker = compute_subset_from_gram(
            G, emb, sec_idx, sent_idx,
            mask_marker,
            dup_threshold, cluster_threshold, plateau_min_len, loop_eps
        )
        result["marker_r_dup"] = stats_marker["r_dup"]
        result["marker_cluster_count"] = stats_marker["cluster_count"]
        result["marker_cluster_size_max"] = stats_marker["cluster_size_max"]
        result["marker_rho_max"] = stats_marker["rho_max"]
        result["marker_loopiness"] = stats_marker["loopiness"]
        result["marker_plateau_len_max"] = stats_marker["plateau_len_max"]

    # ---- wait-like 句子 ----
    mask_wait = wait_like.astype(bool)
    if mask_wait.any():
        stats_wait = compute_subset_from_gram(
            G, emb, sec_idx, sent_idx,
            mask_wait,
            dup_threshold, cluster_threshold, plateau_min_len, loop_eps
        )
        result["wait_r_dup"] = stats_wait["r_dup"]
        result["wait_cluster_count"] = stats_wait["cluster_count"]
        result["wait_cluster_size_max"] = stats_wait["cluster_size_max"]
        result["wait_rho_max"] = stats_wait["rho_max"]
        result["wait_loopiness"] = stats_wait["loopiness"]
        result["wait_plateau_len_max"] = stats_wait["plateau_len_max"]

    return result


def compute_rambling_score(
    r_self_sent: float,
    marker_r_dup: float,
    marker_loopiness: float,
    marker_plateau_len_max: float,
    H_lang: float,
    weights: Dict[str, float],
    loopiness_cap: float,
    plateau_k: int,
) -> float:
    """
    RamblingScore = 线性可调组合:
      - 自我打断比例 r_self_sent
      - marker 区域重复度 marker_r_dup
      - marker 区域 loopiness (裁剪)
      - marker plateau >= k 的指示
      - 语言熵 H_lang
    """
    if math.isnan(r_self_sent) or math.isnan(marker_r_dup):
        return float("nan")

    loop_component = 0.0
    if loopiness_cap > 0 and not math.isnan(marker_loopiness):
        loop_component = min(marker_loopiness, loopiness_cap) / loopiness_cap

    plateau_component = 1.0 if (not math.isnan(marker_plateau_len_max) and marker_plateau_len_max >= plateau_k) else 0.0
    H_term = H_lang if not math.isnan(H_lang) else 0.0

    score = (
        weights["self"] * r_self_sent +
        weights["dup"] * marker_r_dup +
        weights["loop"] * loop_component +
        weights["plateau"] * plateau_component +
        weights["lang"] * H_term
    )
    return float(score)


# ===================== 多进程 worker =====================

def _init_worker(segments_path: str,
                 emb_path: str,
                 valid_sample_list: List[int],
                 params: Dict[str, Any]):
    """
    spawn 模式下每个子进程启动时调用一次，把必要的全局变量写进去
    """
    global GLOBAL_SEGMENTS_PATH, GLOBAL_EMB_PATH, GLOBAL_VALID_SAMPLE_SET, GLOBAL_PARAMS
    GLOBAL_SEGMENTS_PATH = segments_path
    GLOBAL_EMB_PATH = emb_path
    # 用 set 加速 sample_idx 过滤
    GLOBAL_VALID_SAMPLE_SET = set(valid_sample_list)
    GLOBAL_PARAMS = params

def _worker_rowgroup(rg_idx: int) -> pd.DataFrame:
    """
    每个 worker 处理一个 row group:
      - 读 segments & embeddings 的该 row group
      - 过滤 sample_idx ∈ GLOBAL_VALID_SAMPLE_SET
      - 对每个 sample_idx 计算语言 + embedding 指标
    返回的 DataFrame:
      - sample_idx
      - H_lang, num_lang_major, code_switch_count
      - all_* / marker_* / wait_* / embedding_truncated
    """
    global PF_SEG, PF_EMB
    if GLOBAL_SEGMENTS_PATH is None or GLOBAL_EMB_PATH is None or GLOBAL_VALID_SAMPLE_SET is None or GLOBAL_PARAMS is None:
        raise RuntimeError("Global paths/params not initialized in worker")

    # Lazy 初始化 ParquetFile（每个进程只打开一次）
    if PF_SEG is None or PF_EMB is None:
        PF_SEG = pq.ParquetFile(GLOBAL_SEGMENTS_PATH)
        PF_EMB = pq.ParquetFile(GLOBAL_EMB_PATH)

    table_seg = PF_SEG.read_row_group(rg_idx)
    table_emb = PF_EMB.read_row_group(rg_idx)
    df_seg = table_seg.to_pandas()
    df_emb = table_emb.to_pandas()
    if len(df_seg) != len(df_emb):
        raise ValueError(f"Row group {rg_idx} segments vs embeddings row mismatch: {len(df_seg)} vs {len(df_emb)}")

    df = df_seg.copy()
    df["emb"] = df_emb["emb"]

    # 只保留我们要分析的 sample（kept/correct 过滤在主进程决定）
    df = df[df["sample_idx"].isin(GLOBAL_VALID_SAMPLE_SET)]
    # >>> 新增：打印一下这个 row group 的大小情况
    print(
        f"[Worker {os.getpid()}] rg={rg_idx}: rows_seg={len(df_seg)}, "
        f"rows_valid={len(df)}, samples_valid={df['sample_idx'].nunique()}",
        flush=True,
    )
    # <<<
    if df.empty:
        return pd.DataFrame(columns=WORKER_METRIC_COLUMNS)

    rows: List[Dict[str, Any]] = []

    for sid, sub in df.groupby("sample_idx"):
        # 1) 语言统计
        H_lang, num_lang_major, code_switch_count = compute_lang_stats(sub)

        # 2) embedding 指标
        metrics_emb = compute_embedding_metrics_for_sample(
            sub,
            dup_threshold=GLOBAL_PARAMS["dup_threshold"],
            cluster_threshold=GLOBAL_PARAMS["cluster_threshold"],
            plateau_min_len=GLOBAL_PARAMS["plateau_min_len"],
            loop_eps=GLOBAL_PARAMS["loop_eps"],
            max_sent_per_sample=GLOBAL_PARAMS["max_sent_per_sample"],
        )

        row = {
            "sample_idx": int(sid),
            "H_lang": H_lang,
            "num_lang_major": int(num_lang_major),
            "code_switch_count": int(code_switch_count),
        }
        row.update(metrics_emb)
        rows.append(row)

    # rows 可能为空（理论上不太会发生），也没关系，DataFrame 会是空表但列名一致
    return pd.DataFrame(rows, columns=WORKER_METRIC_COLUMNS)


# ===================== 主函数 =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess-dir", type=Path, required=True,
                    help="Dir containing features.parquet & segments.parquet")
    ap.add_argument("--embed-dir", type=Path, required=True,
                    help="Dir containing sentence_embeddings.parquet")
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 8) // 2),
                    help="并行进程数")

    ap.add_argument("--dup-threshold", type=float, default=0.90)
    ap.add_argument("--cluster-threshold", type=float, default=0.85)
    ap.add_argument("--plateau-min-len", type=int, default=3)
    ap.add_argument("--loop-eps", type=float, default=1e-4)
    ap.add_argument("--max-sent-per-sample", type=int, default=128,
                    help="每个 sample 做 embedding 分析时最多使用多少句子 (>0 即子采样)")

    ap.add_argument("--only-kept", action="store_true",
                    help="只对 kept_flag==1 的 sample 做分析")
    ap.add_argument("--only-correct", action="store_true",
                    help="只对 cls_id==1 的 sample 做分析")

    # RamblingScore 的权重 & 超参
    ap.add_argument("--rambling-weight-self", type=float, default=1.0)
    ap.add_argument("--rambling-weight-dup", type=float, default=1.0)
    ap.add_argument("--rambling-weight-loop", type=float, default=1.0)
    ap.add_argument("--rambling-weight-plateau", type=float, default=1.0)
    ap.add_argument("--rambling-weight-lang", type=float, default=0.1)
    ap.add_argument("--rambling-loopiness-cap", type=float, default=5.0)
    ap.add_argument("--rambling-plateau-k", type=int, default=4)

    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = args.preprocess_dir / "features.parquet"
    segments_path = args.preprocess_dir / "segments.parquet"
    emb_path = args.embed_dir / "sentence_embeddings.parquet"
    out_features_path = out_dir / "features_analysis.parquet"
    meta_path = out_dir / "meta_analysis.json"

    print(f"[Stage3] Loading sample-level features from {samples_path}")
    t0 = time.time()
    samples_df = pd.read_parquet(samples_path)
    n_samples = len(samples_df)
    print(f"[Stage3] Loaded {n_samples:,} samples in {time.time()-t0:.1f}s")

    if "sample_idx" not in samples_df.columns:
        raise ValueError("features.parquet must contain 'sample_idx' column")

    # 选择需要做 embedding 分析的 sample
    mask_valid = np.ones(n_samples, dtype=bool)
    if args.only_kept and "kept_flag" in samples_df.columns:
        mask_valid &= (samples_df["kept_flag"].to_numpy().astype(int) == 1)
    if args.only_correct and "cls_id" in samples_df.columns:
        mask_valid &= (samples_df["cls_id"].to_numpy().astype(int) == 1)
    valid_sample_idxs = samples_df.loc[mask_valid, "sample_idx"].astype(int).unique().tolist()
    print(f"[Stage3] Will analyze {len(valid_sample_idxs):,} samples (after kept/correct filter)")

    # 先准备一个 metrics DataFrame 的骨架: 所有列为 NaN/0，后面覆盖 valid 部分
    # 这些列名要与 worker 输出一致 + rambling_score
    metric_cols = [
        "H_lang", "num_lang_major", "code_switch_count",
        "all_r_dup", "all_cluster_count", "all_cluster_size_max", "all_rho_max", "all_loopiness", "all_plateau_len_max",
        "marker_r_dup", "marker_cluster_count", "marker_cluster_size_max", "marker_rho_max", "marker_loopiness", "marker_plateau_len_max",
        "wait_r_dup", "wait_cluster_count", "wait_cluster_size_max", "wait_rho_max", "wait_loopiness", "wait_plateau_len_max",
        "embedding_truncated",
    ]

    # metrics_all：最后按 sample_idx 对齐
    metrics_all = pd.DataFrame({
        "sample_idx": samples_df["sample_idx"].astype(int).values
    })
    for col in metric_cols:
        if col in {"num_lang_major", "code_switch_count"}:
            metrics_all[col] = np.zeros(n_samples, dtype=np.int32)
        else:
            metrics_all[col] = np.full(n_samples, np.nan, dtype=np.float32)

    # # 全局参数 (给子进程)
    # global GLOBAL_SEGMENTS_PATH, GLOBAL_EMB_PATH, GLOBAL_VALID_SAMPLE_SET, GLOBAL_PARAMS
    # GLOBAL_SEGMENTS_PATH = str(segments_path)
    # GLOBAL_EMB_PATH = str(emb_path)
    # GLOBAL_VALID_SAMPLE_SET = set(valid_sample_idxs)
    # GLOBAL_PARAMS = {
    #     "dup_threshold": args.dup_threshold,
    #     "cluster_threshold": args.cluster_threshold,
    #     "plateau_min_len": args.plateau_min_len,
    #     "loop_eps": args.loop_eps,
    #     "max_sent_per_sample": args.max_sent_per_sample,
    # }
    # 准备传给 worker 的参数（spawn 模式下通过 initializer 传递）
    worker_params = {
        "dup_threshold": args.dup_threshold,
        "cluster_threshold": args.cluster_threshold,
        "plateau_min_len": args.plateau_min_len,
        "loop_eps": args.loop_eps,
        "max_sent_per_sample": args.max_sent_per_sample,
    }

    # 主进程先读一遍 Parquet metadata，拿 num_row_groups
    pf_seg_main = pq.ParquetFile(str(segments_path))
    pf_emb_main = pq.ParquetFile(str(emb_path))
    if pf_seg_main.num_row_groups != pf_emb_main.num_row_groups:
        raise ValueError("segments.parquet and sentence_embeddings.parquet have different #row_groups")
    num_row_groups = pf_seg_main.num_row_groups
    print(f"[Stage3] segments / embeddings row groups = {num_row_groups}")

    # # 多进程跑 rowgroup
    # print(f"[Stage3] Running analysis with {args.procs} processes ...")
    # t1 = time.time()
    # start_method = "spawn" if sys.platform == "win32" else "fork"
    # ctx = mp.get_context(start_method)

    # rg_indices = list(range(num_row_groups))
    # partial_metrics_list: List[pd.DataFrame] = []

    # with ctx.Pool(processes=args.procs) as pool:
    #     for i, df_rg in enumerate(pool.imap_unordered(_worker_rowgroup, rg_indices, chunksize=1)):
    # 多进程跑 rowgroup
    # 1) 固定用 spawn，避免 pyarrow + fork 的潜在坑
    # 2) 进程数不超过 row_groups 数量
    max_procs = min(args.procs, num_row_groups)
    print(f"[Stage3] Running analysis with {max_procs} processes (row_groups={num_row_groups}) ...")
    t1 = time.time()
    ctx = mp.get_context("spawn")

    rg_indices = list(range(num_row_groups))
    partial_metrics_list: List[pd.DataFrame] = []

    # 通过 initializer 把路径 & 参数传给每个子进程
    with ctx.Pool(
        processes=max_procs,
        initializer=_init_worker,
        initargs=(str(segments_path), str(emb_path), valid_sample_idxs, worker_params),
    ) as pool:
        for i, df_rg in enumerate(pool.imap_unordered(_worker_rowgroup, rg_indices, chunksize=1)):
            if not df_rg.empty:
                partial_metrics_list.append(df_rg)
            if (i + 1) % 5 == 0 or (i + 1) == num_row_groups:
                print(f"  [Stage3] row groups processed: {i+1}/{num_row_groups}")

    if partial_metrics_list:
        metrics_concat = pd.concat(partial_metrics_list, ignore_index=True)
        # 对 sample_idx 去重（理论上不会重复）
        metrics_concat = metrics_concat.drop_duplicates(subset=["sample_idx"], keep="last")
        metrics_concat.set_index("sample_idx", inplace=True)

        # 按 sample_idx 对齐覆盖
        idx_series = samples_df["sample_idx"].astype(int)
        for col in metric_cols:
            if col not in metrics_concat.columns:
                continue
            aligned = metrics_concat[col].reindex(idx_series).to_numpy()
            metrics_all[col] = aligned
    else:
        print("[Stage3] No metrics computed (no valid samples?).")

    print(f"[Stage3] Embedding-based metrics computed in {time.time()-t1:.1f}s")

    # 把 metrics_all merge 回 samples_df
    for col in metric_cols:
        samples_df[col] = metrics_all[col].values

    # 计算 RamblingScore（在主进程做）
    weights = {
        "self": args.rambling_weight_self,
        "dup": args.rambling_weight_dup,
        "loop": args.rambling_weight_loop,
        "plateau": args.rambling_weight_plateau,
        "lang": args.rambling_weight_lang,
    }
    loop_cap = args.rambling_loopiness_cap
    plateau_k = args.rambling_plateau_k

    print("[Stage3] Computing rambling_score ...")
    # 准备数组
    rambling_scores = np.full(n_samples, np.nan, dtype=np.float32)

    num_sent_arr = samples_df.get("num_sentences", pd.Series(np.nan, index=samples_df.index)).to_numpy().astype(float)
    num_self_arr = samples_df.get("num_self_reflect_sentences", pd.Series(np.nan, index=samples_df.index)).to_numpy().astype(float)
    H_lang_arr = samples_df["H_lang"].to_numpy().astype(float)
    marker_r_dup_arr = samples_df["marker_r_dup"].to_numpy().astype(float)
    marker_loopiness_arr = samples_df["marker_loopiness"].to_numpy().astype(float)
    marker_plateau_arr = samples_df["marker_plateau_len_max"].to_numpy().astype(float)

    for i in range(n_samples):
        if not mask_valid[i]:
            continue
        num_sent = num_sent_arr[i]
        num_self = num_self_arr[i]
        if num_sent > 0 and not math.isnan(num_self):
            r_self = num_self / num_sent
        else:
            r_self = float("nan")

        rs = compute_rambling_score(
            r_self,
            marker_r_dup_arr[i],
            marker_loopiness_arr[i],
            marker_plateau_arr[i],
            H_lang_arr[i],
            weights=weights,
            loopiness_cap=loop_cap,
            plateau_k=plateau_k,
        )
        rambling_scores[i] = rs

    samples_df["rambling_score"] = rambling_scores

    print(f"[Stage3] Writing features_analysis.parquet to {out_features_path}")
    t2 = time.time()
    samples_df.to_parquet(out_features_path, index=False)
    print(f"[Stage3] Wrote features_analysis in {time.time()-t2:.1f}s")

    meta = {
        "preprocess_dir": str(args.preprocess_dir),
        "embed_dir": str(args.embed_dir),
        "out_features_path": str(out_features_path),
        "dup_threshold": args.dup_threshold,
        "cluster_threshold": args.cluster_threshold,
        "plateau_min_len": args.plateau_min_len,
        "loop_eps": args.loop_eps,
        "max_sent_per_sample": args.max_sent_per_sample,
        "only_kept": args.only_kept,
        "only_correct": args.only_correct,
        "rambling_weights": weights,
        "rambling_loopiness_cap": args.rambling_loopiness_cap,
        "rambling_plateau_k": args.rambling_plateau_k,
        "procs": args.procs,
        "build_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Stage3] Wrote meta: {meta_path}")

    print("[Stage3 Summary]")
    print(f"  samples total      = {n_samples:,}")
    print(f"  samples analyzed   = {mask_valid.sum():,}")
    print(f"  output path        = {out_features_path}")

if __name__ == "__main__":
    main()
