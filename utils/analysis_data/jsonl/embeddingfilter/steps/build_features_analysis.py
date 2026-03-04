#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features_analysis.py

Stage 3: 从
  - features.parquet          (sample-level, from build_features_preprocess.py)
  - segments.parquet          (sentence-level, from build_features_preprocess.py)
  - sentence_embeddings.parquet (sentence-level, from build_features_embedding.py)

中加载数据，对每个 sample 做 embedding-based 语义分析和统计：
  - 语言分布: H_lang, num_lang_major, code_switch_count
  - 冗余/聚类指标: R_dup, cluster_count, rho_max 等
  - 轨迹 loopiness (L / D), plateau_len_max 等
  - marker 子集 (有 SELF_REFLECT/CONTRAST/CONCLUDE 的句子) 的同样指标
  - 一个简单可调的 RamblingScore

输出:
  - features_analysis.parquet: 在原有 features 的基础上增加这些新列
"""

from __future__ import annotations
from pathlib import Path
import argparse, os, time, math, json
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

# bitmask 定义要和 preprocess 对齐
MARKER_SELF_REFLECT = 1
MARKER_CONTRAST     = 2
MARKER_CONCLUDE     = 4

def compute_lang_stats(sub: pd.DataFrame) -> Tuple[float, int, int]:
    """
    对一个 sample 的 sentence-level 子 DataFrame:
      - lang_main: str
      - sent_char_len: float
    计算:
      - H_lang: 语言熵
      - num_lang_major: 占比 >0.1 的语言数
      - code_switch_count: 相邻句子语言不同的次数
    """
    if sub.empty:
        return (float("nan"), 0, 0)

    langs = sub["lang_main"].fillna("unk").astype(str).values
    char_lens = sub["sent_char_len"].fillna(0.0).values.astype(float)
    char_lens[char_lens <= 0] = 1.0  # 避免全 0

    # 统计每种语言按字符数加权的占比
    total_chars = char_lens.sum()
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

    # code_switch_count: 按 sentence 顺序统计 lang_main 变化次数
    code_switch = 0
    if len(langs) > 1:
        for i in range(1, len(langs)):
            if langs[i] != langs[i-1]:
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
    从全局 Gram 矩阵 G 中抽取 mask 子集的指标:
      - R_dup: 近重复比例
      - cluster_count / cluster_size_max / rho_max
      - loopiness
      - plateau_len_max
    注意:
      - emb: [N_valid, d]
      - sec_idx/sent_idx: [N_valid]
      - mask: [N_valid] bool, 表示对哪个点取子集
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

    # subset Gram
    G_sub = G[np.ix_(idx, idx)]
    emb_sub = emb[idx]
    sec_sub = sec_idx[idx]
    sent_sub = sent_idx[idx]

    if m >= 2:
        # R_dup
        G_tmp = G_sub.copy()
        np.fill_diagonal(G_tmp, -1.0)
        max_sim = G_tmp.max(axis=1)
        r_dup = float((max_sim >= dup_threshold).mean())

        # cluster: adjacency by threshold
        adj = (G_sub >= cluster_threshold)
        np.fill_diagonal(adj, False)
        visited = np.zeros(m, dtype=bool)
        cluster_labels = -np.ones(m, dtype=int)
        cluster_sizes = []
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

        # 轨迹 loopiness
        order = np.lexsort((sent_sub, sec_sub))  # 按 section_idx, sent_idx 排序
        seq = emb_sub[order]  # [m, d]
        # 邻接弧长
        dot_next = (seq[:-1] * seq[1:]).sum(axis=1)
        dot_next = np.clip(dot_next, -1.0, 1.0)
        d = np.arccos(dot_next)
        L = float(d.sum())
        # 端点位移
        dot_end = float(np.clip((seq[0] * seq[-1]).sum(), -1.0, 1.0))
        D = float(np.arccos(dot_end))
        loopiness = L / (D + loop_eps)

        # plateau: cluster_labels 按顺序
        cl_seq = cluster_labels[order]
        plateau_max = 1
        cur_len = 1
        for i in range(1, m):
            if cl_seq[i] == cl_seq[i-1]:
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
    对一个 sample 的 sentence-level + embedding 子 DataFrame:
      - emb: list[float], len=d
      - section_idx, sent_idx, marker_mask

    计算:
      - all_sent_*: 全句子子集的 R_dup, cluster, loopiness, plateau
      - marker_sent_*: marker_mask != 0 子集的同类指标
      - wait_sent_*: marker_wait_like==1 子集指标（如果你在 preprocess 里有该列）

    max_sent_per_sample: 超过这个上限时子采样，避免 O(N^2) 爆炸。
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

    # 把 embedding 列转成矩阵
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

    # 如果句子过多，子采样
    if max_sent_per_sample > 0 and N_valid > max_sent_per_sample:
        idx_sub = np.linspace(0, N_valid - 1, num=max_sent_per_sample, dtype=int)
        emb = emb[idx_sub]
        sec_idx = sec_idx[idx_sub]
        sent_idx = sent_idx[idx_sub]
        marker_mask = marker_mask[idx_sub]
        wait_like = wait_like[idx_sub]
        N_valid = emb.shape[0]
        result["embedding_truncated"] = 1.0

    # Gram matrix
    G = emb @ emb.T  # cos，因为之前 Stage2 做了 L2 normalize

    # all sentences
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

    # marker sentences
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

    # wait-like sentences
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
    一个简单的可调 RamblingScore:
      - 自我打断比例 r_self_sent
      - marker 区域的重复度 marker_r_dup
      - marker 区域 loopiness (裁剪到 loopiness_cap)
      - marker plateau_len_max 是否 >= plateau_k
      - 语言熵 H_lang
    """
    if math.isnan(r_self_sent) or math.isnan(marker_r_dup):
        return float("nan")

    loop_component = min(marker_loopiness, loopiness_cap) / loopiness_cap if loopiness_cap > 0 and not math.isnan(marker_loopiness) else 0.0
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess-dir", type=Path, required=True,
                    help="Directory containing features.parquet and segments.parquet")
    ap.add_argument("--embed-dir", type=Path, required=True,
                    help="Directory containing sentence_embeddings.parquet")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dup-threshold", type=float, default=0.90)
    ap.add_argument("--cluster-threshold", type=float, default=0.85)
    ap.add_argument("--plateau-min-len", type=int, default=3)
    ap.add_argument("--loop-eps", type=float, default=1e-4)
    ap.add_argument("--max-sent-per-sample", type=int, default=128,
                    help="上限超过时在 embedding 分析中对子采样，避免 O(N^2) 过大")
    ap.add_argument("--only-kept", action="store_true",
                    help="只对 kept_flag==1 的 sample 做 embedding 分析")
    ap.add_argument("--only-correct", action="store_true",
                    help="只对 cls_id==1 (正确样本) 做 embedding 分析")
    # RamblingScore 权重
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

    # 为新指标预先分配列 (np.nan)
    new_cols = {
        "H_lang": np.full(n_samples, np.nan, dtype=np.float32),
        "num_lang_major": np.zeros(n_samples, dtype=np.int32),
        "code_switch_count": np.zeros(n_samples, dtype=np.int32),

        "all_r_dup": np.full(n_samples, np.nan, dtype=np.float32),
        "all_cluster_count": np.full(n_samples, np.nan, dtype=np.float32),
        "all_cluster_size_max": np.full(n_samples, np.nan, dtype=np.float32),
        "all_rho_max": np.full(n_samples, np.nan, dtype=np.float32),
        "all_loopiness": np.full(n_samples, np.nan, dtype=np.float32),
        "all_plateau_len_max": np.full(n_samples, np.nan, dtype=np.float32),

        "marker_r_dup": np.full(n_samples, np.nan, dtype=np.float32),
        "marker_cluster_count": np.full(n_samples, np.nan, dtype=np.float32),
        "marker_cluster_size_max": np.full(n_samples, np.nan, dtype=np.float32),
        "marker_rho_max": np.full(n_samples, np.nan, dtype=np.float32),
        "marker_loopiness": np.full(n_samples, np.nan, dtype=np.float32),
        "marker_plateau_len_max": np.full(n_samples, np.nan, dtype=np.float32),

        "wait_r_dup": np.full(n_samples, np.nan, dtype=np.float32),
        "wait_cluster_count": np.full(n_samples, np.nan, dtype=np.float32),
        "wait_cluster_size_max": np.full(n_samples, np.nan, dtype=np.float32),
        "wait_rho_max": np.full(n_samples, np.nan, dtype=np.float32),
        "wait_loopiness": np.full(n_samples, np.nan, dtype=np.float32),
        "wait_plateau_len_max": np.full(n_samples, np.nan, dtype=np.float32),

        "embedding_truncated": np.zeros(n_samples, dtype=np.float32),
        "rambling_score": np.full(n_samples, np.nan, dtype=np.float32),
    }

    # 索引：sample_idx -> row index in samples_df
    # 我们在 preprocess 中 sample_idx = line_no-1，且 features.parquet 里也有 sample_idx
    if "sample_idx" not in samples_df.columns:
        raise ValueError("features.parquet 必须包含 'sample_idx' 列")
    sample_idx_to_row = dict(zip(samples_df["sample_idx"].astype(int).tolist(), range(n_samples)))

    pf_seg = pq.ParquetFile(str(segments_path))
    pf_emb = pq.ParquetFile(str(emb_path))
    if pf_seg.num_row_groups != pf_emb.num_row_groups:
        raise ValueError("segments.parquet 与 sentence_embeddings.parquet 的 row group 数量不一致")

    weights = {
        "self": args.rambling_weight_self,
        "dup": args.rambling_weight_dup,
        "loop": args.rambling_weight_loop,
        "plateau": args.rambling_weight_plateau,
        "lang": args.rambling_weight_lang,
    }

    print(f"[Stage3] Start per-sample analysis (row-group streaming). Row groups = {pf_seg.num_row_groups}")
    t1 = time.time()
    n_samples_processed = 0

    for rg_idx in range(pf_seg.num_row_groups):
        print(f"[Stage3] Row group {rg_idx+1}/{pf_seg.num_row_groups} ...")
        table_seg = pf_seg.read_row_group(rg_idx)
        table_emb = pf_emb.read_row_group(rg_idx)
        df_seg = table_seg.to_pandas()
        df_emb = table_emb.to_pandas()

        if len(df_seg) != len(df_emb):
            raise ValueError(f"Row group {rg_idx} 中 segments 与 embeddings 行数不一致")

        # 合并句子信息和 embedding（按行对齐）
        df = df_seg.copy()
        df["emb"] = df_emb["emb"]

        # 按 sample_idx 分组
        for sid, sub in df.groupby("sample_idx"):
            if sid not in sample_idx_to_row:
                continue
            row_idx = sample_idx_to_row[sid]

            kept_flag = int(samples_df.iloc[row_idx].get("kept_flag", -1))
            cls_id = int(samples_df.iloc[row_idx].get("cls_id", 2))
            if args.only_kept and kept_flag != 1:
                continue
            if args.only_correct and cls_id != 1:
                continue

            # 1) 语言统计
            H_lang, num_lang_major, code_switch_count = compute_lang_stats(sub)
            new_cols["H_lang"][row_idx] = H_lang
            new_cols["num_lang_major"][row_idx] = num_lang_major
            new_cols["code_switch_count"][row_idx] = code_switch_count

            # 2) embedding-based 指标
            metrics_emb = compute_embedding_metrics_for_sample(
                sub,
                dup_threshold=args.dup_threshold,
                cluster_threshold=args.cluster_threshold,
                plateau_min_len=args.plateau_min_len,
                loop_eps=args.loop_eps,
                max_sent_per_sample=args.max_sent_per_sample,
            )

            for k, v in metrics_emb.items():
                if k in new_cols:
                    new_cols[k][row_idx] = v

            # 3) RamblingScore
            # 需要 r_self_sent, marker_r_dup, marker_loopiness, marker_plateau_len_max, H_lang
            num_sentences = float(samples_df.iloc[row_idx].get("num_sentences", np.nan))
            num_self = float(samples_df.iloc[row_idx].get("num_self_reflect_sentences", np.nan))
            if num_sentences > 0 and not math.isnan(num_self):
                r_self_sent = num_self / num_sentences
            else:
                r_self_sent = float("nan")

            marker_r_dup = metrics_emb.get("marker_r_dup", float("nan"))
            marker_loopiness = metrics_emb.get("marker_loopiness", float("nan"))
            marker_plateau_len_max = metrics_emb.get("marker_plateau_len_max", float("nan"))

            rambling_score = compute_rambling_score(
                r_self_sent,
                marker_r_dup,
                marker_loopiness,
                marker_plateau_len_max,
                H_lang,
                weights=weights,
                loopiness_cap=args.rambling_loopiness_cap,
                plateau_k=args.rambling_plateau_k,
            )
            new_cols["rambling_score"][row_idx] = rambling_score

            n_samples_processed += 1

        print(f"  [Stage3] processed samples so far: {n_samples_processed:,}")

    # 把新列写回 samples_df
    for col, arr in new_cols.items():
        samples_df[col] = arr

    print(f"[Stage3] Writing analysis features to {out_features_path}")
    t2 = time.time()
    samples_df.to_parquet(out_features_path, index=False)
    print(f"[Stage3] Wrote features_analysis.parquet in {time.time()-t2:.1f}s")
    print(f"[Stage3] Total analysis time: {time.time()-t1:.1f}s")

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
        "build_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Stage3] Wrote meta: {meta_path}")

    print("[Stage3 Summary]")
    print(f"  samples (total)        = {n_samples:,}")
    print(f"  samples (processed)    = {n_samples_processed:,}")
    print(f"  features_analysis path = {out_features_path}")

if __name__ == "__main__":
    main()
