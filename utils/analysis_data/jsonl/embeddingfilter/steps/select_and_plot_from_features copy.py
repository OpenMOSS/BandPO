#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_and_plot_from_features.py

基于 build_features_preprocess.py / build_features_analysis.py 生成的 Parquet 做：

1) 样本筛选：
   - --cls {any,correct,wrong,no_judge}
   - --kept-or-dropped {all,kept,dropped}
   - --chinese-or-english {all,chinese,english}
   - --step-min / --step-max
   - --where 'pandas 表达式'（可用所有列）
   - --ge / --le / --eq 'col:val'（同 select_samples_from_features.py）
   - --dedup-uid
   - --limit / --per-step-limit

2) 导出筛选结果的 JSONL：
   - 从 meta_preprocess.json 中定位原始 JSONL
   - 按 (global_step, byte_offset) 排序
   - 用 os.pread() 按 byte_offset 精确 copy 行

3) 可视化（可选，用 --no-plot 关闭）：
   - 横轴：global_step
   - 纵轴：各种 metrics（默认一组预设指标，或 --metrics 自己选）
   - 每个样本一个散点（jitter），按 cls_id 上色（correct / wrong / no_judge）
   - 每个 step 的 agg 值（mean / median / sum）画一条黑色折线

如果你不指定任何过滤条件（所有开关都是默认值），就是“全数据”筛选 + 全数据可视化。
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, os, sys, time, math, hashlib, re
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============== 小工具函数（和之前脚本保持风格） ===============

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
    # map salt to [0,1) 再映射到 [-scale, scale]
    r = (salts % 10_000_019) / 10_000_019.0
    return (r - 0.5) * 2.0 * scale


# =============== 画图相关 ===============

COLOR_CORRECT = "blue"
COLOR_WRONG   = "red"
COLOR_UNKNOWN = "purple"
COLOR_AGG     = "black"

def aggregate_per_step(values: np.ndarray,
                       steps: np.ndarray,
                       agg: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 filtered 子集的 (steps, values) 进行 per-step 聚合。
    values 中的 NaN 会被忽略；若某个 step 全是 NaN，则跳过。
    """
    df = pd.DataFrame({"step": steps, "val": values})
    # 只保留非 NaN
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
    画一张“沙箱图”：scatter (jitter) + per-step 聚合折线。
    """
    t0 = time.time()
    fig, ax = plt.subplots(figsize=(14, 6))

    # 只画 finite 的点
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

    # wrong / correct / no_judge 分颜色 + marker
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

    # legend 去重
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="best", fontsize="small", frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    dt = time.time() - t0
    print(f"[PLOT] {metric_name} -> {out_png}  (took {dt:.1f}s)")


# =============== 主逻辑：筛选 + 导出 + 画图 ===============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, required=True,
                    help="build_features_preprocess / analysis 的输出目录 (包含 features*.parquet 和 meta_preprocess.json)")
    ap.add_argument("--features-file", type=str, default=None,
                    help="默认优先用 features_analysis.parquet；若不存在则用 features.parquet。也可手动指定文件名。")
    ap.add_argument("--out-jsonl", type=Path, required=True,
                    help="筛选后的样本 JSONL 输出路径（原始行的 byte 级 copy）")
    ap.add_argument("--plot-dir", type=Path, default=None,
                    help="可视化输出目录（默认为 --features-dir）")

    # 筛选条件
    ap.add_argument("--cls", type=str, default="any",
                    choices=["any","correct","wrong","no_judge"],
                    help="按 cls_id 过滤：1=correct,0=wrong,2=no_judge")
    ap.add_argument("--kept-or-dropped", type=str, default="all",
                    choices=["all","kept","dropped"],
                    help="按 filter_state 过滤：kept/dropped/all（features 中用 kept_flag）")
    ap.add_argument("--chinese-or-english", type=str, default="all",
                    choices=["all","chinese","english"],
                    help="基于 contain_chinese 列 (1/0/-1)")
    ap.add_argument("--step-min", type=int, default=None)
    ap.add_argument("--step-max", type=int, default=None)
    ap.add_argument("--where", type=str, default=None,
                    help="pandas query 表达式，例如 'non_ascii_ratio>0.15 and repeat_word_run_max>10'")
    ap.add_argument("--ge", action="append", default=[],
                    help="阈值条件 'col:val' 表示 col >= val，可以多次使用")
    ap.add_argument("--le", action="append", default=[],
                    help="阈值条件 'col:val' 表示 col <= val，可以多次使用")
    ap.add_argument("--eq", action="append", default=[],
                    help="阈值条件 'col:val' 表示 col == val，可以多次使用")
    ap.add_argument("--logic", type=str, default="and", choices=["and","or"],
                    help="--ge/--le/--eq 多个条件之间用 AND 还是 OR 组合")
    ap.add_argument("--dedup-uid", action="store_true",
                    help="按 uid 去重（同一个 uid 保留 (global_step, byte_offset) 最小的那条）")
    ap.add_argument("--limit", type=int, default=None,
                    help="全局最多保留多少条样本（按排序之后截断）")
    ap.add_argument("--per-step-limit", type=int, default=None,
                    help="每一个 global_step 最多保留多少条样本（在排序之后，groupby head）")

    # 可视化参数
    ap.add_argument("--no-plot", action="store_true",
                    help="只筛选 + 导出 JSONL，不画图")
    ap.add_argument("--metrics", type=str, default="all",
                    help="要可视化的列列表：'all' 或 逗号分隔的列名，如 'ce_self_seq_mean,rambling_score'")
    ap.add_argument("--agg", type=str, default="mean",
                    choices=["mean","median","sum"],
                    help="per-step 聚合方式")
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--point-size", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.25)

    args = ap.parse_args()

    features_dir = args.features_dir
    plot_dir = args.plot_dir or features_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. 选定 features parquet & meta ----
    if args.features_file is not None:
        features_path = features_dir / args.features_file
    else:
        cand1 = features_dir / "features_analysis.parquet"
        cand2 = features_dir / "features.parquet"
        if cand1.exists():
            features_path = cand1
        elif cand2.exists():
            features_path = cand2
        else:
            raise FileNotFoundError("在 --features-dir 下没有找到 features_analysis.parquet 或 features.parquet，"
                                    "请用 --features-file 手动指定")

    meta_pre = features_dir / "meta_preprocess.json"
    if not meta_pre.exists():
        # 向后兼容旧的 meta.json
        meta_pre = features_dir / "meta.json"
    if not meta_pre.exists():
        raise FileNotFoundError("未找到 meta_preprocess.json 或 meta.json，用于解析 source_jsonl 和 offset 信息")

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

    # sample_idx 主要用于稳定 jitter 种子，如果没有就用行号
    if "sample_idx" in df.columns:
        jitter_seed_col = df["sample_idx"].astype(int).to_numpy()
    elif "uid" in df.columns:
        jitter_seed_col = np.array([salt64_from_uid(u) for u in df["uid"].astype(str)], dtype=np.int64)
    else:
        jitter_seed_col = np.arange(len(df), dtype=np.int64)

    # ---- 2. 构建筛选 mask ----
    n = len(df)
    mask = np.ones(n, dtype=bool)
    filter_applied = False  # 是否启用了任何过滤条件

    # kept / dropped
    if args.kept_or_dropped == "kept":
        mask &= (df["kept_flag"] == 1)
        filter_applied = True
    elif args.kept_or_dropped == "dropped":
        mask &= (df["kept_flag"] == 0)
        filter_applied = True

    # cls_id
    if args.cls != "any":
        cls_map = {"correct":1, "wrong":0, "no_judge":2}
        mask &= (df["cls_id"] == cls_map[args.cls])
        filter_applied = True

    # 中文 / 非中文
    if args.chinese_or_english != "all":
        if args.chinese_or_english == "chinese":
            mask &= (df["contain_chinese"] == 1)
        else:  # "english"
            mask &= (df["contain_chinese"] == 0)
        filter_applied = True

    # step 范围
    if args.step_min is not None:
        mask &= (df["global_step"] >= int(args.step_min))
        filter_applied = True
    if args.step_max is not None:
        mask &= (df["global_step"] <= int(args.step_max))
        filter_applied = True

    # ge / le / eq
    ge_list = parse_cmp_list(args.ge)
    le_list = parse_cmp_list(args.le)
    eq_list = parse_cmp_list(args.eq)
    if ge_list or le_list or eq_list:
        filter_applied = True
        cur = np.zeros(n, dtype=bool) if args.logic == "or" else np.ones(n, dtype=bool)
        def _comb(cur_mask, cond_mask):
            return (cur_mask | cond_mask) if args.logic == "or" else (cur_mask & cond_mask)

        for col, val in ge_list:
            if col not in df.columns:
                raise KeyError(f"--ge: unknown column {col}")
            cond = (df[col].astype(float) >= float(val))
            cur = _comb(cur, cond.to_numpy())
        for col, val in le_list:
            if col not in df.columns:
                raise KeyError(f"--le: unknown column {col}")
            cond = (df[col].astype(float) <= float(val))
            cur = _comb(cur, cond.to_numpy())
        for col, val in eq_list:
            if col not in df.columns:
                raise KeyError(f"--eq: unknown column {col}")
            cond = (df[col].astype(float) == float(val))
            cur = _comb(cur, cond.to_numpy())
        mask &= cur

    # where 表达式
    if args.where:
        filter_applied = True
        try:
            idx_where = df.query(args.where, engine="python").index
            cond = df.index.to_series().isin(idx_where).to_numpy()
            mask &= cond
        except Exception as e:
            print(f"[ERROR] 解析 --where 失败: {e}", file=sys.stderr)
            sys.exit(2)

    df_sel = df[mask].copy()
    print(f"[INFO] After filtering: rows={len(df_sel):,} (filter_applied={filter_applied})")

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

    # 全局 limit
    if args.limit is not None and args.limit > 0:
        df_sel = df_sel.sort_values(["global_step","byte_offset"], kind="mergesort")
        df_sel = df_sel.head(int(args.limit))
        print(f"[INFO] After limit={args.limit}: rows={len(df_sel):,}")
    else:
        # 默认排序
        df_sel = df_sel.sort_values(["global_step","byte_offset"], kind="mergesort")

    # ---- 3. 导出 JSONL （原始行 copy）----
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

    # ---- 4. 可视化（基于 df_plot）----
    if args.no_plot:
        print("[INFO] --no-plot set, skip plotting.")
        return

    if df_sel.empty:
        if not filter_applied:
            # 理论上不会发生：没过滤 && 空
            print("[WARN] 选出来 0 行，且 filter_applied=False，跳过绘图。")
            return
        else:
            print("[WARN] 过滤条件太严格，选出来 0 行，跳过绘图。")
            return

    print("[INFO] Start plotting …")
    steps = df_sel["global_step"].astype(int).to_numpy()
    cls_id = df_sel["cls_id"].astype(int).to_numpy()
    # jitter 种子：沿用原 df 的 jitter_seed_col
    jitter_seeds_sel = jitter_seed_col[df_sel.index.to_numpy()]

    # 默认 metrics 列表：你可以按需要再加
    default_metrics = [
        # RL 训练数值
        "ce_self_seq_mean",
        "ce_ref_seq_mean",
        "reward_seq_sum",
        "kl_seq_mean_log_ratio",
        "entropy_seq_mean",
        "response_len_tokens",

        # 文本结构
        "resp_char_len",
        "non_ascii_ratio",
        "control_private_unassigned_ratio",
        "repeat_word_run_max",
        "trigram_repeat_ratio",
        "compress_ratio",

        # 语言与 embedding 分析
        "H_lang",
        "num_lang_major",
        "code_switch_count",
        "all_r_dup",
        "marker_r_dup",
        "all_loopiness",
        "marker_loopiness",
        "rambling_score",
    ]

    if args.metrics.strip().lower() == "all":
        metric_names = [m for m in default_metrics if m in df_sel.columns]
    else:
        metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
        # 过滤掉不存在的列
        metric_names = [m for m in metric_names if m in df_sel.columns]

    if not metric_names:
        print("[WARN] 没有任何可画的 metrics（列不存在），跳过绘图。")
        return

    for m in metric_names:
        vals = df_sel[m].astype("float64").to_numpy()
        out_png = plot_dir / f"{m}.png"
        scatter_one_metric(
            out_png=out_png,
            metric_name=m,
            values=vals,
            steps=steps,
            cls_id=cls_id,
            jitter_seeds=jitter_seeds_sel,
            agg=args.agg,
            jitter=args.jitter,
            point_size=args.point_size,
            alpha=args.alpha,
            y_label=m,
            title=m,
        )

    print(f"[DONE] Selection + plotting finished. JSONL saved to {args.out_jsonl}, plots in {plot_dir}")

if __name__ == "__main__":
    main()
