#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize sequence-level Cross-Entropy (CE) from veRL/DAPO rollout JSONL logs.
(See the top of the file for a full description.)
"""
from __future__ import annotations
from pathlib import Path
import json
import argparse
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

def get_nested(d: Dict[str, Any], path: str, default: Any=None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur

def to_boolish(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v > 0)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "yes", "y", "correct", "right", "pass", "ok", "passed", "success"}:
            return True
        if s in {"false", "f", "no", "n", "wrong", "incorrect", "fail", "failed"}:
            return False
        try:
            fv = float(s)
            return fv > 0
        except Exception:
            return None
    return None

def classify_record(rec: Dict[str, Any], correct_key: Optional[str], use_reward_sum_fallback: bool = True) -> str:
    v = None
    if correct_key:
        v = get_nested(rec, correct_key, default=None)
    if v is None:
        rei = rec.get("reward_extra_infos")
        if rei is None:
            rei = rec.get("reward_extra_infos_dict")
        if isinstance(rei, dict):
            for k in ["is_correct", "correct", "equal", "any_of_three", "pass", "ok", "acc", "accuracy"]:
                if k in rei:
                    v = rei[k]
                    break
            if v is None and "label" in rei:
                v = rei["label"]
    if v is None and use_reward_sum_fallback:
        rs = rec.get("reward_seq_sum", None)
        if isinstance(rs, (int, float)):
            v = (rs > 0)
    b = to_boolish(v)
    if b is None:
        return "no_judge"
    return "correct" if b else "wrong"

def parse_line(line: str):
    try:
        return json.loads(line)
    except Exception:
        return None

def compute_ce_fields(rec: Dict[str, Any]) -> Dict[str, Optional[float]]:
    def neg(x):
        return None if x is None else (-float(x))
    old_mean = rec.get("old_logprob_mean")
    old_sum  = rec.get("old_logprob_sum")
    ref_mean = rec.get("ref_logprob_mean")
    ref_sum  = rec.get("ref_logprob_sum")
    return {
        "ce_self_seq_mean": neg(old_mean),
        "ce_self_seq_sum":  neg(old_sum),
        "ce_ref_seq_mean":  neg(ref_mean),
        "ce_ref_seq_sum":   neg(ref_sum),
    }

def jitter(uid: Any, step: int, scale: float=0.08) -> float:
    s = f"{uid}|{step}"
    import hashlib
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
    return ((h % 2001) / 2000.0 - 0.5) * 2.0 * scale

def load_records(jsonl_path: Path):
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = parse_line(line)
            if not isinstance(obj, dict):
                continue
            keep = {
                "global_step": obj.get("global_step"),
                "uid": obj.get("uid"),
                "old_logprob_mean": obj.get("old_logprob_mean"),
                "old_logprob_sum": obj.get("old_logprob_sum"),
                "ref_logprob_mean": obj.get("ref_logprob_mean"),
                "ref_logprob_sum": obj.get("ref_logprob_sum"),
                "reward_seq_sum": obj.get("reward_seq_sum"),
                "reward_extra_infos": obj.get("reward_extra_infos"),
                "reward_extra_infos_dict": obj.get("reward_extra_infos_dict"),
            }
            records.append(keep)
    return records

def group_by_step(records, correct_key: Optional[str], use_reward_sum_fallback: bool):
    step_map = {}
    for rec in records:
        step = rec.get("global_step", None)
        if step is None:
            continue
        try:
            step = int(step)
        except Exception:
            continue
        ce = compute_ce_fields(rec)
        cls = classify_record(rec, correct_key=correct_key, use_reward_sum_fallback=use_reward_sum_fallback)
        item = {"uid": rec.get("uid"), "cls": cls, **ce}
        step_map.setdefault(step, []).append(item)
    return step_map

COLOR_CORRECT = "blue"
COLOR_WRONG   = "red"
COLOR_UNKNOWN = "purple"
COLOR_AGG     = "black"
#（可选）统一管理 marker 形状
MARKER_CORRECT = "o"   # 空心圆
MARKER_WRONG   = "^"   # 叉叉
MARKER_UNKNOWN = "s"   # 空心三角
MARKER_AGG     = "o"

def _scatter_and_agg(
    ax,
    step_map: Dict[int, List[Dict[str, Any]]],
    value_key: str,
    agg: str,
    jitter_scale: float,
    point_size: int,
    alpha: float,
):
    """Draw all sample points with color-coding and the aggregator polyline."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    steps_sorted = sorted(step_map.keys())

    # Scatter by class per step
    for step in steps_sorted:
        items = step_map[step]
        xs_c, ys_c = [], []
        xs_w, ys_w = [], []
        xs_u, ys_u = [], []

        for it in items:
            y = it.get(value_key, None)
            if y is None:
                continue
            x = step + jitter(it.get("uid"), step, jitter_scale)
            if it["cls"] == "correct":
                xs_c.append(x); ys_c.append(y)
            elif it["cls"] == "wrong":
                xs_w.append(x); ys_w.append(y)
            else:
                xs_u.append(x); ys_u.append(y)

        # if xs_w: ax.scatter(xs_w, ys_w, s=point_size, alpha=alpha, marker=MARKER_WRONG, color=COLOR_WRONG, linewidths=1.2,   label="wrong")
        if xs_w: ax.scatter(xs_w, ys_w, s=point_size, alpha=alpha, marker=MARKER_WRONG,   facecolors="none", edgecolors=COLOR_WRONG,   linewidths=1.2, label="wrong")
        if xs_c: ax.scatter(xs_c, ys_c, s=point_size, alpha=alpha, marker=MARKER_CORRECT, facecolors="none", edgecolors=COLOR_CORRECT, linewidths=1.2, label="correct")
        if xs_u: ax.scatter(xs_u, ys_u, s=point_size, alpha=alpha, marker=MARKER_UNKNOWN, facecolors="none", edgecolors=COLOR_UNKNOWN, linewidths=1.2, label="no_judge")

    # Aggregator per step
    agg_x, agg_y = [], []
    for step in steps_sorted:
        vals = [it[value_key] for it in step_map[step] if it.get(value_key) is not None]
        if not vals:
            continue
        if   agg == "mean":   y = float(np.mean(vals))
        elif agg == "median": y = float(np.median(vals))
        elif agg == "sum":    y = float(np.sum(vals))
        else:                 y = float(np.mean(vals))
        agg_x.append(step); agg_y.append(y)

    if agg_x:
        ax.plot(agg_x, agg_y, marker="o", linewidth=1.5, color=COLOR_AGG, label=f"{agg} per step")

    # === 横轴仅整数刻度 ===
    if steps_sorted:
        ax.set_xlim(min(steps_sorted) - 0.5, max(steps_sorted) + 0.5)
        # 强制主刻度为整数（避免 266.5 之类的小数刻度）
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # 如果你希望“只显示存在的那些 step”，可改用下面一行（二选一）：
        # ax.set_xticks(steps_sorted)

    # === 横排 legend 放右下角 ===
    # handles = [
    #     Line2D([0],[0], marker='x', linestyle='', color=COLOR_WRONG,   label='wrong'),
    #     Line2D([0],[0], marker='o', linestyle='', color=COLOR_CORRECT, label='correct'),
    #     Line2D([0],[0], marker='o', linestyle='', color=COLOR_UNKNOWN, label='no_judge'),
    #     Line2D([0],[0], marker='o', color=COLOR_AGG, label=f'{agg} per step')
    # ]
    handles = [
        Line2D([0],[0], marker=MARKER_WRONG,   linestyle='', markerfacecolor='none', markeredgecolor=COLOR_WRONG,   color='none', label='wrong'),
        Line2D([0],[0], marker=MARKER_CORRECT, linestyle='', markerfacecolor='none', markeredgecolor=COLOR_CORRECT, color='none', label='correct'),
        Line2D([0],[0], marker=MARKER_UNKNOWN, linestyle='', markerfacecolor='none', markeredgecolor=COLOR_UNKNOWN, color='none', label='no_judge'),
        Line2D([0],[0], marker=MARKER_AGG, color=COLOR_AGG, label=f'{agg} per step'),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",     # 右下角
        ncol=4,                # 横向 1×4
        frameon=True,
        borderaxespad=0.6,
        handlelength=1.2,
        columnspacing=1.2,
        fontsize="small",
    )

    ax.set_xlabel("global_step")
    ax.set_ylabel("CE (nats)")

def plot_four_figures(step_map, agg: str, jitter_scale: float = 0.08, point_size: int = 14, alpha: float = 0.7):
    fig1, ax1 = plt.subplots()
    _scatter_and_agg(ax1, step_map, "ce_self_seq_mean", agg, jitter_scale, point_size, alpha)
    ax1.set_title(f"CE self (mean)")
    plt.show()

    fig2, ax2 = plt.subplots()
    _scatter_and_agg(ax2, step_map, "ce_self_seq_sum", agg, jitter_scale, point_size, alpha)
    ax2.set_title(f"CE self (sum)")
    plt.show()

    fig3, ax3 = plt.subplots()
    _scatter_and_agg(ax3, step_map, "ce_ref_seq_mean", agg, jitter_scale, point_size, alpha)
    ax3.set_title(f"CE ref (mean)")
    plt.show()

    fig4, ax4 = plt.subplots()
    _scatter_and_agg(ax4, step_map, "ce_ref_seq_sum", agg, jitter_scale, point_size, alpha)
    ax4.set_title(f"CE ref (sum)")
    plt.show()

def plot_grid_2x2_and_save(step_map, agg: str, out_path: Path, jitter_scale: float = 0.08, point_size: int = 14, alpha: float = 0.7, figsize=(14,10)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax = axes[0,0]
    _scatter_and_agg(ax, step_map, "ce_self_seq_mean", agg, jitter_scale, point_size, alpha)
    ax.set_title("CE self (mean)")

    ax = axes[0,1]
    _scatter_and_agg(ax, step_map, "ce_self_seq_sum", agg, jitter_scale, point_size, alpha)
    ax.set_title("CE self (sum)")

    ax = axes[1,0]
    _scatter_and_agg(ax, step_map, "ce_ref_seq_mean", agg, jitter_scale, point_size, alpha)
    ax.set_title("CE ref (mean)")

    ax = axes[1,1]
    _scatter_and_agg(ax, step_map, "ce_ref_seq_sum", agg, jitter_scale, point_size, alpha)
    ax.set_title("CE ref (sum)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[Saved] 2x2 grid to: {out_path}")

def main():
    p = argparse.ArgumentParser(description="Visualize sequence-level CE from veRL/DAPO rollout JSONL.")
    p.add_argument("--jsonl", type=Path, required=True, help="Path to rollout record JSONL.")
    p.add_argument("--agg", type=str, default="mean", choices=["mean","median","sum"], help="Per-step aggregator for black polyline.")
    p.add_argument("--correct-key", type=str, default=None, help="Dot-path to a boolean-like field indicating correctness, e.g., 'reward_extra_infos.is_correct'.")
    p.add_argument("--no-reward-sum-fallback", action="store_true", help="Disable fallback classification based on reward_seq_sum > 0 when correctness key is missing.")
    p.add_argument("--jitter", type=float, default=0.08, help="Horizontal jitter scale per point (to reduce overlap).")
    p.add_argument("--point-size", type=int, default=14, help="Scatter marker size.")
    p.add_argument("--alpha", type=float, default=0.7, help="Scatter alpha.")
    p.add_argument("--grid-out", type=Path, default=None, help="If set, additionally save a 2x2 grid figure to this PNG path.")
    args = p.parse_args()

    records = load_records(args.jsonl)
    step_map = group_by_step(records, correct_key=args.correct_key, use_reward_sum_fallback=not args.no_reward_sum_fallback)

    plot_four_figures(step_map, agg=args.agg, jitter_scale=args.jitter, point_size=args.point_size, alpha=args.alpha)

    if args.grid_out is not None:
        out_path = args.grid_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_grid_2x2_and_save(step_map, agg=args.agg, out_path=out_path, jitter_scale=args.jitter, point_size=args.point_size, alpha=args.alpha)

if __name__ == "__main__":
    main()
