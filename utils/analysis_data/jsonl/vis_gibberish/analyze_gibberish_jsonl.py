#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze 'gibberish / meltdown' symptoms from veRL/DAPO rollout JSONL logs, step by step.

- Stream-read a huge JSONL (20G+ ok) in CHUNKS.
- Multi-process parse + feature extraction (CPU-bound -> processes > threads).
- Per-sample metrics include:
  * From log: CE self/ref (mean/sum), reward_seq_sum, KL (log-ratio approx), entropy_seq_mean,
    response_len_tokens (already provided by your writer).
  * From text (response_text): char/word length, non-ASCII ratio, replacement-char ratio,
    control/private/unassigned-cat ratio, longest repeated char/word runs,
    unique-word ratio, top-1 word proportion, repeated trigram ratio,
    zlib compression ratio, character-level Shannon entropy (bits/char).

- Per-step aggregation: mean / median / sum (configurable via --agg).
- Scatter all points ("sandbox") + per-step black line for aggregation.
- Plot order: fast -> medium -> heavy; save after each figure.

Usage:
  python analyze_gibberish_jsonl.py \
    --jsonl /path/to/sample_traces.jsonl \
    --out-dir /path/to/out_imgs \
    --agg mean \
    --procs 64 \
    --chunk-lines 20000 \
    --correct-key reward_extra_infos.acc \
    --jitter 0.25 --point-size 3 --alpha 0.25

"""

from __future__ import annotations
from pathlib import Path
import argparse
import time
import math
import zlib
import unicodedata
import hashlib
import sys
import os
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------- fast json --------------------
try:
    import orjson
    def _json_loads(s: str):
        return orjson.loads(s)
except Exception:
    import json
    def _json_loads(s: str):
        return json.loads(s)

# -------------------- helpers --------------------
def get_nested(d: Dict[str, Any], path: Optional[str], default: Any=None) -> Any:
    if not path:
        return default
    cur: Any = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
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
        if s in {"true","t","yes","y","correct","right","pass","ok","passed","success"}:
            return True
        if s in {"false","f","no","n","wrong","incorrect","fail","failed"}:
            return False
        try:
            return float(s) > 0
        except Exception:
            return None
    return None

def classify_record(rec: Dict[str, Any], correct_key: Optional[str], use_reward_sum_fallback: bool=True) -> int:
    """
    Return class id: 1=correct, 0=wrong, 2=no_judge
    """
    v = None
    if correct_key:
        v = get_nested(rec, correct_key, default=None)
    if v is None:
        rei = rec.get("reward_extra_infos") or rec.get("reward_extra_infos_dict")
        if isinstance(rei, dict):
            for k in ["is_correct","correct","equal","any_of_three","pass","ok","acc","accuracy","label"]:
                if k in rei:
                    v = rei[k]; break
    if v is None and use_reward_sum_fallback:
        rs = rec.get("reward_seq_sum", None)
        if isinstance(rs, (int,float)):
            v = (rs > 0)
    b = to_boolish(v)
    if b is None:
        return 2
    return 1 if b else 0

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        xf = float(x)
        # weed out inf/nan
        if math.isfinite(xf):
            return xf
        return None
    except Exception:
        return None

def neg(x: Optional[float]) -> Optional[float]:
    return None if x is None else -x

def salt64_from_uid(uid: Any) -> int:
    s = str(uid)
    h = hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(h, 'little', signed=False)

def jitter_from_salt(salt: np.ndarray, scale: float) -> np.ndarray:
    # deterministic in [-scale, scale]
    # map salt to [0,1) then to [-scale, scale]
    r = (salt % 10_000_003) / 10_000_003.0
    return (r - 0.5) * 2.0 * scale

# -------------------- text metrics --------------------
SPACE_CHARS = {" ", "\t", "\n", "\r", "\v", "\f"}
REPL_CHAR = "\uFFFD"  # �

def text_metrics(s: str) -> Dict[str, float]:
    # bytes / chars
    b = s.encode('utf-8', errors='replace')
    char_len = len(s)
    byte_len = len(b)

    # unicode categories
    cnt = len(s)
    if cnt == 0:
        return {
            "resp_char_len": 0.0,
            "resp_byte_len": 0.0,
            "non_ascii_ratio": 0.0,
            "replacement_char_ratio": 0.0,
            "control_private_unassigned_ratio": 0.0,
            "repeat_char_run_max": 0.0,
            "repeat_word_run_max": 0.0,
            "unique_word_ratio": 0.0,
            "top1_word_prop": 0.0,
            "trigram_repeat_ratio": 0.0,
            "compress_ratio": 1.0,
            "char_entropy_bits_per_char": 0.0,
        }

    non_ascii = sum(1 for ch in s if ord(ch) > 127)
    repl_ratio = s.count(REPL_CHAR) / cnt

    # Control/Private/Unassigned categories (C* or Co/Cs/Cn)
    bad_cat = 0
    for ch in s:
        cat = unicodedata.category(ch)  # e.g., 'Lu','Ll','Nd','Zs','So','Cf',...
        if cat.startswith('C') or cat in {'Co','Cs','Cn'}:
            bad_cat += 1
    cpu_ratio = bad_cat / cnt

    # repeat runs (char + word)
    max_char_run = 1
    cur_run = 1
    for i in range(1, cnt):
        if s[i] == s[i-1]:
            cur_run += 1
            if cur_run > max_char_run: max_char_run = cur_run
        else:
            cur_run = 1

    # word-level
    words = [w for w in s.split() if w]
    wlen = len(words)
    if wlen == 0:
        uniq_ratio = 0.0
        top1_prop = 0.0
        max_word_run = float(max_char_run)  # fallback
        trigram_rep = 0.0
    else:
        uniq = len(set(words))
        uniq_ratio = uniq / wlen
        # top1
        # 若样本特别长，这一步 O(n) 仍可接受
        from collections import Counter
        wc = Counter(words)
        top1 = wc.most_common(1)[0][1]
        top1_prop = top1 / wlen
        # max word run
        max_word_run = 1
        cur = 1
        for i in range(1, wlen):
            if words[i] == words[i-1]:
                cur += 1
                if cur > max_word_run: max_word_run = cur
            else:
                cur = 1
        # trigram repeat ratio
        if wlen >= 3:
            tgs = [tuple(words[i:i+3]) for i in range(wlen-2)]
            from collections import Counter
            c3 = Counter(tgs)
            rep = sum(v for v in c3.values() if v > 1)
            trigram_rep = rep / len(tgs)
        else:
            trigram_rep = 0.0

    # compress ratio (smaller => more repetitive)
    try:
        comp = zlib.compress(b, level=6)
        comp_ratio = len(comp) / max(1, len(b))
    except Exception:
        comp_ratio = 1.0

    # char entropy (bits per char)
    from collections import Counter
    cc = Counter(s)
    probs = [c / cnt for c in cc.values()]
    ent = -sum(p * math.log2(p) for p in probs)

    return {
        "resp_char_len": float(char_len),
        "resp_byte_len": float(byte_len),
        "non_ascii_ratio": float(non_ascii / cnt),
        "replacement_char_ratio": float(repl_ratio),
        "control_private_unassigned_ratio": float(cpu_ratio),
        "repeat_char_run_max": float(max_char_run),
        "repeat_word_run_max": float(max_word_run),
        "unique_word_ratio": float(uniq_ratio),
        "top1_word_prop": float(top1_prop),
        "trigram_repeat_ratio": float(trigram_rep),
        "compress_ratio": float(comp_ratio),
        "char_entropy_bits_per_char": float(ent),
    }

# -------------------- per-line → minimal fields --------------------
NUM_KEYS = (
    # CE / KL / Entropy from your writer
    "old_logprob_mean","old_logprob_sum",
    "ref_logprob_mean","ref_logprob_sum",
    "kl_seq_mean_log_ratio","entropy_seq_mean",
    "reward_seq_sum",
    "response_len_tokens",
)

def extract_minimal_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    rec = {
        "global_step": obj.get("global_step"),
        "uid": obj.get("uid"),
        "filter_state": obj.get("filter_state"),  # kept/dropped if present
        "reward_seq_sum": obj.get("reward_seq_sum"),
        "response_len_tokens": obj.get("response_len_tokens"),
        "kl_seq_mean_log_ratio": obj.get("kl_seq_mean_log_ratio"),
        "entropy_seq_mean": obj.get("entropy_seq_mean"),
        "old_logprob_mean": obj.get("old_logprob_mean"),
        "old_logprob_sum": obj.get("old_logprob_sum"),
        "ref_logprob_mean": obj.get("ref_logprob_mean"),
        "ref_logprob_sum": obj.get("ref_logprob_sum"),
        "reward_extra_infos": obj.get("reward_extra_infos"),
        "reward_extra_infos_dict": obj.get("reward_extra_infos_dict"),
        "response_text": obj.get("response_text"),
    }
    return rec

# -------------------- worker --------------------
def process_chunk(lines: List[str], correct_key: Optional[str], use_reward_sum_fallback: bool) -> Dict[str, Any]:
    steps: List[int] = []
    salts: List[int] = []
    cls_id: List[int] = []
    kept_flag: List[int] = []  # 1=kept, 0=dropped, -1=unknown

    # numeric arrays
    out_num: Dict[str, List[float]] = {k: [] for k in [
        "ce_self_seq_mean","ce_self_seq_sum","ce_ref_seq_mean","ce_ref_seq_sum",
        "reward_seq_sum","kl_seq_mean_log_ratio","entropy_seq_mean","response_len_tokens"
    ]}
    # text arrays
    out_txt: Dict[str, List[float]] = {k: [] for k in [
        "resp_char_len","resp_byte_len","non_ascii_ratio","replacement_char_ratio",
        "control_private_unassigned_ratio","repeat_char_run_max","repeat_word_run_max",
        "unique_word_ratio","top1_word_prop","trigram_repeat_ratio","compress_ratio",
        "char_entropy_bits_per_char"
    ]}

    n_total = 0
    n_parsed = 0
    n_with_step = 0

    for line in lines:
        n_total += 1
        s = line.strip()
        if not s:
            continue
        try:
            obj = _json_loads(s)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        n_parsed += 1
        rec = extract_minimal_fields(obj)
        step = rec.get("global_step", None)
        if step is None:
            continue
        try:
            step = int(step)
        except Exception:
            continue
        n_with_step += 1

        # class id
        cid = classify_record(rec, correct_key=correct_key, use_reward_sum_fallback=use_reward_sum_fallback)

        # salt for jitter
        salt = salt64_from_uid(rec.get("uid"))

        # numeric
        ce_self_mean = neg(safe_float(rec.get("old_logprob_mean")))
        ce_self_sum  = neg(safe_float(rec.get("old_logprob_sum")))
        ce_ref_mean  = neg(safe_float(rec.get("ref_logprob_mean")))
        ce_ref_sum   = neg(safe_float(rec.get("ref_logprob_sum")))
        kllog        = safe_float(rec.get("kl_seq_mean_log_ratio"))
        ent_mean     = safe_float(rec.get("entropy_seq_mean"))
        rew_sum      = safe_float(rec.get("reward_seq_sum"))
        resp_len_tok = safe_float(rec.get("response_len_tokens"))

        # append
        steps.append(step); salts.append(salt); cls_id.append(cid)
        fs = rec.get("filter_state")
        kept_flag.append(1 if fs=="kept" else (0 if fs=="dropped" else -1))

        out_num["ce_self_seq_mean"].append(ce_self_mean)
        out_num["ce_self_seq_sum"].append(ce_self_sum)
        out_num["ce_ref_seq_mean"].append(ce_ref_mean)
        out_num["ce_ref_seq_sum"].append(ce_ref_sum)
        out_num["reward_seq_sum"].append(rew_sum)
        out_num["kl_seq_mean_log_ratio"].append(kllog)
        out_num["entropy_seq_mean"].append(ent_mean)
        out_num["response_len_tokens"].append(resp_len_tok)

        # text metrics (if any)
        rt = rec.get("response_text")
        if isinstance(rt, str):
            tm = text_metrics(rt)
            for k in out_txt.keys():
                out_txt[k].append(tm[k])
        else:
            for k in out_txt.keys():
                out_txt[k].append(None)

    return {
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_with_step": n_with_step,
        "steps": steps,
        "salts": salts,
        "cls_id": cls_id,
        "kept_flag": kept_flag,
        "num": out_num,
        "txt": out_txt,
    }

# -------------------- plotting --------------------
COLOR_CORRECT = "blue"
COLOR_WRONG   = "red"
COLOR_UNKNOWN = "pink"
COLOR_AGG     = "black"
# 新增：all 模式下区分 kept / dropped 的颜色
COLOR_DROPPED_WRONG   = "black"   # 按你的要求
COLOR_DROPPED_CORRECT = "purple"  # 按你的要求（注意：和 no_judge 同色，保持一致）
# dropped 的 no_judge 仍用 COLOR_UNKNOWN（紫色）

def aggregate_per_step(values: np.ndarray, step_idx: Dict[int, np.ndarray], agg: str, eligible_mask: np.ndarray) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for st in sorted(step_idx.keys()):
        idx = step_idx[st]
        # 只聚合“符合选择（kept/dropped/all）且非 NaN”的样本
        idx = idx[eligible_mask[idx]]
        if idx.size == 0:
            continue
        v = values[idx]
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        if   agg == "median": y = float(np.median(v))
        elif agg == "sum":    y = float(np.sum(v))
        else:                 y = float(np.mean(v))
        xs.append(st); ys.append(y)
    return xs, ys

def scatter_one_metric(
    out_png: Path,
    metric_name: str,
    values: np.ndarray,     # shape [N]
    steps: np.ndarray,      # shape [N]
    salts: np.ndarray,      # shape [N]
    cls_id: np.ndarray,     # 1=correct, 0=wrong, 2=no_judge
    kepts: np.ndarray,      # 1=kept, 0=dropped, -1=unknown
    step_idx: Dict[int, np.ndarray],
    agg: str,
    jitter: float,
    point_size: int,
    alpha: float,
    y_label: str,
    title: str,
    kept_or_dropped: str,
    agg_point_size: float,
):
    t0 = time.time()
    fig, ax = plt.subplots(figsize=(14, 6))

    # —— 选择要绘制与聚合的子集（eligible）——
    eligible = np.isfinite(values)
    if kept_or_dropped == "kept":
        eligible &= (kepts == 1)
    elif kept_or_dropped == "dropped":
        eligible &= (kepts == 0)
    else:  # "all"
        pass  # 全部（含 -1 未知也会参与）

    # —— 画散点：按 all/kept/dropped 分别处理颜色 —— 
    # 形状保持：wrong="^", correct="o", no_judge="s"
    def _scatter(mask, marker, edgec, label):
        if not np.any(mask):
            return
        xs = steps[mask].astype(float)
        js = jitter_from_salt(salts[mask], jitter)
        ys = values[mask]
        ax.scatter(xs + js, ys, s=point_size, alpha=alpha, marker=marker,
                   facecolors="none", edgecolors=edgec, linewidths=0.8, label=label)

    if kept_or_dropped == "all":
        # kept 部分
        kept_mask = eligible & (kepts == 1)
        _scatter(kept_mask & (cls_id == 0), "^", COLOR_WRONG,   "kept wrong")
        _scatter(kept_mask & (cls_id == 1), "o", COLOR_CORRECT, "kept correct")
        _scatter(kept_mask & (cls_id == 2), "s", COLOR_UNKNOWN, "kept no_judge")

        # dropped 部分（你指定的颜色）
        drop_mask = eligible & (kepts == 0)
        _scatter(drop_mask & (cls_id == 0), "^", COLOR_DROPPED_WRONG,   "dropped wrong")
        _scatter(drop_mask & (cls_id == 1), "o", COLOR_DROPPED_CORRECT, "dropped correct")
        _scatter(drop_mask & (cls_id == 2), "s", COLOR_UNKNOWN,         "dropped no_judge")
    else:
        # 只画 kept 或只画 dropped，颜色沿用 all 逻辑
        if kept_or_dropped == "kept":
            base = eligible & (kepts == 1)
            _scatter(base & (cls_id == 0), "^", COLOR_WRONG,   "kept wrong")
            _scatter(base & (cls_id == 1), "o", COLOR_CORRECT, "kept correct")
            _scatter(base & (cls_id == 2), "s", COLOR_UNKNOWN, "kept no_judge")
        else:  # "dropped"
            base = eligible & (kepts == 0)
            _scatter(base & (cls_id == 0), "^", COLOR_DROPPED_WRONG,   "dropped wrong")
            _scatter(base & (cls_id == 1), "o", COLOR_DROPPED_CORRECT, "dropped correct")
            _scatter(base & (cls_id == 2), "s", COLOR_UNKNOWN,         "dropped no_judge")

    # —— per-step 聚合黑线：只对 eligible 的样本做聚合 —— 
    agg_x, agg_y = aggregate_per_step(values, step_idx, agg=agg, eligible_mask=eligible)
    if agg_x:
        ax.plot(agg_x, agg_y, marker="o", markersize=agg_point_size, linewidth=1.5,
                color=COLOR_AGG, label=f"{agg} per step")

    ax.set_xlabel("global_step")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    # legend：自动去重
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", ncol=3, fontsize="small", frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    dt = time.time() - t0
    print(f"[Saved] {metric_name} -> {out_png}  (took {dt:.1f}s)")

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--agg", type=str, default="mean", choices=["mean","median","sum"])
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--chunk-lines", type=int, default=20000)
    ap.add_argument("--correct-key", type=str, default=None)
    ap.add_argument("--no-reward-sum-fallback", action="store_true")
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--point-size", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--kept-or-dropped", type=str, default="all", choices=["all", "kept", "dropped"], help="Select which samples to plot: kept only, dropped only, or both (all).")
    ap.add_argument("--agg-point-size", type=float, default=2.0, help="Marker size for the per-step aggregation line (black dots).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # -------- reading + parallel processing --------
    print(f"[Info] Reading JSONL: {args.jsonl}")
    tot_lines = 0
    t0 = time.time()

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex, args.jsonl.open("r", encoding="utf-8") as f:
        pending = []
        buf: List[str] = []
        for line in f:
            buf.append(line)
            tot_lines += 1
            if len(buf) >= args.chunk_lines:
                fut = ex.submit(process_chunk, buf, args.correct_key, not args.no_reward_sum_fallback)
                pending.append(fut)
                buf = []
                # 回压，避免 pending 太多占内存
                if len(pending) >= args.procs * 2:
                    for fu in as_completed(pending[:args.procs]):
                        results.append(fu.result())
                        pending.remove(fu)
        if buf:
            fut = ex.submit(process_chunk, buf, args.correct_key, not args.no_reward_sum_fallback)
            pending.append(fut)
        for fu in as_completed(pending):
            results.append(fu.result())

    # -------- merge --------
    print("[Info] Finished reading. Now merging…")
    steps_all   = []
    salts_all   = []
    clsid_all   = []
    kept_all    = []
    num_arrays: Dict[str, List[Optional[float]]] = {}
    txt_arrays: Dict[str, List[Optional[float]]] = {}
    for r in results:
        steps_all.extend(r["steps"])
        salts_all.extend(r["salts"])
        clsid_all.extend(r["cls_id"])
        kept_all.extend(r["kept_flag"])
        for k, arr in r["num"].items():
            num_arrays.setdefault(k, []).extend(arr)
        for k, arr in r["txt"].items():
            txt_arrays.setdefault(k, []).extend(arr)

    steps = np.array(steps_all, dtype=np.int32)
    salts = np.array(salts_all, dtype=np.uint64)
    clsid = np.array(clsid_all, dtype=np.int8)
    kepts = np.array(kept_all, dtype=np.int8)

    def to_float_np(lst: List[Optional[float]]) -> np.ndarray:
        a = np.array([np.nan if (v is None) else float(v) for v in lst], dtype=np.float64)
        return a

    num_np = {k: to_float_np(v) for k, v in num_arrays.items()}
    txt_np = {k: to_float_np(v) for k, v in txt_arrays.items()}

    # -------- first print: raw counts --------
    n_total = sum(r["n_total"] for r in results)
    n_parsed = sum(r["n_parsed"] for r in results)
    n_with_step = sum(r["n_with_step"] for r in results)
    uniq_steps = np.unique(steps)
    monotonic = np.all(np.diff(steps) >= 0)
    # 大致检测“是否成块（连续 4096 的块）”：统计相邻不同 step 的 run-length
    run_lengths = []
    if monotonic and steps.size > 0:
        cur = steps[0]; cnt = 1
        for s in steps[1:]:
            if s == cur:
                cnt += 1
            else:
                run_lengths.append(cnt); cur = s; cnt = 1
        run_lengths.append(cnt)

    print("=== PASS 1 / Read Summary ===")
    print(f"lines_total={n_total} parsed_ok={n_parsed} have_step={n_with_step}")
    if steps.size:
        print(f"steps: min={int(steps.min())} max={int(steps.max())} unique={len(uniq_steps)} monotonic_nondec={monotonic}")
        if run_lengths:
            q = np.percentile(run_lengths, [10,50,90,99])
            print(f"run_length_per_step (10/50/90/99%): {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}/{q[3]:.0f}  (ideal≈4096)")
    cls_counts = { "wrong": int(np.sum(clsid==0)),
                   "correct": int(np.sum(clsid==1)),
                   "no_judge": int(np.sum(clsid==2)) }
    kept_counts = { "kept": int(np.sum(kepts==1)),
                    "dropped": int(np.sum(kepts==0)),
                    "unknown": int(np.sum(kepts==-1)) }
    print(f"class_counts={cls_counts}")
    print(f"kept_counts={kept_counts}")

    # -------- group index per step (second print after grouping) --------
    step_idx: Dict[int, np.ndarray] = {}
    for st in uniq_steps:
        step_idx[int(st)] = np.nonzero(steps == st)[0]
    print("=== PASS 2 / Grouping Done ===")
    # 给个 sanity：每步样本数分布
    per_step_sizes = np.array([len(step_idx[s]) for s in sorted(step_idx.keys())], dtype=np.int32)
    q2 = np.percentile(per_step_sizes, [10,50,90,99])
    print(f"samples_per_step (10/50/90/99%): {q2[0]:.0f}/{q2[1]:.0f}/{q2[2]:.0f}/{q2[3]:.0f}")

    # -------- plotting order: fast -> medium -> heavy --------
    # FAST (pure numeric)
    fast_specs = [
        ("ce_self_seq_mean", num_np["ce_self_seq_mean"], "CE self (mean, nats)"),
        ("ce_self_seq_sum",  num_np["ce_self_seq_sum"],  "CE self (sum, nats)"),
        ("ce_ref_seq_mean",  num_np["ce_ref_seq_mean"],  "CE ref (mean, nats)"),
        ("ce_ref_seq_sum",   num_np["ce_ref_seq_sum"],   "CE ref (sum, nats)"),
        ("reward_seq_sum",   num_np["reward_seq_sum"],   "Reward (sum)"),
        ("kl_seq_mean_log_ratio", num_np["kl_seq_mean_log_ratio"], "KL approx (mean log-ratio)"),
        ("entropy_seq_mean", num_np["entropy_seq_mean"], "Token entropy (mean)"),
        ("response_len_tokens", num_np["response_len_tokens"], "Response length (tokens)"),
    ]

    # MEDIUM (light text)
    med_specs = [
        ("resp_char_len", txt_np["resp_char_len"], "Response length (chars)"),
        ("unique_word_ratio", txt_np["unique_word_ratio"], "Unique words ratio"),
        ("top1_word_prop", txt_np["top1_word_prop"], "Top-1 word proportion"),
        ("repeat_char_run_max", txt_np["repeat_char_run_max"], "Max repeated char run"),
        ("repeat_word_run_max", txt_np["repeat_word_run_max"], "Max repeated word run"),
        ("non_ascii_ratio", txt_np["non_ascii_ratio"], "Non-ASCII ratio"),
        ("replacement_char_ratio", txt_np["replacement_char_ratio"], "Replacement char (�) ratio"),
        ("control_private_unassigned_ratio", txt_np["control_private_unassigned_ratio"], "Ctrl/Private/Unassigned cat ratio"),
    ]

    # HEAVY
    heavy_specs = [
        ("trigram_repeat_ratio", txt_np["trigram_repeat_ratio"], "Repeated trigram ratio"),
        ("compress_ratio", txt_np["compress_ratio"], "Zlib compression ratio (↓ more repetitive)"),
        ("char_entropy_bits_per_char", txt_np["char_entropy_bits_per_char"], "Char entropy (bits/char)"),
    ]

    # paint & save
    def paint(specs, tag):
        for metric_name, arr, ylabel in specs:
            out_png = args.out_dir / f"{metric_name}.png"
            scatter_one_metric(
                out_png=out_png,
                metric_name=metric_name,
                values=arr,
                steps=steps,
                salts=salts,
                cls_id=clsid,
                kepts=kepts,
                step_idx=step_idx,
                agg=args.agg,
                jitter=args.jitter,
                point_size=args.point_size,
                alpha=args.alpha,
                y_label=ylabel,
                title=f"{ylabel}",
                kept_or_dropped=args.kept_or_dropped,
                agg_point_size=args.agg_point_size,
            )


    print("[Plot] FAST metrics …")
    paint(fast_specs, "fast")
    print("[Plot] MEDIUM text metrics …")
    paint(med_specs, "medium")
    print("[Plot] HEAVY text metrics …")
    paint(heavy_specs, "heavy")

    dt = time.time() - t0
    print(f"[Done] All finished in {dt/60:.1f} min; output dir = {args.out_dir}")

if __name__ == "__main__":
    main()
