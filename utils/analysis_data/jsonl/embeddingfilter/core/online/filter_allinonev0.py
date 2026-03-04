#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_allinone.py

单脚本版本：
从 RL 训练的超大 JSONL 中直接完成整套流程：

1) 扫描 JSONL，构建 (line_no, byte_offset, byte_len) 索引；
2) 多进程读取 JSONL，计算 sample-level & sentence-level CPU 指标：
   - 文本质量 text_metrics 全套
   - 中文检测 contains_chinese
   - 句子切分 + discourse markers + langid
   - cls_id / kept_flag / ce_* / reward_seq_sum 等
3) 用 vLLM (task='embed') 计算 sentence embedding；
4) 基于 embedding 计算语言分布 + 冗余/聚类 + loopiness + plateau + RamblingScore；
5) 按 pattern 规则过滤 target 子集（通常是 kept & correct），得到 bad / good；
6) 从原始 JSONL 按 byte_offset 读取原始行，附加 filter_debug 信息，
   分别写入 good_samples.jsonl / bad_samples.jsonl。
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse, json, os, sys, time, math, hashlib, re
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# vLLM
from vllm import LLM

# ---------- fast json ----------
try:
    import orjson
    def _json_loads(b: bytes):
        return orjson.loads(b)
except Exception:
    import json as _json
    def _json_loads(b: bytes):
        return _json.loads(b.decode("utf-8", errors="replace"))

# 用户可配置：哪些 metrics 需要真正计算 / 输出到 filter_debug.metrics。
# 注释掉某一项即可关闭它（会影响 Stage1/2/3 中的重型计算；简单的 ce_* 等
# 仍然会被读取，但不会出现在 metrics 里）。
ENABLED_METRICS = [
    # text_metrics
    "resp_char_len", "resp_byte_len",
    "non_ascii_ratio", "replacement_char_ratio",
    "control_private_unassigned_ratio",
    "repeat_char_run_max", "repeat_word_run_max",
    "unique_word_ratio", "top1_word_prop",
    "trigram_repeat_ratio",
    "compress_ratio",
    "char_entropy_bits_per_char",

    # section / marker 统计
    "num_sections", "num_sentences",
    "num_marker_sentences",
    "num_self_reflect_sentences",
    "num_contrast_sentences",
    "num_conclude_sentences",
    "num_marker_sections",

    # 中文检测
    "contain_chinese",

    # ce / reward / token 级（便宜，但你也可以通过这里控制是否写进 debug）
    "ce_self_seq_mean", "ce_self_seq_sum",
    "ce_ref_seq_mean", "ce_ref_seq_sum",
    "reward_seq_sum",
    "kl_seq_mean_log_ratio",
    "entropy_seq_mean",
    "response_len_tokens",

    # embedding / 语言分布相关
    "H_lang", "num_lang_major", "code_switch_count",
    "all_r_dup", "all_cluster_count", "all_cluster_size_max", "all_rho_max", "all_loopiness", "all_plateau_len_max",
    "marker_r_dup", "marker_cluster_count", "marker_cluster_size_max", "marker_rho_max", 
    "marker_loopiness",
    "marker_plateau_len_max", "wait_r_dup", "wait_cluster_count", "wait_cluster_size_max", "wait_rho_max", "wait_loopiness", "wait_plateau_len_max",
    "embedding_truncated",
    "rambling_score",
]

# ---------- Stage1 segmentation / lang / marker 开关 ----------
# 这两个开关是“用户手动”偏好，真正是否会执行由 resolve_metric_dependencies()
# 根据 ENABLED_METRICS 和依赖关系做一次修正。
ENABLE_DETECT_MARKERS: bool = True
ENABLE_DETECT_LANG: bool = True
# ENABLE_DETECT_LANG: bool = False

# ---------- text metrics ----------
import unicodedata, zlib

REPL_CHAR = "\uFFFD"

# ---------- Chinese detector ----------
_CHN_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3000-\u303F"
    r"\U00020000-\U0002A6DF\U0002A700-\U0002B73F\U0002B740-\U0002B81F"
    r"\U0002B820-\U0002CEAF\U0002CEB0-\U0002EBEF]"
)

def contains_chinese(s: str) -> bool:
    if not s:
        return False
    return bool(_CHN_RE.search(s))

def text_metrics(s: str) -> Dict[str, float]:
    """字符级 text_metrics，与原实现对齐。"""
    try:
        b = s.encode("utf-8", errors="replace")
        cnt = len(s)
        if cnt == 0:
            return {
                "resp_char_len": 0.0, "resp_byte_len": 0.0, "non_ascii_ratio": 0.0,
                "replacement_char_ratio": 0.0, "control_private_unassigned_ratio": 0.0,
                "repeat_char_run_max": 0.0, "repeat_word_run_max": 0.0, "unique_word_ratio": 0.0,
                "top1_word_prop": 0.0, "trigram_repeat_ratio": 0.0, "compress_ratio": 1.0,
                "char_entropy_bits_per_char": 0.0,
            }

        non_ascii = sum(1 for ch in s if ord(ch) > 127)
        repl_ratio = s.count(REPL_CHAR) / cnt
        bad_cat = 0
        for ch in s:
            cat = unicodedata.category(ch)
            if cat.startswith("C") or cat in {"Co", "Cs", "Cn"}:
                bad_cat += 1
        cpu_ratio = bad_cat / cnt

        # char run
        max_char_run = 1
        run = 1
        for i in range(1, cnt):
            if s[i] == s[i-1]:
                run += 1
                if run > max_char_run:
                    max_char_run = run
            else:
                run = 1

        words = [w for w in s.split() if w]
        if not words:
            uniq_ratio = 0.0
            top1_prop = 0.0
            max_word_run = float(max_char_run)
            trigram_rep = 0.0
        else:
            from collections import Counter
            wlen = len(words)
            uniq_ratio = len(set(words)) / wlen
            wc = Counter(words)
            top1_prop = wc.most_common(1)[0][1] / wlen

            max_word_run = 1
            run = 1
            for i in range(1, wlen):
                if words[i] == words[i-1]:
                    run += 1
                    if run > max_word_run:
                        max_word_run = run
                else:
                    run = 1

            if wlen >= 3:
                tgs = [tuple(words[i:i+3]) for i in range(wlen-2)]
                c3 = Counter(tgs)
                trigram_rep = sum(v for v in c3.values() if v > 1) / len(tgs)
            else:
                trigram_rep = 0.0

        try:
            comp = zlib.compress(b, level=6)
            comp_ratio = len(comp) / max(1, len(b))
        except Exception:
            comp_ratio = 1.0

        from collections import Counter
        cc = Counter(s)
        probs = [c / cnt for c in cc.values()]
        ent = -sum(p * math.log2(p) for p in probs)

        return {
            "resp_char_len": float(cnt),
            "resp_byte_len": float(len(b)),
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
    except Exception:
        return {
            "resp_char_len": np.nan, "resp_byte_len": np.nan, "non_ascii_ratio": np.nan,
            "replacement_char_ratio": np.nan, "control_private_unassigned_ratio": np.nan,
            "repeat_char_run_max": np.nan, "repeat_word_run_max": np.nan, "unique_word_ratio": np.nan,
            "top1_word_prop": np.nan, "trigram_repeat_ratio": np.nan, "compress_ratio": np.nan,
            "char_entropy_bits_per_char": np.nan,
        }

# ---------- nested key helper ----------
def get_nested(d: Dict[str, Any], path: str, default: Any=None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def classify_record(obj: Dict[str, Any], correct_key: Optional[str], use_reward_sum_fallback=True) -> int:
    """1=correct, 0=wrong, 2=no_judge."""
    v = None
    if correct_key:
        v = get_nested(obj, correct_key, None)
    if v is None:
        rei = obj.get("reward_extra_infos") or obj.get("reward_extra_infos_dict")
        if isinstance(rei, dict):
            for k in ["is_correct","correct","equal","any_of_three","pass","ok","acc","accuracy","label"]:
                if k in rei:
                    v = rei[k]
                    break
    if v is None and use_reward_sum_fallback:
        rs = obj.get("reward_seq_sum", None)
        if isinstance(rs, (int,float)):
            v = (rs > 0)

    def _to_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int,float)):
            return x > 0
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"true","t","yes","y","correct","right","pass","ok","passed","success"}:
                return True
            if s in {"false","f","no","n","wrong","incorrect","fail","failed"}:
                return False
            try:
                return float(s) > 0
            except Exception:
                return None
        return None

    b = _to_bool(v)
    if b is None:
        return 2
    return 1 if b else 0

def neg(x):
    try:
        xf = float(x)
        if math.isfinite(xf):
            return -xf
        return np.nan
    except Exception:
        return np.nan

@dataclass
class BuildConfig:
    jsonl: Path
    correct_key: Optional[str]
    min_lang_chars: int
    cpu_procs: int
    chunk_lines: int
    compute_sha256: bool
    step_min: Optional[int]
    step_max: Optional[int]
    debug: bool
    # metric / Stage1 开关（由 resolve_metric_dependencies() 决定）
    enable_text_metrics: bool = True
    enable_contains_chinese: bool = True
    enable_split_sections_and_sentences: bool = True
    enable_detect_markers: bool = True
    enable_detect_lang: bool = True

SCHEMA_VERSION = 3

# ---------- discourse markers ----------
SELF_REFLECT_PATTERNS = [
    r"\bwait\b",
    r"\bwait a second\b",
    r"\bhold on\b",
    r"\bhang on\b",
    r"\blet me think\b",
    r"\bgive me a moment\b",
    r"\bjust a second\b",
    r"等一下",
    r"等一等",
    r"先等等",
    r"稍等",
    r"moment mal",
    r"einen moment",
    r"подожди",
    r"подождите",
    r"секундочку",
]

CONTRAST_PATTERNS = [
    r"\bhowever\b",
    r"\bbut\b",
    r"\bon the other hand\b",
    r"\bnevertheless\b",
    r"\bnonetheless\b",
    r"但是",
    r"然而",
    r"不过",
    r"另一方面",
    r"doch",
    r"aber",
    r"jedoch",
    r"однако",
]

CONCLUDE_PATTERNS = [
    r"\bso\b",
    r"\btherefore\b",
    r"\bthus\b",
    r"\bhence\b",
    r"\bin conclusion\b",
    r"\bas a result\b",
    r"所以",
    r"因此",
    r"综上",
    r"由此可见",
    r"itak",
    r"итак",
    r"значит",
]

SELF_REFLECT_RE = re.compile("|".join(SELF_REFLECT_PATTERNS), re.IGNORECASE)
CONTRAST_RE     = re.compile("|".join(CONTRAST_PATTERNS),     re.IGNORECASE)
CONCLUDE_RE     = re.compile("|".join(CONCLUDE_PATTERNS),     re.IGNORECASE)
WAITLIKE_RE     = re.compile(r"\bwait\b|\bhold on\b|\bhang on\b|等一下|稍等|подожди|подождите", re.IGNORECASE)

MARKER_SELF_REFLECT = 1
MARKER_CONTRAST     = 2
MARKER_CONCLUDE     = 4

def detect_markers(text: str) -> Tuple[int, int, int, int, int]:
    """返回 (mask, self_flag, contrast_flag, conclude_flag, waitlike_flag)."""
    mask = 0
    sr = 1 if SELF_REFLECT_RE.search(text) else 0
    ct = 1 if CONTRAST_RE.search(text) else 0
    cd = 1 if CONCLUDE_RE.search(text) else 0
    if sr:
        mask |= MARKER_SELF_REFLECT
    if ct:
        mask |= MARKER_CONTRAST
    if cd:
        mask |= MARKER_CONCLUDE
    waitlike = 1 if WAITLIKE_RE.search(text) else 0
    return mask, sr, ct, cd, waitlike

# ---------- sentence segmentation (pysbd) ----------
try:
    import pysbd
    _SEGMENTER = pysbd.Segmenter(language="en", clean=False, char_span=True)
    _SEGMENTER_CHARSPAN = True
except Exception:
    try:
        import pysbd  # type: ignore
        _SEGMENTER = pysbd.Segmenter(language="en", clean=False, char_span=False)
        _SEGMENTER_CHARSPAN = False
    except Exception:
        _SEGMENTER = None
        _SEGMENTER_CHARSPAN = False

def split_sections(text: str) -> List[Tuple[str, int, int]]:
    """按 \n+ 切 section，保持原逻辑。"""
    if not text:
        return []
    n = len(text)
    sections: List[Tuple[str, int, int]] = []
    start = 0
    i = 0
    while i < n:
        if text[i] == "\n":
            j = i
            while j < n and text[j] == "\n":
                j += 1
            sec_text = text[start:j]
            sections.append((sec_text, start, j))
            start = j
            i = j
        else:
            i += 1
    if start < n:
        sections.append((text[start:n], start, n))
    if not sections:
        return [(text, 0, n)]
    return sections

def split_sentences(section_text: str, sec_start: int) -> List[Tuple[str, int, int]]:
    """在一个 section 内用 pysbd 切分 sentence。"""
    if not section_text:
        return []
    if _SEGMENTER is None:
        return [(section_text, sec_start, sec_start + len(section_text))]
    try:
        spans = _SEGMENTER.segment(section_text)
        results: List[Tuple[str, int, int]] = []
        if spans and _SEGMENTER_CHARSPAN and hasattr(spans[0], "start"):
            for sp in spans:
                s = getattr(sp, "start")
                e = getattr(sp, "end")
                if not isinstance(s, int) or not isinstance(e, int):
                    continue
                global_s = sec_start + s
                global_e = sec_start + e
                sent_text = section_text[s:e]
                results.append((sent_text, global_s, global_e))
        else:
            pos = 0
            for sent in spans:
                if not isinstance(sent, str):
                    continue
                idx = section_text.find(sent, pos)
                if idx == -1:
                    idx = pos
                s = idx
                e = idx + len(sent)
                global_s = sec_start + s
                global_e = sec_start + e
                results.append((sent, global_s, global_e))
                pos = e
        if not results:
            return [(section_text, sec_start, sec_start + len(section_text))]
        return results
    except Exception:
        return [(section_text, sec_start, sec_start + len(section_text))]

# ---------- sentence language detection ----------
try:
    import langid
    _HAS_LANGID = True
    langid.set_languages(['en', 'zh', 'de', 'ru', 'it', 'fr', 'es', 'ja'])
except Exception:
    _HAS_LANGID = False

def detect_lang(text: str, min_len: int = 5) -> Tuple[str, float]:
    if not text or len(text) < min_len or not _HAS_LANGID:
        return "unk", 0.0
    try:
        lang, conf = langid.classify(text)
        return str(lang), float(conf)
    except Exception:
        return "unk", 0.0

# ---------- Stage0: scan offsets ----------
def scan_offsets(cfg: BuildConfig, kept_or_dropped: str, cls_filter: str,):
    """
    Stage0:
      - 单次遍历 JSONL:
        * 统计: total / kept / dropped / kept&correct / kept&wrong
        * 根据 step_min/step_max + kept_or_dropped + cls_filter 选子集
      - 返回选中的 (line_no, offset, length) 分块
    """
    jsonl = cfg.jsonl
    chunk_lines = cfg.chunk_lines
    compute_sha256 = cfg.compute_sha256

    offsets: List[Tuple[int, int, int]] = []
    chunks: List[List[Tuple[int, int, int]]] = []
    hasher = hashlib.sha256() if compute_sha256 else None

    stat = jsonl.stat()
    total_bytes = stat.st_size

    # 统计计数
    n_total = 0
    num_selected = 0
    num_kept = 0
    num_dropped = 0
    num_kept_correct = 0
    num_kept_wrong = 0

    with jsonl.open("rb", buffering=1024 * 1024) as f:
        pbar = tqdm(total=total_bytes, desc="[Scan] JSONL + subset", unit="B", unit_scale=True)
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                if offsets:
                    chunks.append(offsets)
                    offsets = []
                break

            ln = len(line)
            pbar.update(ln)
            if hasher:
                hasher.update(line)
            n_total += 1

            kept_flag = -1  # 1=kept,0=dropped,-1=其它
            cls_id = 2      # 1=correct,0=wrong,2=no_judge
            gs_int = -1

            # 解析 JSON，用于统计和过滤
            try:
                obj = _json_loads(line)
            except Exception:
                obj = None

            if isinstance(obj, dict):
                # kept/dropped
                fs = obj.get("filter_state")
                if fs == "kept":
                    kept_flag = 1
                    num_kept += 1
                elif fs == "dropped":
                    kept_flag = 0
                    num_dropped += 1
                else:
                    kept_flag = -1

                # cls_id（和 Stage1 完全同一逻辑）
                cls_id = classify_record(obj, correct_key=cfg.correct_key)

                # kept 中 correct / wrong 统计
                if kept_flag == 1:
                    if cls_id == 1:
                        num_kept_correct += 1
                    elif cls_id == 0:
                        num_kept_wrong += 1

                # global_step
                gs = get_nested(obj, "global_step", -1)
                try:
                    gs_int = int(gs)
                except Exception:
                    gs_int = -1

            # -------- 子集过滤逻辑 --------
            include = True

            # 步数过滤：在 scan 阶段就完成
            if cfg.step_min is not None and gs_int < cfg.step_min:
                include = False
            if cfg.step_max is not None and gs_int > cfg.step_max:
                include = False

            # kept / dropped 子集
            if kept_or_dropped == "kept" and kept_flag != 1:
                include = False
            elif kept_or_dropped == "dropped" and kept_flag != 0:
                include = False

            # cls 子集
            if cls_filter == "correct" and cls_id != 1:
                include = False
            elif cls_filter == "wrong" and cls_id != 0:
                include = False
            elif cls_filter == "no_judge" and cls_id != 2:
                include = False

            if include:
                offsets.append((n_total, off, ln))
                num_selected += 1
                if len(offsets) >= chunk_lines:
                    chunks.append(offsets)
                    offsets = []

        pbar.close()

    sha256 = hasher.hexdigest() if hasher else ""
    stats = {
        "num_total": n_total,
        "num_kept": num_kept,
        "num_dropped": num_dropped,
        "num_kept_correct": num_kept_correct,
        "num_kept_wrong": num_kept_wrong,
        "num_selected": num_selected,
    }
    return chunks, sha256, stat.st_size, int(stat.st_mtime), n_total, stats

# ---------- Stage1-like worker ----------
def _worker_process_stage1(
    jsonl_path: str,
    offsets: List[Tuple[int,int,int]],
    cfg: BuildConfig,
) -> Dict[str, Any]:
    """
    对一块 offsets 做 sample & segments 特征计算。
    返回:
      - df_samples
      - df_segments
    """
    fd = os.open(jsonl_path, os.O_RDONLY)
    rows_samples: List[Dict[str, Any]] = []
    rows_segments: List[Dict[str, Any]] = []

    for line_no, off, ln in offsets:
        try:
            b = os.pread(fd, ln, off)
        except Exception:
            obj = None
        else:
            try:
                obj = _json_loads(b)
            except Exception:
                obj = None

        sample_idx = line_no - 1

        if not isinstance(obj, dict):
            # 无法解析，给空行
            rec = {
                "sample_idx": np.int64(sample_idx),
                "line_no": np.int64(line_no),
                "byte_offset": np.int64(off),
                "byte_len": np.int32(ln),
                "valid": np.int8(0),
                "global_step": np.int32(-1),
                "uid": "",
                "kept_flag": np.int8(-1),
                "ce_self_seq_mean": np.float32(np.nan),
                "ce_self_seq_sum":  np.float32(np.nan),
                "ce_ref_seq_mean":  np.float32(np.nan),
                "ce_ref_seq_sum":   np.float32(np.nan),
                "reward_seq_sum":   np.float32(np.nan),
                "kl_seq_mean_log_ratio": np.float32(np.nan),
                "entropy_seq_mean":      np.float32(np.nan),
                "response_len_tokens":   np.float32(np.nan),
                "cls_id": np.int8(2),
                "contain_chinese": np.int8(-1),
            }
            tm = {k: np.nan for k in [
                "resp_char_len","resp_byte_len","non_ascii_ratio","replacement_char_ratio",
                "control_private_unassigned_ratio","repeat_char_run_max","repeat_word_run_max",
                "unique_word_ratio","top1_word_prop","trigram_repeat_ratio","compress_ratio",
                "char_entropy_bits_per_char"
            ]}
            for k, v in tm.items():
                rec[k] = np.float32(v)
            rec.update({
                "num_sections": np.int32(0),
                "num_sentences": np.int32(0),
                "num_marker_sentences": np.int32(0),
                "num_self_reflect_sentences": np.int32(0),
                "num_contrast_sentences": np.int32(0),
                "num_conclude_sentences": np.int32(0),
                "num_marker_sections": np.int32(0),
            })
            rows_samples.append(rec)
            continue

        # Step 过滤：如果有 step_min/step_max，则非范围内 sample 可以直接跳过后续计算（但仍记录基本信息）
        gs = get_nested(obj, "global_step", -1)
        if not isinstance(gs, int):
            try:
                gs = int(gs)
            except Exception:
                gs = -1

        kept_flag = 1 if obj.get("filter_state") == "kept" else (0 if obj.get("filter_state") == "dropped" else -1)

        # rec: Dict[str, Any] = {
        #     "sample_idx": np.int64(sample_idx),
        #     "line_no": np.int64(line_no),
        #     "byte_offset": np.int64(off),
        #     "byte_len": np.int32(ln),
        #     "valid": np.int8(1),
        #     "global_step": np.int32(gs),
        #     "uid": str(obj.get("uid")) if ("uid" in obj) else "",
        #     "kept_flag": np.int8(kept_flag),
        #     "ce_self_seq_mean": np.float32(neg(obj.get("old_logprob_mean"))),
        #     "ce_self_seq_sum":  np.float32(neg(obj.get("old_logprob_sum"))),
        #     "ce_ref_seq_mean":  np.float32(neg(obj.get("ref_logprob_mean"))),
        #     "ce_ref_seq_sum":   np.float32(neg(obj.get("ref_logprob_sum"))),
        #     "reward_seq_sum":   np.float32(obj.get("reward_seq_sum") if "reward_seq_sum" in obj else np.nan),
        #     "kl_seq_mean_log_ratio": np.float32(obj.get("kl_seq_mean_log_ratio") if "kl_seq_mean_log_ratio" in obj else np.nan),
        #     "entropy_seq_mean":      np.float32(obj.get("entropy_seq_mean") if "entropy_seq_mean" in obj else np.nan),
        #     "response_len_tokens":   np.float32(obj.get("response_len_tokens") if "response_len_tokens" in obj else np.nan),
        #     "cls_id": np.int8(classify_record(obj, correct_key=cfg.correct_key)),
        # }
        
        # ---- sequence-level CE stats (self/ref) ----
        # 优先使用 JSON 中已经写好的 ce_* 字段；
        # 若不存在，则从 *_logprob_mean / *_logprob_sum 推导（兼容旧版离线 JSON）。
        ce_self_mean = obj.get("ce_self_seq_mean", None)
        ce_self_sum  = obj.get("ce_self_seq_sum", None)
        ce_ref_mean  = obj.get("ce_ref_seq_mean", None)
        ce_ref_sum   = obj.get("ce_ref_seq_sum", None)

        if ce_self_mean is None and "old_logprob_mean" in obj:
            ce_self_mean = neg(obj.get("old_logprob_mean"))
        if ce_self_sum is None and "old_logprob_sum" in obj:
            ce_self_sum = neg(obj.get("old_logprob_sum"))

        if ce_ref_mean is None and "ref_logprob_mean" in obj:
            ce_ref_mean = neg(obj.get("ref_logprob_mean"))
        if ce_ref_sum is None and "ref_logprob_sum" in obj:
            ce_ref_sum = neg(obj.get("ref_logprob_sum"))

        rec: Dict[str, Any] = {
            "sample_idx": np.int64(sample_idx),
            "line_no": np.int64(line_no),
            "byte_offset": np.int64(off),
            "byte_len": np.int32(ln),
            "valid": np.int8(1),
            "global_step": np.int32(gs),
            "uid": str(obj.get("uid")) if ("uid" in obj) else "",
            "kept_flag": np.int8(kept_flag),
            "ce_self_seq_mean": np.float32(ce_self_mean if ce_self_mean is not None else np.nan),
            "ce_self_seq_sum":  np.float32(ce_self_sum  if ce_self_sum  is not None else np.nan),
            "ce_ref_seq_mean":  np.float32(ce_ref_mean  if ce_ref_mean  is not None else np.nan),
            "ce_ref_seq_sum":   np.float32(ce_ref_sum   if ce_ref_sum   is not None else np.nan),
            "reward_seq_sum":   np.float32(obj.get("reward_seq_sum") if "reward_seq_sum" in obj else np.nan),
            "kl_seq_mean_log_ratio": np.float32(obj.get("kl_seq_mean_log_ratio") if "kl_seq_mean_log_ratio" in obj else np.nan),
            "entropy_seq_mean":      np.float32(obj.get("entropy_seq_mean") if "entropy_seq_mean" in obj else np.nan),
            "response_len_tokens":   np.float32(obj.get("response_len_tokens") if "response_len_tokens" in obj else np.nan),
            "cls_id": np.int8(classify_record(obj, correct_key=cfg.correct_key)),
        }

        rt = obj.get("response_text")
        pt = obj.get("prompt_text")

        # # 是否包含中文
        # if isinstance(rt, str) or isinstance(pt, str):
        #     has_cn = (contains_chinese(rt) if isinstance(rt, str) else False) \
        #              or (contains_chinese(pt) if isinstance(pt, str) else False)
        #     rec["contain_chinese"] = np.int8(1 if has_cn else 0)
        # else:
        #     rec["contain_chinese"] = np.int8(-1)
        # 是否包含中文（可关闭）
        if cfg.enable_contains_chinese and (isinstance(rt, str) or isinstance(pt, str)):
            has_cn = (contains_chinese(rt) if isinstance(rt, str) else False) \
                     or (contains_chinese(pt) if isinstance(pt, str) else False)
            rec["contain_chinese"] = np.int8(1 if has_cn else 0)
        else:
            rec["contain_chinese"] = np.int8(-1)

        # # 文本指标
        # if isinstance(rt, str):
        #     tm = text_metrics(rt)
        # else:
        #     tm = {k: np.nan for k in [
        #         "resp_char_len","resp_byte_len","non_ascii_ratio","replacement_char_ratio",
        #         "control_private_unassigned_ratio","repeat_char_run_max","repeat_word_run_max",
        #         "unique_word_ratio","top1_word_prop","trigram_repeat_ratio","compress_ratio",
        #         "char_entropy_bits_per_char"
        #     ]}
        # 文本指标（可关闭）
        if cfg.enable_text_metrics and isinstance(rt, str):
            tm = text_metrics(rt)
        else:
            tm = {k: np.nan for k in [
                "resp_char_len","resp_byte_len","non_ascii_ratio","replacement_char_ratio",
                "control_private_unassigned_ratio","repeat_char_run_max","repeat_word_run_max",
                "unique_word_ratio","top1_word_prop","trigram_repeat_ratio","compress_ratio",
                "char_entropy_bits_per_char"
            ]}

        for k, v in tm.items():
            rec[k] = np.float32(v)

        # # section / sentence 切分
        # num_sections = 0
        # num_sentences = 0
        # num_marker_sent = 0
        # num_self_sent = 0
        # num_contrast_sent = 0
        # num_conclude_sent = 0
        # num_marker_sec = 0

        # if isinstance(rt, str) and rt:
        #     sections = split_sections(rt)
        #     num_sections = len(sections)
        #     for sec_idx, (sec_text, sec_start, sec_end) in enumerate(sections):
        #         section_has_marker = False
        #         sent_list = split_sentences(sec_text, sec_start)
        #         if not sent_list:
        #             sent_list = [(sec_text, sec_start, sec_end)]
        #         for local_sent_idx, (sent_text, global_s, global_e) in enumerate(sent_list):
        #             sent_idx = num_sentences

        #             marker_mask, sr, ct, cd, waitlike = detect_markers(sent_text)
        #             if marker_mask != 0:
        #                 section_has_marker = True
        #                 num_marker_sent += 1
        #             if sr:
        #                 num_self_sent += 1
        #             if ct:
        #                 num_contrast_sent += 1
        #             if cd:
        #                 num_conclude_sent += 1

        #             lang_main, lang_conf = detect_lang(sent_text, min_len=cfg.min_lang_chars)
        #             sent_char_len = float(len(sent_text))

        #             rows_segments.append({
        #                 "sample_idx": np.int64(sample_idx),
        #                 "line_no": np.int64(line_no),
        #                 "section_idx": np.int32(sec_idx),
        #                 "sent_idx": np.int32(sent_idx),
        #                 "section_local_sent_idx": np.int32(local_sent_idx),
        #                 "char_start": np.int32(global_s),
        #                 "char_end": np.int32(global_e),
        #                 "sent_char_len": np.float32(sent_char_len),
        #                 "text": sent_text,
        #                 "marker_mask": np.int8(marker_mask),
        #                 "marker_self_reflect": np.int8(sr),
        #                 "marker_contrast": np.int8(ct),
        #                 "marker_conclude": np.int8(cd),
        #                 "marker_wait_like": np.int8(waitlike),
        #                 "lang_main": lang_main,
        #                 "lang_conf": np.float32(lang_conf),
        #             })
        #             num_sentences += 1
        #         if section_has_marker:
        #             num_marker_sec += 1

        # section / sentence 切分 + markers + lang（可关闭）
        num_sections = 0
        num_sentences = 0
        num_marker_sent = 0
        num_self_sent = 0
        num_contrast_sent = 0
        num_conclude_sent = 0
        num_marker_sec = 0

        if isinstance(rt, str) and rt and cfg.enable_split_sections_and_sentences:
            sections = split_sections(rt)
            num_sections = len(sections)
            for sec_idx, (sec_text, sec_start, sec_end) in enumerate(sections):
                section_has_marker = False
                sent_list = split_sentences(sec_text, sec_start)
                if not sent_list:
                    sent_list = [(sec_text, sec_start, sec_end)]
                for local_sent_idx, (sent_text, global_s, global_e) in enumerate(sent_list):
                    sent_idx = num_sentences

                    # discourse markers
                    if cfg.enable_detect_markers:
                        marker_mask, sr, ct, cd, waitlike = detect_markers(sent_text)
                    else:
                        marker_mask, sr, ct, cd, waitlike = 0, 0, 0, 0, 0

                    if marker_mask != 0:
                        section_has_marker = True
                        num_marker_sent += 1
                    if sr:
                        num_self_sent += 1
                    if ct:
                        num_contrast_sent += 1
                    if cd:
                        num_conclude_sent += 1

                    # sentence-level langid
                    if cfg.enable_detect_lang:
                        lang_main, lang_conf = detect_lang(sent_text, min_len=cfg.min_lang_chars)
                    else:
                        lang_main, lang_conf = "unk", 0.0

                    sent_char_len = float(len(sent_text))

                    rows_segments.append({
                        "sample_idx": np.int64(sample_idx),
                        "line_no": np.int64(line_no),
                        "section_idx": np.int32(sec_idx),
                        "sent_idx": np.int32(sent_idx),
                        "section_local_sent_idx": np.int32(local_sent_idx),
                        "char_start": np.int32(global_s),
                        "char_end": np.int32(global_e),
                        "sent_char_len": np.float32(sent_char_len),
                        "text": sent_text,
                        "marker_mask": np.int8(marker_mask),
                        "marker_self_reflect": np.int8(sr),
                        "marker_contrast": np.int8(ct),
                        "marker_conclude": np.int8(cd),
                        "marker_wait_like": np.int8(waitlike),
                        "lang_main": lang_main,
                        "lang_conf": np.float32(lang_conf),
                    })
                    num_sentences += 1
                if section_has_marker:
                    num_marker_sec += 1
        # 如果 enable_split_sections_and_sentences=False，则保持 num_* 为 0，segments_df 中没有该 sample 的行

        rec["num_sections"] = np.int32(num_sections)
        rec["num_sentences"] = np.int32(num_sentences)
        rec["num_marker_sentences"] = np.int32(num_marker_sent)
        rec["num_self_reflect_sentences"] = np.int32(num_self_sent)
        rec["num_contrast_sentences"] = np.int32(num_contrast_sent)
        rec["num_conclude_sentences"] = np.int32(num_conclude_sent)
        rec["num_marker_sections"] = np.int32(num_marker_sec)

        rows_samples.append(rec)

    os.close(fd)

    df_samples = pd.DataFrame(rows_samples)
    df_segments = pd.DataFrame(rows_segments)
    # dtypes 收紧
    if not df_samples.empty:
        df_samples["sample_idx"] = df_samples["sample_idx"].astype(np.int64)
        df_samples["line_no"] = df_samples["line_no"].astype(np.int64)
        df_samples["byte_offset"] = df_samples["byte_offset"].astype(np.int64)
        df_samples["byte_len"] = df_samples["byte_len"].astype(np.int32)
        df_samples["global_step"] = df_samples["global_step"].astype(np.int32)
        df_samples["kept_flag"] = df_samples["kept_flag"].astype(np.int8)
        df_samples["cls_id"] = df_samples["cls_id"].astype(np.int8)
        df_samples["contain_chinese"] = df_samples["contain_chinese"].astype(np.int8)
        for col in [
            "num_sections","num_sentences","num_marker_sentences",
            "num_self_reflect_sentences","num_contrast_sentences",
            "num_conclude_sentences","num_marker_sections"
        ]:
            df_samples[col] = df_samples[col].astype(np.int32)

    if not df_segments.empty:
        df_segments["sample_idx"] = df_segments["sample_idx"].astype(np.int64)
        df_segments["line_no"] = df_segments["line_no"].astype(np.int64)
        df_segments["section_idx"] = df_segments["section_idx"].astype(np.int32)
        df_segments["sent_idx"] = df_segments["sent_idx"].astype(np.int32)
        df_segments["section_local_sent_idx"] = df_segments["section_local_sent_idx"].astype(np.int32)
        df_segments["char_start"] = df_segments["char_start"].astype(np.int32)
        df_segments["char_end"] = df_segments["char_end"].astype(np.int32)
        df_segments["sent_char_len"] = df_segments["sent_char_len"].astype(np.float32)
        df_segments["marker_mask"] = df_segments["marker_mask"].astype(np.int8)
        df_segments["marker_self_reflect"] = df_segments["marker_self_reflect"].astype(np.int8)
        df_segments["marker_contrast"] = df_segments["marker_contrast"].astype(np.int8)
        df_segments["marker_conclude"] = df_segments["marker_conclude"].astype(np.int8)
        df_segments["marker_wait_like"] = df_segments["marker_wait_like"].astype(np.int8)
        df_segments["lang_conf"] = df_segments["lang_conf"].astype(np.float32)

    return {"samples": df_samples, "segments": df_segments}

def _worker_entry_stage1(args):
    return _worker_process_stage1(*args)

# ---------- Stage2: embedding with vLLM ----------
def extract_embedding_from_output(output: Any) -> List[float]:
    outs = getattr(output, "outputs", None)
    if outs is None:
        return []
    if hasattr(outs, "embedding"):
        try:
            return list(outs.embedding)
        except Exception:
            pass
    if hasattr(outs, "data"):
        try:
            return outs.data.tolist()
        except Exception:
            pass
    if isinstance(outs, (list, tuple)) and len(outs) > 0:
        o0 = outs[0]
        if hasattr(o0, "embedding"):
            try:
                return list(o0.embedding)
            except Exception:
                pass
        if hasattr(o0, "data"):
            try:
                return o0.data.tolist()
            except Exception:
                pass
    return []

def run_vllm_embedding(
    segments_df: pd.DataFrame,
    model_name: str,
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
    enforce_eager: bool,
) -> pd.DataFrame:
    if segments_df.empty:
        segments_df["emb"] = [[] for _ in range(len(segments_df))]
        return segments_df

    print(f"[Stage2-vLLM] Loading LLM(model={model_name}, task='embed') ...")
    t0 = time.time()
    llm = LLM(
        model=model_name,
        task="embed",
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        enforce_eager=enforce_eager,
    )
    print(f"[Stage2-vLLM] Model loaded in {time.time()-t0:.1f}s")

    texts = segments_df["text"].astype(str).tolist()
    n = len(texts)
    print(f"[Stage2-vLLM] Embedding {n:,} sentences ...")
    t1 = time.time()
    outputs = llm.embed(texts)
    if len(outputs) != n:
        raise RuntimeError(f"llm.embed returned {len(outputs)} outputs, expected {n}")
    embs: List[List[float]] = []
    for out in outputs:
        embs.append(extract_embedding_from_output(out))
    segments_df = segments_df.copy()
    segments_df["emb"] = embs
    print(f"[Stage2-vLLM] Embedding done in {time.time()-t1:.1f}s")
    return segments_df

def _hf_embed_single_device(
    texts: List[str],
    model_name: str,
    device_str: str,
    hf_batch_size: int,
    hf_max_length: int,
    hf_torch_dtype: str,
    trust_remote_code: bool,
) -> List[List[float]]:
    """
    在单个 device 上做 sentence embedding：
    - 会根据 hf_batch_size 做分 batch；<=0 表示不分 batch 一次性跑完。
    - 返回与 texts 一一对应的 embedding 列表。
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    from torch import nn

    device = torch.device(device_str)
    print(f"[Stage2-HF] Using device = {device}")

    # dtype 选择
    torch_dtype = None
    if hf_torch_dtype != "auto":
        try:
            torch_dtype = getattr(torch, hf_torch_dtype)
        except AttributeError:
            raise ValueError(
                f"Invalid --hf-torch-dtype={hf_torch_dtype}, "
                "可选: auto,float32,float16,bfloat16"
            )
    if device.type == "cpu" and torch_dtype in (
        getattr(torch, "float16", None),
        getattr(torch, "bfloat16", None),
    ):
        print("[Stage2-HF][WARN] CPU 上使用 float16/bfloat16 可能不支持，自动退回 float32")
        torch_dtype = torch.float32

    print(f"[Stage2-HF] Loading HF model on {device}: {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if torch_dtype is None:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
    model.to(device)
    model.eval()
    print(f"[Stage2-HF] HF model on {device} loaded in {time.time()-t0:.1f}s")

    n = len(texts)
    # hf_batch_size <= 0 表示不切 batch，一次性送入模型（显存不够会 OOM）
    if hf_batch_size is None or hf_batch_size <= 0:
        bs = max(1, n)
        print(f"[Stage2-HF][{device}] hf_batch_size <= 0，使用单 batch size = {bs}")
    else:
        bs = max(1, int(hf_batch_size))

    embs: List[List[float]] = []
    norm_layer = nn.functional.normalize

    with torch.inference_mode():
        for start in tqdm(
            range(0, n, bs),
            desc=f"[Stage2-HF][{device}] Batches",
        ):
            batch_texts = texts[start:start + bs]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=hf_max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**enc)
            if hasattr(outputs, "last_hidden_state"):
                token_embeddings = outputs.last_hidden_state   # [B, T, D]
            else:
                token_embeddings = outputs[0]

            attention_mask = enc["attention_mask"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
            sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
            sent_emb = sum_embeddings / sum_mask  # [B, D]

            # L2 normalize
            sent_emb = norm_layer(sent_emb, p=2, dim=1)

            embs.extend(sent_emb.detach().cpu().numpy().tolist())

    if len(embs) != n:
        raise RuntimeError(
            f"[Stage2-HF] 得到的 embedding 数量 {len(embs)} != 文本数量 {n}"
        )

    return embs

def _hf_embedding_worker(args):
    """
    多进程 worker：在指定 device 上对一段文本做 embedding。
    args: (device_str, model_name, texts_slice, hf_batch_size, hf_max_length, hf_torch_dtype, trust_remote_code)
    """
    (
        device_str,
        model_name,
        texts,
        hf_batch_size,
        hf_max_length,
        hf_torch_dtype,
        trust_remote_code,
    ) = args
    return _hf_embed_single_device(
        texts=texts,
        model_name=model_name,
        device_str=device_str,
        hf_batch_size=hf_batch_size,
        hf_max_length=hf_max_length,
        hf_torch_dtype=hf_torch_dtype,
        trust_remote_code=trust_remote_code,
    )

# def run_hf_embedding(
#     segments_df: pd.DataFrame,
#     model_name: str,
#     hf_device: str,
#     hf_batch_size: int,
#     hf_max_length: int,
#     hf_torch_dtype: str,
#     trust_remote_code: bool,
# ) -> pd.DataFrame:
#     """
#     使用 HuggingFace Transformers 做 sentence embedding。

#     - 支持单卡 / 多卡：
#       * hf_device in {'auto','cuda'} 且 GPU 数 > 1 时，自动多卡；
#       * 其它情况（'cuda:0', 'cpu' 等）走单卡。
#     - 多卡模式下：
#       * 每张卡一个进程 + 一个模型（像多进程），texts 按 GPU 数均匀分块；
#       * 每个进程内部再按 hf_batch_size 做 micro-batch。
#     """
#     if segments_df.empty:
#         segments_df = segments_df.copy()
#         segments_df["emb"] = [[] for _ in range(len(segments_df))]
#         return segments_df

#     try:
#         import torch
#     except ImportError as e:
#         raise RuntimeError(
#             "HuggingFace backend 需要安装 `torch` 和 `transformers`，"
#             "请先 `pip install torch transformers`."
#         ) from e

#     texts = segments_df["text"].astype(str).tolist()
#     n = len(texts)
#     if n == 0:
#         segments_df = segments_df.copy()
#         segments_df["emb"] = [[] for _ in range(len(segments_df))]
#         return segments_df

#     # 判断是否走多卡
#     multi_gpu = False
#     num_gpus = 0
#     device_str = "cpu"

#     if hf_device in ("auto", "cuda"):
#         if torch.cuda.is_available():
#             num_gpus = torch.cuda.device_count()
#             if num_gpus > 1:
#                 multi_gpu = True
#             # 单卡 fallback 时默认用 cuda:0
#             device_str = "cuda:0"
#         else:
#             device_str = "cpu"
#     elif hf_device.startswith("cuda:"):
#         # 明确指定某一张卡 -> 单卡模式
#         device_str = hf_device
#         num_gpus = 1
#     else:
#         # 'cpu' 或其它
#         device_str = hf_device
#         num_gpus = 0

#     # ---------- 多卡模式 ----------
#     if multi_gpu and n > 1:
#         import multiprocessing as mp

#         num_gpus = torch.cuda.device_count()
#         # 不需要的 GPU 不起 worker
#         n_workers = min(num_gpus, n)
#         print(f"[Stage2-HF] Multi-GPU embedding: {n_workers} workers over {num_gpus} visible GPUs, total texts = {n:,}")

#         # 均匀切分文本给不同 GPU
#         # indices: [0, ..., n]
#         split_indices = np.linspace(0, n, num=n_workers + 1, dtype=int)

#         tasks = []
#         for worker_id in range(n_workers):
#             start = int(split_indices[worker_id])
#             end = int(split_indices[worker_id + 1])
#             if start >= end:
#                 continue
#             dev_str = f"cuda:{worker_id}"
#             texts_slice = texts[start:end]
#             tasks.append(
#                 (
#                     dev_str,
#                     model_name,
#                     texts_slice,
#                     hf_batch_size,
#                     hf_max_length,
#                     hf_torch_dtype,
#                     trust_remote_code,
#                 )
#             )

#         if not tasks:
#             # 退化为单卡
#             embs = _hf_embed_single_device(
#                 texts=texts,
#                 model_name=model_name,
#                 device_str=device_str,
#                 hf_batch_size=hf_batch_size,
#                 hf_max_length=hf_max_length,
#                 hf_torch_dtype=hf_torch_dtype,
#                 trust_remote_code=trust_remote_code,
#             )
#         else:
#             t0 = time.time()
#             # 为了 CUDA 安全，显式用 spawn
#             ctx = mp.get_context("spawn")
#             with ctx.Pool(processes=len(tasks)) as pool:
#                 # map 保证返回顺序与 tasks 相同，因此拼接后顺序不乱
#                 results = pool.map(_hf_embedding_worker, tasks)

#             print(f"[Stage2-HF] Multi-GPU embedding finished in {time.time()-t0:.1f}s")

#             embs: List[List[float]] = []
#             for part in results:
#                 embs.extend(part)

#             if len(embs) != n:
#                 raise RuntimeError(
#                     f"[Stage2-HF][Multi-GPU] 得到的 embedding 数量 {len(embs)} != 文本数量 {n}"
#                 )

#     # ---------- 单卡模式 ----------
#     else:
#         print(f"[Stage2-HF] Single-device HF embedding on {device_str}, total texts = {n:,}")
#         embs = _hf_embed_single_device(
#             texts=texts,
#             model_name=model_name,
#             device_str=device_str,
#             hf_batch_size=hf_batch_size,
#             hf_max_length=hf_max_length,
#             hf_torch_dtype=hf_torch_dtype,
#             trust_remote_code=trust_remote_code,
#         )

#     segments_df = segments_df.copy()
#     segments_df["emb"] = embs
#     return segments_df
def run_hf_embedding(
    segments_df: pd.DataFrame,
    model_name: str,
    hf_device: str,
    hf_batch_size: int,
    hf_max_length: int,
    hf_torch_dtype: str,
    trust_remote_code: bool,
    hf_max_workers: int,
) -> pd.DataFrame:
    """
    使用 HuggingFace Transformers 做 sentence embedding。

    - 支持单卡 / 多卡：
      * hf_device in {'auto','cuda'} 且 GPU 数 > 1 时，自动多卡；
      * 其它情况（'cuda:0', 'cpu' 等）走单卡。
    - 多卡模式下：
      * 每张卡一个进程 + 一个模型（像多进程），texts 按 GPU 数/worker 数均匀分块；
      * 每个进程内部再按 hf_batch_size 做 micro-batch。
    - hf_max_workers:
      * <=0: 使用所有可见 GPU；
      * >0: 最多只起 hf_max_workers 个 worker/GPU。
    """
    if segments_df.empty:
        segments_df = segments_df.copy()
        segments_df["emb"] = [[] for _ in range(len(segments_df))]
        return segments_df

    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "HuggingFace backend 需要安装 `torch` 和 `transformers`，"
            "请先 `pip install torch transformers`."
        ) from e

    texts = segments_df["text"].astype(str).tolist()
    n = len(texts)
    if n == 0:
        segments_df = segments_df.copy()
        segments_df["emb"] = [[] for _ in range(len(segments_df))]
        return segments_df

    # 判断是否走多卡
    multi_gpu = False
    num_gpus = 0
    device_str = "cpu"

    if hf_device in ("auto", "cuda"):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            # 如果只想强制 1 worker，就不要多卡
            if num_gpus > 1 and not (hf_max_workers == 1):
                multi_gpu = True
            device_str = "cuda:0"
        else:
            device_str = "cpu"
    elif hf_device.startswith("cuda:"):
        # 明确指定某一张卡 -> 单卡模式
        device_str = hf_device
        num_gpus = 1
    else:
        # 'cpu' 或其它
        device_str = hf_device
        num_gpus = 0

    # ---------- 多卡模式 ----------
    if multi_gpu and n > 1:
        import multiprocessing as mp

        num_gpus = torch.cuda.device_count()
        # 根据 hf_max_workers 限制 worker 数
        if hf_max_workers and hf_max_workers > 0:
            max_workers = min(hf_max_workers, num_gpus)
        else:
            max_workers = num_gpus

        # 不能比样本数多
        n_workers = min(max_workers, n)
        print(
            f"[Stage2-HF] Multi-GPU embedding: {n_workers} workers over {num_gpus} visible GPUs "
            f"(hf_max_workers={hf_max_workers}), total texts = {n:,}"
        )

        if n_workers <= 1:
            # 没必要起多进程，退化成单卡
            print("[Stage2-HF] n_workers <= 1, fallback to single-device mode.")
            embs = _hf_embed_single_device(
                texts=texts,
                model_name=model_name,
                device_str=device_str,
                hf_batch_size=hf_batch_size,
                hf_max_length=hf_max_length,
                hf_torch_dtype=hf_torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            # 均匀切分文本给不同 worker / GPU
            split_indices = np.linspace(0, n, num=n_workers + 1, dtype=int)

            tasks = []
            # 这里简单用前 n_workers 个 GPU：cuda:0, cuda:1, ...
            for worker_id in range(n_workers):
                start = int(split_indices[worker_id])
                end = int(split_indices[worker_id + 1])
                if start >= end:
                    continue
                dev_str = f"cuda:{worker_id}"
                texts_slice = texts[start:end]
                tasks.append(
                    (
                        dev_str,
                        model_name,
                        texts_slice,
                        hf_batch_size,
                        hf_max_length,
                        hf_torch_dtype,
                        trust_remote_code,
                    )
                )

            if not tasks:
                # 理论上不会发生，但保险起见再 fallback 一次
                embs = _hf_embed_single_device(
                    texts=texts,
                    model_name=model_name,
                    device_str=device_str,
                    hf_batch_size=hf_batch_size,
                    hf_max_length=hf_max_length,
                    hf_torch_dtype=hf_torch_dtype,
                    trust_remote_code=trust_remote_code,
                )
            else:
                t0 = time.time()
                # 为了 CUDA 安全，显式用 spawn
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=len(tasks)) as pool:
                    # map 保证返回顺序与 tasks 相同，因此拼接后顺序不乱
                    results = pool.map(_hf_embedding_worker, tasks)

                print(f"[Stage2-HF] Multi-GPU embedding finished in {time.time()-t0:.1f}s")

                embs: List[List[float]] = []
                for part in results:
                    embs.extend(part)

                if len(embs) != n:
                    raise RuntimeError(
                        f"[Stage2-HF][Multi-GPU] 得到的 embedding 数量 {len(embs)} != 文本数量 {n}"
                    )

    # ---------- 单卡模式 ----------
    else:
        print(f"[Stage2-HF] Single-device HF embedding on {device_str}, total texts = {n:,}")
        embs = _hf_embed_single_device(
            texts=texts,
            model_name=model_name,
            device_str=device_str,
            hf_batch_size=hf_batch_size,
            hf_max_length=hf_max_length,
            hf_torch_dtype=hf_torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    segments_df = segments_df.copy()
    segments_df["emb"] = embs
    return segments_df

# ---------- Stage3: embedding-based metrics ----------
MARKER_SELF_REFLECT = 1
MARKER_CONTRAST     = 2
MARKER_CONCLUDE     = 4

def _default_embedding_metric_result() -> Dict[str, float]:
    return {
        "all_r_dup": float("nan"),
        "all_cluster_count": float("nan"),
        "all_cluster_size_max": float("nan"),
        "all_rho_max": float("nan"),
        "all_loopiness": float("nan"),
        "all_plateau_len_max": float("nan"),
        "marker_r_dup": float("nan"),
        "marker_cluster_count": float("nan"),
        "marker_cluster_size_max": float("nan"),
        "marker_rho_max": float("nan"),
        "marker_loopiness": float("nan"),
        "marker_plateau_len_max": float("nan"),
        "wait_r_dup": float("nan"),
        "wait_cluster_count": float("nan"),
        "wait_cluster_size_max": float("nan"),
        "wait_rho_max": float("nan"),
        "wait_loopiness": float("nan"),
        "wait_plateau_len_max": float("nan"),
        "embedding_truncated": 0.0,
    }

def compute_lang_stats(sub: pd.DataFrame) -> Tuple[float, int, int]:
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
        G_tmp = G_sub.copy()
        np.fill_diagonal(G_tmp, -1.0)
        max_sim = G_tmp.max(axis=1)
        r_dup = float((max_sim >= dup_threshold).mean())
        # cluster
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
        # loopiness
        order = np.lexsort((sent_sub, sec_sub))
        seq = emb_sub[order]
        dot_next = (seq[:-1] * seq[1:]).sum(axis=1)
        dot_next = np.clip(dot_next, -1.0, 1.0)
        d = np.arccos(dot_next)
        L = float(d.sum())
        dot_end = float(np.clip((seq[0] * seq[-1]).sum(), -1.0, 1.0))
        D = float(np.arccos(dot_end))
        loopiness = L / (D + loop_eps)
        # plateau
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
        result.update({
            "r_dup": 0.0,
            "cluster_count": 1.0,
            "cluster_size_max": 1.0,
            "rho_max": 1.0,
            "loopiness": 1.0,
            "plateau_len_max": 1.0,
        })
    return result

# def compute_embedding_metrics_for_sample(
#     sub_emb: pd.DataFrame,
#     dup_threshold: float,
#     cluster_threshold: float,
#     plateau_min_len: int,
#     loop_eps: float,
#     max_sent_per_sample: int,
# ) -> Dict[str, float]:
#     result = {
#         "all_r_dup": float("nan"),
#         "all_cluster_count": float("nan"),
#         "all_cluster_size_max": float("nan"),
#         "all_rho_max": float("nan"),
#         "all_loopiness": float("nan"),
#         "all_plateau_len_max": float("nan"),
#         "marker_r_dup": float("nan"),
#         "marker_cluster_count": float("nan"),
#         "marker_cluster_size_max": float("nan"),
#         "marker_rho_max": float("nan"),
#         "marker_loopiness": float("nan"),
#         "marker_plateau_len_max": float("nan"),
#         "wait_r_dup": float("nan"),
#         "wait_cluster_count": float("nan"),
#         "wait_cluster_size_max": float("nan"),
#         "wait_rho_max": float("nan"),
#         "wait_loopiness": float("nan"),
#         "wait_plateau_len_max": float("nan"),
#         "embedding_truncated": 0.0,
#     }
def compute_embedding_metrics_for_sample(
    sub_emb: pd.DataFrame,
    dup_threshold: float,
    cluster_threshold: float,
    plateau_min_len: int,
    loop_eps: float,
    max_sent_per_sample: int,
    do_compute: bool = True,
) -> Dict[str, float]:
    result = _default_embedding_metric_result()
    if not do_compute:
        return result
    if sub_emb.empty:
        return result
    emb_list = []
    valid_idx = []
    for i, e in enumerate(sub_emb["emb"].tolist()):
        if isinstance(e, (list, tuple, np.ndarray)) and len(e) > 0:
            arr = np.asarray(e, dtype=np.float32)
            emb_list.append(arr)
            valid_idx.append(i)
    if not emb_list:
        return result
    emb = np.stack(emb_list, axis=0)
    N_valid = emb.shape[0]
    sec_idx = sub_emb["section_idx"].to_numpy()[valid_idx].astype(np.int32)
    sent_idx = sub_emb["sent_idx"].to_numpy()[valid_idx].astype(np.int32)
    marker_mask = sub_emb["marker_mask"].to_numpy()[valid_idx].astype(np.int32)
    wait_like = sub_emb["marker_wait_like"].to_numpy()[valid_idx].astype(np.int32)
    if max_sent_per_sample > 0 and N_valid > max_sent_per_sample:
        idx_sub = np.linspace(0, N_valid-1, num=max_sent_per_sample, dtype=int)
        emb = emb[idx_sub]
        sec_idx = sec_idx[idx_sub]
        sent_idx = sent_idx[idx_sub]
        marker_mask = marker_mask[idx_sub]
        wait_like = wait_like[idx_sub]
        N_valid = emb.shape[0]
        result["embedding_truncated"] = 1.0
    G = emb @ emb.T
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

# def run_embedding_analysis_for_all_samples(
#     samples_df: pd.DataFrame,
#     segments_df: pd.DataFrame,
#     dup_threshold: float,
#     cluster_threshold: float,
#     plateau_min_len: int,
#     loop_eps: float,
#     max_sent_per_sample: int,
#     rambling_weights: Dict[str, float],
#     rambling_loopiness_cap: float,
#     rambling_plateau_k: int,
# ) -> pd.DataFrame:
def run_embedding_analysis_for_all_samples(
    samples_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    dup_threshold: float,
    cluster_threshold: float,
    plateau_min_len: int,
    loop_eps: float,
    max_sent_per_sample: int,
    rambling_weights: Dict[str, float],
    rambling_loopiness_cap: float,
    rambling_plateau_k: int,
    enable_embedding_metrics: bool = True,
    enable_rambling_score: bool = True,
) -> pd.DataFrame:
    if segments_df.empty or samples_df.empty:
        # 无数据直接补 NaN
        for col in [
            "H_lang", "num_lang_major", "code_switch_count",
            "all_r_dup", "all_cluster_count", "all_cluster_size_max", "all_rho_max", "all_loopiness", "all_plateau_len_max",
            "marker_r_dup", "marker_cluster_count", "marker_cluster_size_max", "marker_rho_max", "marker_loopiness", "marker_plateau_len_max",
            "wait_r_dup", "wait_cluster_count", "wait_cluster_size_max", "wait_rho_max", "wait_loopiness", "wait_plateau_len_max",
            "embedding_truncated", "rambling_score",
        ]:
            if col in ["num_lang_major", "code_switch_count"]:
                samples_df[col] = 0
            else:
                samples_df[col] = np.nan
        return samples_df

    print("[Stage3] Computing embedding-based metrics per sample ...")
    t0 = time.time()
    metrics_rows: List[Dict[str, Any]] = []

    # 以 sample_idx 分组
    grouped = segments_df.groupby("sample_idx")
    for sid, sub in grouped:
        sid_int = int(sid)
        row: Dict[str, Any] = {"sample_idx": sid_int}
        # 语言统计
        H_lang, num_lang_major, code_switch_count = compute_lang_stats(sub)
        row["H_lang"] = H_lang
        row["num_lang_major"] = int(num_lang_major)
        row["code_switch_count"] = int(code_switch_count)
        # # embedding metrics
        # metrics_emb = compute_embedding_metrics_for_sample(
        #     sub,
        #     dup_threshold=dup_threshold,
        #     cluster_threshold=cluster_threshold,
        #     plateau_min_len=plateau_min_len,
        #     loop_eps=loop_eps,
        #     max_sent_per_sample=max_sent_per_sample,
        # )
        # embedding metrics（可关闭）
        metrics_emb = compute_embedding_metrics_for_sample(
            sub,
            dup_threshold=dup_threshold,
            cluster_threshold=cluster_threshold,
            plateau_min_len=plateau_min_len,
            loop_eps=loop_eps,
            max_sent_per_sample=max_sent_per_sample,
            do_compute=(enable_embedding_metrics or enable_rambling_score),
        )
        row.update(metrics_emb)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        for col in [
            "H_lang", "num_lang_major", "code_switch_count",
            "all_r_dup", "all_cluster_count", "all_cluster_size_max", "all_rho_max", "all_loopiness", "all_plateau_len_max",
            "marker_r_dup", "marker_cluster_count", "marker_cluster_size_max", "marker_rho_max", "marker_loopiness", "marker_plateau_len_max",
            "wait_r_dup", "wait_cluster_count", "wait_cluster_size_max", "wait_rho_max", "wait_loopiness", "wait_plateau_len_max",
            "embedding_truncated", "rambling_score",
        ]:
            if col in ["num_lang_major", "code_switch_count"]:
                samples_df[col] = 0
            else:
                samples_df[col] = np.nan
        return samples_df

    metrics_df = metrics_df.drop_duplicates(subset=["sample_idx"], keep="last")
    metrics_df.set_index("sample_idx", inplace=True)

    # 先把 embedding metrics merge 回 samples_df
    samples_df = samples_df.copy()
    idx_series = samples_df["sample_idx"].astype(int)
    for col in [
        "H_lang", "num_lang_major", "code_switch_count",
        "all_r_dup", "all_cluster_count", "all_cluster_size_max", "all_rho_max", "all_loopiness", "all_plateau_len_max",
        "marker_r_dup", "marker_cluster_count", "marker_cluster_size_max", "marker_rho_max", "marker_loopiness", "marker_plateau_len_max",
        "wait_r_dup", "wait_cluster_count", "wait_cluster_size_max", "wait_rho_max", "wait_loopiness", "wait_plateau_len_max",
        "embedding_truncated",
    ]:
        if col in metrics_df.columns:
            aligned = metrics_df[col].reindex(idx_series).to_numpy()
        else:
            if col in ["num_lang_major", "code_switch_count"]:
                aligned = np.zeros(len(samples_df), dtype=np.int32)
            else:
                aligned = np.full(len(samples_df), np.nan, dtype=np.float32)
        samples_df[col] = aligned

    # # RamblingScore
    # print("[Stage3] Computing rambling_score ...")
    # n_samples = len(samples_df)
    # rambling_scores = np.full(n_samples, np.nan, dtype=np.float32)
    # num_sent_arr = samples_df.get("num_sentences", pd.Series(np.nan, index=samples_df.index)).to_numpy().astype(float)
    # num_self_arr = samples_df.get("num_self_reflect_sentences", pd.Series(np.nan, index=samples_df.index)).to_numpy().astype(float)
    # H_lang_arr = samples_df["H_lang"].to_numpy().astype(float)
    # marker_r_dup_arr = samples_df["marker_r_dup"].to_numpy().astype(float)
    # marker_loopiness_arr = samples_df["marker_loopiness"].to_numpy().astype(float)
    # marker_plateau_arr = samples_df["marker_plateau_len_max"].to_numpy().astype(float)

    # for i in range(n_samples):
    #     num_sent = num_sent_arr[i]
    #     num_self = num_self_arr[i]
    #     if num_sent > 0 and not math.isnan(num_self):
    #         r_self = num_self / num_sent
    #     else:
    #         r_self = float("nan")
    #     rs = compute_rambling_score(
    #         r_self,
    #         marker_r_dup_arr[i],
    #         marker_loopiness_arr[i],
    #         marker_plateau_arr[i],
    #         H_lang_arr[i],
    #         weights=rambling_weights,
    #         loopiness_cap=rambling_loopiness_cap,
    #         plateau_k=rambling_plateau_k,
    #     )
    #     rambling_scores[i] = rs

    # samples_df["rambling_score"] = rambling_scores

    # RamblingScore
    if enable_rambling_score:
        print("[Stage3] Computing rambling_score ...")
        n_samples = len(samples_df)
        rambling_scores = np.full(n_samples, np.nan, dtype=np.float32)
        num_sent_arr = samples_df.get("num_sentences", pd.Series(np.nan, index=samples_df.index)).to_numpy().astype(float)
        num_self_arr = samples_df.get("num_self_reflect_sentences", pd.Series(np.nan, index=samples_df.index)).to_numpy().astype(float)
        H_lang_arr = samples_df["H_lang"].to_numpy().astype(float)
        marker_r_dup_arr = samples_df["marker_r_dup"].to_numpy().astype(float)
        marker_loopiness_arr = samples_df["marker_loopiness"].to_numpy().astype(float)
        marker_plateau_arr = samples_df["marker_plateau_len_max"].to_numpy().astype(float)

        for i in range(n_samples):
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
                weights=rambling_weights,
                loopiness_cap=rambling_loopiness_cap,
                plateau_k=rambling_plateau_k,
            )
            rambling_scores[i] = rs

        samples_df["rambling_score"] = rambling_scores
    else:
        # 保持列存在但全 NaN，方便 downstream 统一处理
        samples_df["rambling_score"] = np.nan


    print(f"[Stage3] Embedding-based metrics done in {time.time()-t0:.1f}s")
    return samples_df

# ---------- pattern-based filter ----------
# 全局 pattern 并行度（文本规则）
PATTERN_WORKERS: int = 1

# tandem repetition 参数
MIN_TOKENS_FOR_REPEAT_CHECK = 30
REPEAT_NGRAM_MIN = 1
REPEAT_NGRAM_MAX = 16
MIN_REPEAT_BLOCKS = 16
MIN_REPEAT_TOTAL_TOKENS = max(20, MIN_REPEAT_BLOCKS)

def _tokens_from_text(resp: str) -> List[str]:
    return re.findall(r"\S+", resp)

def _tokens_to_ids(tokens: List[str]) -> List[int]:
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
    n = len(seq)
    if n < 2 * min_block_len:
        return 1, 0
    max_block_len = min(max_block_len, n // 2)
    if max_block_len < min_block_len:
        return 1, 0

    MOD_MASK = (1 << 64) - 1
    BASE = 911382323

    prefix = [0] * (n + 1)
    pow_base = [1] * (n + 1)
    for i, x in enumerate(seq):
        prefix[i + 1] = (prefix[i] * BASE + (x + 1)) & MOD_MASK
        pow_base[i + 1] = (pow_base[i] * BASE) & MOD_MASK

    def _hash(l: int, r: int) -> int:
        return (prefix[r] - (prefix[l] * pow_base[r - l] & MOD_MASK)) & MOD_MASK

    best_blocks = 1
    best_total_len = 0

    for k in range(min_block_len, max_block_len + 1):
        i = 0
        while i + 2 * k <= n:
            h1 = _hash(i, i + k)
            h2 = _hash(i + k, i + 2 * k)
            if h1 != h2:
                i += 1
                continue
            blocks = 2
            j = i + 2 * k
            while j + k <= n and _hash(j, j + k) == h1:
                blocks += 1
                j += k
            total_len = blocks * k
            if blocks > best_blocks or (blocks == best_blocks and total_len > best_total_len):
                best_blocks = blocks
                best_total_len = total_len
            i = j
    return best_blocks, best_total_len

def _rule_tandem_repetition(resp: str) -> bool:
    if not isinstance(resp, str):
        return False
    s = resp.strip()
    if not s:
        return False
    tokens = _tokens_from_text(s)
    if len(tokens) < MIN_TOKENS_FOR_REPEAT_CHECK:
        return False
    ids = _tokens_to_ids(tokens)
    max_blocks, max_total_len = _max_tandem_repeat_blocks(ids)
    if max_blocks >= MIN_REPEAT_BLOCKS and max_total_len >= MIN_REPEAT_TOTAL_TOKENS:
        return True
    return False

def _is_redundant_noisy_response(resp: str) -> bool:
    if not isinstance(resp, str) or resp == "":
        return False
    RULE_FNS = [
        _rule_tandem_repetition,
        # 以后可以在这里继续加其他 rule
    ]
    for fn in RULE_FNS:
        if fn(resp):
            return True
    return False

def _check_redundant_chunk(responses: List[str]) -> List[bool]:
    out: List[bool] = []
    for resp in responses:
        out.append(_is_redundant_noisy_response(resp))
    return out

def pattern_redundant_answer_noise(df: pd.DataFrame) -> pd.Series:
    print("[Pattern] pattern_redundant_answer_noise ...")
    if "response_text" not in df.columns:
        return pd.Series(False, index=df.index)
    responses = df["response_text"].astype(str).tolist()
    n = len(responses)
    if n == 0:
        return pd.Series(False, index=df.index)
    workers = max(1, int(PATTERN_WORKERS or 1))
    if workers == 1 or n < 1000:
        flags = [_is_redundant_noisy_response(resp) for resp in responses]
    else:
        chunk_size = max(1, n // (workers * 4))
        chunks: List[List[str]] = []
        for i in range(0, n, chunk_size):
            chunks.append(responses[i:i+chunk_size])
        flags: List[bool] = []
        with mp.Pool(processes=workers) as pool:
            for part in pool.map(_check_redundant_chunk, chunks):
                flags.extend(part)
    return pd.Series(flags, index=df.index)

def pattern_ce_ref_mean_too_large(df: pd.DataFrame) -> pd.Series:
    print("[Pattern] pattern_ce_ref_mean_too_large ...")
    if "ce_ref_seq_mean" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["ce_ref_seq_mean"].astype(float) > 1.8

def pattern_repeat_word_run_max_too_large(df: pd.DataFrame) -> pd.Series:
    print("[Pattern] pattern_repeat_word_run_max_too_large ...")
    if "repeat_word_run_max" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["repeat_word_run_max"].astype(float) > 25

def pattern_num_self_reflect_sentences_too_more(df: pd.DataFrame) -> pd.Series:
    print("[Pattern] pattern_num_self_reflect_sentences_too_more ...")
    if "num_self_reflect_sentences" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["num_self_reflect_sentences"].astype(float) > 40

def pattern_marker_loopiness_too_more(df: pd.DataFrame) -> pd.Series:
    print("[Pattern] pattern_marker_loopiness_too_more ...")
    if "marker_loopiness" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["marker_loopiness"].astype(float) > 50

# 这里集中管理 pattern 函数，后续你只需要注释/取消注释就能开关
PATTERN_FUNCS = [
    ("ce_ref_mean_too_large", pattern_ce_ref_mean_too_large),
    ("repeat_word_run_max_too_large", pattern_repeat_word_run_max_too_large),
    ("num_self_reflect_sentences_too_more", pattern_num_self_reflect_sentences_too_more),
    ("marker_loopiness_too_more", pattern_marker_loopiness_too_more),
    ("redundant_answer_noise", pattern_redundant_answer_noise),
    # 以后新增 pattern 在这里加入 ("name", func)
]

def build_filter_mask_by_patterns(
    df: pd.DataFrame,
    target_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    返回:
      keep_mask: True=保留
      bad_mask: True=被 pattern 命中
      pattern_hits: dict[name -> bool array]
    """
    n = len(df)
    if n == 0:
        return np.ones(0, dtype=bool), np.zeros(0, dtype=bool), {}
    if target_mask is None:
        target_mask_series = pd.Series(True, index=df.index)
    else:
        if len(target_mask) != n:
            raise ValueError("target_mask length mismatch")
        target_mask_series = pd.Series(target_mask, index=df.index)

    bad_mask = pd.Series(False, index=df.index)
    pattern_hits: Dict[str, np.ndarray] = {}

    for name, fn in PATTERN_FUNCS:
        m = fn(df)
        if m is None:
            continue
        if len(m) != n:
            raise ValueError(f"Pattern {name} returned mask of wrong length")
        hit = (m & target_mask_series)
        pattern_hits[name] = hit.to_numpy()
        bad_mask |= hit

    keep_mask = ~bad_mask
    return keep_mask.to_numpy(), bad_mask.to_numpy(), pattern_hits

# ---------- debug metrics list ----------
TEXT_METRIC_COLUMNS = [
    "resp_char_len", "resp_byte_len",
    "non_ascii_ratio", "replacement_char_ratio",
    "control_private_unassigned_ratio",
    "repeat_char_run_max", "repeat_word_run_max",
    "unique_word_ratio", "top1_word_prop",
    "trigram_repeat_ratio",
    "compress_ratio",
    "char_entropy_bits_per_char",
    "num_sections", "num_sentences",
    "num_marker_sentences",
    "num_self_reflect_sentences",
    "num_contrast_sentences",
    "num_conclude_sentences",
    "num_marker_sections",
    "contain_chinese",
    "ce_self_seq_mean", "ce_self_seq_sum",
    "ce_ref_seq_mean", "ce_ref_seq_sum",
    "reward_seq_sum",
    "kl_seq_mean_log_ratio",
    "entropy_seq_mean",
    "response_len_tokens",
]

EMBEDDING_METRIC_COLUMNS = [
    "H_lang", "num_lang_major", "code_switch_count",
    "all_r_dup", "all_cluster_count", "all_cluster_size_max", "all_rho_max", "all_loopiness", "all_plateau_len_max",
    "marker_r_dup", "marker_cluster_count", "marker_cluster_size_max", "marker_rho_max", "marker_loopiness", "marker_plateau_len_max",
    "wait_r_dup", "wait_cluster_count", "wait_cluster_size_max", "wait_rho_max", "wait_loopiness", "wait_plateau_len_max",
    "embedding_truncated",
    "rambling_score",
]

FILTER_RELATED_COLUMNS = TEXT_METRIC_COLUMNS + EMBEDDING_METRIC_COLUMNS

@dataclass
class MetricSwitches:
    enabled_metrics: set
    enable_text_metrics: bool
    enable_contains_chinese: bool
    enable_split_sections_and_sentences: bool
    enable_detect_markers: bool
    enable_detect_lang: bool
    enable_embedding: bool
    enable_embedding_metrics: bool
    enable_rambling_score: bool


def resolve_metric_dependencies() -> MetricSwitches:
    enabled = set(ENABLED_METRICS)

    # 纯 text_metrics
    text_metric_base = {
        "resp_char_len", "resp_byte_len",
        "non_ascii_ratio", "replacement_char_ratio",
        "control_private_unassigned_ratio",
        "repeat_char_run_max", "repeat_word_run_max",
        "unique_word_ratio", "top1_word_prop",
        "trigram_repeat_ratio",
        "compress_ratio",
        "char_entropy_bits_per_char",
    }
    # section / marker 相关计数
    section_metrics = {
        "num_sections", "num_sentences",
        "num_marker_sentences",
        "num_self_reflect_sentences",
        "num_contrast_sentences",
        "num_conclude_sentences",
        "num_marker_sections",
    }
    # 基于 langid 的 sample 级指标
    lang_metrics = {"H_lang", "num_lang_major", "code_switch_count"}
    # 纯 embedding-based 指标（不含 H_lang / rambling_score）
    embedding_only_metrics = {
        "all_r_dup", "all_cluster_count", "all_cluster_size_max", "all_rho_max",
        "all_loopiness", "all_plateau_len_max",
        "marker_r_dup", "marker_cluster_count", "marker_cluster_size_max",
        "marker_rho_max", "marker_loopiness", "marker_plateau_len_max",
        "wait_r_dup", "wait_cluster_count", "wait_cluster_size_max",
        "wait_rho_max", "wait_loopiness", "wait_plateau_len_max",
        "embedding_truncated",
    }

    need_text = bool(enabled & text_metric_base)
    need_contains_chinese = "contain_chinese" in enabled
    need_rambling = "rambling_score" in enabled
    need_lang_stats = bool(enabled & lang_metrics) or need_rambling
    need_embedding_metrics = bool(enabled & embedding_only_metrics) or need_rambling

    # 需要 markers 的指标（包括 marker_* / wait_* 以及 marker 计数）
    metrics_requiring_markers = {
        "num_marker_sentences",
        "num_self_reflect_sentences",
        "num_contrast_sentences",
        "num_conclude_sentences",
        "num_marker_sections",
        "marker_r_dup", "marker_cluster_count", "marker_cluster_size_max",
        "marker_rho_max", "marker_loopiness", "marker_plateau_len_max",
        "wait_r_dup", "wait_cluster_count", "wait_cluster_size_max",
        "wait_rho_max", "wait_loopiness", "wait_plateau_len_max",
    }
    need_markers = bool(enabled & metrics_requiring_markers) or need_rambling

    # 是否需要做 split_sections/split_sentences
    need_split = bool(
        enabled & section_metrics
        or need_markers
        or need_lang_stats
        or need_embedding_metrics
    )

    # 应用用户手动开关 + 依赖修正
    detect_markers_flag = ENABLE_DETECT_MARKERS
    detect_lang_flag = ENABLE_DETECT_LANG

    if need_markers and not detect_markers_flag:
        print("[WARN] metrics need discourse markers, but ENABLE_DETECT_MARKERS=False; force enabling detect_markers.")
        detect_markers_flag = True
    if need_lang_stats and not detect_lang_flag:
        print("[WARN] metrics need sentence-level langid, but ENABLE_DETECT_LANG=False; force enabling detect_lang.")
        detect_lang_flag = True

    if not need_split and not detect_markers_flag and not detect_lang_flag:
        enable_split = False
    else:
        enable_split = True

    # Stage2 / Stage3 embedding
    enable_embedding = need_embedding_metrics
    enable_embedding_metrics = need_embedding_metrics

    print("[INFO] Metric switches:")
    print(f"  text_metrics            = {need_text}")
    print(f"  contain_chinese         = {need_contains_chinese}")
    print(f"  split/markers/lang      = {enable_split} (markers={detect_markers_flag}, lang={detect_lang_flag})")
    print(f"  embedding (vLLM)        = {enable_embedding}")
    print(f"  embedding metrics       = {enable_embedding_metrics}")
    print(f"  rambling_score          = {need_rambling}")

    return MetricSwitches(
        enabled_metrics=enabled,
        enable_text_metrics=need_text,
        enable_contains_chinese=need_contains_chinese,
        enable_split_sections_and_sentences=enable_split,
        enable_detect_markers=detect_markers_flag,
        enable_detect_lang=detect_lang_flag,
        enable_embedding=enable_embedding,
        enable_embedding_metrics=enable_embedding_metrics,
        enable_rambling_score=need_rambling,
    )

# ---------- main ----------
import multiprocessing as mp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True, help="输入的 RL 训练 JSONL")
    ap.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    ap.add_argument("--cpu-procs", type=int, default=max(1, (os.cpu_count() or 8)//2))
    ap.add_argument("--chunk-lines", type=int, default=20000)
    ap.add_argument("--compute-sha256", action="store_true")
    ap.add_argument("--correct-key", type=str, default="reward_extra_infos.acc")
    ap.add_argument("--min-lang-chars", type=int, default=5)
    ap.add_argument("--step-min", type=int, default=None)
    ap.add_argument("--step-max", type=int, default=None)
    # 预加载阶段子集选择
    ap.add_argument("--kept-or-dropped", type=str, default="all", choices=["all", "kept", "dropped"], help="在加载阶段选择 kept/dropped 子集（默认 all 不过滤）",)
    ap.add_argument("--cls",type=str,default="any",choices=["any", "correct", "wrong", "no_judge"],help="在加载阶段按 cls_id 选择子集（默认 any 不过滤）",)
    # embedding backend 选择
    ap.add_argument(
        "--embed-backend",
        type=str,
        default="vllm",
        choices=["vllm", "hf"],
        help="选择 embedding 后端: vllm 或 huggingface(transformers)，默认 vllm",
    )
    # embedding model
    ap.add_argument("--embed-model-name", type=str, default="BAAI/bge-m3")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    # HuggingFace backend 相关参数
    ap.add_argument(
        "--hf-max-workers",
        type=int,
        default=0,
        help="HF 多卡模式下最多使用的 worker 数（也即使用的 GPU 数）。"
             "0 或负数表示用完所有可见 GPU。",
    )
    ap.add_argument(
        "--hf-device",
        type=str,
        default="auto",
        help="HF 后端的 device: 'auto','cuda','cpu' 等，默认 auto 自动检测",
    )
    ap.add_argument(
        "--hf-batch-size",
        type=int,
        default=64,
        help="HF embedding 的 batch size，默认 64；<=0 表示不切 batch，一次性送入模型（显存不足可能会 OOM）",
    )
    ap.add_argument(
        "--hf-max-length",
        type=int,
        default=512,
        help="HF tokenizer 的 max_length（截断），默认 512",
    )
    ap.add_argument(
        "--hf-torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="HF 模型的 torch_dtype，默认 auto 由 transformers 自己决定",
    )
    # embedding analysis
    ap.add_argument("--dup-threshold", type=float, default=0.90)
    ap.add_argument("--cluster-threshold", type=float, default=0.85)
    ap.add_argument("--plateau-min-len", type=int, default=3)
    ap.add_argument("--loop-eps", type=float, default=1e-4)
    ap.add_argument("--max-sent-per-sample", type=int, default=128)

    # RamblingScore
    ap.add_argument("--rambling-weight-self", type=float, default=1.0)
    ap.add_argument("--rambling-weight-dup", type=float, default=1.0)
    ap.add_argument("--rambling-weight-loop", type=float, default=1.0)
    ap.add_argument("--rambling-weight-plateau", type=float, default=1.0)
    ap.add_argument("--rambling-weight-lang", type=float, default=0.1)
    ap.add_argument("--rambling-loopiness-cap", type=float, default=5.0)
    ap.add_argument("--rambling-plateau-k", type=int, default=4)

    # pattern workers
    ap.add_argument("--pattern-workers", type=int, default=0)

    # debug
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    t_pipeline_start = time.time()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl = args.jsonl
    if not jsonl.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl}")

    # pattern 并行度
    global PATTERN_WORKERS
    cpu_total = os.cpu_count() or 1
    if args.pattern_workers <= 0:
        PATTERN_WORKERS = max(1, min(cpu_total, 32))
    else:
        PATTERN_WORKERS = max(1, min(args.pattern_workers, cpu_total))
    print(f"[INFO] pattern_workers = {PATTERN_WORKERS} (available CPUs = {cpu_total})")

    # ---------- Metric switches & dependency check ----------
    metric_switches = resolve_metric_dependencies()

    # ---------- Stage0: scan ----------
    # cfg = BuildConfig(
    #     jsonl=jsonl,
    #     correct_key=args.correct_key,
    #     min_lang_chars=args.min_lang_chars,
    #     cpu_procs=int(args.cpu_procs),
    #     chunk_lines=int(args.chunk_lines),
    #     compute_sha256=bool(args.compute_sha256),
    #     step_min=args.step_min,
    #     step_max=args.step_max,
    #     debug=bool(args.debug),
    # )
    cfg = BuildConfig(
        jsonl=jsonl,
        correct_key=args.correct_key,
        min_lang_chars=args.min_lang_chars,
        cpu_procs=int(args.cpu_procs),
        chunk_lines=int(args.chunk_lines),
        compute_sha256=bool(args.compute_sha256),
        step_min=args.step_min,
        step_max=args.step_max,
        debug=bool(args.debug),

        # 根据 resolve_metric_dependencies() 的结果设置 Stage1 开关
        enable_text_metrics=metric_switches.enable_text_metrics,
        enable_contains_chinese=metric_switches.enable_contains_chinese,
        enable_split_sections_and_sentences=metric_switches.enable_split_sections_and_sentences,
        enable_detect_markers=metric_switches.enable_detect_markers,
        enable_detect_lang=metric_switches.enable_detect_lang,
    )


    # ---------- Stage0: scan + 预过滤子集 ----------
    print(f"[Scan] Scanning offsets in JSONL (with pre-filter): {cfg.jsonl}")
    t0 = time.time()
    chunks, sha256, fsize, mtime, n_lines, scan_stats = scan_offsets(
        cfg,
        kept_or_dropped=args.kept_or_dropped,
        cls_filter=args.cls,
    )

    print(f"[Scan] File total lines        = {scan_stats['num_total']:,}")
    print(f"[Scan]   kept samples          = {scan_stats['num_kept']:,}")
    print(f"[Scan]   dropped samples       = {scan_stats['num_dropped']:,}")
    print(f"[Scan]   kept & correct        = {scan_stats['num_kept_correct']:,}")
    print(f"[Scan]   kept & wrong          = {scan_stats['num_kept_wrong']:,}")
    print(f"[Scan] Selected subset lines   = {scan_stats['num_selected']:,} "
          f"(after step/kept/cls pre-filter)")
    print(f"[Scan] Chunks to process       = {len(chunks):,}; "
          f"scan took {time.time()-t0:.1f}s")
    if cfg.compute_sha256:
        print(f"[Scan] sha256={sha256}")

    # ---------- Stage1-like: CPU features ----------
    print(f"[Stage1] Processing chunks with {cfg.cpu_procs} processes ...")
    t1 = time.time()
    tasks = [(str(cfg.jsonl), offs, cfg) for offs in chunks]
    start_method = "spawn" if sys.platform == "win32" else "fork"
    ctx = mp.get_context(start_method)
    samples_list: List[pd.DataFrame] = []
    segments_list: List[pd.DataFrame] = []

    with ctx.Pool(processes=cfg.cpu_procs) as pool:
        for i, ret in enumerate(
            tqdm(pool.imap_unordered(_worker_entry_stage1, tasks, chunksize=1),
                 total=len(tasks),
                 desc="[Stage1] Processing chunks")
        ):
            df_s = ret["samples"]
            df_seg = ret["segments"]
            if not df_s.empty:
                samples_list.append(df_s)
            if not df_seg.empty:
                segments_list.append(df_seg)

    if samples_list:
        samples_df = pd.concat(samples_list, ignore_index=True)
    else:
        samples_df = pd.DataFrame()

    if segments_list:
        segments_df = pd.concat(segments_list, ignore_index=True)
    else:
        segments_df = pd.DataFrame()


    print(f"[Stage1] Done: samples rows = {len(samples_df):,}, segments rows = {len(segments_df):,}, time = {time.time()-t1:.1f}s")

    # 注意：step / kept / cls 的子集已经在 Stage0 完成，这里不再做过滤，
    # 后续所有 embedding / pattern 过滤都在这个子集上工作。
    num_kept = int((samples_df["kept_flag"] == 1).sum())
    is_kept = (samples_df["kept_flag"] == 1)

    # ---------- 从 JSONL 中补充 response_text（for pattern_redundant_answer_noise） ----------
    if "response_text" not in samples_df.columns:
        print("[INFO] Attaching response_text from source JSONL ...")
        t_txt0 = time.time()
        resp_list: List[str] = []
        with cfg.jsonl.open("rb", buffering=1024*1024) as fsrc:
            fileno = fsrc.fileno()
            for _, row in samples_df.iterrows():
                off = int(row["byte_offset"])
                ln = int(row["byte_len"])
                b = os.pread(fileno, ln, off)
                try:
                    j = _json_loads(b)
                except Exception:
                    resp_list.append("")
                    continue
                resp_list.append(j.get("response_text", ""))
        samples_df["response_text"] = resp_list
        print(f"[INFO] response_text attached, took {time.time()-t_txt0:.1f}s")

    # # ---------- Stage2: vLLM embedding ----------
    # t2 = time.time()
    # segments_df = run_vllm_embedding(
    #     segments_df,
    #     model_name=args.embed_model_name,
    #     tensor_parallel_size=args.tensor_parallel_size,
    #     dtype=args.dtype,
    #     gpu_memory_utilization=args.gpu_memory_utilization,
    #     trust_remote_code=args.trust_remote_code,
    #     enforce_eager=args.enforce_eager,
    # )
    # print(f"[Stage2] Embedding stage finished in {time.time()-t2:.1f}s")

    # # ---------- Stage2: vLLM embedding ----------
    # if metric_switches.enable_embedding and not segments_df.empty:
    #     t2 = time.time()
    #     segments_df = run_vllm_embedding(
    #         segments_df,
    #         model_name=args.embed_model_name,
    #         tensor_parallel_size=args.tensor_parallel_size,
    #         dtype=args.dtype,
    #         gpu_memory_utilization=args.gpu_memory_utilization,
    #         trust_remote_code=args.trust_remote_code,
    #         enforce_eager=args.enforce_eager,
    #     )
    #     print(f"[Stage2] Embedding stage finished in {time.time()-t2:.1f}s")
    # else:
    #     print("[Stage2] Embedding stage is disabled or there is no segment; skip vLLM.embed().")
    # ---------- Stage2: embedding ----------
    if metric_switches.enable_embedding and not segments_df.empty:
        t2 = time.time()
        if args.embed_backend == "vllm":
            print("[Stage2] Using vLLM backend for embedding ...")
            segments_df = run_vllm_embedding(
                segments_df,
                model_name=args.embed_model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=args.trust_remote_code,
                enforce_eager=args.enforce_eager,
            )
        else:
            print("[Stage2] Using HuggingFace (transformers) backend for embedding ...")
            segments_df = run_hf_embedding(
                segments_df,
                model_name=args.embed_model_name,
                hf_device=args.hf_device,
                hf_batch_size=args.hf_batch_size,
                hf_max_length=args.hf_max_length,
                hf_torch_dtype=args.hf_torch_dtype,
                trust_remote_code=args.trust_remote_code,
                hf_max_workers=args.hf_max_workers,
            )
        print(f"[Stage2] Embedding stage finished in {time.time()-t2:.1f}s")
    else:
        print("[Stage2] Embedding stage is disabled or there is no segment; skip embedding.")

    # ---------- Stage3: embedding-based metrics ----------
    rambling_weights = {
        "self": args.rambling_weight_self,
        "dup": args.rambling_weight_dup,
        "loop": args.rambling_weight_loop,
        "plateau": args.rambling_weight_plateau,
        "lang": args.rambling_weight_lang,
    }
    # samples_df = run_embedding_analysis_for_all_samples(
    #     samples_df,
    #     segments_df,
    #     dup_threshold=args.dup_threshold,
    #     cluster_threshold=args.cluster_threshold,
    #     plateau_min_len=args.plateau_min_len,
    #     loop_eps=args.loop_eps,
    #     max_sent_per_sample=args.max_sent_per_sample,
    #     rambling_weights=rambling_weights,
    #     rambling_loopiness_cap=args.rambling_loopiness_cap,
    #     rambling_plateau_k=args.rambling_plateau_k,
    # )
    samples_df = run_embedding_analysis_for_all_samples(
        samples_df,
        segments_df,
        dup_threshold=args.dup_threshold,
        cluster_threshold=args.cluster_threshold,
        plateau_min_len=args.plateau_min_len,
        loop_eps=args.loop_eps,
        max_sent_per_sample=args.max_sent_per_sample,
        rambling_weights=rambling_weights,
        rambling_loopiness_cap=args.rambling_loopiness_cap,
        rambling_plateau_k=args.rambling_plateau_k,
        enable_embedding_metrics=metric_switches.enable_embedding_metrics,
        enable_rambling_score=metric_switches.enable_rambling_score,
    )

    # ---------- Pattern filtering ----------
    print("[Filter] Running pattern-based filtering on current subset ...")
    t3 = time.time()
    keep_mask, bad_mask, pattern_hits = build_filter_mask_by_patterns(
        samples_df,
        target_mask=None,  # 在 Stage0 已经选好子集，这里对当前子集全量应用 pattern
    )
    print(f"[Filter] Pattern filtering finished in {time.time()-t3:.1f}s")

    selected_cnt = int(bad_mask.sum())
    bad_in_kept = int((bad_mask & is_kept.to_numpy()).sum())
    ratio_bad_in_kept = (bad_in_kept / num_kept) if num_kept > 0 else 0.0
    print(f"[STATS] pattern-matched samples (overall)  = {selected_cnt:,}")
    print(f"[STATS] pattern-matched within kept       = {bad_in_kept:,} ({ratio_bad_in_kept:.4%} of kept)")

    df_bad = samples_df[bad_mask].copy()
    df_good = samples_df[~bad_mask].copy()

    print(f"[Filter] df_bad rows  = {len(df_bad):,}")
    print(f"[Filter] df_good rows = {len(df_good):,}")

    # ---------- 写出 good / bad JSONL（附带 debug_filter） ----------
    out_good = out_dir / "good_samples.jsonl"
    out_bad = out_dir / "bad_samples.jsonl"
    print(f"[Output] Writing good samples to {out_good}")
    print(f"[Output] Writing bad samples to  {out_bad}")

    # 把 pattern_hits 转成 per-row 的列表，便于 debug_filter
    pattern_names = [name for (name, _) in PATTERN_FUNCS]
    # per-row: 哪些 pattern hit
    hit_patterns_per_row: List[List[str]] = [[] for _ in range(len(samples_df))]
    for pname in pattern_names:
        if pname not in pattern_hits:
            continue
        mask_p = pattern_hits[pname]
        for i, flag in enumerate(mask_p):
            if flag:
                hit_patterns_per_row[i].append(pname)

    # 构建一个快捷索引: row_idx -> metrics dict
    # def build_metrics_for_row(row: pd.Series) -> Dict[str, Any]:
    #     m: Dict[str, Any] = {}
    #     for col in FILTER_RELATED_COLUMNS:
    #         if col in row:
    #             val = row[col]
    #             if isinstance(val, (np.generic,)):
    #                 val = val.item()
    #             m[col] = val
    #     return m
    def build_metrics_for_row(row: pd.Series) -> Dict[str, Any]:
        m: Dict[str, Any] = {}
        enabled = set(ENABLED_METRICS)
        for col in FILTER_RELATED_COLUMNS:
            if col not in enabled:
                continue
            if col in row:
                val = row[col]
                if isinstance(val, (np.generic,)):
                    val = val.item()
                m[col] = val
        return m

    with cfg.jsonl.open("rb", buffering=1024*1024) as fsrc, \
         out_good.open("w", encoding="utf-8") as f_good, \
         out_bad.open("w", encoding="utf-8") as f_bad:
        fileno = fsrc.fileno()

        # 但这里我们其实按 samples_df 的顺序遍历更方便
        for global_idx, row in samples_df.iterrows():
            off = int(row["byte_offset"])
            ln = int(row["byte_len"])
            b = os.pread(fileno, ln, off)
            try:
                j = _json_loads(b)
            except Exception:
                j = {}
            # 构建 debug_filter 信息
            row_bad = bool(bad_mask[global_idx])
            hit_list = hit_patterns_per_row[global_idx]
            metrics = build_metrics_for_row(row)
            debug_info = {
                "hit": row_bad,
                "hit_patterns": hit_list,
                "cls_id": int(row.get("cls_id", 2)),
                "kept_flag": int(row.get("kept_flag", -1)),
                "global_step": int(row.get("global_step", -1)),
                "sample_idx": int(row.get("sample_idx", -1)),
                "metrics": metrics,
            }
            if args.debug:
                debug_info["debug_note"] = "pattern filtering debug info"

            # 附加到原始 JSON
            if isinstance(j, dict):
                j["filter_debug"] = debug_info
                line_out = json.dumps(j, ensure_ascii=False)
            else:
                # 原始不是 dict，直接包装
                j2 = {"raw": str(j), "filter_debug": debug_info}
                line_out = json.dumps(j2, ensure_ascii=False)

            if row_bad:
                f_bad.write(line_out + "\n")
            else:
                f_good.write(line_out + "\n")

    t_pipeline_end = time.time()
    print("[DONE] All pipeline finished.")
    print(f"  total samples    = {scan_stats['num_selected']:,}")
    print(f"  bad samples      = {selected_cnt:,}")
    print(f"  bad in kept      = {bad_in_kept:,} ({ratio_bad_in_kept:.4%} of kept)")
    print(f"  good samples     = {len(df_good):,}")
    print(f"  JSONL in         = {cfg.jsonl}")
    print(f"  good_samples     = {out_good}")
    print(f"  bad_samples      = {out_bad}")
    print(f"  Total pipeline time = {t_pipeline_end - t_pipeline_start:.1f}s")

if __name__ == "__main__":
    main()
