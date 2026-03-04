#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features_preprocess.py

Stage 1: 从超大 JSONL 构建基础特征 + 文本切分（section/sentence）+ discourse marker + 句子级语言信息。

输出到 --out-dir:
  - features.parquet : 每个 sample 一行（沿用你原来的 features 设计 + 新增字段）
  - segments.parquet : 每个 sentence 一行（后续 embedding/分析用）
  - meta_preprocess.json : 元信息（schema/version/参数等）

设计要点：
- 单次顺序扫描 JSONL，先记录 (line_no, offset, length)。
- 多进程 worker，各自 reopen 源文件，用 pread() 按 offset 解析 JSON，并计算特征。
- 保留原先 build_features_from_jsonl 的所有 feature 计算逻辑。
- 新增分段/分句/marker/language 信息，写入 segments.parquet。
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse, json, os, sys, time, math, hashlib, re
from typing import Any, Dict, List, Tuple, Optional
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # 新增：进度条
# ---------- fast json ----------
try:
    import orjson
    def _json_loads(b: bytes):
        return orjson.loads(b)
except Exception:
    import json as _json
    def _json_loads(b: bytes):
        return _json.loads(b.decode("utf-8", errors="replace"))

# ---------- text metrics (沿用你的实现) ----------
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
    """沿用你原来的字符级指标."""
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

def sanitize_snap_col(path: str) -> str:
    return "snap__" + path.replace(".", "__")

@dataclass
class BuildConfig:
    jsonl: Path
    out_dir: Path
    procs: int
    chunk_lines: int
    correct_key: Optional[str]
    snapshot_keys: List[str]
    compute_sha256: bool
    min_lang_chars: int

SCHEMA_VERSION = 2  # 新 schema

# ---------- discourse markers (多语言) ----------
# SELF_REFLECT / CONTRAST / CONCLUDE
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
    r"itak",  # итак 的常见拉丁转写
    r"итак",
    r"значит",
]

SELF_REFLECT_RE = re.compile("|".join(SELF_REFLECT_PATTERNS), re.IGNORECASE)
CONTRAST_RE     = re.compile("|".join(CONTRAST_PATTERNS),     re.IGNORECASE)
CONCLUDE_RE     = re.compile("|".join(CONCLUDE_PATTERNS),     re.IGNORECASE)

# 单独标记 "wait-like"（更细一点）
WAITLIKE_RE = re.compile(r"\bwait\b|\bhold on\b|\bhang on\b|等一下|稍等|подожди|подождите", re.IGNORECASE)

# marker bitmask: bit0=self_reflect, bit1=contrast, bit2=conclude
MARKER_SELF_REFLECT = 1
MARKER_CONTRAST     = 2
MARKER_CONCLUDE     = 4

def detect_markers(text: str) -> Tuple[int, int, int, int]:
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
    """
    按 \n+ 切成 section。
    每个 section 包含前面的 \n run。
    返回 list[(section_text, start_idx, end_idx)]，end_idx 为原字符串切片的 end（不包含）。
    """
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
            # [i, j) 是一段连续的 \n，属于前一个 section 的结尾
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
    """
    在一个 section 内用 pysbd 切分 sentence。
    返回 list[(sent_text, global_start, global_end)]。
    """
    if not section_text:
        return []
    if _SEGMENTER is None:
        # 没有 pysbd 时，整个 section 当成一个句子
        return [(section_text, sec_start, sec_start + len(section_text))]

    try:
        spans = _SEGMENTER.segment(section_text)
        results: List[Tuple[str, int, int]] = []
        if spans and _SEGMENTER_CHARSPAN and hasattr(spans[0], "start"):
            # 新版 pysbd: 返回带 start/end 的对象
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
            # 老版：只返回字符串，手动对齐
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
    # 可选：限制语言集合，加速 + 稳一点
    langid.set_languages(['en', 'zh', 'de', 'ru', 'it', 'fr', 'es', 'ja'])
except Exception:
    _HAS_LANGID = False

def detect_lang(text: str, min_len: int = 5) -> Tuple[str, float]:
    """
    用 langid 检测句子主语言。
    返回 (lang_code, confidence)，如果无法检测则 ('unk', 0.0)。
    对特别短的句子/噪声，直接返回 unk。
    """
    if not text or len(text) < min_len or not _HAS_LANGID:
        return "unk", 0.0
    try:
        lang, conf = langid.classify(text)
        return str(lang), float(conf)
    except Exception:
        return "unk", 0.0

# ---------- worker ----------
def _worker_process(jsonl_path: str,
                    offsets: List[Tuple[int,int,int]],
                    correct_key: Optional[str],
                    snapshot_keys: List[str],
                    min_lang_chars: int) -> Dict[str, Any]:
    """
    offsets: list of (line_no, offset, length)
    返回：
      - df_samples: 每样本一行
      - df_segments: 每 sentence 一行
    """
    fd = os.open(jsonl_path, os.O_RDONLY)
    rows_samples: List[Dict[str, Any]] = []
    rows_segments: List[Dict[str, Any]] = []

    snap_cols = [(k, sanitize_snap_col(k)) for k in snapshot_keys]

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

        sample_idx = line_no - 1  # 全局 sample 索引（0-based）

        rec: Dict[str, Any] = {
            "sample_idx": np.int64(sample_idx),
            "line_no": np.int64(line_no),
            "byte_offset": np.int64(off),
            "byte_len": np.int32(ln),
            "valid": np.int8(1 if isinstance(obj, dict) else 0),
            "global_step": np.int32(get_nested(obj, "global_step", -1)) if obj else np.int32(-1),
            "uid": str(obj.get("uid")) if (obj and "uid" in obj) else "",
            "kept_flag": np.int8(1 if (obj and obj.get("filter_state")=="kept")
                             else (0 if (obj and obj.get("filter_state")=="dropped") else -1)),
            "ce_self_seq_mean": np.float32(neg(obj.get("old_logprob_mean") if obj else None)),
            "ce_self_seq_sum":  np.float32(neg(obj.get("old_logprob_sum") if obj else None)),
            "ce_ref_seq_mean":  np.float32(neg(obj.get("ref_logprob_mean") if obj else None)),
            "ce_ref_seq_sum":   np.float32(neg(obj.get("ref_logprob_sum") if obj else None)),
            "reward_seq_sum":   np.float32(obj.get("reward_seq_sum") if obj else np.nan),
            "kl_seq_mean_log_ratio": np.float32(obj.get("kl_seq_mean_log_ratio") if obj else np.nan),
            "entropy_seq_mean":      np.float32(obj.get("entropy_seq_mean") if obj else np.nan),
            "response_len_tokens":   np.float32(obj.get("response_len_tokens") if obj else np.nan),
            "cls_id": np.int8(classify_record(obj, correct_key=correct_key) if obj else 2),
        }

        rt = obj.get("response_text") if isinstance(obj, dict) else None
        pt = obj.get("prompt_text") if isinstance(obj, dict) else None

        # 是否包含中文（prompt/response）
        if isinstance(rt, str) or isinstance(pt, str):
            has_cn = (contains_chinese(rt) if isinstance(rt, str) else False) \
                     or (contains_chinese(pt) if isinstance(pt, str) else False)
            rec["contain_chinese"] = np.int8(1 if has_cn else 0)
        else:
            rec["contain_chinese"] = np.int8(-1)

        # 文本级指标（response）
        if isinstance(rt, str):
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

        # snapshots
        if isinstance(obj, dict):
            for path, col in snap_cols:
                try:
                    val = get_nested(obj, path, None)
                except Exception:
                    val = None
                rec[col] = val
        else:
            for _, col in snap_cols:
                rec[col] = None

        # ---------- section / sentence 切分 & marker & language ----------
        num_sections = 0
        num_sentences = 0
        num_marker_sent = 0
        num_self_sent = 0
        num_contrast_sent = 0
        num_conclude_sent = 0
        num_marker_sec = 0

        if isinstance(rt, str) and rt:
            sections = split_sections(rt)
            num_sections = len(sections)

            for sec_idx, (sec_text, sec_start, sec_end) in enumerate(sections):
                section_has_marker = False
                sent_list = split_sentences(sec_text, sec_start)
                if not sent_list:
                    sent_list = [(sec_text, sec_start, sec_end)]
                for local_sent_idx, (sent_text, global_s, global_e) in enumerate(sent_list):
                    sent_idx = num_sentences  # 全局 sentence 序号（在该 response 内）

                    marker_mask, sr, ct, cd, waitlike = detect_markers(sent_text)
                    if marker_mask != 0:
                        section_has_marker = True
                        num_marker_sent += 1
                    if sr:
                        num_self_sent += 1
                    if ct:
                        num_contrast_sent += 1
                    if cd:
                        num_conclude_sent += 1

                    lang_main, lang_conf = detect_lang(sent_text, min_len=min_lang_chars)

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

def _worker_entry(args):
    return _worker_process(*args)

def _scan_offsets(jsonl: Path, chunk_lines: int, compute_sha256: bool):
    """
    Sequentially iterate binary lines to get (line_no, offset, length).
    Also compute sha256 if requested (in-stream).

    这里加一个按字节数的 tqdm 进度条，方便观察扫描进度和预估时间。
    """
    offsets: List[Tuple[int,int,int]] = []
    chunks: List[List[Tuple[int,int,int]]] = []
    hasher = hashlib.sha256() if compute_sha256 else None

    # 先拿到文件大小，用于 tqdm 的 total
    stat = jsonl.stat()
    total_bytes = stat.st_size

    with jsonl.open("rb", buffering=1024*1024) as f:
        line_no = 0
        processed_bytes = 0

        pbar = tqdm(
            total=total_bytes,
            desc="[Stage1] Scanning JSONL",
            unit="B",
            unit_scale=True,
        )

        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                if offsets:
                    chunks.append(offsets)
                    offsets = []
                break

            line_no += 1
            if hasher:
                hasher.update(line)
            ln = len(line)
            processed_bytes += ln
            pbar.update(ln)

            offsets.append((line_no, off, ln))
            if len(offsets) >= chunk_lines:
                chunks.append(offsets)
                offsets = []

        pbar.close()

    sha256 = hasher.hexdigest() if hasher else ""
    # stat 前面已经取过，这里直接用
    return chunks, sha256, stat.st_size, int(stat.st_mtime)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 8)//2))
    ap.add_argument("--chunk-lines", type=int, default=20000)
    ap.add_argument("--correct-key", type=str, default=None,
                    help="Dot path for correctness label, e.g., reward_extra_infos.acc")
    ap.add_argument("--snapshot-keys", type=str, default="epoch,global_step,uid,filter_state,prompt_text,response_text,prompt_len_tokens,response_len_tokens,total_len_tokens,reward_seq_sum,reward_extra_infos.acc",
                    help="Comma separated dot-keys to snapshot for fast export.")
    ap.add_argument("--compute-sha256", action="store_true",
                    help="Compute sha256 of source file for strong validation (costly on very large files).")
    ap.add_argument("--min-lang-chars", type=int, default=5,
                    help="Min char length of a sentence before running langid (shorter -> 'unk').")
    args = ap.parse_args()

    cfg = BuildConfig(
        jsonl=args.jsonl,
        out_dir=args.out_dir,
        procs=int(args.procs),
        chunk_lines=int(args.chunk_lines),
        correct_key=args.correct_key,
        snapshot_keys=[s.strip() for s in args.snapshot_keys.split(",") if s.strip()],
        compute_sha256=bool(args.compute_sha256),
        min_lang_chars=int(args.min_lang_chars),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = cfg.out_dir / "features.parquet"
    segments_path = cfg.out_dir / "segments.parquet"
    meta_path = cfg.out_dir / "meta_preprocess.json"

    print(f"[Stage1] Scanning offsets in JSONL: {cfg.jsonl}")
    t0 = time.time()
    chunks, sha256, fsize, mtime = _scan_offsets(cfg.jsonl, cfg.chunk_lines, cfg.compute_sha256)
    n_lines = sum(len(c) for c in chunks)
    print(f"[Stage1] Found {n_lines:,} lines, {len(chunks)} chunks; scan took {time.time()-t0:.1f}s")

    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"[Stage1] Processing chunks with {cfg.procs} processes ...")
    t1 = time.time()
    writer_samples = None
    writer_segments = None
    n_rows_samples = 0
    n_rows_segments = 0

    start_method = "spawn" if sys.platform == "win32" else "fork"
    ctx = mp.get_context(start_method)
    tasks = [(str(cfg.jsonl), offs, cfg.correct_key, cfg.snapshot_keys, cfg.min_lang_chars) for offs in chunks]

    # with ctx.Pool(processes=cfg.procs) as pool:
    #     for i, ret in enumerate(pool.imap_unordered(_worker_entry, tasks, chunksize=1)):
    #         df_s = ret["samples"]
    #         df_seg = ret["segments"]

    #         if not df_s.empty:
    #             table_s = pa.Table.from_pandas(df_s, preserve_index=False)
    #             if writer_samples is None:
    #                 writer_samples = pq.ParquetWriter(samples_path, table_s.schema, compression="zstd", version="2.6")
    #             writer_samples.write_table(table_s)
    #             n_rows_samples += len(df_s)

    #         if not df_seg.empty:
    #             table_seg = pa.Table.from_pandas(df_seg, preserve_index=False)
    #             if writer_segments is None:
    #                 writer_segments = pq.ParquetWriter(segments_path, table_seg.schema, compression="zstd", version="2.6")
    #             writer_segments.write_table(table_seg)
    #             n_rows_segments += len(df_seg)

    #         if (i+1) % 5 == 0:
    #             print(f"  [Stage1] processed {i+1}/{len(chunks)} chunks; samples={n_rows_samples:,}, segments={n_rows_segments:,}")
    with ctx.Pool(processes=cfg.procs) as pool:
        # 用 tqdm 包裹 imap_unordered，按 chunk 显示进度和 ETA
        for i, ret in enumerate(
            tqdm(
                pool.imap_unordered(_worker_entry, tasks, chunksize=1),
                total=len(tasks),
                desc="[Stage1] Processing chunks",
            )
        ):
            df_s = ret["samples"]
            df_seg = ret["segments"]

            if not df_s.empty:
                table_s = pa.Table.from_pandas(df_s, preserve_index=False)
                if writer_samples is None:
                    writer_samples = pq.ParquetWriter(samples_path, table_s.schema, compression="zstd", version="2.6")
                writer_samples.write_table(table_s)
                n_rows_samples += len(df_s)

            if not df_seg.empty:
                table_seg = pa.Table.from_pandas(df_seg, preserve_index=False)
                if writer_segments is None:
                    writer_segments = pq.ParquetWriter(segments_path, table_seg.schema, compression="zstd", version="2.6")
                writer_segments.write_table(table_seg)
                n_rows_segments += len(df_seg)

            # 可选：保留原来每 5 个 chunk 打一行 summary 的行为
            if (i + 1) % 5 == 0:
                print(
                    f"  [Stage1] processed {i+1}/{len(chunks)} chunks; "
                    f"samples={n_rows_samples:,}, segments={n_rows_segments:,}"
                )

    if writer_samples is not None:
        writer_samples.close()
    if writer_segments is not None:
        writer_segments.close()

    print(f"[Stage1] Wrote samples features: {samples_path} (rows={n_rows_samples:,})")
    print(f"[Stage1] Wrote segments (sentence-level): {segments_path} (rows={n_rows_segments:,})")
    print(f"[Stage1] Total time: {time.time()-t1:.1f}s")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "source_jsonl": str(cfg.jsonl),
        "source_size": fsize,
        "source_mtime": mtime,
        "source_sha256": sha256,
        "samples_path": str(samples_path),
        "segments_path": str(segments_path),
        "rows_samples": n_rows_samples,
        "rows_segments": n_rows_segments,
        "snapshot_keys": cfg.snapshot_keys,
        "snapshot_columns": {k: sanitize_snap_col(k) for k in cfg.snapshot_keys},
        "build_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {
            "procs": cfg.procs,
            "chunk_lines": cfg.chunk_lines,
            "correct_key": cfg.correct_key,
            "compute_sha256": cfg.compute_sha256,
            "min_lang_chars": cfg.min_lang_chars,
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Stage1] Wrote meta: {meta_path}")

    print("[Stage1 Summary]")
    print(f"  samples rows = {n_rows_samples:,}")
    print(f"  segments rows = {n_rows_segments:,}")

if __name__ == "__main__":
    main()
