#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a semantic feature store + random-access index for huge rollout JSONL.

在你原先 build_features_from_jsonl.py 的基础上增加：
  - response_text 的 section/sentence 两级切分（按 '\n' + pysbd）
  - SELF_REFLECT / CONTRAST / CONCLUDE 多语言 discourse marker 标记
  - 句子级语言识别（langid），response 级语言分布统计
  - 基于 embedding 的语义指标（重复度/cluster/loopiness/plateau/gibberish_semantic_ratio）

输出目录 (--out-dir) 下：
  - features.parquet : 每条 sample 一行（原 schema + 新增语义特征）
  - meta.json        : 元信息 + 语义相关配置

注意：
  - embedding 计算很重，可以通过 --no-embedding 关掉，或者用 --max-sentences-for-embedding 控制每个样本最多 embedding 的句子数。
  - 如果 embedding 用 GPU，建议把 --procs 设得比较小（甚至 1），避免多进程重复载入模型占爆显存。
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import argparse, json, os, sys, time, math, hashlib
from typing import Any, Dict, List, Tuple, Optional
import multiprocessing as mp

import numpy as np
import pandas as pd
import unicodedata, zlib, re

# ---------- fast json ----------
try:
    import orjson
    def _json_loads(b: bytes):
        return orjson.loads(b)
except Exception:
    import json as _json
    def _json_loads(b: bytes):
        return _json.loads(b.decode("utf-8", errors="replace"))

# ---------- Chinese detector ----------
REPL_CHAR = "\uFFFD"

_CHN_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3000-\u303F"
    r"\U00020000-\U0002A6DF\U0002A700-\U0002B73F\U0002B740-\U0002B81F"
    r"\U0002B820-\U0002CEAF\U0002CEB0-\U0002EBEF]"
)


def contains_chinese(s: str) -> bool:
    if not s:
        return False
    return bool(_CHN_RE.search(s))


# ---------- text metrics（与你原先的一致） ----------
def text_metrics(s: str) -> Dict[str, float]:
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
        # do not crash the builder on weird strings
        return {
            "resp_char_len": np.nan, "resp_byte_len": np.nan, "non_ascii_ratio": np.nan,
            "replacement_char_ratio": np.nan, "control_private_unassigned_ratio": np.nan,
            "repeat_char_run_max": np.nan, "repeat_word_run_max": np.nan, "unique_word_ratio": np.nan,
            "top1_word_prop": np.nan, "trigram_repeat_ratio": np.nan, "compress_ratio": np.nan,
            "char_entropy_bits_per_char": np.nan,
        }


def get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def classify_record(obj: Dict[str, Any], correct_key: Optional[str], use_reward_sum_fallback=True) -> int:
    # 1=correct, 0=wrong, 2=no_judge
    v = None
    if correct_key:
        v = get_nested(obj, correct_key, None)
    if v is None:
        rei = obj.get("reward_extra_infos") or obj.get("reward_extra_infos_dict")
        if isinstance(rei, dict):
            for k in ["is_correct", "correct", "equal", "any_of_three", "pass", "ok", "acc", "accuracy", "label"]:
                if k in rei:
                    v = rei[k]
                    break
    if v is None and use_reward_sum_fallback:
        rs = obj.get("reward_seq_sum", None)
        if isinstance(rs, (int, float)):
            v = (rs > 0)

    def _to_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return x > 0
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"true", "t", "yes", "y", "correct", "right", "pass", "ok", "passed", "success"}:
                return True
            if s in {"false", "f", "no", "n", "wrong", "incorrect", "fail", "failed"}:
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
    # dotpath -> safe column: "snap__a.b.c" -> "snap__a__b__c"
    return "snap__" + path.replace(".", "__")


# ---------- Semantic / embedding config ----------

@dataclass
class SemanticConfig:
    enable_semantic: bool = True
    enable_embedding: bool = False
    embed_model: Optional[str] = None
    # 新增：embedding backend，"st"=sentence-transformers，"vllm"=vLLM
    embed_backend: str = "st"
    embed_device: str = "cuda"
    embed_batch_size: int = 32
    max_sentences_for_embedding: int = 128
    sim_thresh_dup: float = 0.9
    sim_thresh_cluster: float = 0.85
    nn_low_sim_thresh: float = 0.4
    gibberish_non_ascii_thresh: float = 0.5
    gibberish_repeat_char_run_thresh: int = 8
    gibberish_repl_char_ratio_thresh: float = 0.1


@dataclass
class BuildConfig:
    jsonl: Path
    out_dir: Path
    procs: int
    chunk_lines: int
    correct_key: Optional[str]
    snapshot_keys: List[str]
    compute_sha256: bool
    semantic: SemanticConfig


SCHEMA_VERSION = 2  # 扩展了 schema


# ---------- sentence segmentation (pysbd) ----------

SEGMENTER = None


def get_segmenter():
    global SEGMENTER
    if SEGMENTER is None:
        import pysbd
        # clean=False 尽量避免改写原文
        SEGMENTER = pysbd.Segmenter(language="en", clean=False)
    return SEGMENTER


def split_sections(resp: str) -> List[str]:
    """
    按 '\n' 的连续 run 切成 section：
    - 每一串连续的 '\n\n\n' 归入前一个 section 的末尾。
    - 不做 strip/清洗。
    """
    if not isinstance(resp, str) or not resp:
        return []

    n = len(resp)
    sections: List[str] = []
    start = 0
    i = 0
    while i < n:
        if resp[i] == "\n":
            j = i
            while j < n and resp[j] == "\n":
                j += 1
            # [start, j) 作为一个 section（包括前面内容 + 这串 \n）
            sections.append(resp[start:j])
            start = j
            i = j
        else:
            i += 1
    if start < n:
        sections.append(resp[start:])

    # 如果开头就是一串纯 \n，可以和后一个 section 合并，避免出现“空 section”
    if len(sections) >= 2 and sections[0].strip("\n") == "":
        sections[1] = sections[0] + sections[1]
        sections = sections[1:]
    return sections or [resp]


# ---------- discourse markers (多语言) ----------

SELF_REFLECT_PATTERNS = [
    r"\bwait\b",
    r"\bhold on\b",
    r"\bhang on\b",
    r"\bgive me a moment\b",
    r"\bjust a second\b",
    r"\blet me think\b",
    r"等一下", r"先等等", r"稍等", r"等等看",
    r"moment mal",
    r"\bwarte mal\b", r"\bwarte\b",
    r"подожди", r"подождите",
]

CONTRAST_PATTERNS = [
    r"\bhowever\b",
    r"\bbut\b",
    r"\bon the other hand\b",
    r"\bnevertheless\b",
    r"\bnonetheless\b",
    r"\bby contrast\b",
    r"然而", r"但是", r"不过", r"相反地",
    r"\bdoch\b", r"\baber\b", r"\bjedoch\b",
    r"однако", r"с другой стороны",
]

CONCLUDE_PATTERNS = [
    r"\bso\b",
    r"\btherefore\b",
    r"\bthus\b",
    r"\bhence\b",
    r"\bin conclusion\b",
    r"\bas a result\b",
    r"所以", r"因此", r"综上", r"综上所述", r"总之", r"由此可见",
    r"значит", r"в итоге", r"итог", r"следовательно",
]

SELF_REFLECT_RE = re.compile("(" + "|".join(SELF_REFLECT_PATTERNS) + ")", re.IGNORECASE)
CONTRAST_RE = re.compile("(" + "|".join(CONTRAST_PATTERNS) + ")", re.IGNORECASE)
CONCLUDE_RE = re.compile("(" + "|".join(CONCLUDE_PATTERNS) + ")", re.IGNORECASE)


def detect_markers(text: str) -> Tuple[bool, bool, bool]:
    if not text:
        return False, False, False
    s = text
    self_flag = bool(SELF_REFLECT_RE.search(s))
    contrast_flag = bool(CONTRAST_RE.search(s))
    conclude_flag = bool(CONCLUDE_RE.search(s))
    return self_flag, contrast_flag, conclude_flag


# ---------- language detection (langid) ----------

_LANGID_INITIALIZED = False

try:
    import langid as _langid_mod
    _LANGID_AVAILABLE = True
except Exception:
    _LANGID_AVAILABLE = False
    _langid_mod = None


def init_langid():
    global _LANGID_INITIALIZED
    if not _LANGID_AVAILABLE or _LANGID_INITIALIZED:
        return
    # 可选：限制语言集合；这里先用默认
    _LANGID_INITIALIZED = True


def detect_lang(text: str) -> Tuple[str, float]:
    if not _LANGID_AVAILABLE or not isinstance(text, str) or not text.strip():
        return "unk", 0.0
    try:
        init_langid()
        lang, conf = _langid_mod.classify(text)
        return str(lang), float(conf)
    except Exception:
        return "unk", 0.0


# ---------- embedding (SentenceTransformer) ----------

EMBEDDER = None
EMBED_CFG: Optional[SemanticConfig] = None


class SentenceEmbedder:
    """
    统一的 embedding 封装：
      - backend="st": 使用 sentence-transformers
      - backend="vllm": 使用 vLLM 的 LLM(task="embed").embed(texts)
    """
    def __init__(self, cfg: SemanticConfig):
        self.cfg = cfg
        self.backend = getattr(cfg, "embed_backend", "st")
        self.batch_size = cfg.embed_batch_size

        if self.backend == "vllm":
            # vLLM backend
            try:
                from vllm import LLM as VLLModel
            except ImportError:
                raise ImportError(
                    "embed_backend='vllm' but vllm is not installed. "
                    "请先 `pip install vllm`，或者改回 `--embed-backend st`。"
                )
            print(
                f"[Worker {os.getpid()}] Loading vLLM embedding model '{cfg.embed_model}' ...",
                file=sys.stderr,
            )
            # 这里使用 vLLM 官方 embedding 接口：task="embed" + embed() :contentReference[oaicite:3]{index=3}
            self.client = VLLModel(
                model=cfg.embed_model,
                task="embed",
                max_num_seqs=cfg.embed_batch_size,
                tensor_parallel_size=1,
                trust_remote_code=True,
                dtype="auto",
            )
        else:
            # sentence-transformers backend（保持原有行为）
            from sentence_transformers import SentenceTransformer
            print(
                f"[Worker {os.getpid()}] Loading embedding model '{cfg.embed_model}' on {cfg.embed_device} ...",
                file=sys.stderr,
            )
            self.model = SentenceTransformer(cfg.embed_model, device=cfg.embed_device)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        if self.backend == "vllm":
            # 使用 vLLM 的 embed API
            outputs = self.client.embed(texts)
            # 每个 output.outputs.embedding 是一个向量
            emb_list = [o.outputs.embedding for o in outputs]
            E = np.asarray(emb_list, dtype=np.float32)
            # 下游假设是归一化的向量，这里自行做 L2 normalize
            norms = np.linalg.norm(E, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            E = E / norms
            return E

        # 默认：sentence-transformers
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)


def get_embedder(cfg: SemanticConfig) -> Optional[SentenceEmbedder]:
    global EMBEDDER, EMBED_CFG
    if not cfg.enable_embedding or not cfg.embed_model:
        return None
    if EMBEDDER is None:
        try:
            EMBED_CFG = cfg
            EMBEDDER = SentenceEmbedder(cfg)
        except Exception as e:
            print(f"[Worker {os.getpid()}] ERROR: failed to load embed model: {e}", file=sys.stderr)
            EMBEDDER = None
    return EMBEDDER


# ---------- semantic feature helpers ----------

def compute_lang_stats(sent_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sent_infos:
        return {
            "lang_main": "",
            "lang_entropy": np.nan,
            "num_lang_major": 0,
            "code_switch_count": 0,
        }
    lang_char: Dict[str, int] = {}
    total_chars = 0
    prev_lang = None
    code_switch_count = 0
    for info in sent_infos:
        txt = info["text"] or ""
        lang = info.get("lang", "unk") or "unk"
        c = len(txt)
        total_chars += c
        lang_char[lang] = lang_char.get(lang, 0) + c
        if prev_lang is not None and lang != prev_lang:
            code_switch_count += 1
        prev_lang = lang

    if total_chars <= 0:
        return {
            "lang_main": "",
            "lang_entropy": np.nan,
            "num_lang_major": 0,
            "code_switch_count": code_switch_count,
        }

    # 分布
    items = list(lang_char.items())
    items.sort(key=lambda x: x[1], reverse=True)
    main_lang = items[0][0]
    probs = [cnt / total_chars for _, cnt in items]
    H = -sum(p * math.log(p + 1e-12) for p in probs)
    num_lang_major = sum(1 for p in probs if p > 0.1)

    return {
        "lang_main": main_lang,
        "lang_entropy": float(H),
        "num_lang_major": int(num_lang_major),
        "code_switch_count": int(code_switch_count),
    }


def compute_marker_stats(sent_infos: List[Dict[str, Any]], num_sections: int) -> Dict[str, Any]:
    N_sent = len(sent_infos)
    if N_sent == 0:
        return {
            "num_sections": int(num_sections),
            "num_sentences": 0,
            "num_self_sent": 0,
            "num_contrast_sent": 0,
            "num_conclude_sent": 0,
            "ratio_self_sent": np.nan,
            "ratio_contrast_sent": np.nan,
            "ratio_conclude_sent": np.nan,
            "num_self_sec": 0,
            "num_contrast_sec": 0,
            "num_conclude_sec": 0,
            "ratio_self_sec": np.nan,
            "ratio_contrast_sec": np.nan,
            "ratio_conclude_sec": np.nan,
        }

    num_self_sent = sum(1 for s in sent_infos if s.get("marker_self"))
    num_contrast_sent = sum(1 for s in sent_infos if s.get("marker_contrast"))
    num_conclude_sent = sum(1 for s in sent_infos if s.get("marker_conclude"))

    sec_self = [False] * max(1, num_sections)
    sec_contrast = [False] * max(1, num_sections)
    sec_conclude = [False] * max(1, num_sections)
    for s in sent_infos:
        sec_idx = int(s.get("section_idx", 0))
        if 0 <= sec_idx < len(sec_self):
            if s.get("marker_self"):
                sec_self[sec_idx] = True
            if s.get("marker_contrast"):
                sec_contrast[sec_idx] = True
            if s.get("marker_conclude"):
                sec_conclude[sec_idx] = True

    num_self_sec = sum(1 for x in sec_self if x)
    num_contrast_sec = sum(1 for x in sec_contrast if x)
    num_conclude_sec = sum(1 for x in sec_conclude if x)

    return {
        "num_sections": int(num_sections),
        "num_sentences": int(N_sent),
        "num_self_sent": int(num_self_sent),
        "num_contrast_sent": int(num_contrast_sent),
        "num_conclude_sent": int(num_conclude_sent),
        "ratio_self_sent": float(num_self_sent / N_sent),
        "ratio_contrast_sent": float(num_contrast_sent / N_sent),
        "ratio_conclude_sent": float(num_conclude_sent / N_sent),
        "num_self_sec": int(num_self_sec),
        "num_contrast_sec": int(num_contrast_sec),
        "num_conclude_sec": int(num_conclude_sec),
        "ratio_self_sec": float(num_self_sec / max(1, num_sections)),
        "ratio_contrast_sec": float(num_contrast_sec / max(1, num_sections)),
        "ratio_conclude_sec": float(num_conclude_sec / max(1, num_sections)),
    }


def _compute_pairwise_sim(E: np.ndarray) -> np.ndarray:
    if E.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    # E assumed normalized
    sim = E @ E.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim


def _dup_ratio_from_sim(sim: np.ndarray, thresh: float) -> Tuple[float, np.ndarray]:
    m = sim.shape[0]
    if m <= 1:
        return np.nan, np.zeros((m,), dtype=np.float32)
    sim_no_diag = sim.copy()
    np.fill_diagonal(sim_no_diag, -np.inf)
    nn_sim = sim_no_diag.max(axis=1)
    dup_ratio = float(np.mean(nn_sim >= thresh))
    return dup_ratio, nn_sim


def _cluster_stats_from_sim(sim: np.ndarray, thresh: float) -> Tuple[int, int, float, np.ndarray]:
    m = sim.shape[0]
    if m == 0:
        return 0, 0, np.nan, np.zeros((0,), dtype=np.int32)
    adj = (sim >= thresh)
    np.fill_diagonal(adj, False)
    visited = np.zeros(m, dtype=bool)
    cluster_ids = np.full(m, -1, dtype=np.int32)
    cluster_sizes: List[int] = []
    cid = 0
    for i in range(m):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            cluster_ids[i] = cid
            size = 0
            while stack:
                u = stack.pop()
                size += 1
                # neighbors
                neigh = np.nonzero(adj[u])[0]
                for v in neigh:
                    if not visited[v]:
                        visited[v] = True
                        cluster_ids[v] = cid
                        stack.append(v)
            cluster_sizes.append(size)
            cid += 1
    if not cluster_sizes:
        return 0, 0, np.nan, cluster_ids
    max_size = max(cluster_sizes)
    rho_max = float(max_size / m)
    return cid, max_size, rho_max, cluster_ids


def _loopiness_from_embeddings(E: np.ndarray) -> float:
    n = E.shape[0]
    if n <= 1:
        return np.nan
    dots = np.sum(E[:-1] * E[1:], axis=1)
    np.clip(dots, -1.0, 1.0, out=dots)
    dists = np.arccos(dots)
    L = float(np.sum(dists))
    d0 = float(np.arccos(np.clip(float(np.sum(E[0] * E[-1])), -1.0, 1.0)))
    eps = 1e-4
    return float(L / (d0 + eps))


def _plateau_len_max(cluster_ids: np.ndarray) -> int:
    n = len(cluster_ids)
    if n == 0:
        return 0
    max_run = 1
    run = 1
    for i in range(1, n):
        if cluster_ids[i] == cluster_ids[i - 1]:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1
    return int(max_run)


def compute_embedding_metrics(
    sent_infos: List[Dict[str, Any]],
    cfg: SemanticConfig,
    embedder: Optional[SentenceEmbedder],
) -> Dict[str, Any]:
    """
    对一个 sample 的句子级 embedding 做语义指标：
      - dup_ratio_all / dup_ratio_marker
      - cluster_count / max_size / rho_max（all / marker）
      - loopiness_all / loopiness_marker
      - plateau_len_max_marker
      - gibberish_semantic_ratio
    """
    N = len(sent_infos)
    if not cfg.enable_embedding or embedder is None or N == 0:
        return {
            "dup_ratio_all": np.nan,
            "dup_ratio_marker": np.nan,
            "cluster_count_all": 0,
            "cluster_max_size_all": 0,
            "rho_max_all": np.nan,
            "cluster_count_marker": 0,
            "cluster_max_size_marker": 0,
            "rho_max_marker": np.nan,
            "loopiness_all": np.nan,
            "loopiness_marker": np.nan,
            "plateau_len_max_marker": 0,
            "gibberish_semantic_ratio": np.nan,
        }

    # 选择要做 embedding 的 sentence index（尽量保留顺序）
    all_idx = list(range(N))
    if cfg.max_sentences_for_embedding > 0 and N > cfg.max_sentences_for_embedding:
        step = N / cfg.max_sentences_for_embedding
        chosen = sorted({int(i * step) for i in range(cfg.max_sentences_for_embedding)})
        selected_idx = [all_idx[i] for i in chosen if 0 <= i < N]
    else:
        selected_idx = all_idx

    if not selected_idx:
        return {
            "dup_ratio_all": np.nan,
            "dup_ratio_marker": np.nan,
            "cluster_count_all": 0,
            "cluster_max_size_all": 0,
            "rho_max_all": np.nan,
            "cluster_count_marker": 0,
            "cluster_max_size_marker": 0,
            "rho_max_marker": np.nan,
            "loopiness_all": np.nan,
            "loopiness_marker": np.nan,
            "plateau_len_max_marker": 0,
            "gibberish_semantic_ratio": np.nan,
        }

    texts = [sent_infos[i]["text"] or "" for i in selected_idx]
    E = embedder.encode(texts)
    if E.ndim != 2:
        return {
            "dup_ratio_all": np.nan,
            "dup_ratio_marker": np.nan,
            "cluster_count_all": 0,
            "cluster_max_size_all": 0,
            "rho_max_all": np.nan,
            "cluster_count_marker": 0,
            "cluster_max_size_marker": 0,
            "rho_max_marker": np.nan,
            "loopiness_all": np.nan,
            "loopiness_marker": np.nan,
            "plateau_len_max_marker": 0,
            "gibberish_semantic_ratio": np.nan,
        }

    # map: global_idx -> local_idx in E
    idx_map: Dict[int, int] = {g: i for i, g in enumerate(selected_idx)}
    m = E.shape[0]
    sim_all = _compute_pairwise_sim(E)
    dup_ratio_all, nn_sim_all = _dup_ratio_from_sim(sim_all, cfg.sim_thresh_dup)
    cluster_count_all, cluster_max_size_all, rho_max_all, cluster_ids_all = _cluster_stats_from_sim(
        sim_all, cfg.sim_thresh_cluster
    )
    loopiness_all = _loopiness_from_embeddings(E)

    # marker subset
    marker_global_idx = [i for i, s in enumerate(sent_infos) if s.get("marker_self") or s.get("marker_contrast") or s.get("marker_conclude")]
    marker_local_idx = [idx_map[g] for g in marker_global_idx if g in idx_map]
    if len(marker_local_idx) >= 2:
        sim_marker = sim_all[np.ix_(marker_local_idx, marker_local_idx)]
        dup_ratio_marker, _ = _dup_ratio_from_sim(sim_marker, cfg.sim_thresh_dup)
        cc_m, maxsize_m, rho_m, cluster_ids_m = _cluster_stats_from_sim(sim_marker, cfg.sim_thresh_cluster)
        # plateau：marker 句子按原来顺序已经是有序的（marker_global_idx）
        plateau_len_max_marker = _plateau_len_max(cluster_ids_m)
        # loopiness：按 marker 句子顺序取 embedding
        E_marker = E[marker_local_idx]
        loopiness_marker = _loopiness_from_embeddings(E_marker)
    else:
        dup_ratio_marker = np.nan
        cc_m, maxsize_m, rho_m = 0, 0, np.nan
        plateau_len_max_marker = 0
        loopiness_marker = np.nan

    # gibberish_semantic_ratio：embedding + 字面指标
    # 只在 selected_idx 范围内做
    if m <= 1:
        gibberish_ratio = np.nan
    else:
        # sentence-level text metrics（只用几个粗指标）
        non_ascii = np.zeros((m,), dtype=np.float32)
        repl_ratio = np.zeros((m,), dtype=np.float32)
        repeat_char_run = np.zeros((m,), dtype=np.float32)
        for li, g in enumerate(selected_idx):
            tm = text_metrics(sent_infos[g]["text"] or "")
            non_ascii[li] = tm["non_ascii_ratio"]
            repl_ratio[li] = tm["replacement_char_ratio"]
            repeat_char_run[li] = tm["repeat_char_run_max"]

        # 用 nn_sim_all（对应 selected_idx）
        gibberish_flags = (
            (nn_sim_all < cfg.nn_low_sim_thresh)
            & (
                (non_ascii > cfg.gibberish_non_ascii_thresh)
                | (repl_ratio > cfg.gibberish_repl_char_ratio_thresh)
                | (repeat_char_run >= cfg.gibberish_repeat_char_run_thresh)
            )
        )
        gibberish_ratio = float(np.mean(gibberish_flags)) if m > 0 else np.nan

    return {
        "dup_ratio_all": float(dup_ratio_all),
        "dup_ratio_marker": float(dup_ratio_marker),
        "cluster_count_all": int(cluster_count_all),
        "cluster_max_size_all": int(cluster_max_size_all),
        "rho_max_all": float(rho_max_all) if not math.isnan(rho_max_all) else np.nan,
        "cluster_count_marker": int(cc_m),
        "cluster_max_size_marker": int(maxsize_m),
        "rho_max_marker": float(rho_m) if not math.isnan(rho_m) else np.nan,
        "loopiness_all": float(loopiness_all) if not math.isnan(loopiness_all) else np.nan,
        "loopiness_marker": float(loopiness_marker) if not math.isnan(loopiness_marker) else np.nan,
        "plateau_len_max_marker": int(plateau_len_max_marker),
        "gibberish_semantic_ratio": float(gibberish_ratio) if not math.isnan(gibberish_ratio) else np.nan,
    }


# ---------- worker ----------

def _worker_process(
    jsonl_path: str,
    offsets: List[Tuple[int, int, int]],
    correct_key: Optional[str],
    snapshot_keys: List[str],
    semantic_cfg: SemanticConfig,
) -> Dict[str, Any]:
    """
    offsets: list of (line_no, offset, length)
    """
    fd = os.open(jsonl_path, os.O_RDONLY)
    rows = []

    # prepare snap col names
    snap_cols = [(k, sanitize_snap_col(k)) for k in snapshot_keys]

    # lazy init embedder per process
    embedder = get_embedder(semantic_cfg) if semantic_cfg.enable_embedding else None

    segmenter = get_segmenter() if semantic_cfg.enable_semantic else None

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

        rec: Dict[str, Any] = {
            "line_no": line_no,
            "byte_offset": off,
            "byte_len": ln,
            "valid": 1 if isinstance(obj, dict) else 0,
            "global_step": np.int32(get_nested(obj, "global_step", -1)) if obj else np.int32(-1),
            "uid": str(obj.get("uid")) if (obj and "uid" in obj) else "",
            "kept_flag": np.int8(
                1
                if (obj and obj.get("filter_state") == "kept")
                else (0 if (obj and obj.get("filter_state") == "dropped") else -1)
            ),
            # CE/KL/Entropy/Reward/length
            "ce_self_seq_mean": np.float32(neg(obj.get("old_logprob_mean") if obj else None)),
            "ce_self_seq_sum": np.float32(neg(obj.get("old_logprob_sum") if obj else None)),
            "ce_ref_seq_mean": np.float32(neg(obj.get("ref_logprob_mean") if obj else None)),
            "ce_ref_seq_sum": np.float32(neg(obj.get("ref_logprob_sum") if obj else None)),
            "reward_seq_sum": np.float32(obj.get("reward_seq_sum") if obj else np.nan),
            "kl_seq_mean_log_ratio": np.float32(obj.get("kl_seq_mean_log_ratio") if obj else np.nan),
            "entropy_seq_mean": np.float32(obj.get("entropy_seq_mean") if obj else np.nan),
            "response_len_tokens": np.float32(obj.get("response_len_tokens") if obj else np.nan),
            "cls_id": np.int8(classify_record(obj, correct_key=correct_key) if obj else 2),
        }

        # contain_chinese + response-level text metrics（原逻辑）
        rt = obj.get("response_text") if isinstance(obj, dict) else None
        pt = obj.get("prompt_text") if isinstance(obj, dict) else None

        if isinstance(rt, str) or isinstance(pt, str):
            has_cn = (contains_chinese(rt) if isinstance(rt, str) else False) or (
                contains_chinese(pt) if isinstance(pt, str) else False
            )
            rec["contain_chinese"] = np.int8(1 if has_cn else 0)
        else:
            rec["contain_chinese"] = np.int8(-1)

        if isinstance(rt, str):
            tm = text_metrics(rt)
        else:
            tm = {
                k: np.nan
                for k in [
                    "resp_char_len",
                    "resp_byte_len",
                    "non_ascii_ratio",
                    "replacement_char_ratio",
                    "control_private_unassigned_ratio",
                    "repeat_char_run_max",
                    "repeat_word_run_max",
                    "unique_word_ratio",
                    "top1_word_prop",
                    "trigram_repeat_ratio",
                    "compress_ratio",
                    "char_entropy_bits_per_char",
                ]
            }
        for k, v in tm.items():
            rec[k] = np.float32(v)

        # snapshots (any dotpath)
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

        # ---------- semantic features ----------
        if semantic_cfg.enable_semantic and isinstance(rt, str) and rt:
            # 1) section & sentence 切分 + marker + language
            sections = split_sections(rt)
            num_sections = len(sections)
            sent_infos: List[Dict[str, Any]] = []
            global_sent_idx = 0
            for sec_idx, sec in enumerate(sections):
                if not sec:
                    continue
                try:
                    sents = segmenter.segment(sec)
                except Exception:
                    sents = [sec]
                if not isinstance(sents, list):
                    sents = [sec]
                for local_idx, sent in enumerate(sents):
                    if not isinstance(sent, str):
                        continue
                    lang, lang_conf = detect_lang(sent)
                    m_self, m_contrast, m_conclude = detect_markers(sent)
                    sent_infos.append(
                        {
                            "text": sent,
                            "section_idx": sec_idx,
                            "sent_idx": local_idx,
                            "global_sent_idx": global_sent_idx,
                            "lang": lang,
                            "lang_conf": lang_conf,
                            "marker_self": bool(m_self),
                            "marker_contrast": bool(m_contrast),
                            "marker_conclude": bool(m_conclude),
                        }
                    )
                    global_sent_idx += 1

            marker_stats = compute_marker_stats(sent_infos, num_sections)
            lang_stats = compute_lang_stats(sent_infos)
            for k, v in marker_stats.items():
                if isinstance(v, int):
                    rec[k] = np.int32(v)
                else:
                    rec[k] = np.float32(v)
            rec["lang_main"] = lang_stats["lang_main"]
            rec["lang_entropy"] = np.float32(lang_stats["lang_entropy"])
            rec["num_lang_major"] = np.int32(lang_stats["num_lang_major"])
            rec["code_switch_count"] = np.int32(lang_stats["code_switch_count"])

            # 2) embedding-based metrics
            emb_metrics = compute_embedding_metrics(sent_infos, semantic_cfg, embedder)
            for k, v in emb_metrics.items():
                if isinstance(v, int):
                    rec[k] = np.int32(v)
                else:
                    rec[k] = np.float32(v)
        else:
            # 填补 semantic 特征为 NaN/0
            for k, v in {
                "num_sections": np.int32(0),
                "num_sentences": np.int32(0),
                "num_self_sent": np.int32(0),
                "num_contrast_sent": np.int32(0),
                "num_conclude_sent": np.int32(0),
                "ratio_self_sent": np.nan,
                "ratio_contrast_sent": np.nan,
                "ratio_conclude_sent": np.nan,
                "num_self_sec": np.int32(0),
                "num_contrast_sec": np.int32(0),
                "num_conclude_sec": np.int32(0),
                "ratio_self_sec": np.nan,
                "ratio_contrast_sec": np.nan,
                "ratio_conclude_sec": np.nan,
                "lang_main": "",
                "lang_entropy": np.nan,
                "num_lang_major": np.int32(0),
                "code_switch_count": np.int32(0),
                "dup_ratio_all": np.nan,
                "dup_ratio_marker": np.nan,
                "cluster_count_all": np.int32(0),
                "cluster_max_size_all": np.int32(0),
                "rho_max_all": np.nan,
                "cluster_count_marker": np.int32(0),
                "cluster_max_size_marker": np.int32(0),
                "rho_max_marker": np.nan,
                "loopiness_all": np.nan,
                "loopiness_marker": np.nan,
                "plateau_len_max_marker": np.int32(0),
                "gibberish_semantic_ratio": np.nan,
            }.items():
                rec[k] = v

        rows.append(rec)

    os.close(fd)
    df = pd.DataFrame(rows)
    # compact dtypes
    df["line_no"] = df["line_no"].astype(np.int64)
    df["byte_offset"] = df["byte_offset"].astype(np.int64)
    df["byte_len"] = df["byte_len"].astype(np.int32)
    df["global_step"] = df["global_step"].astype(np.int32)
    df["kept_flag"] = df["kept_flag"].astype(np.int8)
    df["cls_id"] = df["cls_id"].astype(np.int8)
    if "contain_chinese" in df.columns:
        df["contain_chinese"] = df["contain_chinese"].astype(np.int8)
    # 一些计数字段也强制为 int32
    for col in [
        "num_sections",
        "num_sentences",
        "num_self_sent",
        "num_contrast_sent",
        "num_conclude_sent",
        "num_self_sec",
        "num_contrast_sec",
        "num_conclude_sec",
        "num_lang_major",
        "code_switch_count",
        "cluster_count_all",
        "cluster_max_size_all",
        "cluster_count_marker",
        "cluster_max_size_marker",
        "plateau_len_max_marker",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(np.int32)

    return {"df": df}


def _worker_entry(args):
    jsonl_path, offsets, correct_key, snapshot_keys, semantic_cfg = args
    return _worker_process(jsonl_path, offsets, correct_key, snapshot_keys, semantic_cfg)


# ---------- main ----------

def _scan_offsets(jsonl: Path, chunk_lines: int, compute_sha256: bool):
    """
    Sequentially iterate binary lines to get (line_no, offset, length).
    Also compute sha256 if requested (in-stream).
    """
    offsets: List[Tuple[int, int, int]] = []
    chunks: List[List[Tuple[int, int, int]]] = []
    hasher = hashlib.sha256() if compute_sha256 else None

    with jsonl.open("rb", buffering=1024 * 1024) as f:
        line_no = 0
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
            offsets.append((line_no, off, ln))
            if len(offsets) >= chunk_lines:
                chunks.append(offsets)
                offsets = []
    sha256 = hasher.hexdigest() if hasher else ""
    stat = jsonl.stat()
    return chunks, sha256, stat.st_size, int(stat.st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--chunk-lines", type=int, default=20000)
    ap.add_argument(
        "--correct-key",
        type=str,
        default=None,
        help="Dot path for correctness label, e.g., reward_extra_infos.acc",
    )
    ap.add_argument(
        "--snapshot-keys",
        type=str,
        default="response_text,prompt_text",
        help="Comma separated dot-keys to snapshot for fast export.",
    )
    ap.add_argument(
        "--compute-sha256",
        action="store_true",
        help="Compute sha256 of source file for strong validation (costly on very large files).",
    )
    # semantic / embedding 相关参数
    ap.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable all semantic features (section/sentence/marker/lang/embedding).",
    )
    ap.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable embedding-based semantic metrics (keep only structural/lang features).",
    )
    ap.add_argument(
        "--embed-model",
        type=str,
        default=None,
        help="SentenceTransformer model name/path for embeddings, e.g. 'BAAI/bge-m3'.",
    )
    # 新增：选择 embedding backend
    ap.add_argument(
        "--embed-backend",
        type=str,
        default="st",
        choices=["st", "vllm"],
        help="Embedding backend: 'st' (sentence-transformers) or 'vllm' (vLLM pooling model).",
    )
    ap.add_argument(
        "--embed-device",
        type=str,
        default="cuda",
        help="Device for embedding model: 'cuda', 'cuda:0', 'cpu', etc.",
    )
    ap.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding model.",
    )
    ap.add_argument(
        "--max-sentences-for-embedding",
        type=int,
        default=128,
        help="Max sentences per sample for embedding (<=0 means no cap).",
    )
    ap.add_argument(
        "--sim-thresh-dup",
        type=float,
        default=0.9,
        help="Cosine similarity threshold for duplicate detection.",
    )
    ap.add_argument(
        "--sim-thresh-cluster",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for cluster graph edges.",
    )
    ap.add_argument(
        "--nn-low-sim-thresh",
        type=float,
        default=0.4,
        help="Threshold on nearest-neighbor similarity for semantic gibberish detection.",
    )
    ap.add_argument(
        "--gibberish-non-ascii-thresh",
        type=float,
        default=0.5,
        help="non_ascii_ratio threshold for semantic gibberish detection.",
    )
    ap.add_argument(
        "--gibberish-repeat-char-run-thresh",
        type=int,
        default=8,
        help="repeat_char_run_max threshold for semantic gibberish detection.",
    )
    ap.add_argument(
        "--gibberish-repl-char-ratio-thresh",
        type=float,
        default=0.1,
        help="replacement_char_ratio threshold for semantic gibberish detection.",
    )

    args = ap.parse_args()

    semantic_cfg = SemanticConfig(
        enable_semantic=not args.no_semantic,
        enable_embedding=(not args.no_embedding) and bool(args.embed_model),
        embed_model=args.embed_model,
        embed_backend=args.embed_backend,  # 新增
        embed_device=args.embed_device,
        embed_batch_size=int(args.embed_batch_size),
        max_sentences_for_embedding=int(args.max_sentences_for_embedding),
        sim_thresh_dup=float(args.sim_thresh_dup),
        sim_thresh_cluster=float(args.sim_thresh_cluster),
        nn_low_sim_thresh=float(args.nn_low_sim_thresh),
        gibberish_non_ascii_thresh=float(args.gibberish_non_ascii_thresh),
        gibberish_repeat_char_run_thresh=int(args.gibberish_repeat_char_run_thresh),
        gibberish_repl_char_ratio_thresh=float(args.gibberish_repl_char_ratio_thresh),
    )


    cfg = BuildConfig(
        jsonl=args.jsonl,
        out_dir=args.out_dir,
        procs=int(args.procs),
        chunk_lines=int(args.chunk_lines),
        correct_key=args.correct_key,
        snapshot_keys=[s.strip() for s in args.snapshot_keys.split(",") if s.strip()],
        compute_sha256=bool(args.compute_sha256),
        semantic=semantic_cfg,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    features_path = cfg.out_dir / "features.parquet"
    meta_path = cfg.out_dir / "meta.json"

    print(f"[INFO] Source JSONL: {cfg.jsonl}")
    print(f"[INFO] Output dir   : {cfg.out_dir}")
    print(f"[INFO] Procs={cfg.procs}, chunk_lines={cfg.chunk_lines}")
    print(f"[INFO] Semantic enabled: {cfg.semantic.enable_semantic}")
    print(f"[INFO] Embedding enabled: {cfg.semantic.enable_embedding}")
    if cfg.semantic.enable_embedding:
        print(
            f"[INFO]  Embed model='{cfg.semantic.embed_model}', device='{cfg.semantic.embed_device}', "
            f"batch_size={cfg.semantic.embed_batch_size}, max_sentences_for_embedding={cfg.semantic.max_sentences_for_embedding}"
        )
        print(
            f"[INFO]  sim_thresh_dup={cfg.semantic.sim_thresh_dup}, "
            f"sim_thresh_cluster={cfg.semantic.sim_thresh_cluster}, "
            f"nn_low_sim_thresh={cfg.semantic.nn_low_sim_thresh}"
        )

    print(f"[INFO] Scanning offsets: {cfg.jsonl}")
    t0 = time.time()
    chunks, sha256, fsize, mtime = _scan_offsets(cfg.jsonl, cfg.chunk_lines, cfg.compute_sha256)
    total_lines = sum(len(c) for c in chunks)
    print(f"[INFO] Found {total_lines:,} lines, chunks={len(chunks)}, took {time.time() - t0:.1f}s")

    # process chunks in parallel and append to Parquet
    print(f"[INFO] Processing chunks with {cfg.procs} processes …")
    t1 = time.time()
    writer = None
    n_rows = 0
    import pyarrow as pa, pyarrow.parquet as pq

    with mp.get_context("fork").Pool(processes=cfg.procs) as pool:
        tasks = [(str(cfg.jsonl), offs, cfg.correct_key, cfg.snapshot_keys, cfg.semantic) for offs in chunks]
        for i, ret in enumerate(pool.imap_unordered(_worker_entry, tasks, chunksize=1)):
            df: pd.DataFrame = ret["df"]
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(features_path, table.schema, compression="zstd", version="2.6")
            writer.write_table(table)
            n_rows += len(df)
            if (i + 1) % 10 == 0:
                print(f"  [Write] {i + 1}/{len(chunks)} chunks, rows={n_rows:,}")

    if writer is not None:
        writer.close()

    print(f"[INFO] Wrote features: {features_path} (rows={n_rows:,})  in {time.time() - t1:.1f}s")

    # meta.json
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source_jsonl": str(cfg.jsonl),
        "source_size": fsize,
        "source_mtime": mtime,
        "source_sha256": sha256,
        "features_path": str(features_path),
        "rows": n_rows,
        "snapshot_keys": cfg.snapshot_keys,
        "snapshot_columns": {k: sanitize_snap_col(k) for k in cfg.snapshot_keys},
        "build_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {
            "procs": cfg.procs,
            "chunk_lines": cfg.chunk_lines,
            "correct_key": cfg.correct_key,
            "compute_sha256": cfg.compute_sha256,
        },
        "semantic_config": {
            "enable_semantic": cfg.semantic.enable_semantic,
            "enable_embedding": cfg.semantic.enable_embedding,
            "embed_model": cfg.semantic.embed_model,
            "embed_backend": getattr(cfg.semantic, "embed_backend", "st"),
            "embed_device": cfg.semantic.embed_device,
            "embed_batch_size": cfg.semantic.embed_batch_size,
            "max_sentences_for_embedding": cfg.semantic.max_sentences_for_embedding,
            "sim_thresh_dup": cfg.semantic.sim_thresh_dup,
            "sim_thresh_cluster": cfg.semantic.sim_thresh_cluster,
            "nn_low_sim_thresh": cfg.semantic.nn_low_sim_thresh,
            "gibberish_non_ascii_thresh": cfg.semantic.gibberish_non_ascii_thresh,
            "gibberish_repeat_char_run_thresh": cfg.semantic.gibberish_repeat_char_run_thresh,
            "gibberish_repl_char_ratio_thresh": cfg.semantic.gibberish_repl_char_ratio_thresh,
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote meta: {meta_path}")

    print("[SUMMARY]")
    print(f"  rows={n_rows:,}")
    print(f"  features={features_path}")
    print(f"  meta={meta_path}")


if __name__ == "__main__":
    main()
