#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
online_rambling_filter_server.py

一个围绕现有 filter_allinone.py 的 HTTP 服务封装：

- 预加载 embedding 模型（vLLM 或 HF 多 GPU）；
- 每次请求传入一个 JSONL 路径（ray 那边写出的临时文件）；
- 复用 filter_allinone 里的全套指标 + pattern 逻辑，输出 keep_mask；
- 可选写 debug JSONL（good / bad），写到单独的 debug 目录。

配合你的 OnlineRamblingFilterClient 使用：

    client = OnlineRamblingFilterClient("http://filter-host:8000")
    keep_mask = client.filter_jsonl(jsonl_path, cls="correct")

冒烟测试：

    curl -s "${BASE_URL}/smoke" | jq .

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import multiprocessing as mp

import numpy as np
import pandas as pd

# ---- Web 框架 ----
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    raise RuntimeError(
        "online_rambling_filter_server 需要 fastapi + uvicorn + pydantic，"
        "请先安装: pip install fastapi uvicorn pydantic"
    ) from e

# ---- 导入你现有的 filter_allinone 作为库复用算法逻辑 ----
import filter_allinone as fa


# =============================================================================
# 配置 dataclass
# =============================================================================

@dataclass
class ServerConfig:
    # 核心 filter 配置
    correct_key: str
    min_lang_chars: int
    cpu_procs: int
    chunk_lines: int
    compute_sha256: bool

    # scan 阶段的子集选择
    kept_or_dropped: str        # "kept" / "dropped" / "all"
    scan_cls: str               # "correct" / "wrong" / "no_judge" / "any"

    # embedding backend 选择
    embed_backend: str          # "vllm" or "hf"
    embed_model_name: str

    # vLLM 配置
    tensor_parallel_size: int
    dtype: str
    gpu_memory_utilization: float
    trust_remote_code: bool
    enforce_eager: bool

    # HF 配置
    hf_max_workers: int
    hf_device: str
    hf_batch_size: int
    hf_max_length: int
    hf_torch_dtype: str

    # embedding analysis 配置
    dup_threshold: float
    cluster_threshold: float
    plateau_min_len: int
    loop_eps: float
    max_sent_per_sample: int

    # RamblingScore 配置
    rambling_weight_self: float
    rambling_weight_dup: float
    rambling_weight_loop: float
    rambling_weight_plateau: float
    rambling_weight_lang: float
    rambling_loopiness_cap: float
    rambling_plateau_k: int

    # pattern 并行度
    pattern_workers: int

    # debug JSONL 输出目录
    debug_out_dir: Optional[Path]
    debug_jsonl_enabled: bool

    # 全局 debug 开关（不从 client 传）
    debug: bool

    # 日志
    quiet: bool = False

    # >>> 新增：服务监听配置，用于启动完成时打印 URL <<<
    host: str = "0.0.0.0"
    port: int = 8000


# =============================================================================
# HF embedding worker
# =============================================================================

def _hf_worker_loop(
    worker_id: int,
    device_str: str,
    model_name: str,
    hf_max_length: int,
    hf_batch_size: int,
    hf_torch_dtype: str,
    trust_remote_code: bool,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
):
    """
    长驻的 HF embedding worker：
    - 每个 worker 固定在一张 GPU 上，启动时加载一次 tokenizer + model；
    - 循环从 task_queue 里拿 (job_id, texts)，返回 (job_id, embs)；
    - 退出条件：收到 job_id 为 None。
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    from torch import nn

    device = torch.device(device_str)
    print(f"[HF-Worker-{worker_id}] Start on device={device}", flush=True)

    # 解析 dtype
    torch_dtype = None
    if hf_torch_dtype != "auto":
        try:
            torch_dtype = getattr(torch, hf_torch_dtype)
        except AttributeError:
            raise ValueError(
                f"[HF-Worker-{worker_id}] Invalid hf_torch_dtype={hf_torch_dtype}, "
                "可选: auto,float32,float16,bfloat16"
            )
    if device.type == "cpu" and torch_dtype in (
        getattr(torch, "float16", None),
        getattr(torch, "bfloat16", None),
    ):
        print(f"[HF-Worker-{worker_id}] WARN: CPU 上使用 {hf_torch_dtype} 不安全，退回 float32", flush=True)
        torch_dtype = torch.float32

    # 加载模型
    print(f"[HF-Worker-{worker_id}] Loading HF model {model_name} ...", flush=True)
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
    print(f"[HF-Worker-{worker_id}] Model loaded in {time.time()-t0:.1f}s", flush=True)

    norm_layer = nn.functional.normalize

    while True:
        job = task_queue.get()
        if job is None:
            print(f"[HF-Worker-{worker_id}] Got stop signal. Exit.", flush=True)
            break

        job_id, texts = job
        if not isinstance(texts, list):
            texts = list(texts)

        n = len(texts)
        if n == 0:
            result_queue.put((job_id, []))
            continue

        if hf_batch_size is None or hf_batch_size <= 0:
            bs = max(1, n)
        else:
            bs = max(1, int(hf_batch_size))

        embs: List[List[float]] = []
        with torch.inference_mode():
            for start in range(0, n, bs):
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
                    token_embeddings = outputs.last_hidden_state
                else:
                    token_embeddings = outputs[0]

                attention_mask = enc["attention_mask"]
                expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

                sum_embeddings = (token_embeddings * expanded_mask).sum(dim=1)
                sum_mask = expanded_mask.sum(dim=1).clamp(min=1e-9)
                sent_emb = sum_embeddings / sum_mask

                sent_emb = norm_layer(sent_emb, p=2, dim=1)

                embs.extend(sent_emb.detach().cpu().numpy().tolist())

        result_queue.put((job_id, embs))


# =============================================================================
# Filter Service：封装完整 pipeline
# =============================================================================

class FilterService:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg

        # 解析 metrics 开关（完全复用你原来的逻辑）
        self.metric_switches = fa.resolve_metric_dependencies()

        # pattern 并行度
        cpu_total = os.cpu_count() or 1
        if cfg.pattern_workers <= 0:
            fa.PATTERN_WORKERS = max(1, min(cpu_total, 32))
        else:
            fa.PATTERN_WORKERS = max(1, min(cfg.pattern_workers, cpu_total))
        if not cfg.quiet:
            print(f"[Service] pattern_workers = {fa.PATTERN_WORKERS} (available CPUs = {cpu_total})")

        # debug 输出目录 & 标志
        self.debug_out_dir = cfg.debug_out_dir
        self.debug = cfg.debug
        # debug_jsonl 必须同时满足：全局 debug + 没有被 no-debug-jsonl 禁用
        self.debug_jsonl_enabled = cfg.debug_jsonl_enabled and cfg.debug
        if self.debug_out_dir is not None:
            self.debug_out_dir.mkdir(parents=True, exist_ok=True)
            if not cfg.quiet:
                print(f"[Service] Debug JSONL dir = {self.debug_out_dir}")
                print(f"[Service] Debug JSONL enabled = {self.debug_jsonl_enabled}")

        # embedding backend 初始化锁 -> 保证每次只由一个请求使用 embed backend
        self._embed_lock = threading.Lock()

        # 初始化 embedding backend
        self.embed_backend = cfg.embed_backend
        if self.embed_backend == "vllm":
            self._init_vllm()
        elif self.embed_backend == "hf":
            self._init_hf()
        else:
            raise ValueError(f"Unknown embed_backend={self.embed_backend}")

    # -------------------------------------------------------------------------
    # vLLM backend
    # -------------------------------------------------------------------------

    def _init_vllm(self):
        from vllm import LLM

        print(f"[Service] Initializing vLLM embedding model: {self.cfg.embed_model_name}", flush=True)
        t0 = time.time()
        self.vllm_llm = LLM(
            model=self.cfg.embed_model_name,
            task="embed",
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            dtype=self.cfg.dtype,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            trust_remote_code=self.cfg.trust_remote_code,
            enforce_eager=self.cfg.enforce_eager,
        )
        print(f"[Service] vLLM model loaded in {time.time()-t0:.1f}s", flush=True)

    def _embed_with_vllm(self, texts: List[str]) -> List[List[float]]:
        """
        使用预加载的 vLLM LLM(task='embed') 做 sentence embedding。
        """
        with self._embed_lock:
            if not texts:
                return []
            outputs = self.vllm_llm.embed(texts)
        if len(outputs) != len(texts):
            raise RuntimeError(f"vLLM embed returned {len(outputs)} outputs, expected {len(texts)}")
        embs: List[List[float]] = []
        for out in outputs:
            embs.append(fa.extract_embedding_from_output(out))
        return embs

    # -------------------------------------------------------------------------
    # HF backend（多 GPU，多进程，常驻）
    # -------------------------------------------------------------------------

    def _init_hf(self):
        try:
            import torch  # noqa
        except ImportError as e:
            raise RuntimeError(
                "HF backend 需要 torch + transformers，"
                "请先 pip install torch transformers"
            ) from e

        # 决定 GPU 数 & worker 数
        import torch
        if self.cfg.hf_device in ("auto", "cuda"):
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if num_gpus <= 0:
                    print("[Service] HF backend: torch.cuda.is_available=True but device_count=0, fallback to CPU")
                    devices = ["cpu"]
                else:
                    if self.cfg.hf_max_workers and self.cfg.hf_max_workers > 0:
                        n_workers = min(self.cfg.hf_max_workers, num_gpus)
                    else:
                        n_workers = num_gpus
                    devices = [f"cuda:{i}" for i in range(n_workers)]
            else:
                print("[Service] HF backend: cuda not available, use CPU")
                devices = ["cpu"]
        elif self.cfg.hf_device.startswith("cuda:"):
            devices = [self.cfg.hf_device]
        else:
            devices = [self.cfg.hf_device]

        self.hf_devices = devices
        self.hf_task_queues: List[mp.Queue] = []
        self.hf_result_queues: List[mp.Queue] = []
        self.hf_workers: List[mp.Process] = []

        ctx = mp.get_context("spawn")
        for wid, dev in enumerate(devices):
            tq = ctx.Queue()
            rq = ctx.Queue()
            p = ctx.Process(
                target=_hf_worker_loop,
                args=(
                    wid,
                    dev,
                    self.cfg.embed_model_name,
                    self.cfg.hf_max_length,
                    self.cfg.hf_batch_size,
                    self.cfg.hf_torch_dtype,
                    self.cfg.trust_remote_code,
                    tq,
                    rq,
                ),
                daemon=True,
            )
            p.start()
            self.hf_task_queues.append(tq)
            self.hf_result_queues.append(rq)
            self.hf_workers.append(p)

        print(
            f"[Service] HF embedding backend init done: {len(self.hf_workers)} workers on devices={devices}",
            flush=True,
        )

    def _embed_with_hf(self, texts: List[str]) -> List[List[float]]:
        """
        使用常驻 HF worker，多 GPU 并行做 embedding。
        - 简单同步调度：一个请求内，所有 workers 只处理这一次任务；
        - 为避免混乱，用 self._embed_lock 保证单次只有一个请求在跑。
        """
        with self._embed_lock:
            n = len(texts)
            if n == 0:
                return []

            n_workers = len(self.hf_workers)
            if n_workers == 0:
                raise RuntimeError("HF backend not initialized")

            splits = np.linspace(0, n, num=n_workers + 1, dtype=int)
            tasks: List[Tuple[int, int, int]] = []  # (worker_idx, start, end)
            for wid in range(n_workers):
                start = int(splits[wid])
                end = int(splits[wid + 1])
                if start >= end:
                    continue
                tasks.append((wid, start, end))

            if not tasks:
                wid = 0
                tasks = [(wid, 0, n)]

            # 发送 job
            for wid, start, end in tasks:
                job_id = start  # 用起始 index 作为 job_id
                slice_texts = texts[start:end]
                self.hf_task_queues[wid].put((job_id, slice_texts))

            # 收集结果
            results: Dict[int, List[List[float]]] = {}
            for wid, start, end in tasks:
                job_id_expected = start
                job_id, embs = self.hf_result_queues[wid].get()
                if job_id != job_id_expected:
                    print(
                        f"[Service] WARN: HF worker {wid} returned unexpected job_id={job_id}, "
                        f"expected {job_id_expected}",
                        flush=True,
                    )
                results[job_id] = embs

            final_embs: List[List[float]] = [None] * n  # type: ignore
            for wid, start, end in tasks:
                embs = results.get(start, [])
                if len(embs) != (end - start):
                    raise RuntimeError(
                        f"HF worker result length mismatch: worker={wid}, start={start}, end={end}, "
                        f"expected {end-start}, got {len(embs)}"
                    )
                for i, e in enumerate(embs):
                    final_embs[start + i] = e

            for i, e in enumerate(final_embs):
                if e is None:
                    raise RuntimeError(f"HF embedding result missing at index {i}")

            return final_embs  # type: ignore

    # -------------------------------------------------------------------------
    # 冒烟测试 / 健康检查：检查 embedding backend 是否正常
    # -------------------------------------------------------------------------

    def smoke_test(self) -> Tuple[bool, str]:
        """
        做一次较完整的健康检查：

        - 尝试用当前 embedding backend 跑一次最小 embedding；
        - embed-backend = "hf" 时，额外检查 HF worker 是否全部存活；
        - 返回 (ok, message)，不抛异常，方便 /health 直接使用。
        """
        try:
            test_text = ["smoke test"]
            # 1) 尝试跑一次 embedding
            if self.embed_backend == "vllm":
                _ = self._embed_with_vllm(test_text)
            else:
                _ = self._embed_with_hf(test_text)

            # 2) 如果是 HF backend，顺便看看 worker 进程有没有挂
            if self.embed_backend == "hf":
                dead_workers = []
                for idx, proc in enumerate(self.hf_workers):
                    if proc is not None and not proc.is_alive():
                        dead_workers.append(idx)
                if dead_workers:
                    return (
                        False,
                        f"HF embedding workers not alive: {dead_workers}",
                    )

            # 一切正常
            return True, "embedding backend healthy"

        except Exception as e:
            # 把异常封装成 message 返回，不往外抛
            return False, f"{type(e).__name__}: {e}"

    # -------------------------------------------------------------------------
    # 主过滤逻辑：给定 jsonl_path，跑完整 pipeline，返回 keep_mask
    # -------------------------------------------------------------------------

    def run_filter_on_jsonl(
        self,
        jsonl_path: str,
        req_cls: str = "correct",
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, str]]:
        """
        读入 jsonl_path（ray 写出的临时 JSONL），运行完整 pipeline。

        req_cls: 本次请求希望在哪个 cls 上做 pattern 过滤（correct / wrong / no_judge / any）。
        scan 阶段使用的是 server 启动时的 self.cfg.scan_cls。
        """
        t_start = time.time()
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL not found: {path}")

        if not self.cfg.quiet:
            print(
                f"[Service] === Filter request on {path} "
                f"(req_cls={req_cls}, debug={self.debug}) ===",
                flush=True,
            )

        # ------- Metric switches -------
        ms = self.metric_switches

        # ------- Stage0: scan offsets -------
        cfg = fa.BuildConfig(
            jsonl=path,
            correct_key=self.cfg.correct_key,
            min_lang_chars=self.cfg.min_lang_chars,
            cpu_procs=int(self.cfg.cpu_procs),
            chunk_lines=int(self.cfg.chunk_lines),
            compute_sha256=bool(self.cfg.compute_sha256),
            step_min=None,
            step_max=None,
            debug=bool(self.debug),
            enable_text_metrics=ms.enable_text_metrics,
            enable_contains_chinese=ms.enable_contains_chinese,
            enable_split_sections_and_sentences=ms.enable_split_sections_and_sentences,
            enable_detect_markers=ms.enable_detect_markers,
            enable_detect_lang=ms.enable_detect_lang,
        )

        if not self.cfg.quiet:
            print(f"[Service][Scan] Scanning offsets in JSONL: {cfg.jsonl}", flush=True)
            print(
                f"[Service][Scan] kept_or_dropped={self.cfg.kept_or_dropped}, "
                f"cls_filter={self.cfg.scan_cls}",
                flush=True,
            )

        chunks, sha256, fsize, mtime, n_lines, scan_stats = fa.scan_offsets(
            cfg,
            kept_or_dropped=self.cfg.kept_or_dropped,
            cls_filter=self.cfg.scan_cls,
        )

        if not self.cfg.quiet:
            print(f"[Service][Scan] File lines      = {scan_stats['num_total']:,}")
            print(f"[Service][Scan] Selected lines  = {scan_stats['num_selected']:,}")
            print(f"[Service][Scan] Chunks          = {len(chunks):,}")

        if scan_stats["num_selected"] == 0:
            # 没有任何可解析行，直接返回“全部保留”（不做过滤）
            return (
                np.ones(scan_stats["num_total"], dtype=bool),
                {
                    "num_total": scan_stats["num_total"],
                    "num_selected": 0,
                    "num_bad": 0,
                },
                {},
            )

        # ------- Stage1: CPU metrics + segmentation -------
        if not self.cfg.quiet:
            print(f"[Service][Stage1] Processing chunks with {cfg.cpu_procs} processes ...", flush=True)

        tasks = [(str(cfg.jsonl), offs, cfg) for offs in chunks]
        start_method = "spawn" if sys.platform == "win32" else "fork"
        ctx = mp.get_context(start_method)
        samples_list: List[pd.DataFrame] = []
        segments_list: List[pd.DataFrame] = []

        with ctx.Pool(processes=cfg.cpu_procs) as pool:
            for ret in pool.imap_unordered(fa._worker_entry_stage1, tasks, chunksize=1):
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

        # 按 sample_idx 排序
        if not samples_df.empty:
            samples_df.sort_values(by=["sample_idx"], inplace=True)
            samples_df.reset_index(drop=True, inplace=True)

        if not segments_df.empty:
            segments_df.sort_values(
                by=["sample_idx", "section_idx", "sent_idx"],
                inplace=True,
            )
            segments_df.reset_index(drop=True, inplace=True)

        if not self.cfg.quiet:
            print(
                f"[Service][Stage1] Done: samples={len(samples_df):,}, "
                f"segments={len(segments_df):,}",
                flush=True,
            )

        # ------- correct_key Warning 逻辑 -------
        if "cls_id" not in samples_df.columns:
            print(
                f"[WARN] correct_key='{self.cfg.correct_key}' 未能生成 'cls_id' 列，"
                "所有样本将被视为 no_judge (cls_id=2)。",
                flush=True,
            )
            samples_df["cls_id"] = 2
        else:
            n_total_cls = len(samples_df)
            n_valid_cls = int(
                ((samples_df["cls_id"] == 0) | (samples_df["cls_id"] == 1)).sum()
            )
            if n_valid_cls == 0 and n_total_cls > 0:
                print(
                    f"[WARN] correct_key='{self.cfg.correct_key}' 未在任何样本上产生 "
                    "cls_id ∈ {0,1}；可能 key 路径错误或字段缺失。"
                    "pattern 中基于 cls 的过滤可能不会生效。",
                    flush=True,
                )

        # ------- 补充 response_text -------
        if "response_text" not in samples_df.columns:
            if not self.cfg.quiet:
                print("[Service] Attaching response_text from JSONL ...", flush=True)
            t_txt0 = time.time()
            resp_list: List[str] = []
            with cfg.jsonl.open("rb", buffering=1024 * 1024) as fsrc:
                fileno = fsrc.fileno()
                for _, row in samples_df.iterrows():
                    off = int(row["byte_offset"])
                    ln = int(row["byte_len"])
                    b = os.pread(fileno, ln, off)
                    try:
                        j = fa._json_loads(b)
                    except Exception:
                        resp_list.append("")
                        continue
                    resp_list.append(j.get("response_text", ""))
            samples_df["response_text"] = resp_list
            if not self.cfg.quiet:
                print(f"[Service] response_text attached in {time.time()-t_txt0:.1f}s", flush=True)

        # ------- Stage2: sentence embedding -------
        if ms.enable_embedding and not segments_df.empty:
            t2 = time.time()
            texts = segments_df["text"].astype(str).tolist()
            if self.embed_backend == "vllm":
                if not self.cfg.quiet:
                    print("[Service][Stage2] Using vLLM backend for embedding ...", flush=True)
                embs = self._embed_with_vllm(texts)
            else:
                if not self.cfg.quiet:
                    print(
                        f"[Service][Stage2] Using HF backend for embedding "
                        f"(workers={len(self.hf_workers)}, devices={self.hf_devices}) ...",
                        flush=True,
                    )
                embs = self._embed_with_hf(texts)

            if len(embs) != len(segments_df):
                raise RuntimeError(
                    f"Embedding result row mismatch: got {len(embs)} embs, "
                    f"segments_df has {len(segments_df)} rows."
                )

            segments_df = segments_df.copy()
            segments_df["emb"] = embs

            if not self.cfg.quiet:
                print(f"[Service][Stage2] Embedding done in {time.time()-t2:.1f}s", flush=True)
        else:
            if not self.cfg.quiet:
                print("[Service][Stage2] Embedding disabled or no segments; skip embedding.", flush=True)

        # ------- Stage3: embedding-based metrics + rambling_score -------
        rambling_weights = {
            "self": self.cfg.rambling_weight_self,
            "dup": self.cfg.rambling_weight_dup,
            "loop": self.cfg.rambling_weight_loop,
            "plateau": self.cfg.rambling_weight_plateau,
            "lang": self.cfg.rambling_weight_lang,
        }

        samples_df = fa.run_embedding_analysis_for_all_samples(
            samples_df,
            segments_df,
            dup_threshold=self.cfg.dup_threshold,
            cluster_threshold=self.cfg.cluster_threshold,
            plateau_min_len=self.cfg.plateau_min_len,
            loop_eps=self.cfg.loop_eps,
            max_sent_per_sample=self.cfg.max_sent_per_sample,
            rambling_weights=rambling_weights,
            rambling_loopiness_cap=self.cfg.rambling_loopiness_cap,
            rambling_plateau_k=self.cfg.rambling_plateau_k,
            enable_embedding_metrics=ms.enable_embedding_metrics,
            enable_rambling_score=ms.enable_rambling_score,
        )

        # ------- Pattern filtering -------
        req_cls = (req_cls or "any").lower()
        if req_cls == "correct":
            target_mask = (samples_df["cls_id"].astype(int) == 1).to_numpy()
        elif req_cls == "wrong":
            target_mask = (samples_df["cls_id"].astype(int) == 0).to_numpy()
        elif req_cls == "no_judge":
            target_mask = (samples_df["cls_id"].astype(int) == 2).to_numpy()
        else:
            target_mask = None  # any

        if not self.cfg.quiet:
            print(f"[Service][Filter] Running pattern-based filtering (req_cls={req_cls}) ...", flush=True)

        keep_mask, bad_mask, pattern_hits = fa.build_filter_mask_by_patterns(
            samples_df,
            target_mask=target_mask,
        )

        num_total = len(samples_df)
        num_bad = int(bad_mask.sum())
        num_good = num_total - num_bad

        if not self.cfg.quiet:
            print(f"[Service][Filter] total={num_total:,}, bad={num_bad:,}, good={num_good:,}", flush=True)

        # ------- Debug JSONL 输出（可选） -------
        debug_files: Dict[str, str] = {}
        if self.debug and self.debug_jsonl_enabled and self.debug_out_dir is not None:
            if not self.cfg.quiet:
                print("[Service][Debug] Writing debug JSONL (good/bad) ...", flush=True)
            debug_files = self._write_debug_jsonl(
                cfg=cfg,
                samples_df=samples_df,
                bad_mask=bad_mask,
                pattern_hits=pattern_hits,
                jsonl_path=path,
            )

        t_end = time.time()
        if not self.cfg.quiet:
            print(f"[Service] Filter pipeline finished in {t_end-t_start:.1f}s", flush=True)

        stats = {
            "num_total": num_total,
            "num_bad": num_bad,
            "num_good": num_good,
        }

        return keep_mask, stats, debug_files

    # -------------------------------------------------------------------------
    # debug JSONL 写出
    # -------------------------------------------------------------------------

    def _write_debug_jsonl(
        self,
        cfg: fa.BuildConfig,
        samples_df: pd.DataFrame,
        bad_mask: np.ndarray,
        pattern_hits: Dict[str, np.ndarray],
        jsonl_path: Path,
    ) -> Dict[str, str]:
        """
        将 good / bad 样本分别写出到 debug_out_dir 下的 JSONL：
        - 文件名：<basename>.good.jsonl / <basename>.bad.jsonl
        - 在每行附加 "filter_debug" 字段，包含 metrics + 命中的 pattern 名。
        """
        # per-row: 命中的 pattern 名
        pattern_names = [name for (name, _) in fa.PATTERN_FUNCS]
        hit_patterns_per_row: List[List[str]] = [[] for _ in range(len(samples_df))]
        for pname in pattern_names:
            if pname not in pattern_hits:
                continue
            m = pattern_hits[pname]
            for i, flag in enumerate(m):
                if flag:
                    hit_patterns_per_row[i].append(pname)

        enabled_metrics = set(fa.ENABLED_METRICS)

        def build_metrics_for_row(row: pd.Series) -> Dict[str, Any]:
            m: Dict[str, Any] = {}
            for col in fa.FILTER_RELATED_COLUMNS:
                if col not in enabled_metrics:
                    continue
                if col in row:
                    val = row[col]
                    if isinstance(val, (np.generic,)):
                        val = val.item()
                    m[col] = val
            return m

        base = jsonl_path.name
        out_good = self.debug_out_dir / f"{base}.good.jsonl"
        out_bad = self.debug_out_dir / f"{base}.bad.jsonl"

        with cfg.jsonl.open("rb", buffering=1024 * 1024) as fsrc, \
                out_good.open("w", encoding="utf-8") as f_good, \
                out_bad.open("w", encoding="utf-8") as f_bad:
            fileno = fsrc.fileno()

            for idx, row in samples_df.iterrows():
                off = int(row["byte_offset"])
                ln = int(row["byte_len"])
                b = os.pread(fileno, ln, off)
                try:
                    j = fa._json_loads(b)
                except Exception:
                    j = {}

                row_bad = bool(bad_mask[idx])
                metrics = build_metrics_for_row(row)
                debug_info = {
                    "hit": row_bad,
                    "hit_patterns": hit_patterns_per_row[idx],
                    "cls_id": int(row.get("cls_id", 2)),
                    "kept_flag": int(row.get("kept_flag", -1)),
                    "global_step": int(row.get("global_step", -1)),
                    "sample_idx": int(row.get("sample_idx", -1)),
                    "metrics": metrics,
                }
                if cfg.debug:
                    debug_info["debug_note"] = "online pattern filtering debug info"

                if isinstance(j, dict):
                    j["filter_debug"] = debug_info
                    line_out = json.dumps(j, ensure_ascii=False)
                else:
                    j2 = {"raw": str(j), "filter_debug": debug_info}
                    line_out = json.dumps(j2, ensure_ascii=False)

                if row_bad:
                    f_bad.write(line_out + "\n")
                else:
                    f_good.write(line_out + "\n")

        return {
            "good": str(out_good),
            "bad": str(out_bad),
        }


# =============================================================================
# FastAPI request/response schema
# =============================================================================

class FilterRequest(BaseModel):
    jsonl_path: str
    cls: str = "correct"       # "correct" / "wrong" / "no_judge" / "any"


class FilterResponse(BaseModel):
    ok: bool
    message: Optional[str] = None
    num_samples: int = 0
    num_bad: int = 0
    keep_mask: Optional[List[bool]] = None
    debug_files: Optional[Dict[str, str]] = None


# =============================================================================
# 创建 app
# =============================================================================

def create_app(config: ServerConfig) -> FastAPI:
    service = FilterService(config)

    app = FastAPI(title="Online Rambling Filter Server")

    @app.on_event("startup")
    async def _on_startup():
        """
        应用启动完成后的回调：
        此时 FastAPI 已经完成初始化、路由挂载，说明服务已经 ready。
        host/port 与 main() 里 uvicorn.run 传入的保持一致。
        """
        print(
            f"[Server] Online Rambling Filter Server is ready at "
            f"http://{config.host}:{config.port}",
            flush=True,
        )

    @app.get("/health")
    def health(deep: bool = False):
        """
        健康检查端点：

        - deep = False（默认）：只检查进程是否存活，返回一个快速 ok；
        - deep = True：做一次完整冒烟测试（embedding backend 等），
          等价于原来的 /smoke。
        """
        if not deep:
            # 进程能响应这个接口就说明“活着”
            return {
                "ok": True,
                "message": "online rambling filter is alive",
            }

        ok, msg = service.smoke_test()
        return {
            "ok": ok,
            "message": msg,
        }

    @app.get("/smoke")
    def smoke():
        """
        兼容旧用法：/smoke 相当于 /health?deep=true
        """
        ok, msg = service.smoke_test()
        return {
            "ok": ok,
            "message": msg,
        }

    @app.post("/filter", response_model=FilterResponse)
    def filter_endpoint(req: FilterRequest):
        try:
            keep_mask, stats, debug_files = service.run_filter_on_jsonl(
                jsonl_path=req.jsonl_path,
                req_cls=req.cls,
            )
        except Exception as e:
            return FilterResponse(
                ok=False,
                message=f"{type(e).__name__}: {e}",
                num_samples=0,
                num_bad=0,
                keep_mask=None,
                debug_files=None,
            )

        mask_list = [bool(x) for x in keep_mask.tolist()]
        return FilterResponse(
            ok=True,
            message="ok",
            num_samples=len(mask_list),
            num_bad=int((~keep_mask).sum()),
            keep_mask=mask_list,
            debug_files=debug_files or None,
        )

    return app


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Online rambling filter server (wraps filter_allinone.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- server 相关 ----
    ap.add_argument("--host", type=str, default="0.0.0.0", help="HTTP 监听地址")
    ap.add_argument("--port", type=int, default=8000, help="HTTP 监听端口")
    ap.add_argument("--log-level", type=str, default="info", help="uvicorn 日志级别")
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="减少 filter 内部打印",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="开启全局 debug 模式（BuildConfig.debug / debug JSONL）",
    )

    # ---- filter 核心参数 ----
    ap.add_argument("--correct-key", type=str, default="reward_extra_infos.acc")
    ap.add_argument(
        "--min-lang-chars",
        type=int,
        default=5,
        help="句子长度小于该值则不做 langid",
    )
    ap.add_argument(
        "--cpu-procs",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 2),
        help="Stage1 CPU 多进程数",
    )
    ap.add_argument("--chunk-lines", type=int, default=20000)
    ap.add_argument("--compute-sha256", action="store_true")

    # ---- scan 子集参数：保留你原来 offline 脚本的接口 ----
    ap.add_argument(
        "--kept-or-dropped",
        type=str,
        default="kept",
        choices=["kept", "dropped", "all"],
        help="scan_offsets 阶段选择 kept/dropped/all",
    )
    ap.add_argument(
        "--cls",
        type=str,
        default="correct",
        choices=["correct", "wrong", "no_judge", "any"],
        help="scan_offsets 阶段选择 cls 过滤（与请求里的 cls 解耦）",
    )

    # ---- embedding backend ----
    ap.add_argument(
        "--embed-backend",
        type=str,
        default="vllm",
        choices=["vllm", "hf"],
        help="选择 embedding 后端: vllm 或 hf",
    )
    ap.add_argument(
        "--embed-model-name",
        type=str,
        default="BAAI/bge-m3",
        help="embedding 模型名或本地路径",
    )

    # vLLM
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")

    # HF
    ap.add_argument(
        "--hf-max-workers",
        type=int,
        default=0,
        help="HF 多 GPU worker 数；<=0 表示使用所有可见 GPU",
    )
    ap.add_argument(
        "--hf-device",
        type=str,
        default="auto",
        help="'auto', 'cuda', 'cuda:0', 'cpu' 等",
    )
    ap.add_argument(
        "--hf-batch-size",
        type=int,
        default=128,
        help="HF embedding batch size；<=0 表示单 batch 全部送入（可能 OOM）",
    )
    ap.add_argument(
        "--hf-max-length",
        type=int,
        default=1024,
        help="HF tokenizer max_length（截断）",
    )
    ap.add_argument(
        "--hf-torch-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
    )

    # ---- embedding analysis 参数 ----
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
    ap.add_argument(
        "--pattern-workers",
        type=int,
        default=0,
        help="pattern 规则并行度；<=0 表示自动选择",
    )

    # debug JSONL 输出目录
    ap.add_argument(
        "--debug-out-dir",
        type=str,
        default=None,
        help="如果设置，则在此目录下写出 good/bad debug JSONL；需要配合 --debug 启用",
    )
    ap.add_argument(
        "--no-debug-jsonl",
        action="store_true",
        help="即使有 debug-out-dir 也不写 debug JSONL",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    debug_out_dir = Path(args.debug_out_dir) if args.debug_out_dir else None

    cfg = ServerConfig(
        correct_key=args.correct_key,
        min_lang_chars=args.min_lang_chars,
        cpu_procs=args.cpu_procs,
        chunk_lines=args.chunk_lines,
        compute_sha256=args.compute_sha256,
        kept_or_dropped=args.kept_or_dropped,
        scan_cls=args.cls.lower(),
        embed_backend=args.embed_backend,
        embed_model_name=args.embed_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        hf_max_workers=args.hf_max_workers,
        hf_device=args.hf_device,
        hf_batch_size=args.hf_batch_size,
        hf_max_length=args.hf_max_length,
        hf_torch_dtype=args.hf_torch_dtype,
        dup_threshold=args.dup_threshold,
        cluster_threshold=args.cluster_threshold,
        plateau_min_len=args.plateau_min_len,
        loop_eps=args.loop_eps,
        max_sent_per_sample=args.max_sent_per_sample,
        rambling_weight_self=args.rambling_weight_self,
        rambling_weight_dup=args.rambling_weight_dup,
        rambling_weight_loop=args.rambling_weight_loop,
        rambling_weight_plateau=args.rambling_weight_plateau,
        rambling_weight_lang=args.rambling_weight_lang,
        rambling_loopiness_cap=args.rambling_loopiness_cap,
        rambling_plateau_k=args.rambling_plateau_k,
        pattern_workers=args.pattern_workers,
        debug_out_dir=debug_out_dir,
        debug_jsonl_enabled=(not args.no_debug_jsonl),
        debug=bool(args.debug),
        quiet=bool(args.quiet),
        # >>> 新增：把 CLI 的 host/port 传进配置 <<<
        host=args.host,
        port=args.port,
    )

    app = create_app(cfg)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
