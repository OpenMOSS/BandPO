#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_server.py

在线过滤服务：
- 启动时预加载 embedding 模型（HF 或 vLLM）。
- 每次收到请求，传入一个 JSONL 路径（和离线 filter_allinone 兼容的字段格式）。
- 跑一遍 Stage0 + Stage1 + Embedding + Stage3 + pattern filter，
  返回与 JSONL 行顺序一致的 keep_mask / bad_mask（0/1），可选写 debug good/bad jsonl。

启动示例（HF 后端）：

python filter_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --cpu-procs 8 \
  --chunk-lines 512 \
  --embed-backend hf \
  --embed-model-name "/remote-home1/yli/Workspace/BandPO/data/models/BAAI/bge-m3" \
  --hf-device cuda \
  --hf-max-workers 1 \
  --hf-batch-size 128 \
  --hf-max-length 1024 \
  --hf-torch-dtype bfloat16 \
  --dup-threshold 0.90 \
  --cluster-threshold 0.85 \
  --plateau-min-len 3 \
  --max-sent-per-sample 128 \
  --rambling-weight-self 1.0 \
  --rambling-weight-dup 1.0 \
  --rambling-weight-loop 1.0 \
  --rambling-weight-plateau 1.0 \
  --rambling-weight-lang 0.1 \
  --rambling-loopiness-cap 5.0 \
  --rambling-plateau-k 4 \
  --pattern-workers 16 \
  --debug-out-dir "/path/to/filter_debug_dir"

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 直接复用 filter_allinone 里的算法和结构
from filter_allinone import (
    BuildConfig,
    ENABLED_METRICS,
    EMBEDDING_METRIC_COLUMNS,
    FILTER_RELATED_COLUMNS,
    PATTERN_FUNCS,
    PATTERN_WORKERS,
    SCHEMA_VERSION,
    TEXT_METRIC_COLUMNS,
    _json_loads,
    _worker_entry_stage1,
    build_filter_mask_by_patterns,
    resolve_metric_dependencies,
    run_embedding_analysis_for_all_samples,
    scan_offsets,
)

import multiprocessing as mp


# ==================== 嵌入模型封装 ====================

class BaseEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class VllmEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        dtype: str,
        gpu_memory_utilization: float,
        trust_remote_code: bool,
        enforce_eager: bool,
    ) -> None:
        from vllm import LLM
        from filter_allinone import extract_embedding_from_output

        self._extract_embedding_from_output = extract_embedding_from_output
        print(f"[Server][vLLM] Loading model {model_name} ...")
        t0 = time.time()
        self.llm = LLM(
            model=model_name,
            task="embed",
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
        )
        print(f"[Server][vLLM] Model loaded in {time.time() - t0:.1f}s")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        outs = self.llm.embed(texts)
        embs: List[List[float]] = []
        for out in outs:
            embs.append(self._extract_embedding_from_output(out))
        return embs


class HFEmbedder(BaseEmbedder):
    """
    简化版 HF 嵌入器：单进程、单 device。
    多卡情况下可以通过起多个 filter_server 进程，由外层做负载均衡。
    如果以后需要多 GPU/多进程，可以在此基础上扩展一个持久 worker pool。
    """

    def __init__(
        self,
        model_name: str,
        device_str: str,
        batch_size: int,
        max_length: int,
        torch_dtype_str: str,
        trust_remote_code: bool,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device_str)
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)

        if torch_dtype_str == "auto":
            torch_dtype = None
        else:
            try:
                torch_dtype = getattr(torch, torch_dtype_str)
            except AttributeError:
                raise ValueError(
                    f"Invalid hf-torch-dtype={torch_dtype_str}, "
                    "must be one of: auto,float32,float16,bfloat16"
                )
        if self.device.type == "cpu" and torch_dtype in (
            getattr(torch, "float16", None),
            getattr(torch, "bfloat16", None),
        ):
            print("[Server][HF][WARN] CPU 上不建议 float16/bfloat16，回退到 float32")
            torch_dtype = torch.float32

        print(f"[Server][HF] Loading tokenizer/model {model_name} on {self.device} ...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if torch_dtype is None:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
            )
        self.model.to(self.device)
        self.model.eval()
        print(f"[Server][HF] Model loaded in {time.time() - t0:.1f}s")

    def embed(self, texts: List[str]) -> List[List[float]]:
        import torch
        from torch import nn

        n = len(texts)
        if n == 0:
            return []

        embs: List[List[float]] = []
        norm_layer = nn.functional.normalize

        with torch.inference_mode():
            for start in range(0, n, self.batch_size):
                batch_texts = texts[start:start + self.batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**enc)
                if hasattr(outputs, "last_hidden_state"):
                    token_embeddings = outputs.last_hidden_state
                else:
                    token_embeddings = outputs[0]

                attention_mask = enc["attention_mask"]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
                sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
                sent_emb = sum_embeddings / sum_mask  # [B, D]

                sent_emb = norm_layer(sent_emb, p=2, dim=1)
                embs.extend(sent_emb.detach().cpu().numpy().tolist())

        return embs


# ==================== 请求 / 响应模型 ====================

class FilterRequest(BaseModel):
    jsonl_path: str
    # 哪一类样本参与 pattern：any/correct/wrong/no_judge
    cls: str = "correct"
    debug: bool = False


class FilterResponse(BaseModel):
    ok: bool
    message: str
    num_samples: int
    num_bad: int
    keep_mask: List[int]  # 1=keep, 0=bad
    bad_mask: List[int]
    pattern_counts: Dict[str, int]


# ==================== Filter Service 封装 ====================

@dataclass
class ServerConfig:
    correct_key: str
    min_lang_chars: int
    cpu_procs: int
    chunk_lines: int
    dup_threshold: float
    cluster_threshold: float
    plateau_min_len: int
    loop_eps: float
    max_sent_per_sample: int
    rambling_weights: Dict[str, float]
    rambling_loopiness_cap: float
    rambling_plateau_k: int
    debug_out_dir: Optional[Path]
    base_debug: bool


class FilterService:
    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        # pattern workers
        global PATTERN_WORKERS
        cpu_total = os.cpu_count() or 1
        if args.pattern_workers <= 0:
            PATTERN_WORKERS = max(1, min(cpu_total, 32))
        else:
            PATTERN_WORKERS = max(1, min(int(args.pattern_workers), cpu_total))
        print(f"[Server] pattern_workers = {PATTERN_WORKERS} (available CPUs = {cpu_total})")

        # metric switches
        self.metric_switches = resolve_metric_dependencies()

        debug_out_dir = Path(args.debug_out_dir) if args.debug_out_dir else None
        if debug_out_dir is not None:
            debug_out_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = ServerConfig(
            correct_key=args.correct_key,
            min_lang_chars=args.min_lang_chars,
            cpu_procs=int(args.cpu_procs),
            chunk_lines=int(args.chunk_lines),
            dup_threshold=args.dup_threshold,
            cluster_threshold=args.cluster_threshold,
            plateau_min_len=args.plateau_min_len,
            loop_eps=args.loop_eps,
            max_sent_per_sample=args.max_sent_per_sample,
            rambling_weights={
                "self": args.rambling_weight_self,
                "dup": args.rambling_weight_dup,
                "loop": args.rambling_weight_loop,
                "plateau": args.rambling_weight_plateau,
                "lang": args.rambling_weight_lang,
            },
            rambling_loopiness_cap=args.rambling_loopiness_cap,
            rambling_plateau_k=args.rambling_plateau_k,
            debug_out_dir=debug_out_dir,
            base_debug=bool(args.debug),
        )

        # embedder
        self.embedder: Optional[BaseEmbedder] = None
        if self.metric_switches.enable_embedding:
            if args.embed_backend == "vllm":
                print("[Server] Using vLLM backend.")
                self.embedder = VllmEmbedder(
                    model_name=args.embed_model_name,
                    tensor_parallel_size=args.tensor_parallel_size,
                    dtype=args.dtype,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    trust_remote_code=args.trust_remote_code,
                    enforce_eager=args.enforce_eager,
                )
            else:
                print("[Server] Using HF backend (single device). hf_max_workers 参数在在线模式下暂不启用。")
                self.embedder = HFEmbedder(
                    model_name=args.embed_model_name,
                    device_str=args.hf_device,
                    batch_size=args.hf_batch_size,
                    max_length=args.hf_max_length,
                    torch_dtype_str=args.hf_torch_dtype,
                    trust_remote_code=args.trust_remote_code,
                )
        else:
            print("[Server] Embedding is disabled by metric switches; only text metrics + markers pattern will工作。")

    # ---- 核心：对单个 JSONL 文件做过滤 ----

    def filter_jsonl(self, req: FilterRequest) -> FilterResponse:
        jsonl_path = Path(req.jsonl_path)
        if not jsonl_path.exists():
            raise HTTPException(status_code=400, detail=f"jsonl_path not found: {jsonl_path}")

        t_total0 = time.time()

        # BuildConfig，主要用于 Stage0/Stage1
        bcfg = BuildConfig(
            jsonl=jsonl_path,
            correct_key=self.cfg.correct_key,
            min_lang_chars=self.cfg.min_lang_chars,
            cpu_procs=self.cfg.cpu_procs,
            chunk_lines=self.cfg.chunk_lines,
            compute_sha256=False,
            step_min=None,
            step_max=None,
            debug=self.cfg.base_debug or req.debug,
            enable_text_metrics=self.metric_switches.enable_text_metrics,
            enable_contains_chinese=self.metric_switches.enable_contains_chinese,
            enable_split_sections_and_sentences=self.metric_switches.enable_split_sections_and_sentences,
            enable_detect_markers=self.metric_switches.enable_detect_markers,
            enable_detect_lang=self.metric_switches.enable_detect_lang,
        )

        # Stage0: scan offsets，不做 kept/cls 子集过滤，全部交给 target_mask
        print(f"[Server][Scan] {jsonl_path}")
        t0 = time.time()
        chunks, sha256, fsize, mtime, n_lines, scan_stats = scan_offsets(
            bcfg,
            kept_or_dropped="all",
            cls_filter="any",
        )
        print(
            f"[Server][Scan] total_lines={scan_stats['num_total']}, "
            f"selected={scan_stats['num_selected']} chunks={len(chunks)} "
            f"time={time.time()-t0:.2f}s"
        )

        if scan_stats["num_selected"] == 0:
            return FilterResponse(
                ok=True,
                message="no samples",
                num_samples=0,
                num_bad=0,
                keep_mask=[],
                bad_mask=[],
                pattern_counts={},
            )

        # Stage1: CPU 特征
        print(
            f"[Server][Stage1] cpu_procs={bcfg.cpu_procs}, "
            f"metric_switches={self.metric_switches}"
        )
        tasks = [(str(bcfg.jsonl), offs, bcfg) for offs in chunks]
        start_method = "spawn" if sys.platform == "win32" else "fork"
        ctx = mp.get_context(start_method)

        samples_list: List[pd.DataFrame] = []
        segments_list: List[pd.DataFrame] = []

        with ctx.Pool(processes=bcfg.cpu_procs) as pool:
            for ret in pool.imap_unordered(_worker_entry_stage1, tasks, chunksize=1):
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

        print(
            f"[Server][Stage1] samples={len(samples_df)}, segments={len(segments_df)}, "
            f"time={time.time()-t0:.2f}s"
        )

        if samples_df.empty:
            return FilterResponse(
                ok=True,
                message="no samples after Stage1",
                num_samples=0,
                num_bad=0,
                keep_mask=[],
                bad_mask=[],
                pattern_counts={},
            )

        # 为 pattern_redundant_answer_noise 附上 response_text（如果没有）
        if "response_text" not in samples_df.columns:
            print("[Server] Attaching response_text from JSONL ...")
            t_txt0 = time.time()
            resp_list: List[str] = []
            with bcfg.jsonl.open("rb", buffering=1024 * 1024) as fsrc:
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
            print(f"[Server] response_text attached, took {time.time()-t_txt0:.2f}s")

        # Stage2: embedding（使用预加载好的 embedder）
        if self.metric_switches.enable_embedding and not segments_df.empty and self.embedder is not None:
            print(f"[Server][Stage2] Embedding {len(segments_df)} sentences ...")
            t2 = time.time()
            texts = segments_df["text"].astype(str).tolist()
            embs = self.embedder.embed(texts)
            if len(embs) != len(segments_df):
                raise RuntimeError(
                    f"[Server][Stage2] len(embs)={len(embs)} mismatch segments={len(segments_df)}"
                )
            segments_df = segments_df.copy()
            segments_df["emb"] = embs
            print(f"[Server][Stage2] done in {time.time()-t2:.2f}s")
        else:
            print("[Server][Stage2] Embedding disabled or no segments; skip.")

        # Stage3: embedding-based metrics + RamblingScore
        samples_df = run_embedding_analysis_for_all_samples(
            samples_df,
            segments_df,
            dup_threshold=self.cfg.dup_threshold,
            cluster_threshold=self.cfg.cluster_threshold,
            plateau_min_len=self.cfg.plateau_min_len,
            loop_eps=self.cfg.loop_eps,
            max_sent_per_sample=self.cfg.max_sent_per_sample,
            rambling_weights=self.cfg.rambling_weights,
            rambling_loopiness_cap=self.cfg.rambling_loopiness_cap,
            rambling_plateau_k=self.cfg.rambling_plateau_k,
            enable_embedding_metrics=self.metric_switches.enable_embedding_metrics,
            enable_rambling_score=self.metric_switches.enable_rambling_score,
        )

        # Pattern filter：只在指定 cls 上应用 pattern，其它样本一律视为 keep。
        if req.cls == "any":
            target_mask = None
        elif req.cls == "correct":
            target_mask = (samples_df["cls_id"].astype(int) == 1).to_numpy()
        elif req.cls == "wrong":
            target_mask = (samples_df["cls_id"].astype(int) == 0).to_numpy()
        elif req.cls == "no_judge":
            target_mask = (samples_df["cls_id"].astype(int) == 2).to_numpy()
        else:
            raise HTTPException(status_code=400, detail=f"invalid cls='{req.cls}'")

        print(f"[Server][Filter] cls={req.cls}, target_mask has "
              f"{int(target_mask.sum()) if target_mask is not None else len(samples_df)} candidates.")

        keep_mask_df, bad_mask_df, pattern_hits = build_filter_mask_by_patterns(
            samples_df,
            target_mask=target_mask,
        )

        # 把 df 索引对齐回 sample_idx（原始行顺序）
        max_idx = int(samples_df["sample_idx"].max())
        N = max_idx + 1
        bad_by_idx = np.zeros(N, dtype=bool)

        # DataFrame 的 index 顺序未必是 sample_idx 顺序，所以用 sample_idx 做映射
        for row_idx, row in samples_df.iterrows():
            sid = int(row["sample_idx"])
            if bad_mask_df[row_idx]:
                bad_by_idx[sid] = True

        keep_by_idx = (~bad_by_idx).astype(int).tolist()
        bad_by_idx_int = bad_by_idx.astype(int).tolist()
        num_bad = int(bad_by_idx.sum())

        # pattern 统计
        pattern_counts: Dict[str, int] = {}
        for name, mask in pattern_hits.items():
            pattern_counts[name] = int(mask.sum())

        # 可选写 debug jsonl（good/bad）
        if self.cfg.debug_out_dir is not None and (self.cfg.base_debug or req.debug):
            self._write_debug_jsonl(
                cfg=bcfg,
                samples_df=samples_df,
                bad_mask_df=bad_mask_df,
                pattern_hits=pattern_hits,
            )

        msg = (
            f"filtered {N} samples, bad={num_bad}, "
            f"time_total={time.time()-t_total0:.2f}s"
        )
        print(f"[Server] {msg}")

        return FilterResponse(
            ok=True,
            message=msg,
            num_samples=N,
            num_bad=num_bad,
            keep_mask=keep_by_idx,
            bad_mask=bad_by_idx_int,
            pattern_counts=pattern_counts,
        )

    # ---- debug 输出：写 good/bad_samples.jsonl，结构与 offline 版本兼容 ----

    def _write_debug_jsonl(
        self,
        cfg: BuildConfig,
        samples_df: pd.DataFrame,
        bad_mask_df: np.ndarray,
        pattern_hits: Dict[str, np.ndarray],
    ) -> None:
        out_dir = self.cfg.debug_out_dir
        if out_dir is None:
            return

        base = cfg.jsonl.stem
        out_good = out_dir / f"{base}.good_samples.jsonl"
        out_bad = out_dir / f"{base}.bad_samples.jsonl"
        print(f"[Server][Debug] Writing good -> {out_good}")
        print(f"[Server][Debug] Writing bad  -> {out_bad}")

        pattern_names = [name for (name, _) in PATTERN_FUNCS]
        hit_patterns_per_row: List[List[str]] = [[] for _ in range(len(samples_df))]
        for pname in pattern_names:
            if pname not in pattern_hits:
                continue
            mask_p = pattern_hits[pname]
            for i, flag in enumerate(mask_p):
                if flag:
                    hit_patterns_per_row[i].append(pname)

        enabled = set(ENABLED_METRICS)

        def build_metrics_for_row(row: pd.Series) -> Dict[str, Any]:
            m: Dict[str, Any] = {}
            for col in FILTER_RELATED_COLUMNS:
                if col not in enabled:
                    continue
                if col in row:
                    val = row[col]
                    if isinstance(val, (np.generic,)):
                        val = val.item()
                    m[col] = val
            return m

        with cfg.jsonl.open("rb", buffering=1024 * 1024) as fsrc, \
             out_good.open("w", encoding="utf-8") as f_good, \
             out_bad.open("w", encoding="utf-8") as f_bad:
            fileno = fsrc.fileno()
            for global_idx, row in samples_df.iterrows():
                off = int(row["byte_offset"])
                ln = int(row["byte_len"])
                b = os.pread(fileno, ln, off)
                try:
                    j = _json_loads(b)
                except Exception:
                    j = {}
                row_bad = bool(bad_mask_df[global_idx])
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
                if self.cfg.base_debug:
                    debug_info["debug_note"] = "online filter debug info"

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


# ==================== 启动入口 ====================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)

    # 与 offline filter 对齐的参数
    ap.add_argument("--cpu-procs", type=int, default=max(1, (os.cpu_count() or 8)//2))
    ap.add_argument("--chunk-lines", type=int, default=20000)
    ap.add_argument("--correct-key", type=str, default="reward_extra_infos.acc")
    ap.add_argument("--min-lang-chars", type=int, default=5)

    # embedding backend
    ap.add_argument(
        "--embed-backend",
        type=str,
        default="hf",
        choices=["vllm", "hf"],
        help="embedding backend for online service",
    )
    ap.add_argument("--embed-model-name", type=str, default="BAAI/bge-m3")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")

    # HF backend
    ap.add_argument("--hf-device", type=str, default="cuda")
    ap.add_argument("--hf-max-workers", type=int, default=1,
                    help="在线版本目前只用 1 worker，参数占位。")
    ap.add_argument("--hf-batch-size", type=int, default=128)
    ap.add_argument("--hf-max-length", type=int, default=1024)
    ap.add_argument(
        "--hf-torch-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
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
    ap.add_argument("--debug", action="store_true", help="开启 server 端详细日志")
    ap.add_argument("--debug-out-dir", type=str, default=None,
                    help="debug 模式下 good/bad_samples.jsonl 输出目录")
    return ap


def main():
    parser = build_argparser()
    args = parser.parse_args()

    service = FilterService(args)

    app = FastAPI()

    @app.post("/filter", response_model=FilterResponse)
    def filter_endpoint(req: FilterRequest):
        return service.filter_jsonl(req)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
