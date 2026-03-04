#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
W&B Offline Sync Server

功能：
- HTTP 接口：添加/查看/删除要同步的 offline runs（路径）
- 后台按间隔执行：wandb sync <path>
- 添加时校验路径是否存在

用法：
  python server.py --host 127.0.0.1 --port 8765 --default-interval-minutes 15

客户端示例（bash）：
  curl -sS -X POST http://127.0.0.1:8765/add \
    -H 'Content-Type: application/json' \
    -d '{"paths":["/abs/path/to/wandb/offline-run-xxxx"],"interval_minutes":15}'
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple


@dataclass
class Job:
    path: str
    interval_seconds: int
    next_run_epoch: float
    last_run_epoch: Optional[float] = None
    last_returncode: Optional[int] = None
    last_stdout: str = ""
    last_stderr: str = ""
    running: bool = False


class SyncManager:
    def __init__(
        self,
        default_interval_seconds: int = 15 * 60,
        tick_seconds: int = 2,
        wandb_timeout_seconds: int = 30 * 60,
    ) -> None:
        self.default_interval_seconds = default_interval_seconds
        self.tick_seconds = tick_seconds
        self.wandb_timeout_seconds = wandb_timeout_seconds

        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def add_paths(self, paths: List[str], interval_seconds: Optional[int] = None) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        返回：(added_paths, rejected[(path, reason)])
        """
        if not isinstance(paths, list) or not paths:
            return [], [("", "paths must be a non-empty list")]

        interval = interval_seconds if interval_seconds is not None else self.default_interval_seconds
        if not isinstance(interval, int) or interval <= 0:
            return [], [("", "interval_seconds must be a positive integer")]

        added: List[str] = []
        rejected: List[Tuple[str, str]] = []

        now = time.time()

        with self._lock:
            for p in paths:
                if not isinstance(p, str) or not p.strip():
                    rejected.append((str(p), "path must be a non-empty string"))
                    continue

                abs_path = os.path.abspath(os.path.expanduser(p.strip()))
                if not os.path.exists(abs_path):
                    rejected.append((abs_path, "path does not exist"))
                    continue

                # 如果已存在：更新 interval，并让它尽快执行一次（next_run=now）
                if abs_path in self._jobs:
                    job = self._jobs[abs_path]
                    job.interval_seconds = interval
                    job.next_run_epoch = now
                    added.append(abs_path)
                    continue

                self._jobs[abs_path] = Job(
                    path=abs_path,
                    interval_seconds=interval,
                    next_run_epoch=now,  # 添加后尽快执行第一次 sync
                )
                added.append(abs_path)

        return added, rejected

    def remove_path(self, path: str) -> bool:
        abs_path = os.path.abspath(os.path.expanduser(path.strip()))
        with self._lock:
            return self._jobs.pop(abs_path, None) is not None

    def list_jobs(self) -> List[Dict]:
        with self._lock:
            return [asdict(job) for job in self._jobs.values()]

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._run_due_jobs()
            self._stop_event.wait(self.tick_seconds)

    def _run_due_jobs(self) -> None:
        now = time.time()
        due_paths: List[str] = []

        with self._lock:
            for path, job in self._jobs.items():
                if job.running:
                    continue
                if job.next_run_epoch <= now:
                    job.running = True
                    due_paths.append(path)

        # 逐个执行，避免并发过多导致 IO/网络拥塞；如需并发可改为线程池
        for path in due_paths:
            self._run_one(path)

    def _run_one(self, path: str) -> None:
        start = time.time()
        stdout = ""
        stderr = ""
        rc: Optional[int] = None

        # 再次确认路径存在（运行期间可能被删除/移动）
        if not os.path.exists(path):
            stderr = "path disappeared before sync"
            rc = 2
        else:
            try:
                proc = subprocess.run(
                    ["wandb", "sync", path],
                    capture_output=True,
                    text=True,
                    timeout=self.wandb_timeout_seconds,
                )
                rc = proc.returncode
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
            except subprocess.TimeoutExpired as e:
                rc = 124
                stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
                stderr = "wandb sync timeout"
            except Exception as e:
                rc = 1
                stderr = f"wandb sync failed: {e!r}"

        end = time.time()

        with self._lock:
            job = self._jobs.get(path)
            if job is None:
                # 可能在运行中被 remove 了
                return
            job.last_run_epoch = end
            job.last_returncode = rc
            job.last_stdout = stdout[-20000:]  # 防止内存无限增长，保留尾部
            job.last_stderr = stderr[-20000:]
            job.next_run_epoch = start + job.interval_seconds
            job.running = False


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        return None, "invalid Content-Length"

    if length <= 0:
        return None, "empty body"

    raw = handler.rfile.read(length)
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as e:
        return None, f"invalid json: {e!r}"

    if not isinstance(obj, dict):
        return None, "json body must be an object"
    return obj, None


class RequestHandler(BaseHTTPRequestHandler):
    # 由 main 注入
    manager: SyncManager = None  # type: ignore

    def log_message(self, fmt: str, *args) -> None:
        # 如不想要任何日志，可直接 return
        super().log_message(fmt, *args)

    def do_GET(self) -> None:
        if self.path == "/health":
            _json_response(self, 200, {"ok": True})
            return

        if self.path == "/jobs":
            _json_response(self, 200, {"jobs": self.manager.list_jobs()})
            return

        _json_response(self, 404, {"error": "not found", "paths": ["/health", "/jobs", "/add", "/remove"]})

    def do_POST(self) -> None:
        if self.path == "/add":
            body, err = _read_json(self)
            if err:
                _json_response(self, 400, {"error": err})
                return

            paths = body.get("paths")
            # 可选：interval_minutes 或 interval_seconds
            interval_minutes = body.get("interval_minutes", None)
            interval_seconds = body.get("interval_seconds", None)

            interval: Optional[int] = None
            if interval_seconds is not None:
                if not isinstance(interval_seconds, int) or interval_seconds <= 0:
                    _json_response(self, 400, {"error": "interval_seconds must be a positive integer"})
                    return
                interval = interval_seconds
            elif interval_minutes is not None:
                if not isinstance(interval_minutes, int) or interval_minutes <= 0:
                    _json_response(self, 400, {"error": "interval_minutes must be a positive integer"})
                    return
                interval = interval_minutes * 60

            if not isinstance(paths, list):
                _json_response(self, 400, {"error": "paths must be a list of strings"})
                return

            added, rejected = self.manager.add_paths(paths=paths, interval_seconds=interval)
            _json_response(self, 200, {"added": added, "rejected": [{"path": p, "reason": r} for p, r in rejected]})
            return

        if self.path == "/remove":
            body, err = _read_json(self)
            if err:
                _json_response(self, 400, {"error": err})
                return
            p = body.get("path")
            if not isinstance(p, str) or not p.strip():
                _json_response(self, 400, {"error": "path must be a non-empty string"})
                return
            ok = self.manager.remove_path(p)
            _json_response(self, 200, {"removed": ok, "path": os.path.abspath(os.path.expanduser(p.strip()))})
            return

        _json_response(self, 404, {"error": "not found"})


def main() -> None:
    parser = argparse.ArgumentParser(description="W&B Offline Sync Server")
    parser.add_argument("--host", default="127.0.0.1", help="listen host, default 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="listen port, default 8765")
    parser.add_argument("--default-interval-minutes", type=int, default=15, help="default sync interval in minutes")
    parser.add_argument("--tick-seconds", type=int, default=2, help="scheduler tick seconds")
    parser.add_argument("--wandb-timeout-minutes", type=int, default=30, help="timeout for a single wandb sync")
    args = parser.parse_args()

    if shutil.which("wandb") is None:
        raise SystemExit("ERROR: 'wandb' not found in PATH. Please install wandb and ensure CLI is available.")

    manager = SyncManager(
        default_interval_seconds=args.default_interval_minutes * 60,
        tick_seconds=max(1, args.tick_seconds),
        wandb_timeout_seconds=max(60, args.wandb_timeout_minutes * 60),
    )
    manager.start()

    RequestHandler.manager = manager
    httpd = ThreadingHTTPServer((args.host, args.port), RequestHandler)

    try:
        print(f"Listening on http://{args.host}:{args.port}")
        print("Endpoints: GET /health, GET /jobs, POST /add, POST /remove")
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        manager.stop()


if __name__ == "__main__":
    main()
