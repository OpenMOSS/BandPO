#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check an OpenAI-compatible API (e.g., vLLM) is alive and can run a tiny chat completion.

Example:
  python3 check_openai_api.py --model DeepSeek-R1-0528-Qwen3-8B --host 10.176.58.103 --port 8000
  python3 check_openai_api.py --model DeepSeek-R1-0528-Qwen3-8B --host 10.176.58.103 --port 8000 --base-path /api/openai/v1
"""

from __future__ import annotations
import argparse, json, sys, time, ssl, urllib.request, urllib.error
from typing import Any, Dict, Optional

EXIT_OK = 0
EXIT_CONN = 2
EXIT_TIMEOUT = 3
EXIT_HTTP = 4
EXIT_DEP = 5

def _join_base(root: str, path: str) -> str:
    root = root.rstrip("/")
    path = "/" + path.strip("/")
    return root + path

def _http_get_json(url: str, timeout: float, insecure: bool=False) -> Dict[str, Any]:
    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8", errors="replace")) if data else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} on GET {url}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"GET {url} failed: {e}") from e

def _check_health(root_base: str, timeout: float, insecure: bool) -> str:
    url = _join_base(root_base, "/health")
    try:
        data = _http_get_json(url, timeout, insecure)
        return f"/health ok: {data}" if data else "/health ok: (empty body)"
    except Exception as e:
        return f"/health not available: {e}"

def _list_models(api_base: str, timeout: float, insecure: bool) -> Dict[str, Any]:
    return _http_get_json(_join_base(api_base, "/models"), timeout, insecure)

def _chat_via_openai_sdk(api_base: str, api_key: str, model: str,
                         prompt: str, timeout: float, max_tokens: int) -> Dict[str, Any]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ImportError("openai SDK not available") from e
    client = OpenAI(base_url=api_base, api_key=api_key)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
        stream=False,
        timeout=timeout,
    )
    dt = int(round((time.time() - t0) * 1000))
    content = (resp.choices[0].message.content or "").strip()
    return {"latency_ms": dt, "id": resp.id, "content": content}

def _chat_via_requests(api_base: str, api_key: str, model: str,
                       prompt: str, timeout: float, max_tokens: int, insecure: bool) -> Dict[str, Any]:
    try:
        import requests  # type: ignore
    except Exception as e:
        raise ImportError("requests not available") from e
    url = _join_base(api_base, "/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.2, "max_tokens": max_tokens, "stream": False}
    t0 = time.time()
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout, verify=not insecure)
    except requests.exceptions.ConnectTimeout as e:
        raise TimeoutError(str(e)) from e
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(str(e)) from e
    if r.status_code // 100 != 2:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    data = r.json()
    dt = int(round((time.time() - t0) * 1000))
    content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    return {"latency_ms": dt, "id": data.get("id", ""), "content": content}

def main() -> int:
    ap = argparse.ArgumentParser(description="OpenAI-compatible API checker")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--scheme", choices=["http","https"], default="http")
    ap.add_argument("--base-path", default="/v1", help="API base path (default: /v1)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--prompt", default="Reply with a single word: OK")
    ap.add_argument("--insecure", action="store_true")
    args = ap.parse_args()

    root_base = f"{args.scheme}://{args.host}:{args.port}"
    api_base = _join_base(root_base, args.base_path)
    api_key = args.api_key or "sk-no-key-needed"

    print(f"==> Root base: {root_base}")
    print(f"==> API  base: {api_base}")
    print(f"==> Model: {args.model}")

    # health (best-effort)
    print(f"==> Health: {_check_health(root_base, args.timeout, args.insecure)}")

    # /v1/models (or custom base path)
    try:
        models = _list_models(api_base, args.timeout, args.insecure)
        names = [m.get("id") for m in models.get("data", []) if isinstance(m, dict)]
        if names:
            print(f"==> Models visible ({len(names)}): {'; '.join(names[:8])}{'...' if len(names)>8 else ''}")
            print(f"==> Target model present: {args.model in names}")
        else:
            print("==> WARNING: models list empty or unexpected.")
    except ConnectionError as e:
        print(f"!! Connection error on models: {e}", file=sys.stderr); return EXIT_CONN
    except TimeoutError as e:
        print(f"!! Timeout on models: {e}", file=sys.stderr); return EXIT_TIMEOUT
    except RuntimeError as e:
        print(f"!! HTTP error on models: {e}", file=sys.stderr); return EXIT_HTTP

    # tiny chat completion
    last_err: Optional[Exception] = None
    for attempt in range(1, args.retries + 2):
        try:
            try:
                result = _chat_via_openai_sdk(api_base, api_key, args.model, args.prompt, args.timeout, args.max_tokens)
            except ImportError:
                result = _chat_via_requests(api_base, api_key, args.model, args.prompt, args.timeout, args.max_tokens, args.insecure)
            print(f"==> Chat ok (latency {result['latency_ms']} ms, id={result['id']})")
            preview = (result.get("content") or "").replace("\n", " ")[:120]
            print(f"==> Model reply: {preview}")
            return EXIT_OK
        except (ConnectionError, TimeoutError, RuntimeError, ImportError, Exception) as e:
            last_err = e
            et = type(e).__name__
            print(f"!! {et} (attempt {attempt}): {e}", file=sys.stderr)
            time.sleep(0.8 * attempt)

    if last_err:
        return {ConnectionError: EXIT_CONN, TimeoutError: EXIT_TIMEOUT,
                RuntimeError: EXIT_HTTP, ImportError: EXIT_DEP}.get(type(last_err), EXIT_DEP)
    return EXIT_DEP

if __name__ == "__main__":
    sys.exit(main())
