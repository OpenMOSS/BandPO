#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime


def query_gpu_memory():
    """
    返回:
    [
        {
            "index": 0,
            "name": "NVIDIA H200",
            "used_mib": 12092.0,
            "total_mib": 143771.0,
            "percent": 8.41,
        },
        ...
    ]
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    gpus = []
    for line in result.stdout.strip().splitlines():
        # 例如:
        # 0, NVIDIA H200, 12092, 143771
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue

        idx = int(parts[0])
        name = parts[1]
        used = float(parts[2])
        total = float(parts[3])
        percent = (used / total * 100.0) if total > 0 else 0.0

        gpus.append({
            "index": idx,
            "name": name,
            "used_mib": used,
            "total_mib": total,
            "percent": percent,
        })

    gpus.sort(key=lambda x: x["index"])
    return gpus


def clear_screen():
    # 兼容 SSH 终端
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def print_snapshot(gpus, sample_id, interval, mode):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_percent = sum(g["percent"] for g in gpus) / len(gpus)
    max_gpu = max(gpus, key=lambda x: x["percent"])

    if mode == "refresh":
        clear_screen()

    print(f"[{now_str}] sample={sample_id} interval={interval:.2f}s")
    print(f"avg_mem={avg_percent:.2f}% | max_gpu=GPU {max_gpu['index']} ({max_gpu['percent']:.2f}%)")
    print("-" * 90)

    for g in gpus:
        print(
            f"GPU {g['index']:>2} | "
            f"{g['name']:<20} | "
            f"{g['used_mib']:>9.0f} / {g['total_mib']:<9.0f} MiB | "
            f"{g['percent']:>6.2f}%"
        )

    print("-" * 90)
    print("Ctrl+C 停止采样")
    sys.stdout.flush()


def ensure_csv_header(csv_path):
    need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    f = open(csv_path, "a", newline="")
    writer = csv.writer(f)

    if need_header:
        writer.writerow([
            "unix_time",
            "timestamp",
            "gpu_index",
            "gpu_name",
            "memory_used_mib",
            "memory_total_mib",
            "memory_percent",
        ])
        f.flush()

    return f, writer


def main():
    parser = argparse.ArgumentParser(description="实时采集 GPU 显存并保存到 CSV")
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="采样间隔，单位秒，默认 1")
    parser.add_argument("-o", "--output", type=str, default="gpu_mem.csv", help="输出 CSV 文件")
    parser.add_argument(
        "--mode",
        choices=["refresh", "append"],
        default="refresh",
        help="终端展示模式：refresh=单屏刷新，append=持续追加日志"
    )
    args = parser.parse_args()

    csv_file, csv_writer = ensure_csv_header(args.output)

    sample_id = 0
    try:
        while True:
            t0 = time.time()
            ts = datetime.now()
            unix_ts = t0

            try:
                gpus = query_gpu_memory()
            except subprocess.CalledProcessError as e:
                err = e.stderr.strip() if e.stderr else str(e)
                print(f"[ERROR] nvidia-smi 调用失败: {err}", file=sys.stderr)
                time.sleep(args.interval)
                continue

            if not gpus:
                print("[WARN] 没有拿到 GPU 数据", file=sys.stderr)
                time.sleep(args.interval)
                continue

            timestamp_str = ts.isoformat(timespec="seconds")

            for g in gpus:
                csv_writer.writerow([
                    f"{unix_ts:.3f}",
                    timestamp_str,
                    g["index"],
                    g["name"],
                    f"{g['used_mib']:.0f}",
                    f"{g['total_mib']:.0f}",
                    f"{g['percent']:.4f}",
                ])

            csv_file.flush()
            print_snapshot(gpus, sample_id, args.interval, args.mode)

            sample_id += 1

            elapsed = time.time() - t0
            sleep_time = max(0.0, args.interval - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n已停止采样。")
        print(f"CSV 已保存到: {args.output}")
    finally:
        csv_file.close()


if __name__ == "__main__":
    main()