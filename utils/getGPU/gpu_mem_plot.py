#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # 纯 SSH / 无 GUI 环境必须用这个后端
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def read_csv(csv_path):
    """
    返回两份数据:
    1. per_gpu: 每张 GPU 的时序数据
    2. per_time: 每个时间点上所有 GPU 的聚合数据，用来算平均线
    """
    per_gpu = defaultdict(lambda: {
        "name": None,
        "times": [],
        "percent": [],
        "used": [],
        "total": [],
    })

    per_time = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gpu_idx = int(row["gpu_index"])
            gpu_name = row["gpu_name"]
            ts = datetime.fromisoformat(row["timestamp"])
            used = float(row["memory_used_mib"])
            total = float(row["memory_total_mib"])
            percent = float(row["memory_percent"])

            per_gpu[gpu_idx]["name"] = gpu_name
            per_gpu[gpu_idx]["times"].append(ts)
            per_gpu[gpu_idx]["percent"].append(percent)
            per_gpu[gpu_idx]["used"].append(used)
            per_gpu[gpu_idx]["total"].append(total)

            per_time[ts].append({
                "gpu_index": gpu_idx,
                "percent": percent,
                "used": used,
                "total": total,
            })

    return per_gpu, per_time


def build_avg_series(per_time, metric="percent"):
    """
    基于每个时间点的所有 GPU 数据，生成平均线
    """
    avg_times = []
    avg_values = []

    for ts in sorted(per_time.keys()):
        items = per_time[ts]
        if not items:
            continue

        if metric == "percent":
            value = sum(x["percent"] for x in items) / len(items)
        elif metric == "used":
            value = sum(x["used"] for x in items) / len(items)
        else:
            raise ValueError(f"unsupported metric: {metric}")

        avg_times.append(ts)
        avg_values.append(value)

    return avg_times, avg_values


def main():
    parser = argparse.ArgumentParser(description="根据 GPU 显存采样 CSV 生成折线图")
    parser.add_argument("-i", "--input", type=str, default="gpu_mem.csv", help="输入 CSV 文件")
    parser.add_argument("-o", "--output", type=str, default="gpu_mem_percent.png", help="输出 PNG 文件")
    parser.add_argument(
        "--metric",
        choices=["percent", "used"],
        default="percent",
        help="percent=画使用率，used=画已用显存(MiB)"
    )
    args = parser.parse_args()

    per_gpu, per_time = read_csv(args.input)
    if not per_gpu:
        raise RuntimeError("CSV 里没有有效数据")

    plt.figure(figsize=(14, 7))

    # 先画每张 GPU 的线
    for gpu_idx in sorted(per_gpu.keys()):
        item = per_gpu[gpu_idx]
        x = item["times"]

        if args.metric == "percent":
            y = item["percent"]
            ylabel = "Memory Usage (%)"
            title = "GPU Memory Usage Over Time"
        else:
            y = item["used"]
            ylabel = "Memory Used (MiB)"
            title = "GPU Memory Used Over Time"

        plt.plot(x, y, label=f"GPU {gpu_idx}", alpha=0.8)

    # 再画平均线
    avg_times, avg_values = build_avg_series(per_time, metric=args.metric)
    if avg_times:
        plt.plot(
            avg_times,
            avg_values,
            label="AVG",
            linewidth=2.5,
            linestyle="--"
        )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=30)

    if args.metric == "percent":
        plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"图已保存到: {args.output}")


if __name__ == "__main__":
    main()
