#!/bin/bash
ray stop            # 停止本机所有 Ray 进程
ray stop --force || true
pkill -f ray       # 如果还有残留也可以直接杀掉