#!/bin/bash

# ===========================================
# Python 脚本启动器 - 带时间戳日志保存
# ===========================================

# 配置区域 - 根据需要修改这些变量
PYTHON_SCRIPT="call.py"           # Python 文件路径
LOG_DIR="/remote-home1/yli/Workspace/BandPO/data/logs/tinker"                          # 日志保存目录
CONDA_ENV="tinker"                        # Conda 环境名称
PYTHON_CMD="python"                       # Python 解释器命令（激活环境后使用 python）
SHOW_OUTPUT=true                          # 是否在终端显示输出（true/false）

# 初始化 Conda（根据你的系统配置选择合适的方式）
# 方式1：使用 conda.sh（最常见）
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    # 方式2：使用 conda init hook
    eval "$(conda shell.bash hook)" 2>/dev/null
fi

# 激活 Conda 环境
echo "正在激活 Conda 环境: $CONDA_ENV"
conda activate "$CONDA_ENV"

# 检查环境是否激活成功
if [ $? -ne 0 ]; then
    echo "错误: 无法激活 Conda 环境 '$CONDA_ENV'"
    exit 1
fi
echo "Conda 环境已激活: $CONDA_ENV"
echo ""

# 生成时间戳（格式：YYYY-MM-DD_HH-MM-SS）
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 生成日志文件名
LOG_FILE="$LOG_DIR/${TIMESTAMP}.log"

# 将 Python 源码写入日志文件开头
echo "================================================================================" > "$LOG_FILE"
echo "Python 源代码: $PYTHON_SCRIPT" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"
cat "$PYTHON_SCRIPT" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"
echo "程序输出开始" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 记录开始时间（秒级时间戳）
START_TIME=$(date +%s)

# 打印启动信息
echo "================================================"
echo "启动 Python 脚本: $PYTHON_SCRIPT"
echo "日志文件: $LOG_FILE"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
echo ""

# 运行 Python 脚本，根据配置决定是否在终端显示输出
if [ "$SHOW_OUTPUT" = true ]; then
    # 使用 tee -a 命令：同时在终端显示并追加到日志文件
    $PYTHON_CMD "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
    # 捕获 Python 脚本的退出状态
    EXIT_CODE=${PIPESTATUS[0]}
else
    # 只追加到日志文件，不在终端显示
    echo "（输出已重定向到日志文件，终端不显示）"
    $PYTHON_CMD "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1
    # 捕获 Python 脚本的退出状态
    EXIT_CODE=$?
fi

# 记录结束时间并计算耗时
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# 将秒数转换为时分秒格式
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

# 将耗时信息追加到日志文件末尾
echo "" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"
echo "程序执行统计信息" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"
echo "退出码: $EXIT_CODE" >> "$LOG_FILE"
echo "开始时间: $(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r $START_TIME '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "程序耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒 (总计 ${ELAPSED_TIME} 秒)" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"

# 打印结束信息
echo ""
echo "================================================"
echo "脚本执行完成"
echo "退出码: $EXIT_CODE"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "程序耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒 (总计 ${ELAPSED_TIME} 秒)"
echo "日志已保存到: $LOG_FILE"
echo "================================================"

# 返回 Python 脚本的退出码
exit $EXIT_CODE