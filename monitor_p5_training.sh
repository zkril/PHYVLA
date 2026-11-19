#!/bin/bash
# 监控P5训练状态

echo "========================================="
echo "P5 Training Monitor"
echo "时间: $(date)"
echo "========================================="
echo ""

# 1. 检查训练进程
echo "训练进程:"
ps aux | grep "train_p5_projector.py" | grep -v grep | awk '{print "PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}'
echo ""

# 2. GPU状态
echo "GPU状态 (0,1):"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -2
echo ""

# 3. 最新日志
latest_log=$(ls -t logs/train_p5_*.log 2>/dev/null | head -1)
if [ -f "$latest_log" ]; then
    log_size=$(stat -c%s "$latest_log" 2>/dev/null)
    echo "最新日志: $latest_log"
    echo "大小: $log_size bytes"

    if [ "$log_size" -gt 0 ]; then
        echo ""
        echo "最后30行:"
        tail -30 "$latest_log"
    else
        echo "(日志文件为空，训练可能正在初始化...)"
    fi
else
    echo "未找到日志文件"
fi

echo ""
echo "========================================="
