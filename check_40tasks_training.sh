#!/bin/bash
# 监控40任务训练进度

echo "=== 40任务训练状态 ==="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. 检查训练进程
echo "训练进程:"
TRAIN_PID=$(ps aux | grep "train_p5_projector.py.*suite-mode multi" | grep -v grep | awk '{print $2}')
if [ -n "$TRAIN_PID" ]; then
    ps aux | grep "train_p5_projector.py.*suite-mode multi" | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "运行时间:", $10}'
    echo "  状态: 运行中"
else
    echo "  状态: 未运行"
fi
echo ""

# 2. GPU使用情况
echo "GPU 0,1 使用情况:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -2 | while read line; do
    echo "  GPU $line"
done
echo ""

# 3. 最新日志
LATEST_LOG=$(ls -t logs/train_p5_40tasks_*.log 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    LOG_SIZE=$(stat -c%s "$LATEST_LOG" 2>/dev/null || stat -f%z "$LATEST_LOG" 2>/dev/null)
    echo "最新日志: $LATEST_LOG ($(numfmt --to=iec $LOG_SIZE 2>/dev/null || echo ${LOG_SIZE}))"

    if [ "$LOG_SIZE" -gt 1000 ]; then
        echo ""
        echo "最近训练进度:"
        tail -30 "$LATEST_LOG" | grep -E "epoch|loss|VLM hidden|数据集|训练|验收" | tail -10
    else
        echo "  (日志文件较小，可能还在初始化)"
    fi
fi

echo ""
echo "---"
echo "持续监控: watch -n 5 ./check_40tasks_training.sh"
