#!/bin/bash
# 检查P5训练状态

echo "=== P5 Training Status ==="
echo "Time: $(date)"
echo ""

# 检查训练进程
echo "Training Process:"
ps aux | grep "train_p5_projector.py" | grep -v grep | awk '{print "PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}'
echo ""

# 检查GPU使用
echo "GPU Usage (0,1):"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -2
echo ""

# 检查最新日志文件
latest_log=$(ls -t logs/train_p5_*.log 2>/dev/null | head -1)
if [ -f "$latest_log" ]; then
    log_size=$(stat -f%z "$latest_log" 2>/dev/null || stat -c%s "$latest_log" 2>/dev/null)
    echo "Latest log: $latest_log (size: $log_size bytes)"

    if [ "$log_size" -gt 0 ]; then
        echo ""
        echo "Last 30 lines:"
        tail -30 "$latest_log"
    else
        echo "Log file is empty (output may be buffered)"
    fi
fi

# 检查VLM缓存状态
echo ""
echo "VLM Hidden States Cache:"
for i in {0..9}; do
    cache_file="outputs/libero_spatial/task_${i}/vlm_hidden_demo0.npz"
    if [ -f "$cache_file" ]; then
        echo "  task${i}: ✓ ($(stat -f%z "$cache_file" 2>/dev/null || stat -c%s "$cache_file" 2>/dev/null | numfmt --to=iec) bytes)"
    else
        echo "  task${i}: pending..."
    fi
done
