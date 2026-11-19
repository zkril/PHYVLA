#!/bin/bash
# P5 Phase 1: 批量导出 VLA 动作预测
# 用法: bash scripts/batch_export_vla.sh [start_id] [end_id]

set -e

START=${1:-0}
END=${2:-9}
SUITE="libero_spatial"

echo "========================================"
echo "  P5-1: 批量导出 VLA 动作"
echo "  任务范围: task $START - $END"
echo "  Suite: $SUITE"
echo "========================================"
echo ""

cd /0604-10T-test/jiangnan/PHYVLA

# 检查环境
if [[ "$CONDA_DEFAULT_ENV" != "starVLA" ]]; then
    echo "❌ 错误: 必须在 starVLA 环境中运行"
    echo "   请执行: conda activate starVLA"
    exit 1
fi

success=0
failed=0

for i in $(seq $START $END); do
    echo "--------------------"
    echo "📦 Task $i"
    echo "--------------------"

    if python scripts/export_vla_actions.py \
        --suite "$SUITE" \
        --task-id "$i" \
        --demo-index 0; then

        # 验证输出
        out_file="outputs/task$(printf '%02d' $i)/vla_actions_demo0.npz"
        if [ -f "$out_file" ]; then
            size=$(stat -c%s "$out_file" 2>/dev/null || stat -f%z "$out_file")
            echo "✅ task $i 完成 ($(numfmt --to=iec $size 2>/dev/null || echo ${size}B))"
            ((success++))
        else
            echo "⚠️  task $i: 输出文件未生成"
            ((failed++))
        fi
    else
        echo "❌ task $i 失败"
        ((failed++))
    fi
    echo ""
done

echo "========================================"
echo "  完成统计"
echo "  成功: $success / $((END - START + 1))"
echo "  失败: $failed"
echo "========================================"
