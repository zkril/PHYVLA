#!/bin/bash
# 批量导出10个任务的baseline VLA动作

set -e

PROJECT_ROOT="/0604-10T-test/jiangnan/PHYVLA"
cd $PROJECT_ROOT

echo "======================================================================"
echo "批量导出Baseline VLA动作（10个任务）"
echo "======================================================================"

for i in {0..9}; do
    echo ""
    echo "[Task $i/9] 开始导出..."

    conda run -n starVLA python scripts/export_vla_actions.py \
        --suite libero_spatial \
        --task-id $i \
        --demo-index 0

    if [ $? -eq 0 ]; then
        echo "✅ Task $i 完成"
    else
        echo "❌ Task $i 失败"
        exit 1
    fi
done

echo ""
echo "======================================================================"
echo "✅ 所有10个任务的baseline动作导出完成"
echo "======================================================================"
echo ""

# 验证输出
echo "验证输出文件:"
ls outputs/libero_spatial/task_*/vla_actions_demo0.npz 2>/dev/null | wc -l
echo "个文件已生成"
