#!/bin/bash
# 批量导出缺失的VLA actions（30个任务）
# 位置: scripts/batch_export_missing_vla.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

STARVLA_CKPT="${PROJECT_ROOT}/thirdparty/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
GPU="${GPU:-0}"

echo "========================================="
echo "批量导出缺失的VLA actions"
echo "========================================="
echo "GPU: $GPU"
echo "Checkpoint: $STARVLA_CKPT"
echo ""

# 需要处理的任务
declare -a TASKS=(
    "libero_object:0-9"
    "libero_goal:0-9"
    "libero_10:0-9"
)

total=30
current=0

for task_spec in "${TASKS[@]}"; do
    IFS=':' read -r suite range <<< "$task_spec"
    IFS='-' read -r start end <<< "$range"

    for i in $(seq $start $end); do
        current=$((current + 1))

        echo ""
        echo "[$current/$total] $suite/task_$i"
        echo "----------------------------------------"

        # 检查是否需要处理
        vla_file="outputs/$suite/task_$i/vla_actions_demo0.npz"
        if [ -f "$vla_file" ]; then
            echo "  ⊙ VLA actions已存在，跳过"
            continue
        fi

        # 导出VLA actions
        echo "  → 导出VLA actions..."
        CUDA_VISIBLE_DEVICES=$GPU conda run -n starVLA python scripts/export_vla_actions.py \
            --suite "$suite" \
            --task-ids $i \
            --demo-index 0 \
            --ckpt "$STARVLA_CKPT" \
            --device cuda \
            2>&1 | grep -E "(处理|导出|保存|✓|✗|Error)" || true

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "  ✓ 导出成功"

            # 验证对齐
            python -c "
import numpy as np
from pathlib import Path

out_dir = Path('outputs/$suite/task_$i')
tok = np.load(out_dir / 'physics_tokens_demo0.npz')
vla = np.load(out_dir / 'vla_actions_demo0.npz')

T_tok = tok['tokens'].shape[0]
T_vla = vla['a_vla_norm'].shape[0]

if T_tok == T_vla:
    print(f'  ✓ 对齐检查通过: T={T_tok}')
else:
    print(f'  ✗ 对齐失败: physics={T_tok}, vla={T_vla}')
    exit(1)
" 2>&1 | grep -E "(✓|✗)"
        else
            echo "  ✗ 导出失败"
        fi
    done
done

echo ""
echo "========================================="
echo "批处理完成"
echo "========================================="
echo ""
echo "验证最终状态:"
echo "  python scripts/check_40tasks_data.py"
