#!/bin/bash
# P5 Demo: 使用前 10 个任务训练 PhysicsTokenProjector

set -e

cd /0604-10T-test/jiangnan/PHYVLA

echo "=========================================="
echo "P5 Demo: 10 任务训练流程"
echo "=========================================="
echo ""

# Phase 1: 导出 VLA 动作
echo "[1/3] P5-0: 导出 StarVLA 基准动作..."
echo "预计时间: 20-30 分钟"
echo ""

for task_id in {0..9}; do
    OUTPUT_FILE="outputs/task$(printf '%02d' $task_id)/vla_actions_demo0.npz"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "  Task $task_id: 已存在，跳过"
        continue
    fi

    echo "  Task $task_id: 导出中..."
    python scripts/export_vla_actions.py \
        --suite libero_spatial \
        --task-id $task_id \
        --demo-index 0 \
        --output "$OUTPUT_FILE" \
        > logs/export_vla_task$(printf '%02d' $task_id).log 2>&1

    if [ $? -eq 0 ]; then
        echo "    ✓ 完成"
    else
        echo "    ✗ 失败 (详见 logs/export_vla_task$(printf '%02d' $task_id).log)"
    fi
done

echo ""
echo "[1/3] P5-0 完成"
echo ""

# Phase 2: 预缓存 VLM 隐状态
echo "[2/3] P5-1: 预缓存 VLM 隐状态..."
echo "预计时间: 10-15 分钟"
echo ""

for task_id in {0..9}; do
    OUTPUT_FILE="outputs/task$(printf '%02d' $task_id)/vlm_hidden_demo0.npz"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "  Task $task_id: 已存在，跳过"
        continue
    fi

    echo "  Task $task_id: 缓存中..."
    python scripts/train_p5_projector.py \
        --suite libero_spatial \
        --task-id $task_id \
        --demo-index 0 \
        --cache-only \
        --device cuda \
        > logs/cache_vlm_task$(printf '%02d' $task_id).log 2>&1

    if [ $? -eq 0 ]; then
        echo "    ✓ 完成"
    else
        echo "    ✗ 失败 (详见 logs/cache_vlm_task$(printf '%02d' $task_id).log)"
    fi
done

echo ""
echo "[2/3] P5-1 完成"
echo ""

# Phase 3: 训练 Projector
echo "[3/3] P5-3: 训练 PhysicsTokenProjector..."
echo "预计时间: 1-2 小时"
echo ""

python scripts/train_p5_projector.py \
    --task-ids 0 1 2 3 4 5 6 7 8 9 \
    --suite libero_spatial \
    --epochs 300 \
    --lr 1e-4 \
    --batch-size 8 \
    --use-cached-vlm \
    --device cuda \
    --checkpoint-dir outputs/p5_checkpoints_10tasks \
    --save-every 50 \
    --eval-every 10 \
    --log-file outputs/p5_checkpoints_10tasks/train.log

echo ""
echo "=========================================="
echo "✅ P5 Demo 训练完成！"
echo "=========================================="
echo ""
echo "权重位置: outputs/p5_checkpoints_10tasks/projector_best.pt"
echo "训练日志: outputs/p5_checkpoints_10tasks/train.log"
