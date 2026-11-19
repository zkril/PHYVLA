#!/bin/bash
# 训练40任务Physics Projector
# 位置: scripts/train_40tasks_projector.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 配置
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
GPU="${GPU:-0,1}"
DEMO_INDEX="${DEMO_INDEX:-0}"

STARVLA_CKPT="thirdparty/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
LOG_FILE="logs/train_p5_40tasks_$(date +%Y%m%d_%H%M%S).log"

mkdir -p logs

echo "========================================="
echo "训练40任务Physics Projector"
echo "========================================="
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "GPU: $GPU"
echo "Log: $LOG_FILE"
echo ""

# 检查数据完整性
echo "检查数据完整性..."
python scripts/check_40tasks_data.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 数据不完整，请先运行："
    echo "   bash scripts/process_all_40tasks.sh"
    exit 1
fi

echo ""
echo "开始训练..."
echo ""

# 训练命令
# 使用 --suite-mode multi 和 --task-ids 0-39
CUDA_VISIBLE_DEVICES=$GPU conda run -n starVLA python scripts/train_p5_projector.py \
    --suite-mode multi \
    --task-ids $(seq -s ' ' 0 39) \
    --demo-index $DEMO_INDEX \
    --ckpt "$STARVLA_CKPT" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --device cuda \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "✅ 训练完成"
    echo "========================================="
    echo "Checkpoint: outputs/p5_checkpoints/projector_40tasks_best.pt"
    echo "Summary: outputs/p5_checkpoints/p5_40tasks_summary.json"
    echo "Log: $LOG_FILE"
    echo ""
    echo "下一步: LIBERO-plus评估"
    echo "  BENCHMARK=LIBERO-plus bash scripts/run_baseline_client.sh"
else
    echo "========================================="
    echo "❌ 训练失败"
    echo "========================================="
    echo "查看日志: $LOG_FILE"
    exit $EXIT_CODE
fi
