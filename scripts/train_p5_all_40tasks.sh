#!/bin/bash
# 训练P5 Projector - 全部40个LIBERO任务
# 位置: scripts/train_p5_all_40tasks.sh

set -euo pipefail

PHYVLA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PHYVLA_ROOT}"

CKPT="/0604-10T-test/jiangnan/PHYVLA/thirdparty/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
EPOCHS=200
BATCH_SIZE=16
LR=1e-4
GPU_DEVICES="0,1"

echo "========================================="
echo "P5 Projector训练 - 全部40任务"
echo "========================================="
echo "Checkpoint: ${CKPT}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "GPUs: ${GPU_DEVICES}"
echo ""

# 构建任务ID列表
TASK_IDS=""
for suite in spatial object goal 10; do
    for tid in {0..9}; do
        # 计算全局task ID
        case ${suite} in
            spatial) offset=0 ;;
            object)  offset=10 ;;
            goal)    offset=20 ;;
            10)      offset=30 ;;
        esac
        global_id=$((offset + tid))
        TASK_IDS="${TASK_IDS} ${global_id}"
    done
done

# 生成时间戳
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="logs/train_p5_40tasks_${timestamp}.log"
mkdir -p logs

echo "Task IDs: ${TASK_IDS}"
echo "Log: ${log_file}"
echo ""

# 启动训练
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} conda run -n starVLA python scripts/train_p5_projector.py \
    --suite-mode multi \
    --task-ids ${TASK_IDS} \
    --demo-index 0 \
    --ckpt "${CKPT}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --device cuda 2>&1 | tee "${log_file}"

echo ""
echo "========================================="
echo "训练完成"
echo "========================================="
echo "Checkpoint保存: outputs/p5_checkpoints/projector_40tasks_best.pt"
echo "摘要: outputs/p5_checkpoints/p5_40tasks_summary.json"
