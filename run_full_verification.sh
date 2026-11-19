#!/bin/bash
# 完整的物理Token验证流程
# 10个libero_spatial任务的验证

set -e  # 遇到错误立即退出

PROJECT_ROOT="/0604-10T-test/jiangnan/PHYVLA"
CHECKPOINT="thirdparty/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"

cd $PROJECT_ROOT

echo "======================================================================"
echo "物理Token效果验证 - 完整流程"
echo "======================================================================"
echo "任务: libero_spatial (0-9)"
echo "数据路径: outputs/libero_spatial/task_{0-9}/"
echo "======================================================================"
echo ""

# ============================================================================
# 阶段1: 导出Baseline VLA动作（无物理token）
# ============================================================================
echo ""
echo "======================================================================"
echo "阶段1/3: 导出Baseline VLA动作（预计20-30分钟）"
echo "======================================================================"
echo ""

for i in {0..9}; do
    echo "[Task $i/9] 正在导出baseline VLA动作..."
    python scripts/export_vla_actions.py \
        --suite libero_spatial \
        --task-id $i \
        --demo-index 0 \
        --checkpoint $CHECKPOINT

    if [ $? -ne 0 ]; then
        echo "❌ Task $i 导出失败"
        exit 1
    fi
    echo "✅ Task $i 完成"
    echo ""
done

echo "======================================================================"
echo "✅ 阶段1完成: 所有baseline动作已导出"
echo "======================================================================"
echo ""

# 验证输出
echo "验证输出文件:"
ls outputs/libero_spatial/task_*/vla_actions_demo0.npz | wc -l
echo "个vla_actions文件已生成"
echo ""

# ============================================================================
# 阶段2: 训练PhysicsTokenProjector
# ============================================================================
echo ""
echo "======================================================================"
echo "阶段2/3: 训练PhysicsTokenProjector（预计1-2小时）"
echo "======================================================================"
echo ""

# 选项A: 预缓存VLM hidden states（推荐）
read -p "是否预缓存VLM hidden states？这会节省训练时间但需要额外存储空间 (y/n): " cache_vlm

if [ "$cache_vlm" = "y" ] || [ "$cache_vlm" = "Y" ]; then
    echo ""
    echo "[预缓存] 为10个任务生成VLM hidden states..."
    for i in {0..9}; do
        echo "  缓存 Task $i..."
        python scripts/train_p5_projector.py \
            --suite libero_spatial \
            --task-ids $i \
            --demo-index 0 \
            --ckpt $CHECKPOINT \
            --cache-only \
            --device cuda
    done
    echo "✅ VLM hidden states缓存完成"
    USE_CACHED="--use-cached-vlm"
else
    USE_CACHED=""
fi

# 训练projector
echo ""
echo "[训练] 开始训练PhysicsTokenProjector（10任务联合）..."
echo "配置: epochs=200, batch_size=8, lr=1e-4"
echo ""

python scripts/train_p5_projector.py \
    --suite libero_spatial \
    --task-ids 0 1 2 3 4 5 6 7 8 9 \
    --demo-index 0 \
    --ckpt $CHECKPOINT \
    --epochs 200 \
    --batch-size 8 \
    --lr 1e-4 \
    --device cuda \
    $USE_CACHED

if [ $? -ne 0 ]; then
    echo "❌ 训练失败"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ 阶段2完成: PhysicsTokenProjector训练完成"
echo "======================================================================"
echo ""

# ============================================================================
# 阶段3: 查看结果和验证
# ============================================================================
echo ""
echo "======================================================================"
echo "阶段3/3: 验证结果分析"
echo "======================================================================"
echo ""

echo "检查训练输出:"
if [ -f "outputs/p5_checkpoints/projector_best.pt" ]; then
    echo "✅ Checkpoint: outputs/p5_checkpoints/projector_best.pt"
    ls -lh outputs/p5_checkpoints/projector_best.pt
else
    echo "❌ 未找到训练checkpoint"
fi

if [ -f "outputs/p5_checkpoints/train.log" ]; then
    echo ""
    echo "✅ 训练日志: outputs/p5_checkpoints/train.log"
    echo ""
    echo "最后20行日志:"
    echo "----------------------------------------------------------------------"
    tail -20 outputs/p5_checkpoints/train.log
    echo "----------------------------------------------------------------------"
    echo ""

    # 检查物理敏感性测试结果
    if grep -q "物理敏感性" outputs/p5_checkpoints/train.log; then
        echo ""
        echo "物理敏感性测试结果:"
        echo "----------------------------------------------------------------------"
        grep -A 5 "物理敏感性" outputs/p5_checkpoints/train.log
        echo "----------------------------------------------------------------------"
    fi
else
    echo "❌ 未找到训练日志"
fi

echo ""
echo "======================================================================"
echo "🎉 验证流程全部完成！"
echo "======================================================================"
echo ""
echo "下一步:"
echo "  1. 查看完整训练日志: cat outputs/p5_checkpoints/train.log"
echo "  2. 分析训练曲线（loss是否下降）"
echo "  3. 检查物理敏感性测试是否通过"
echo ""
echo "验证成功的标志:"
echo "  ✅ 训练loss持续下降并收敛"
echo "  ✅ 物理敏感性测试: 修改token后输出变化 > 1e-3"
echo "  ✅ 无NaN或异常值"
echo ""
