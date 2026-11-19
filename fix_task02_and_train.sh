#!/bin/bash
# 修复libero_10/task_2数据对齐问题并重新训练

set -e

echo "========================================="
echo "修复 libero_10/task_2 数据对齐问题"
echo "========================================="
echo ""

# 1. 检查当前状态
echo "1. 检查当前数据状态..."
python3 << 'EOF'
import numpy as np
from pathlib import Path

suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
total_z = 0
total_a = 0
mismatched = []

for suite in suites:
    for task_id in range(10):
        tok_path = Path(f"outputs/{suite}/task_{task_id}/physics_tokens_demo0.npz")
        vla_path = Path(f"outputs/{suite}/task_{task_id}/vla_actions_demo0.npz")

        if tok_path.exists() and vla_path.exists():
            tok = np.load(str(tok_path))
            vla = np.load(str(vla_path))

            T_z = tok['tokens'].shape[0]
            T_a = vla['a_vla_norm'].shape[0]

            total_z += T_z
            total_a += T_a

            if T_z != T_a:
                mismatched.append(f"{suite}/task_{task_id}: z={T_z}, a={T_a}")
                print(f"❌ {suite}/task_{task_id}: physics_tokens={T_z}, vla_actions={T_a}")
            else:
                print(f"✓ {suite}/task_{task_id}: {T_z} frames")

print(f"\n总计: z_phy={total_z}, a_vla={total_a}")
if mismatched:
    print(f"\n不匹配的任务 ({len(mismatched)}):")
    for m in mismatched:
        print(f"  {m}")
else:
    print("\n✅ 所有任务数据对齐！")
EOF

echo ""
echo "2. 重新导出 libero_10/task_2 的 VLA actions..."
CUDA_VISIBLE_DEVICES=1 conda run -n starVLA python scripts/export_vla_actions.py \
    --suite libero_10 \
    --task-ids 2 \
    --demo-index 0 \
    --ckpt /0604-10T-test/jiangnan/PHYVLA/thirdparty/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt \
    --device cuda

echo ""
echo "3. 验证修复结果..."
python3 << 'EOF'
import numpy as np
from pathlib import Path

tok_path = Path("outputs/libero_10/task_2/physics_tokens_demo0.npz")
vla_path = Path("outputs/libero_10/task_2/vla_actions_demo0.npz")

tok = np.load(str(tok_path))
vla = np.load(str(vla_path))

T_z = tok['tokens'].shape[0]
T_a = vla['a_vla_norm'].shape[0]

print(f"libero_10/task_2: physics_tokens={T_z}, vla_actions={T_a}")
if T_z == T_a:
    print("✅ 修复成功！")
else:
    print(f"❌ 仍然不匹配！")
    exit(1)
EOF

echo ""
echo "4. 启动40任务训练..."
CUDA_VISIBLE_DEVICES=0,1 nohup conda run -n starVLA python scripts/train_p5_projector.py \
    --suite-mode multi \
    --task-ids $(seq 0 39) \
    --demo-index 0 \
    --epochs 200 \
    --batch-size 8 \
    --lr 1e-4 \
    --device cuda \
    > logs/train_p5_40tasks_$(date +%Y%m%d_%H%M%S).log 2>&1 &

TRAIN_PID=$!
echo "训练已启动，PID: $TRAIN_PID"
echo ""
echo "监控命令: ./check_40tasks_training.sh"
echo "日志文件: logs/train_p5_40tasks_*.log"
