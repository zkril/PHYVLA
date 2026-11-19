#!/usr/bin/env python3
"""
检查40个LIBERO任务的数据完整性
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 4个suites，每个10个任务
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def check_task(suite: str, task_id: int) -> dict:
    """检查单个任务的数据状态"""
    out_dir = PROJECT_ROOT / "outputs" / suite / f"task_{task_id}"

    status = {
        "suite": suite,
        "task_id": task_id,
        "physics_tokens": False,
        "vla_actions": False,
        "vlm_hidden": False,
        "aligned": False,
        "T_physics": None,
        "T_vla": None,
        "T_vlm": None,
    }

    # 检查physics tokens
    physics_file = out_dir / "physics_tokens_demo0.npz"
    if physics_file.exists():
        status["physics_tokens"] = True
        tok = np.load(physics_file)
        status["T_physics"] = tok["tokens"].shape[0]

    # 检查VLA actions
    vla_file = out_dir / "vla_actions_demo0.npz"
    if vla_file.exists():
        status["vla_actions"] = True
        vla = np.load(vla_file)
        status["T_vla"] = vla["a_vla_norm"].shape[0]

    # 检查VLM hidden states（可选，训练时会自动生成）
    vlm_file = out_dir / "vlm_hidden_demo0.npz"
    if vlm_file.exists():
        status["vlm_hidden"] = True
        vlm = np.load(vlm_file)
        status["T_vlm"] = vlm["h"].shape[0]

    # 检查对齐
    if status["physics_tokens"] and status["vla_actions"]:
        status["aligned"] = (status["T_physics"] == status["T_vla"])

    return status


def main():
    print("=" * 70)
    print("LIBERO 40任务数据状态")
    print("=" * 70)
    print()

    all_status = []
    complete_count = 0
    total_tasks = 40

    for suite in SUITES:
        print(f"\n{suite.upper()}")
        print("-" * 70)

        suite_complete = 0

        for task_id in range(10):
            status = check_task(suite, task_id)
            all_status.append(status)

            # 判断是否完整
            is_complete = status["physics_tokens"] and status["vla_actions"] and status["aligned"]

            if is_complete:
                complete_count += 1
                suite_complete += 1
                icon = "✓"
                color = "\033[92m"  # Green
            else:
                icon = "✗"
                color = "\033[91m"  # Red

            reset = "\033[0m"

            # 显示状态
            phy_str = f"T={status['T_physics']}" if status["physics_tokens"] else "missing"
            vla_str = f"T={status['T_vla']}" if status["vla_actions"] else "missing"
            align_str = "✓" if status["aligned"] else "✗"

            print(f"  {color}{icon}{reset} task_{task_id:02d}: "
                  f"physics={phy_str:8s}  vla={vla_str:8s}  aligned={align_str}")

        print(f"  Suite完成度: {suite_complete}/10")

    # 总结
    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print(f"完整任务数: {complete_count}/{total_tasks} ({100*complete_count/total_tasks:.1f}%)")
    print()

    # 分析缺失
    missing_physics = [s for s in all_status if not s["physics_tokens"]]
    missing_vla = [s for s in all_status if not s["vla_actions"]]
    misaligned = [s for s in all_status if s["physics_tokens"] and s["vla_actions"] and not s["aligned"]]

    if missing_physics:
        print(f"缺少physics tokens: {len(missing_physics)}个任务")
        for s in missing_physics[:5]:
            print(f"  - {s['suite']}/task_{s['task_id']}")
        if len(missing_physics) > 5:
            print(f"  ... 和其他{len(missing_physics)-5}个任务")
        print()

    if missing_vla:
        print(f"缺少VLA actions: {len(missing_vla)}个任务")
        for s in missing_vla[:5]:
            print(f"  - {s['suite']}/task_{s['task_id']}")
        if len(missing_vla) > 5:
            print(f"  ... 和其他{len(missing_vla)-5}个任务")
        print()

    if misaligned:
        print(f"数据未对齐: {len(misaligned)}个任务")
        for s in misaligned[:5]:
            print(f"  - {s['suite']}/task_{s['task_id']}: physics={s['T_physics']}, vla={s['T_vla']}")
        if len(misaligned) > 5:
            print(f"  ... 和其他{len(misaligned)-5}个任务")
        print()

    # 建议
    print("=" * 70)
    if complete_count == total_tasks:
        print("✅ 所有任务数据完整，可以开始训练")
        print()
        print("下一步:")
        print("  bash scripts/train_40tasks_projector.sh")
    else:
        print("❌ 数据不完整，需要处理缺失的任务")
        print()
        print("下一步:")
        print("  bash scripts/process_all_40tasks.sh")
    print("=" * 70)

    # 返回状态码
    return 0 if complete_count == total_tasks else 1


if __name__ == "__main__":
    sys.exit(main())
