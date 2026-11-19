#!/usr/bin/env python3
"""验证所有任务的数据对齐情况"""

import numpy as np
from pathlib import Path

def verify_alignment():
    print("=" * 60)
    print("数据对齐验证 - 完整检查")
    print("=" * 60)
    print()

    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

    all_aligned = True
    total_tasks = 0
    aligned_tasks = 0

    for suite in suites:
        print(f"[{suite}]")
        suite_aligned = 0

        for task_id in range(10):
            tok_path = Path(f"outputs/{suite}/task_{task_id}/physics_tokens_demo0.npz")
            vla_path = Path(f"outputs/{suite}/task_{task_id}/vla_actions_demo0.npz")

            total_tasks += 1

            if tok_path.exists() and vla_path.exists():
                tok = np.load(str(tok_path))
                vla = np.load(str(vla_path))

                T_tok = tok['tokens'].shape[0]
                T_vla = vla['a_vla_norm'].shape[0]

                aligned = (T_tok == T_vla)
                status = "✅" if aligned else "❌"

                print(f"  task_{task_id}: {status} T={T_tok}")

                if aligned:
                    aligned_tasks += 1
                    suite_aligned += 1
                else:
                    all_aligned = False
                    print(f"    ⚠️  不对齐: T_tok={T_tok}, T_vla={T_vla}")
            else:
                print(f"  task_{task_id}: ❌ 文件缺失")
                all_aligned = False

        print(f"  小计: {suite_aligned}/10")
        print()

    print("=" * 60)
    print(f"总计: {aligned_tasks}/{total_tasks} 任务对齐")
    if all_aligned:
        print("✅ 所有数据对齐正确！可以开始训练。")
    else:
        print("❌ 发现数据对齐问题，需要修复。")
    print("=" * 60)

    return all_aligned

if __name__ == "__main__":
    verify_alignment()
