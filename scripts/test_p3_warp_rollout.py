"""P3 mujoco_warp 端到端测试：完整 LIBERO 模型真实物理 rollout。

运行（phyvla-p2 环境）：
    python scripts/test_p3_warp_rollout.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FULL_XML   = PROJECT_ROOT / "outputs/task02/model_raw.xml"
TRACE_NPZ  = PROJECT_ROOT / "outputs/task02/controller_trace_v1.npz"
PHYSICS_NPZ= PROJECT_ROOT / "outputs/physics/task02_demo00_contact_force_data.npz"

K = 8
H = 30
START_T = 40   # 抓取阶段（碗已被夹持，ctrl 影响搬运方向）


def make_candidates(ctrl, t, H, K):
    base = ctrl[t:t+H]
    if len(base) < H:
        base = np.vstack([base, np.zeros((H-len(base), ctrl.shape[1]))])
    np.random.seed(42)
    cands = [
        base.copy(),
        base * 0.5,
        base * 1.5,
        base + np.random.randn(*base.shape) * 2,
        base + np.random.randn(*base.shape) * 5,
        base + np.random.randn(*base.shape) * 10,
        np.zeros_like(base),
        -base,
    ]
    return np.stack(cands[:K], axis=0).astype(np.float32)


def main():
    for p in (FULL_XML, TRACE_NPZ, PHYSICS_NPZ):
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")

    trace   = np.load(str(TRACE_NPZ), allow_pickle=False)
    physics = np.load(str(PHYSICS_NPZ), allow_pickle=False)
    states  = trace["states"]
    ctrl    = trace["ctrl"]

    import mujoco
    tmp = mujoco.MjModel.from_xml_path(str(FULL_XML))
    nq, nv = tmp.nq, tmp.nv

    qpos0 = np.concatenate([
        trace["states"][START_T, 0:7],
        trace["states"][START_T, 7:9],
        physics["bowl_position"][START_T],
        physics["bowl_quat_xyzw"][START_T][[3,0,1,2]],
        physics["plate_position"][START_T],
        physics["plate_quat_xyzw"][START_T][[3,0,1,2]],
    ]).astype(np.float64)
    # 填充剩余的 qpos（其他物体保持默认）
    qpos_full = states[START_T, :nq].copy()
    qpos_full[:len(qpos0)] = qpos0
    qvel0 = np.zeros(nv, dtype=np.float64)
    plate_pos = physics["plate_position"][START_T]

    print("="*60)
    print("P3 mujoco_warp 端到端测试（完整 LIBERO 模型）")
    print("="*60)
    print(f"模型: {FULL_XML.name}  nbody={tmp.nbody}  ngeom={tmp.ngeom}")
    print(f"初始帧: {START_T}  K={K}  H={H}")
    print(f"目标: plate_pos = {plate_pos.round(3)}")

    cand_ctrls = make_candidates(ctrl, START_T, H, K)
    print(f"\n候选 ctrl shape: {cand_ctrls.shape}")

    from simulation.mjwarp_rollout import MJWarpRollout
    rollout = MJWarpRollout(
        str(FULL_XML),
        target_body="akita_black_bowl_1_main",
        goal_body="plate_1_main",
    )
    rollout.reset(qpos_full, qvel0)

    # 预热首步（触发 CUDA 内核编译）
    print("\n预热首步（首次运行会编译 CUDA 内核，约141s）...")
    t0 = time.time()
    _ = rollout.rollout_single(cand_ctrls[0])
    print(f"  首步完成: {time.time()-t0:.1f}s")

    # K候选批量 rollout
    print(f"\nK={K} 候选批量 rollout...")
    t0 = time.time()
    result = rollout.rollout_batch(cand_ctrls)
    dt = time.time()-t0
    print(f"  完成: {dt:.1f}s  shape={result['xpos'].shape}")

    # 物理代价评分
    scores = rollout.physics_cost(result["xpos"], plate_pos)
    names = ["专家","×0.5","×1.5","σ=2","σ=5","σ=10","零动作","反向"][:K]

    print("\n"+"="*60)
    print("候选评分（代价越小越好）")
    print("="*60)
    best = int(np.argmin(scores))
    for i,(n,s) in enumerate(zip(names,scores)):
        marker = " ← 最优" if i==best else ""
        print(f"  [{i}] {n:<8s}  cost={s:.4f}{marker}")

    # 验收
    print("\n"+"="*60)
    print("P3 mujoco_warp 验收")
    print("="*60)
    checks = [
        ("完整模型 ngeom>100",            tmp.ngeom > 100),
        ("batch rollout 成功",            result["xpos"].shape==(K,H,tmp.nbody,3)),
        ("无 NaN",                        not np.any(np.isnan(result["xpos"]))),
        ("无 Inf",                        not np.any(np.isinf(result["xpos"]))),
        ("不同候选有差异",                float(np.std(scores)) > 1e-6),
        ("专家代价合理",                  scores[0] < np.mean(scores)*1.5),
    ]
    all_pass = True
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")
        if not ok: all_pass = False
    print()
    print("结果:", "✅ ALL PASS" if all_pass else "❌ 部分失败")
    print(f"物理模型: 完整 LIBERO ({tmp.ngeom} geom)  vs 之前极简 (22 geom)")


if __name__ == "__main__":
    np.random.seed(42)
    main()
