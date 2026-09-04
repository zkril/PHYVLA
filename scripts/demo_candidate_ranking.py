"""P3-3 候选动作重排序可视化 Demo。

流程：
  1. 加载完整 LIBERO MJCF（271geom）+ 专家状态 + ctrl
  2. 生成 K 条候选（专家 + 扰动）
  3. mujoco_warp 并行 rollout（真实物理精度）
  4. 物理代价评分 + 排序
  5. 渲染最优/最差候选轨迹对比视频

运行（phyvla-p2 环境）：
    python scripts/demo_candidate_ranking.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_XML   = PROJECT_ROOT / "outputs/task02/model_raw.xml"   # 完整 LIBERO MJCF（271geom）
TRACE_NPZ   = PROJECT_ROOT / "outputs/task02/controller_trace_v1.npz"
PHYSICS_NPZ = PROJECT_ROOT / "outputs/physics/task02_demo00_contact_force_data.npz"
OUTPUT_DIR  = PROJECT_ROOT / "outputs/task02"

K        = 8
H        = 30
START_T  = 20    # frame 20（已有抓取）
HEIGHT   = 256
WIDTH    = 256


# ─────────────────────────────────────────────
# 候选生成
# ─────────────────────────────────────────────

def make_candidates(ctrl: np.ndarray, t: int, H: int, K: int) -> np.ndarray:
    base = ctrl[t : t + H]
    if len(base) < H:
        base = np.vstack([base, np.zeros((H - len(base), ctrl.shape[1]))])
    np.random.seed(42)
    cands = [
        base.copy(),                          # 0: 专家
        base * 0.5,                           # 1: 幅值×0.5
        base * 1.5,                           # 2: 幅值×1.5
        base + np.random.randn(*base.shape) * 2,   # 3: σ=2
        base + np.random.randn(*base.shape) * 5,   # 4: σ=5
        base + np.random.randn(*base.shape) * 10,  # 5: σ=10
        np.zeros_like(base),                  # 6: 全零（无动作）
        -base,                                # 7: 反向动作
    ]
    return np.stack(cands[:K], axis=0).astype(np.float32)


# ─────────────────────────────────────────────
# 渲染轨迹
# ─────────────────────────────────────────────

def render_qpos_sequence(
    xml_path: str,
    qpos_seq: np.ndarray,   # [T, nq]
    height: int = HEIGHT,
    width:  int = WIDTH,
) -> np.ndarray:
    import mujoco
    model    = mujoco.MjModel.from_xml_path(xml_path)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames   = []
    for qpos in qpos_seq:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frames.append(renderer.render().copy())
    renderer.close()
    return np.stack(frames)   # [T, H, W, 3]


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main() -> None:
    import cv2

    trace   = np.load(str(TRACE_NPZ),   allow_pickle=False)
    physics = np.load(str(PHYSICS_NPZ), allow_pickle=False)
    ctrl    = trace["ctrl"]

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    nq, nv = model.nq, model.nv

    # 构建初始状态
    qpos0 = np.concatenate([
        trace["states"][START_T, 0:7],
        trace["states"][START_T, 7:9],
        physics["bowl_position"][START_T],
        physics["bowl_quat_xyzw"][START_T][[3, 0, 1, 2]],
        physics["plate_position"][START_T],
        physics["plate_quat_xyzw"][START_T][[3, 0, 1, 2]],
    ]).astype(np.float64)
    qvel0 = np.zeros(nv, dtype=np.float64)
    goal_pos = physics["plate_position"][START_T]

    print("=" * 60)
    print("P3-3 候选动作重排序 Demo")
    print("=" * 60)
    print(f"初始帧: {START_T}  K={K}  H={H}")
    print(f"目标位置: plate = {goal_pos.round(3)}")

    # 候选 ctrl
    cand_ctrls = make_candidates(ctrl, START_T, H, K)
    names = ["专家ctrl", "幅值×0.5", "幅值×1.5", "σ=2扰动",
             "σ=5扰动", "σ=10扰动", "全零动作", "反向动作"][:K]

    # MJX rollout
    from simulation.mjwarp_rollout import MJWarpRollout as MJXRollout
    rollout = MJXRollout(str(MODEL_XML),
                         target_body="akita_black_bowl_1_main",
                         goal_body="plate_1_main")
    rollout.reset(qpos0, qvel0)

    print("\n运行批量 rollout...")
    t0 = time.time()
    result = rollout.rollout_batch(cand_ctrls)
    print(f"完成: {time.time()-t0:.1f}s  xpos:{result['xpos'].shape}")

    # 物理代价评分
    target_id, goal_id = rollout.target_id, rollout.goal_id
    xpos_trajs = result["xpos"]     # [K, H, nbody, 3]
    qpos_trajs = result["qpos"]     # [K, H, nq]

    scores = []
    for k in range(K):
        tp = xpos_trajs[k, :, target_id]     # [H, 3]
        gp = xpos_trajs[k, :, goal_id]       # [H, 3]
        dist = np.linalg.norm(tp - gp, axis=-1)
        delta = tp[1:] - tp[:-1]
        cost = float(dist.mean()) + 0.1 * float(np.mean(delta**2))
        scores.append(cost)
    scores = np.array(scores)

    ranked = np.argsort(scores)
    best_idx  = int(ranked[0])
    worst_idx = int(ranked[-1])

    # 打印排名
    print("\n" + "=" * 60)
    print("候选评分排名（代价越小越好）")
    print("=" * 60)
    for rank, k in enumerate(ranked):
        marker = " ← 最优" if k == best_idx else " ← 最差" if k == worst_idx else ""
        print(f"  #{rank+1}  [{k}] {names[k]:<14s}  cost={scores[k]:.4f}{marker}")

    # 渲染最优 vs 最差 vs 专家 对比视频
    print("\n渲染对比视频...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 初始帧作为参考
    init_frame = render_qpos_sequence(str(MODEL_XML), qpos0[None])[0]

    # 渲染三条轨迹
    for label, idx in [("best", best_idx), ("worst", worst_idx), ("expert", 0)]:
        frames = render_qpos_sequence(str(MODEL_XML), qpos_trajs[idx])
        out_path = OUTPUT_DIR / f"ranking_{label}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 10, (WIDTH, HEIGHT))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  {label}: {out_path}  cost={scores[idx]:.4f}")

    # 并排对比视频（最优 vs 专家 vs 最差）
    all_frames = {
        "best":   render_qpos_sequence(str(MODEL_XML), qpos_trajs[best_idx]),
        "expert": render_qpos_sequence(str(MODEL_XML), qpos_trajs[0]),
        "worst":  render_qpos_sequence(str(MODEL_XML), qpos_trajs[worst_idx]),
    }
    comparison_path = OUTPUT_DIR / "ranking_comparison.mp4"
    writer = cv2.VideoWriter(
        str(comparison_path), cv2.VideoWriter_fourcc(*"mp4v"),
        10, (WIDTH * 3, HEIGHT)
    )

    def add_label(img, text, color=(255, 255, 0)):
        out = img.copy()
        cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out

    for t in range(H):
        row = np.concatenate([
            add_label(all_frames["best"][t],   f"BEST  {names[best_idx]}  {scores[best_idx]:.3f}"),
            add_label(all_frames["expert"][t], f"EXPERT {scores[0]:.3f}"),
            add_label(all_frames["worst"][t],  f"WORST {names[worst_idx]} {scores[worst_idx]:.3f}",
                      color=(0, 0, 255)),
        ], axis=1)
        writer.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"\n  对比视频: {comparison_path}")

    # 验收
    print("\n" + "=" * 60)
    print("P3-3 验收")
    print("=" * 60)
    checks = [
        ("批量 rollout 成功",        result["xpos"].shape == (K, H, model.nbody, 3)),
        ("无 NaN",                   not np.any(np.isnan(result["xpos"]))),
        ("专家代价≤ 最差代价",         scores[0] <= scores[worst_idx]),
        ("最优/最差代价有差距",         float(scores[worst_idx] - scores[best_idx]) > 1e-4),
        ("对比视频已生成",             comparison_path.exists()),
    ]
    all_pass = True
    for name, ok in checks:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
        if not ok:
            all_pass = False
    print()
    print("P3-3 结果:", "✅ ALL PASS" if all_pass else "❌ 部分失败")


if __name__ == "__main__":
    np.random.seed(42)
    main()
