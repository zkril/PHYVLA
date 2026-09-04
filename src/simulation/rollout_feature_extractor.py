"""从 MJX rollout 轨迹中提取 z_dyn 和 z_risk 特征向量。

用于替换 P1 中的：
  z_dyn ← teacher-forcing（真实未来帧差分）
  z_risk ← 规则特征（max_force 等当前帧规则）

输入：MJX rollout 的 xpos 轨迹 [H, nbody, 3]
输出：
  dyn_target    [16]  TARGET 对象的绝对动力学变化
  dyn_rel_cont  [N, 36]  其他对象相对 TARGET 的变化（带 role embedding）
  risk_vec      [6]   物理代价各分量
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────
# z_dyn 特征提取
# ──────────────────────────────────────────────────────

def extract_dyn_target(
    xpos_traj:   np.ndarray,   # [H, nbody, 3]  body 世界坐标
    xquat_traj:  np.ndarray,   # [H, nbody, 4]  body 姿态（wxyz）
    xvel_traj:   Optional[np.ndarray],  # [H, nbody, 6] or None
    target_id:   int,
    goal_id:     int,
) -> np.ndarray:
    """提取 TARGET 对象的绝对动力学变化 [16]。

    Δposition[3] + Δrot_6d[6] + Δlin_vel[3] + Δang_vel[3] + Δdist_to_goal[1]
    """
    from utils.rotations import quat_xyzw_to_6d

    H = xpos_traj.shape[0]
    target_pos_0 = xpos_traj[0,  target_id]     # [3]
    target_pos_H = xpos_traj[H-1, target_id]    # [3]
    goal_pos_0   = xpos_traj[0,  goal_id]
    goal_pos_H   = xpos_traj[H-1, goal_id]

    delta_pos = (target_pos_H - target_pos_0).astype(np.float32)

    # 姿态差分（wxyz→xyzw 转换后做6D差）
    q0 = xquat_traj[0,   target_id][[1, 2, 3, 0]]  # wxyz→xyzw
    qH = xquat_traj[H-1, target_id][[1, 2, 3, 0]]
    rot0 = quat_xyzw_to_6d(q0).astype(np.float32)
    rotH = quat_xyzw_to_6d(qH).astype(np.float32)
    delta_rot = rotH - rot0

    if xvel_traj is not None:
        delta_lv = (xvel_traj[H-1, target_id, :3] - xvel_traj[0, target_id, :3]).astype(np.float32)
        delta_av = (xvel_traj[H-1, target_id, 3:] - xvel_traj[0, target_id, 3:]).astype(np.float32)
    else:
        delta_lv = np.zeros(3, dtype=np.float32)
        delta_av = np.zeros(3, dtype=np.float32)

    dist_0 = float(np.linalg.norm(target_pos_0 - goal_pos_0))
    dist_H = float(np.linalg.norm(target_pos_H - goal_pos_H))
    delta_dist = np.array([dist_H - dist_0], dtype=np.float32)

    return np.concatenate([delta_pos, delta_rot, delta_lv, delta_av, delta_dist])  # [16]


def extract_dyn_rel(
    xpos_traj:  np.ndarray,   # [H, nbody, 3]
    target_id:  int,
    obj_ids:    list,          # 非 TARGET 对象的 body id
    obj_roles:  np.ndarray,   # [N] 角色 int
    role_emb_dim: int = 32,
) -> tuple:
    """提取其他对象相对 TARGET 的变化，返回 (dyn_rel [N,36], dyn_rel_mask [N])。"""
    H = xpos_traj.shape[0]
    results, masks = [], []

    for i, (bid, role) in enumerate(zip(obj_ids, obj_roles)):
        if bid == target_id:
            results.append(np.zeros(36, dtype=np.float32))
            masks.append(0.0)
            continue

        rel_0   = xpos_traj[0,   bid] - xpos_traj[0,   target_id]
        rel_H   = xpos_traj[H-1, bid] - xpos_traj[H-1, target_id]
        d_rel   = rel_H - rel_0
        d_dist  = np.array([np.linalg.norm(rel_H) - np.linalg.norm(rel_0)], np.float32)
        role_oh = np.zeros(role_emb_dim, np.float32)
        if role < role_emb_dim:
            role_oh[role] = 1.0

        results.append(np.concatenate([d_rel.astype(np.float32), d_dist, role_oh]))
        masks.append(1.0)

    if not results:
        return np.zeros((1, 36), np.float32), np.zeros(1, np.float32)

    return np.stack(results), np.array(masks, np.float32)


# ──────────────────────────────────────────────────────
# z_risk 特征提取
# ──────────────────────────────────────────────────────

def extract_risk_from_rollout(
    xpos_traj:  np.ndarray,   # [H, nbody, 3]
    target_id:  int,
    goal_id:    int,
    max_vel:    Optional[np.ndarray] = None,  # [H, nbody, 6] 速度（可选）
) -> np.ndarray:
    """从 rollout 轨迹中提取 6 维风险特征向量。

    特征：
      L_goal      碗到盘的平均距离（目标代价）
      L_drop      碗高度最低值（掉落风险）
      L_smooth    碗位置方差（平滑代价）
      L_drift     末帧碗-盘距离（偏离代价）
      goal_progress  起末碗-盘距离差（进展，越大越好→取负）
      height_gain    碗高度净变化（抬起程度）
    """
    H = xpos_traj.shape[0]
    target_pos = xpos_traj[:, target_id]    # [H, 3]
    goal_pos   = xpos_traj[:, goal_id]      # [H, 3]

    dist_to_goal = np.linalg.norm(target_pos - goal_pos, axis=-1)  # [H]
    L_goal       = float(dist_to_goal.mean())

    bowl_z       = target_pos[:, 2]
    L_drop       = float(max(0.0, 0.9 - bowl_z.min()))   # 低于0.9m记为掉落风险

    # 平滑代价：位置变化的标准差
    delta = target_pos[1:] - target_pos[:-1]
    L_smooth = float(np.mean(delta ** 2))

    L_drift  = float(dist_to_goal[-1])

    goal_progress = float(dist_to_goal[0] - dist_to_goal[-1])  # 正=在靠近

    height_gain = float(bowl_z[-1] - bowl_z[0])

    return np.array([
        L_goal,
        L_drop,
        L_smooth,
        L_drift,
        -goal_progress,    # 取负使"越小越好"一致
        -height_gain,      # 取负使"越小越好"
    ], dtype=np.float32)
