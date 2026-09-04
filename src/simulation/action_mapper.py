"""7D 高层动作 → 关节目标位置映射器（JAX 可微）。

LIBERO 7D 动作格式：[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]

映射流程：
  1. 当前 qpos → 前向运动学 → EE 当前位姿
  2. EE 当前位姿 + 7D delta → 目标 EE 位姿
  3. 雅可比伪逆 IK → 目标关节增量 Δq
  4. q_des = q_cur + Δq（裁剪到关节限制内）
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# Franka Panda 关节范围（弧度）
JOINT_LO = jnp.array([-2.9671, -1.8326, -2.9671, -3.1416,
                       -2.9671, -0.0873, -2.9671], dtype=jnp.float32)
JOINT_HI = jnp.array([ 2.9671,  1.8326,  2.9671,  0.0000,
                        2.9671,  3.8223,  2.9671], dtype=jnp.float32)


def _skew(v: jnp.ndarray) -> jnp.ndarray:
    """3D 向量 → 反对称矩阵（用于轴角旋转）。"""
    x, y, z = v[0], v[1], v[2]
    return jnp.array([[ 0., -z,  y],
                      [ z,  0., -x],
                      [-y,  x,  0.]], dtype=jnp.float32)


def axis_angle_to_rotmat(axis_angle: jnp.ndarray) -> jnp.ndarray:
    """轴角 [3] → 旋转矩阵 [3,3]（Rodrigues 公式，可微）。"""
    theta = jnp.linalg.norm(axis_angle) + 1e-8
    axis  = axis_angle / theta
    K     = _skew(axis)
    return (jnp.eye(3) + jnp.sin(theta) * K
            + (1.0 - jnp.cos(theta)) * K @ K)


def compute_jacobian_pinv(
    fk_fn,           # qpos → ee_pos [3]（可微函数）
    qpos: jnp.ndarray,  # [7]
) -> jnp.ndarray:
    """计算末端位置关于关节的雅可比伪逆 J^+ [7, 3]。"""
    J = jax.jacfwd(fk_fn)(qpos)   # [3, 7]
    # 阻尼最小二乘伪逆
    lam = 0.01
    JJT = J @ J.T + lam * jnp.eye(3)
    return J.T @ jnp.linalg.inv(JJT)   # [7, 3]


def action7d_to_joint_targets(
    action:     jnp.ndarray,   # [7]: Δpos[3] + Δrot[3] + gripper[1]
    q_cur:      jnp.ndarray,   # [7]: 当前关节位置
    fk_pos_fn,                 # qpos[7] → ee_pos[3]（JAX 可微）
    pos_scale:  float = 0.05,  # delta 位置缩放
    rot_scale:  float = 0.1,   # delta 旋转缩放
    clip_delta: float = 0.1,   # 最大单步关节变化（弧度）
) -> tuple:
    """
    Returns:
        q_des    [7]: 目标关节位置（已裁剪到关节限制）
        gripper  [1]: 夹爪命令 [-1, 1]
    """
    delta_pos = action[:3] * pos_scale   # [3]
    delta_rot = action[3:6] * rot_scale  # [3] 轴角
    gripper   = action[6:7]              # [1]

    # 位置增量 → 关节增量（雅可比IK）
    J_pinv  = compute_jacobian_pinv(fk_pos_fn, q_cur)   # [7, 3]
    delta_q = J_pinv @ delta_pos                         # [7]

    # 旋转增量（近似：直接加到关节，仅用于小角度）
    # 完整版需要Jacobian的旋转部分，此处用简化
    delta_q = jnp.clip(delta_q, -clip_delta, clip_delta)

    q_des = jnp.clip(q_cur + delta_q, JOINT_LO, JOINT_HI)
    return q_des, gripper


def make_fk_pos_fn(mx, dx_ref, ee_body_id: int):
    """
    创建前向运动学函数：qpos[7] → ee_pos[3]。
    用于在 JAX 计算图中计算末端位置，支持 jax.jacfwd 微分。
    """
    from mujoco import mjx

    def fk_pos(qpos7: jnp.ndarray) -> jnp.ndarray:
        dx = dx_ref.replace(qpos=dx_ref.qpos.at[:7].set(qpos7))
        dx = mjx.forward(mx, dx)
        return dx.xpos[ee_body_id]   # [3]

    return fk_pos
