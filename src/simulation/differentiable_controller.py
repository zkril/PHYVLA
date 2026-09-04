"""可微 JAX 关节位置 PD 控制器。

将目标关节位置映射为关节力矩（完全可微，支持 jax.grad）：

    τ = Kp * (q_des - q) - Kd * dq

用于 P3/P4 可微 rollout，替代 robosuite 不可微的 OSC 控制器。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# Franka Panda 默认 PD 增益（可在运行时覆盖）
DEFAULT_KP = jnp.array([600., 600., 600., 600., 250., 150., 50.], dtype=jnp.float32)
DEFAULT_KD = jnp.array([ 50.,  50.,  50.,  50.,  30.,  25., 15.], dtype=jnp.float32)

# 夹爪增益
GRIPPER_KP = jnp.array([50., 50.], dtype=jnp.float32)
GRIPPER_KD = jnp.array([ 5.,  5.], dtype=jnp.float32)


def pd_torque(
    q_des:   jnp.ndarray,   # [n_joints]  目标关节位置
    q_cur:   jnp.ndarray,   # [n_joints]  当前关节位置
    dq_cur:  jnp.ndarray,   # [n_joints]  当前关节速度
    kp:      jnp.ndarray,   # [n_joints]  比例增益
    kd:      jnp.ndarray,   # [n_joints]  微分增益
) -> jnp.ndarray:
    """PD 控制力矩（完全可微）。"""
    return kp * (q_des - q_cur) - kd * dq_cur


def arm_torque(
    q_des:  jnp.ndarray,    # [7]  目标7轴关节位置
    q_cur:  jnp.ndarray,    # [7]
    dq_cur: jnp.ndarray,    # [7]
    kp: jnp.ndarray = DEFAULT_KP,
    kd: jnp.ndarray = DEFAULT_KD,
) -> jnp.ndarray:
    return pd_torque(q_des, q_cur, dq_cur, kp, kd)


def gripper_torque(
    open_command: jnp.ndarray,  # [1]  -1=关闭, +1=打开
    g_cur:   jnp.ndarray,       # [2]  当前夹爪关节
    dg_cur:  jnp.ndarray,       # [2]  当前夹爪速度
    max_gap: float = 0.04,
) -> jnp.ndarray:
    """夹爪 PD 控制，将 [-1,1] 命令映射到关节目标。"""
    target = (open_command + 1.0) * 0.5 * max_gap   # [0, max_gap]
    g_des  = jnp.array([target[0], -target[0]], dtype=jnp.float32)
    return pd_torque(g_des, g_cur, dg_cur, GRIPPER_KP, GRIPPER_KD)


def compute_ctrl(
    q_des_arm:     jnp.ndarray,   # [7]
    gripper_cmd:   jnp.ndarray,   # [1]
    q_cur:         jnp.ndarray,   # [9]  arm[7] + gripper[2]
    dq_cur:        jnp.ndarray,   # [9]
    kp_arm:        jnp.ndarray = DEFAULT_KP,
    kd_arm:        jnp.ndarray = DEFAULT_KD,
) -> jnp.ndarray:
    """组装完整9维 ctrl 向量（arm[7] + gripper[2]）。"""
    tau_arm  = arm_torque(q_des_arm, q_cur[:7], dq_cur[:7], kp_arm, kd_arm)
    tau_grip = gripper_torque(gripper_cmd, q_cur[7:], dq_cur[7:])
    return jnp.concatenate([tau_arm, tau_grip])
