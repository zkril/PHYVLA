"""MJXStateAdapter：将 CPU MuJoCo 状态映射到 mjx.Data。

运行环境：phyvla-p2（Python 3.11，JAX + mujoco-mjx）
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class MJXStateAdapter:
    """从 CPU MuJoCo 状态构造 mjx.Data，支持 qpos/qvel/ctrl 同步。"""

    def __init__(self, xml_path: str | Path) -> None:
        import mujoco
        from mujoco import mjx

        self.xml_path = str(xml_path)
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data  = mujoco.MjData(self.mj_model)
        self.mx       = mjx.put_model(self.mj_model)

        print(f"[MJXStateAdapter] 模型加载成功")
        print(f"  nq={self.mj_model.nq}  nv={self.mj_model.nv}  nu={self.mj_model.nu}")

    def state_to_mjx(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: Optional[np.ndarray] = None,
    ):
        """将 qpos/qvel（以及可选的 ctrl）转换为 mjx.Data。

        Args:
            qpos: [nq] 广义坐标
            qvel: [nv] 广义速度
            ctrl: [nu] 执行器控制量（可选）

        Returns:
            dx: mjx.Data（已前向传播）
        """
        import mujoco
        from mujoco import mjx

        # 写入 CPU MjData
        self.mj_data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self.mj_data.qvel[:] = np.asarray(qvel, dtype=np.float64)
        if ctrl is not None:
            self.mj_data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)
        else:
            self.mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 转为 mjx.Data
        dx = mjx.put_data(self.mj_model, self.mj_data)
        return dx

    def mjx_to_numpy(self, dx) -> dict:
        """从 mjx.Data 提取关键物理量为 numpy 数组。"""
        import jax.numpy as jnp
        import numpy as np

        return {
            "qpos":  np.asarray(dx.qpos),
            "qvel":  np.asarray(dx.qvel),
            "xpos":  np.asarray(dx.xpos),   # body 世界坐标位置 [nbody, 3]
            "xquat": np.asarray(dx.xquat),  # body 世界坐标姿态 [nbody, 4]
        }

    def body_id(self, name: str) -> int:
        import mujoco
        bid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if bid < 0:
            raise KeyError(f"body '{name}' 不存在")
        return int(bid)
