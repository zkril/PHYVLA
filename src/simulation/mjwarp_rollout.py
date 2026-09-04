"""mujoco_warp 可微 Rollout 模块（替代 mjx_rollout.py）。

使用 mujoco_warp（CUDA Warp 后端）替代 MJX-JAX，解决复杂操作场景（271geom）OOM 问题。
支持完整 LIBERO 模型的真实物理精度 rollout。

运行环境：phyvla-p2（Python 3.11，mujoco_warp 安装）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _resolve_bodies(xml_path: str, mj_model, target_body: str, goal_body: str):
    """如果 target_body/goal_body 为空，从 task_info.json 自动解析。"""
    if target_body and goal_body:
        return target_body, goal_body   # 显式传入，直接使用

    import json
    info_file = Path(xml_path).parent / "task_info.json"
    if not info_file.exists():
        raise FileNotFoundError(
            f"target_body/goal_body 未指定且 task_info.json 不存在：{info_file}\n"
            "请先运行：conda activate libero_replay && "
            "python scripts/save_task_info.py --task-ids <id>"
        )
    info = json.loads(info_file.read_text())
    return info["target_body"], info["goal_body"]


class MJWarpRollout:
    """mujoco_warp 批量 rollout，支持完整 LIBERO 模型（271geom）。

    相比 MJXRollout 的优势：
    - 无 XLA 静态编译 OOM（CUDA 动态内核）
    - 使用完整 LIBERO 模型，真实接触物理
    - 首次编译 ~141s（后续 <1s，内核已缓存）
    """

    def __init__(
        self,
        xml_path:    str | Path,
        target_body: str = "",
        goal_body:   str = "",
        nconmax:     Optional[int] = None,
    ) -> None:
        """
        target_body / goal_body:
            留空时自动从 outputs/task{id}/task_info.json 解析。
            显式传入时直接使用（向后兼容）。
        """
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        self.xml_path = str(xml_path)
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data  = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.mx = mjw.put_model(self.mj_model)
        self._mjw = mjw
        self._wp  = wp
        self._mu  = mujoco

        # 自动从 task_info.json 解析 body 名称
        target_body, goal_body = _resolve_bodies(
            self.xml_path, self.mj_model, target_body, goal_body
        )

        self.target_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, target_body
        )
        self.goal_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, goal_body
        )

        print(f"[MJWarpRollout] 完整 LIBERO 模型加载完成")
        print(f"  nbody={self.mj_model.nbody}  ngeom={self.mj_model.ngeom}  nu={self.mj_model.nu}")
        print(f"  target: {target_body} (id={self.target_id})")
        print(f"  goal  : {goal_body}  (id={self.goal_id})")

    def reset(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """重置初始状态。"""
        import mujoco
        self.mj_data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self.mj_data.qvel[:] = np.asarray(qvel, dtype=np.float64)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def rollout_single(self, ctrl_seq: np.ndarray) -> dict:
        """单条轨迹 rollout（K=1）。返回 xpos/qpos 轨迹。"""
        result = self.rollout_batch(ctrl_seq[None])   # [1, H, nu]
        return {
            "xpos": result["xpos"][0],   # [H, nbody, 3]
            "qpos": result["qpos"][0],   # [H, nq]
        }

    def rollout_batch(self, ctrl_seqs: np.ndarray,
                      n_substeps: int = 1) -> dict:
        """K 候选并行 rollout。

        Args:
            ctrl_seqs:  [K, H, nu]
            n_substeps: 每个 ctrl 帧执行的物理步数。
                        = round(sim_freq / ctrl_freq)
                        LIBERO: sim=500Hz, ctrl=20Hz → n_substeps=25
                        P3/P4 短时域梯度（默认）: n_substeps=1（快速，梯度方向有效）
                        全轨迹重播（保真）:        n_substeps=25（物理时间对齐）
        Returns:
            dict with xpos [K, H, nbody, 3] and qpos [K, H, nq]
        """
        mjw = self._mjw
        wp  = self._wp

        ctrl_seqs = np.asarray(ctrl_seqs, dtype=np.float32)
        K, H, nu  = ctrl_seqs.shape

        # nconmax=100, naconmax=500 确保接触对不溢出（LIBERO 场景有~87个接触）
        dx = mjw.put_data(self.mj_model, self.mj_data, nworld=K,
                          njmax=300, nconmax=100, naconmax=600)

        xpos_list = []
        qpos_list = []

        for t in range(H):
            # 为每个 world 设置当前时间步的 ctrl
            dx.ctrl.assign(ctrl_seqs[:, t, :])   # [K, nu]
            # 每个 ctrl 帧执行 n_substeps 步物理（对齐仿真与控制频率）
            for _ in range(n_substeps):
                mjw.step(self.mx, dx)
            wp.synchronize()

            xpos_list.append(dx.xpos.numpy().copy())   # [K, nbody, 3]
            qpos_list.append(dx.qpos.numpy().copy())   # [K, nq]

            if t == 0:
                print(f"  首步完成，继续 {H-1} 步...")
            if (t + 1) % 10 == 0 or t == H - 1:
                print(f"  [{t+1}/{H}]")

        return {
            "xpos": np.stack(xpos_list, axis=1),   # [K, H, nbody, 3]
            "qpos": np.stack(qpos_list, axis=1),   # [K, H, nq]
        }

    def physics_cost(
        self,
        xpos_traj: np.ndarray,   # [H, nbody, 3] or [K, H, nbody, 3]
        goal_pos:  np.ndarray,   # [3]
        w_goal:   float = 1.0,
        w_smooth: float = 0.1,
        w_drop:   float = 0.5,
    ) -> np.ndarray:
        """物理代价函数，支持单条 [H,...] 或批量 [K,H,...] 输入。"""
        batched = xpos_traj.ndim == 4   # [K, H, nbody, 3]
        if not batched:
            xpos_traj = xpos_traj[None]   # → [1, H, nbody, 3]

        goal_pos = np.asarray(goal_pos, dtype=np.float32)
        K, H = xpos_traj.shape[:2]
        costs = np.zeros(K, dtype=np.float32)

        for k in range(K):
            tp = xpos_traj[k, :, self.target_id]   # [H, 3]
            gp = xpos_traj[k, :, self.goal_id]     # [H, 3]

            # 目标距离（越小越好）
            dist = np.linalg.norm(tp - goal_pos, axis=-1)
            L_goal = float(dist.mean())

            # 掉落惩罚（碗高度低于阈值）
            bowl_z  = tp[:, 2]
            L_drop  = float(max(0.0, 0.9 - bowl_z.min()))

            # 平滑代价
            delta    = tp[1:] - tp[:-1]
            L_smooth = float(np.mean(delta ** 2))

            costs[k] = w_goal * L_goal + w_drop * L_drop + w_smooth * L_smooth

        return costs if batched else costs[0]
