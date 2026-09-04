"""P4 — Warp 自动微分梯度计算模块。

三种策略（均以 mujoco_warp 为物理后端，不依赖 JAX）：
  1. grad_tape_step()   : wp.Tape 单步 Jacobian（dx.ctrl.requires_grad，真 autodiff）
  2. grad_tape_rollout(): wp.Tape H 步展开反向（全轨迹梯度）
  3. grad_fd_batch()    : 并行有限差分（全轨迹累积 loss，主用，任意区间稳定）

运行环境：phyvla-p2（Python 3.11, mujoco_warp 3.10.0.3, warp 1.15.0）
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _resolve_bodies(xml_path: str, target_body: str, goal_body: str):
    """target_body/goal_body 为空时从 task_info.json 自动解析。"""
    if target_body and goal_body:
        return target_body, goal_body
    info_file = Path(xml_path).parent / "task_info.json"
    if not info_file.exists():
        raise FileNotFoundError(
            f"target_body/goal_body 未指定且 task_info.json 不存在：{info_file}\n"
            "请先运行：conda activate libero_replay && "
            "python scripts/save_task_info.py --task-ids <id>"
        )
    info = json.loads(info_file.read_text())
    return info["target_body"], info["goal_body"]


# ─────────────────────────────────────────────────────────────────────────────
# Warp kernel（模块级延迟定义）
# ─────────────────────────────────────────────────────────────────────────────
_accum_goal_dist = None   # 首次 WarpGradComputer.__init__ 时填充


def _ensure_kernels():
    global _accum_goal_dist
    if _accum_goal_dist is not None:
        return
    import warp as wp

    @wp.kernel
    def _k(
        xpos:      wp.array(dtype=wp.vec3f, ndim=2),  # [nworld, nbody]
        target_id: int,
        goal_id:   int,
        loss:      wp.array(dtype=wp.float32),          # [1]
    ):
        """累加 target→goal 距离到 loss（tid = world index）。"""
        w     = wp.tid()
        t_pos = xpos[w, target_id]
        g_pos = xpos[w, goal_id]
        d     = wp.length(t_pos - g_pos)
        wp.atomic_add(loss, 0, d)

    _accum_goal_dist = _k


# ─────────────────────────────────────────────────────────────────────────────
class WarpGradComputer:
    """∂L_phy/∂ctrl 梯度计算器（Warp 后端）。"""

    def __init__(
        self,
        xml_path:    str,
        target_body: str = "",
        goal_body:   str = "",
    ) -> None:
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        self._mjw = mjw
        self._wp  = wp

        self.xml_path = str(xml_path)
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data  = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.mx = mjw.put_model(self.mj_model)

        target_body, goal_body = _resolve_bodies(self.xml_path, target_body, goal_body)

        self.target_id = int(mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, target_body))
        self.goal_id = int(mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, goal_body))

        # 末端执行器 body（用于计算 approach cost）
        for ee_name in ("gripper0_eef", "robot0_right_hand", "eef", "panda_hand"):
            eid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
            if eid >= 0:
                self.ee_id = eid
                break
        else:
            self.ee_id = self.goal_id   # fallback

        self.nu    = int(self.mj_model.nu)
        self.nq    = int(self.mj_model.nq)
        self.nbody = int(self.mj_model.nbody)

        _ensure_kernels()
        print(f"[WarpGradComputer] nu={self.nu}  nbody={self.nbody}"
              f"  target_id={self.target_id}  goal_id={self.goal_id}")

    # ─────────────────────────────────────────────────────────────────────
    # 工具：创建并初始化 dx（动态计算约束槽）
    # ─────────────────────────────────────────────────────────────────────
    def _make_dx(self, nworld: int, qpos0: np.ndarray, qvel0: np.ndarray):
        import mujoco
        data = mujoco.MjData(self.mj_model)
        data.qpos[:] = qpos0.astype(np.float64)
        data.qvel[:] = qvel0.astype(np.float64)
        mujoco.mj_forward(self.mj_model, data)

        ncon_ref = max(int(data.ncon), 1)
        naconmax = max(nworld * ncon_ref * 3, nworld * 20, 600)
        nconmax  = max(nworld * 16, 100)
        njmax    = max(nworld * 40, 300)

        return self._mjw.put_data(
            self.mj_model, data, nworld=nworld,
            njmax=njmax, nconmax=nconmax, naconmax=naconmax,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 策略一：wp.Tape 单步 Jacobian（dx.ctrl.requires_grad）
    # ─────────────────────────────────────────────────────────────────────
    def grad_tape_step(
        self,
        qpos0:    np.ndarray,   # [nq]
        qvel0:    np.ndarray,   # [nv]
        ctrl_vec: np.ndarray,   # [nu]
    ) -> np.ndarray:
        """wp.Tape 单步 ∂(goal_dist)/∂ctrl [nu]。

        在 dx.ctrl 上设置 requires_grad=True，tape 内运行一步 mjw.step，
        backward 得到梯度。
        """
        wp  = self._wp
        mjw = self._mjw

        dx = self._make_dx(1, qpos0, qvel0)
        dx.ctrl.requires_grad = True

        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            dx.ctrl.assign(ctrl_vec.astype(np.float32)[None])
            mjw.step(self.mx, dx)
            wp.synchronize()
            wp.launch(_accum_goal_dist, dim=1,
                      inputs=[dx.xpos, self.target_id, self.goal_id, loss])

        try:
            tape.backward(loss)
            grad = dx.ctrl.grad.numpy()[0]   # [nu]
            return np.asarray(grad, dtype=np.float32)
        except Exception as exc:
            print(f"  [tape_step] backward 失败: {exc}")
            return np.zeros(self.nu, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # 策略二：wp.Tape H 步展开反向
    # ─────────────────────────────────────────────────────────────────────
    def grad_tape_rollout(
        self,
        qpos0:    np.ndarray,   # [nq]
        qvel0:    np.ndarray,   # [nv]
        ctrl_seq: np.ndarray,   # [H, nu]
    ) -> np.ndarray:
        """wp.Tape 展开 H 步全轨迹，backward 得到 grad [H, nu]。

        每步使用独立的 warp array 作为 ctrl，requires_grad=True，
        tape 记录所有 H 步 mjw.step，backward 后读取各步梯度。
        """
        wp  = self._wp
        mjw = self._mjw

        ctrl_seq = np.asarray(ctrl_seq, dtype=np.float32)
        H, nu    = ctrl_seq.shape

        dx = self._make_dx(1, qpos0, qvel0)

        # 每步 ctrl 独立可微 warp array
        ctrl_wps = []
        for t in range(H):
            c = wp.array(ctrl_seq[t][None].copy(),
                         dtype=wp.float32, requires_grad=True)
            ctrl_wps.append(c)

        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            for t in range(H):
                # 每步通过 kernel copy 传入 ctrl（保留梯度图）
                wp.copy(dx.ctrl, ctrl_wps[t])
                mjw.step(self.mx, dx)
            wp.synchronize()
            # 用最终步 xpos 计算 loss
            wp.launch(_accum_goal_dist, dim=1,
                      inputs=[dx.xpos, self.target_id, self.goal_id, loss])

        try:
            tape.backward(loss)
            grad = np.stack([c.grad.numpy()[0] for c in ctrl_wps])   # [H, nu]
            return grad
        except Exception as exc:
            print(f"  [tape_rollout] backward 失败: {exc}")
            return np.zeros((H, nu), dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # 策略三：并行有限差分（主用）
    # ─────────────────────────────────────────────────────────────────────
    def grad_fd_batch(
        self,
        qpos0:    np.ndarray,   # [nq]
        qvel0:    np.ndarray,   # [nv]
        ctrl_seq: np.ndarray,   # [H, nu]
        eps:      float = 1.0,  # ctrl 量级 ~30-50，eps=1.0 ≈ 2%
        w_goal:   float = 1.0,
        w_drop:   float = 0.5,
    ) -> dict:
        """并行 FD 梯度 ∂L_phy/∂ctrl [H, nu]。

        全轨迹累积 loss（sum_{t=1}^{H} dist(target_t, goal_t)），
        使用 mujoco_warp 多 world 并行所有 2*H*nu 扰动。

        Returns:
            dict:
                'grad'        : [H, nu]
                'loss_center' : scalar
                'grad_norm'   : [H]
                'elapsed_s'   : 耗时
        """
        ctrl_seq = np.asarray(ctrl_seq, dtype=np.float32)
        H, nu    = ctrl_seq.shape
        assert nu == self.nu

        # K_total = 1（center）+ 2*H*nu（±eps）
        n_pert  = 2 * H * nu
        K_total = 1 + n_pert

        ctrl_batch = np.tile(ctrl_seq[None], (K_total, 1, 1))   # [K, H, nu]
        pert_index = []   # list of (t, d, sign)
        idx = 0
        for t in range(H):
            for d in range(nu):
                for sign in (+1, -1):
                    ctrl_batch[1 + idx, t, d] += sign * eps
                    pert_index.append((t, d, sign))
                    idx += 1

        t0 = time.time()
        dx = self._make_dx(K_total, qpos0, qvel0)
        # 全轨迹累积 cost [K_total]
        costs = self._rollout_cumulative_cost(dx, ctrl_batch, w_goal, w_drop)
        elapsed = time.time() - t0

        loss_center = float(costs[0])

        grad = np.zeros((H, nu), dtype=np.float32)
        for idx, (t, d, sign) in enumerate(pert_index):
            k = 1 + idx
            if sign == +1:
                grad[t, d] += costs[k] / (2.0 * eps)
            else:
                grad[t, d] -= costs[k] / (2.0 * eps)

        return {
            "grad":        grad,
            "loss_center": loss_center,
            "grad_norm":   np.linalg.norm(grad, axis=-1),
            "elapsed_s":   elapsed,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 梯度下降动作修正
    # ─────────────────────────────────────────────────────────────────────
    def refine_actions(
        self,
        qpos0:      np.ndarray,
        qvel0:      np.ndarray,
        ctrl_seq:   np.ndarray,   # [H, nu]
        n_iters:    int   = 5,
        step_size:  float = 5.0,   # 归一化后的步长（ctrl 量级）
        ctrl_limit: float = 120.0,
        eps:        float = 1.0,
    ) -> dict:
        """梯度下降迭代优化动作序列（归一化步长）。

        A^{n+1} = clip(A^n - step_size * grad / |grad|, -limit, +limit)
        使用归一化梯度，step_size 直接表示每步 ctrl 移动量，
        独立于梯度绝对大小，在任何梯度量级下均有效。
        """
        ctrl = ctrl_seq.copy().astype(np.float32)
        loss_hist = []
        grad_hist = []

        for i in range(n_iters):
            info = self.grad_fd_batch(qpos0, qvel0, ctrl, eps=eps)
            grad = info["grad"]
            loss = info["loss_center"]
            loss_hist.append(loss)
            grad_hist.append(grad.copy())

            g_norm = float(np.linalg.norm(grad))
            if g_norm > 1e-9:
                grad_normed = grad / g_norm
            else:
                grad_normed = grad
                print(f"  iter {i+1}/{n_iters}  梯度为零，跳过更新")
                continue

            ctrl = np.clip(ctrl - step_size * grad_normed, -ctrl_limit, ctrl_limit)
            print(f"  iter {i+1}/{n_iters}  loss={loss:.5f}"
                  f"  |grad|={g_norm:.6f}  step={step_size:.2f}")

        final = self.grad_fd_batch(qpos0, qvel0, ctrl, eps=eps)
        loss_hist.append(final["loss_center"])
        delta = loss_hist[0] - loss_hist[-1]
        pct   = 100.0 * delta / max(loss_hist[0], 1e-8)
        print(f"  loss: {loss_hist[0]:.5f} → {loss_hist[-1]:.5f}  下降 {pct:.2f}%")

        return {
            "ctrl_refined": ctrl,
            "loss_history": np.array(loss_hist),
            "grad_history": np.stack(grad_hist, axis=0),
        }

    # ─────────────────────────────────────────────────────────────────────
    # 内部：全轨迹累积 cost（[K] float）
    # ─────────────────────────────────────────────────────────────────────
    def _rollout_cumulative_cost(
        self,
        dx,
        ctrl_batch: np.ndarray,   # [K, H, nu]
        w_goal:     float = 1.0,
        w_drop:     float = 0.5,
        w_approach: float = 0.8,   # EE→目标物体距离（自由阶段关键）
    ) -> np.ndarray:
        """全轨迹累积代价。

        L = w_goal×dist(bowl,plate) + w_drop×掉落惩罚 + w_approach×dist(EE,bowl)

        三项联合：
          - 自由阶段：EE→碗 驱动梯度（臂移近碗）
          - 接触/搬运阶段：bowl→plate 驱动梯度（碗移向盘子）
        """
        mjw = self._mjw
        wp  = self._wp
        K, H, nu = ctrl_batch.shape

        costs = np.zeros(K, dtype=np.float32)

        for t in range(H):
            dx.ctrl.assign(ctrl_batch[:, t, :])
            mjw.step(self.mx, dx)
            wp.synchronize()

            xpos_np = dx.xpos.numpy()   # [K, nbody, 3]
            for k in range(K):
                tp    = xpos_np[k, self.target_id]   # 目标物（碗）
                gp    = xpos_np[k, self.goal_id]     # 目标面（盘子）
                ep    = xpos_np[k, self.ee_id]       # 末端执行器

                dist_bg  = float(np.linalg.norm(tp - gp))      # 碗-盘子
                dist_et  = float(np.linalg.norm(ep - tp))      # EE-碗
                drop     = float(max(0.0, 0.9 - tp[2]))

                costs[k] += w_goal * dist_bg + w_drop * drop + w_approach * dist_et

        return costs / H
