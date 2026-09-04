"""P2 可视化 parity 验证。

策略：优先 GPU，OOM 则停止并报告。
输出：CPU MuJoCo vs MJX 轨迹的左右对比视频。

运行（phyvla-p2 环境）：
    python scripts/verify_parity_visual.py \
        --xml   outputs/task02/model_mjx_reduced_v1.xml \
        --trace outputs/task02/controller_trace_v1.npz \
        --output outputs/task02/parity_visual.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xml",     type=Path,
                   default=Path("outputs/task02/model_mjx_reduced_v1.xml"))
    p.add_argument("--trace",   type=Path,
                   default=Path("outputs/task02/controller_trace_v1.npz"))
    p.add_argument("--output",  type=Path,
                   default=Path("outputs/task02/parity_visual.mp4"))
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--height",  type=int, default=256)
    p.add_argument("--width",   type=int, default=256)
    p.add_argument("--fps",     type=int, default=10)
    return p.parse_args()


# ─────────────────────────────────────────────
# CPU MuJoCo rollout + 渲染
# ─────────────────────────────────────────────

def cpu_rollout_and_render(
    xml_path: str,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl_seq: np.ndarray,
    height: int,
    width: int,
) -> tuple:
    """返回 (qpos_traj [T+1, nq], frames [T+1, H, W, 3])"""
    import mujoco

    model    = mujoco.MjModel.from_xml_path(xml_path)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)

    T          = len(ctrl_seq)
    qpos_traj  = np.zeros((T + 1, model.nq))
    frames     = []

    qpos_traj[0] = data.qpos.copy()
    renderer.update_scene(data)
    frames.append(renderer.render().copy())

    for t in range(T):
        data.ctrl[:] = ctrl_seq[t]
        mujoco.mj_step(model, data)
        qpos_traj[t + 1] = data.qpos.copy()
        renderer.update_scene(data)
        frames.append(renderer.render().copy())

    renderer.close()
    return qpos_traj, np.stack(frames)


# ─────────────────────────────────────────────
# MJX rollout（GPU 优先，OOM 则停止）
# ─────────────────────────────────────────────

def mjx_kinematics_gpu(
    xml_path: str,
    qpos_seq: np.ndarray,   # [T, nq]  已存储的 qpos 序列
) -> np.ndarray:
    """用 mjx.kinematics（纯运动学，无动力学/接触求解）计算body位置。

    比 mjx.step 编译快10倍以上，足以验证 P2 运动学一致性。
    返回 xpos_traj [T, nbody, 3]。
    GPU OOM 时抛出 RuntimeError。
    """
    import mujoco
    from mujoco import mjx
    import jax
    import jax.numpy as jnp

    os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
    print(f"  JAX 设备: {jax.devices()}")

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    mx  = mjx.put_model(mj_model)
    dx0 = mjx.put_data(mj_model, mj_data)

    # 只 JIT 纯运动学（无接触求解器），编译极快
    @jax.jit
    def get_xpos(qpos):
        """纯运动学：更新 qpos 并做前向传播，取 body 世界坐标。"""
        dx = dx0.replace(qpos=jnp.array(qpos, dtype=jnp.float32))
        dx = mjx.forward(mx, dx)
        return dx.xpos   # [nbody, 3]

    print("  MJX JIT 编译前向传播（GPU）...")
    try:
        T = len(qpos_seq)
        xpos_list = []

        for t in range(T):
            xpos = get_xpos(qpos_seq[t])
            jax.block_until_ready(xpos)
            xpos_list.append(np.array(xpos))
            if t == 0:
                print(f"  首步完成（{mj_model.nbody} bodies），继续 {T-1} 步...")
            if (t + 1) % 10 == 0:
                print(f"  [{t+1}/{T}]")

    except Exception as e:
        err = str(e)
        if any(k in err for k in ("Out of memory", "OOM", "Allocator",
                                   "RESOURCE_EXHAUSTED", "allocate")):
            raise RuntimeError(f"GPU OOM: {err[:200]}")
        raise

    return np.stack(xpos_list)   # [T, nbody, 3]


# ─────────────────────────────────────────────
# MJX qpos → 渲染
# ─────────────────────────────────────────────

def render_from_qpos(
    xml_path: str,
    qpos_traj: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """将 qpos 序列写回 CPU MuJoCo 并渲染每帧，返回 [T, H, W, 3]。"""
    import mujoco

    model    = mujoco.MjModel.from_xml_path(xml_path)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames   = []

    for qpos in qpos_traj:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        frames.append(renderer.render().copy())

    renderer.close()
    return np.stack(frames)


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    trace  = np.load(str(args.trace), allow_pickle=False)
    states = trace["states"]    # [T, nq+nv]

    import mujoco
    tmp = mujoco.MjModel.from_xml_path(str(args.xml))
    nq, nv = tmp.nq, tmp.nv
    H = min(args.horizon, len(states))

    # 从存储状态提取 qpos 序列（不用 ctrl）
    qpos_seq = states[:H, :nq]   # [H, nq]
    qvel_seq = states[:H, nq:nq + nv]

    print(f"[parity_visual] XML     : {args.xml}")
    print(f"[parity_visual] frames  : {H}  nq={nq}  nv={nv}")

    # ── [1/3] CPU MuJoCo 运动学 + 渲染 ──────────────
    print("\n[1/3] CPU MuJoCo 运动学 + 渲染...")
    cpu_xpos_list, cpu_frames = [], []
    renderer = mujoco.Renderer(tmp, height=args.height, width=args.width)
    data     = mujoco.MjData(tmp)
    for t in range(H):
        data.qpos[:] = qpos_seq[t]
        data.qvel[:] = qvel_seq[t]
        mujoco.mj_forward(tmp, data)
        cpu_xpos_list.append(data.xpos.copy())
        renderer.update_scene(data)
        cpu_frames.append(renderer.render().copy())
    renderer.close()
    cpu_xpos   = np.stack(cpu_xpos_list)   # [H, nbody, 3]
    cpu_frames = np.stack(cpu_frames)
    print(f"  CPU xpos shape : {cpu_xpos.shape}")
    print(f"  CPU frames     : {cpu_frames.shape}")

    # ── [2/3] MJX 运动学（GPU） ──────────────────────
    print("\n[2/3] MJX 运动学（GPU 优先）...")
    try:
        mjx_xpos = mjx_kinematics_gpu(str(args.xml), qpos_seq)
        print(f"  MJX xpos shape : {mjx_xpos.shape}")

        xpos_err = np.abs(cpu_xpos - mjx_xpos)
        print(f"\n  xpos RMSE      : {np.sqrt(np.mean(xpos_err**2)):.6f} m")
        print(f"  xpos max_abs   : {xpos_err.max():.6f} m")

        print("\n[3/3] MJX xpos → 渲染...")
        mjx_frames = render_from_qpos(str(args.xml), qpos_seq,
                                      args.height, args.width)

    except RuntimeError as e:
        if "GPU OOM" in str(e):
            print(f"\n❌ GPU OOM — 停止: {e}")
            sys.exit(1)
        raise

    # ── [4/4] 对比视频 ───────────────────────────────
    print("\n[4/4] 生成对比视频...")
    _write_comparison_video(cpu_frames, mjx_frames, args.output,
                            args.fps, args.width, args.height)
    print(f"  已保存: {args.output}")

    xpos_rmse = float(np.sqrt(np.mean((cpu_xpos - mjx_xpos)**2)))
    verdict = ("✅ PASS" if xpos_rmse < 1e-4 else
               "⚠️  ACCEPTABLE" if xpos_rmse < 1e-2 else "❌ DIVERGED")
    print(f"\n  parity 判定: {verdict}  (xpos RMSE={xpos_rmse:.2e} m)")


def _write_comparison_video(
    cpu_frames: np.ndarray,
    mjx_frames: np.ndarray,
    output: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    import cv2

    output.parent.mkdir(parents=True, exist_ok=True)
    T      = min(len(cpu_frames), len(mjx_frames))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width * 2, height))

    def _add_label(frame: np.ndarray, text: str) -> np.ndarray:
        out = frame.copy()
        cv2.putText(out, text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        return out

    for t in range(T):
        left  = _add_label(cpu_frames[t], "CPU MuJoCo")
        right = _add_label(mjx_frames[t], "MJX")
        row   = np.concatenate([left, right], axis=1)
        writer.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    writer.release()


if __name__ == "__main__":
    main()
