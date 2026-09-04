"""MuJoCo / MJX 轨迹一致性验证脚本。

注意：LIBERO 完整场景在 MJX GPU 模式下需要 ~134 GB 显存（geom 过多），
因此本脚本强制使用 JAX CPU 后端完成 parity 验证。

运行环境：phyvla-p2（Python 3.11，JAX + mujoco-mjx）
"""

from __future__ import annotations

# 必须在导入 JAX 之前设置 CPU 后端
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xml",     type=Path,
                   default=Path("outputs/task02/model_mjx_reduced_v1.xml"))
    p.add_argument("--trace",   type=Path,
                   default=Path("outputs/task02/controller_trace_v1.npz"))
    p.add_argument("--physics", type=Path,
                   default=Path("outputs/physics/task02_demo00_contact_force_data.npz"))
    p.add_argument("--output",  type=Path,
                   default=Path("outputs/task02/parity_report.json"))
    p.add_argument("--horizon", type=int, default=50,
                   help="最多验证的时间步数（降低首次运行时间）")
    return p.parse_args()


# ─────────────────────────────────────────────
# CPU MuJoCo rollout
# ─────────────────────────────────────────────

def cpu_rollout(
    xml_path: str,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl_seq: np.ndarray,   # [T, nu]
) -> dict:
    """在 CPU MuJoCo 中逐步应用 ctrl，记录 qpos 轨迹。"""
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)

    T = len(ctrl_seq)
    qpos_traj = np.zeros((T + 1, model.nq))
    xpos_traj = np.zeros((T + 1, model.nbody, 3))

    qpos_traj[0] = data.qpos.copy()
    xpos_traj[0] = data.xpos.copy()

    for t in range(T):
        data.ctrl[:] = ctrl_seq[t]
        mujoco.mj_step(model, data)
        qpos_traj[t + 1] = data.qpos.copy()
        xpos_traj[t + 1] = data.xpos.copy()

    return {"qpos": qpos_traj, "xpos": xpos_traj, "nbody": model.nbody}


# ─────────────────────────────────────────────
# MJX rollout
# ─────────────────────────────────────────────

def mjx_rollout(
    xml_path: str,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl_seq: np.ndarray,   # [T, nu]
) -> dict:
    """在 MJX（JAX）中 jit+lax.scan rollout，记录轨迹。"""
    import mujoco
    from mujoco import mjx
    import jax
    import jax.numpy as jnp

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)

    mj_data.qpos[:] = qpos0
    mj_data.qvel[:] = qvel0
    mujoco.mj_forward(mj_model, mj_data)

    mx = mjx.put_model(mj_model)
    dx0 = mjx.put_data(mj_model, mj_data)

    ctrl_jax = jnp.array(ctrl_seq, dtype=jnp.float32)

    def step_fn(dx, ctrl):
        dx = dx.replace(ctrl=ctrl)
        dx = mjx.step(mx, dx)
        return dx, (dx.qpos, dx.xpos)

    print("  MJX JIT 编译中...")
    step_jit = jax.jit(lambda dx, ctrl_seq: jax.lax.scan(step_fn, dx, ctrl_seq))
    dx_final, (qpos_traj, xpos_traj) = step_jit(dx0, ctrl_jax)

    # 转 numpy：轨迹包含初始帧 + T 步
    qpos_init = np.array(dx0.qpos)[None]            # [1, nq]
    xpos_init = np.array(dx0.xpos)[None]            # [1, nbody, 3]

    return {
        "qpos": np.concatenate([qpos_init, np.array(qpos_traj)], axis=0),
        "xpos": np.concatenate([xpos_init, np.array(xpos_traj)], axis=0),
        "nbody": mj_model.nbody,
    }


# ─────────────────────────────────────────────
# 误差计算与报告
# ─────────────────────────────────────────────

def compute_parity_report(
    cpu: dict,
    mjx_result: dict,
    xml_path: str,
    bowl_body_name: str = "akita_black_bowl_1_main",
) -> dict:
    import mujoco

    qpos_err = np.abs(cpu["qpos"] - mjx_result["qpos"])
    xpos_err = np.abs(cpu["xpos"] - mjx_result["xpos"])

    # 找到 bowl body id
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    bowl_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, bowl_body_name
    )
    if bowl_id < 0:
        # 尝试找任何包含 bowl 的 body
        for i in range(mj_model.nbody):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if "bowl" in name.lower():
                bowl_id = i
                bowl_body_name = name
                break

    T = qpos_err.shape[0] - 1

    report = {
        "horizon_steps":    T,
        "bowl_body":        bowl_body_name,
        "bowl_body_id":     int(bowl_id) if bowl_id >= 0 else -1,
        "qpos": {
            "rmse":          float(np.sqrt(np.mean(qpos_err ** 2))),
            "mean_abs":      float(qpos_err.mean()),
            "max_abs":       float(qpos_err.max()),
            "per_step_rmse": [float(np.sqrt(np.mean(qpos_err[t]**2)))
                              for t in range(T + 1)],
        },
        "bowl_position": {},
    }

    if bowl_id >= 0:
        bowl_pos_cpu = cpu["xpos"][:, bowl_id]       # [T+1, 3]
        bowl_pos_mjx = mjx_result["xpos"][:, bowl_id]
        bowl_err     = np.abs(bowl_pos_cpu - bowl_pos_mjx)
        report["bowl_position"] = {
            "rmse":     float(np.sqrt(np.mean(bowl_err**2))),
            "max_abs":  float(bowl_err.max()),
            "final_err": float(bowl_err[-1].mean()),
        }

    return report


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 加载 ctrl trace
    trace = np.load(str(args.trace), allow_pickle=False)
    states_cpu = trace["states"]    # [T, nq+nv]  注：qpos+qvel flatten
    ctrl_seq   = trace["ctrl"]      # [T, nu]

    # 从 states 中分离 qpos / qvel
    import mujoco
    model_check = mujoco.MjModel.from_xml_path(str(args.xml))
    nq, nv = model_check.nq, model_check.nv

    # controller_trace 的 states 来自 env.get_sim_state()，格式为 qpos+qvel
    qpos0 = states_cpu[0, :nq]
    qvel0 = states_cpu[0, nq:nq + nv]

    # 截取 horizon
    H = min(args.horizon, len(ctrl_seq))
    ctrl_seq = ctrl_seq[:H]

    print(f"[parity] XML     : {args.xml}")
    print(f"[parity] horizon : {H} 步")
    print(f"[parity] nq={nq}  nv={nv}  nu={model_check.nu}")
    print(f"[parity] qpos0 norm : {np.linalg.norm(qpos0):.4f}")

    # CPU rollout
    print("\n[1/2] CPU MuJoCo rollout...")
    cpu = cpu_rollout(str(args.xml), qpos0, qvel0, ctrl_seq)

    # MJX rollout
    print("[2/2] MJX rollout...")
    mjx_res = mjx_rollout(str(args.xml), qpos0, qvel0, ctrl_seq)

    # 误差报告
    report = compute_parity_report(cpu, mjx_res, str(args.xml))

    print(f"\n{'='*55}")
    print("MuJoCo / MJX 轨迹一致性报告")
    print(f"{'='*55}")
    print(f"  验证步数         : {report['horizon_steps']}")
    print(f"  qpos RMSE        : {report['qpos']['rmse']:.6f}")
    print(f"  qpos max_abs     : {report['qpos']['max_abs']:.6f}")
    bp = report.get("bowl_position", {})
    if bp:
        print(f"  碗位置 RMSE     : {bp['rmse']:.6f} m")
        print(f"  碗位置 max_abs  : {bp['max_abs']:.6f} m")
        print(f"  碗位置 末帧误差 : {bp['final_err']:.6f} m")

    # 判定
    qpos_ok = report["qpos"]["rmse"] < 0.01
    bowl_ok = not bp or bp["rmse"] < 0.005
    verdict = "✅ PASS" if (qpos_ok and bowl_ok) else "⚠️  ACCEPTABLE" \
              if report["qpos"]["rmse"] < 0.05 else "❌ DIVERGED"
    print(f"\n  一致性判定: {verdict}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  报告已保存: {args.output}")


if __name__ == "__main__":
    main()
