"""P4-3 Demo：Warp 梯度动作修正可视化。

流程：
  1. 加载专家 ctrl（task02 demo0，帧T开始，H步）
  2. WarpGradComputer.refine_actions() — n_iters 次梯度下降
  3. 对比：原始 ctrl vs 修正后 ctrl 的 rollout 轨迹
  4. 渲染对比视频 + 保存 loss 曲线

运行（phyvla-p2）：
    cd /home/jiangnan-luxitech/workspace/phyvla
    conda run -n phyvla-p2 python scripts/demo_gradient_refinement.py \
        [--start-frame 20] [--horizon 15] [--n-iters 5] [--lr 0.03]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HEIGHT = 256
WIDTH  = 256


def _resolve_task_paths(task_id, suite="libero_spatial"):
    """根据 task_id 自动推导所有文件路径（统一格式）。"""
    out_dir = PROJECT_ROOT / f"outputs/task{task_id:02d}"
    return {
        "model_xml":   out_dir / "model_raw.xml",
        "trace_npz":   out_dir / "controller_trace_v1.npz",
        "physics_npz": out_dir / f"task{task_id:02d}_demo00_physics.npz",
        "output_dir":  out_dir,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 渲染 qpos 序列
# ─────────────────────────────────────────────────────────────────────────────

def render_qpos_seq(
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


# ─────────────────────────────────────────────────────────────────────────────
# 从 ctrl 序列跑 rollout，返回 qpos 轨迹
# ─────────────────────────────────────────────────────────────────────────────

def rollout_ctrl_trace(
    grad_comp,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl_seq: np.ndarray,   # [H, nu]
) -> np.ndarray:
    """返回 [H, nq] qpos 轨迹。"""
    mjw = grad_comp._mjw
    wp  = grad_comp._wp
    dx  = grad_comp._make_dx(1, qpos0, qvel0)
    H   = ctrl_seq.shape[0]
    qpos_list = []
    for t in range(H):
        dx.ctrl.assign(ctrl_seq[t][None].astype(np.float32))
        mjw.step(grad_comp.mx, dx)
        wp.synchronize()
        qpos_list.append(dx.qpos.numpy()[0].copy())
    return np.stack(qpos_list)   # [H, nq]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args) -> None:
    import cv2
    from simulation.warp_grad import WarpGradComputer

    # 解析任务路径
    task_id = args.task_id if args.task_id is not None else 2
    paths   = _resolve_task_paths(task_id, getattr(args, "suite", "libero_spatial"))
    MODEL_XML  = paths["model_xml"]
    TRACE_NPZ  = paths["trace_npz"]
    OUTPUT_DIR = paths["output_dir"]

    # GradientEncoder 可选（需要 torch）
    try:
        from models.gradient_encoder import GradientEncoder
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        print("⚠️  torch 不可用，跳过 z_grad 编码（梯度修正仍正常运行）")

    trace    = np.load(str(TRACE_NPZ), allow_pickle=False)
    ctrl_all = trace["ctrl"]
    states   = trace["states"]

    import mujoco
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    nq = mj_model.nq
    nv = mj_model.nv

    start_t = args.start_frame
    H       = args.horizon
    assert start_t + H <= len(ctrl_all), "帧范围超出 trace 长度"

    qpos0    = states[start_t, 1:nq + 1].astype(np.float64)
    qvel0    = states[start_t, nq + 1:nq + nv + 1].astype(np.float64)
    ctrl_expert = ctrl_all[start_t:start_t + H].astype(np.float32)

    print("=" * 60)
    print("P4-3 梯度动作修正 Demo（扰动路径优化）")
    print("=" * 60)
    print(f"start_frame={start_t}  H={H}  n_iters={args.n_iters}  step_size={args.step_size}")

    # ─── 生成扰动 ctrl ───
    sigma = getattr(args, "perturb_sigma", 0.0)
    mode  = getattr(args, "perturb_mode", "gaussian")

    np.random.seed(getattr(args, "seed", 42))
    if mode == "zero":
        # 全零：机械臂停止，最简单的"坏"轨迹
        ctrl_perturb = np.zeros_like(ctrl_expert)
        print(f"\n>>> 扰动 ctrl（mode=zero：全零，机械臂停止）")
    elif mode == "reverse":
        # 反向：机械臂朝相反方向运动
        noise = np.random.randn(*ctrl_expert.shape).astype(np.float32) * (sigma * 0.3)
        ctrl_perturb = -ctrl_expert + noise
        print(f"\n>>> 扰动 ctrl（mode=reverse：ctrl 取反）")
    elif mode == "scale":
        # 缩放：减弱控制信号，机械臂移动迟缓
        scale = getattr(args, "perturb_scale", 0.1)
        ctrl_perturb = ctrl_expert * scale
        print(f"\n>>> 扰动 ctrl（mode=scale：缩放到 {scale}x）")
    else:  # gaussian
        if sigma > 0:
            noise = np.random.randn(*ctrl_expert.shape).astype(np.float32) * sigma
            ctrl_perturb = ctrl_expert + noise
            print(f"\n>>> 扰动 ctrl（mode=gaussian, sigma={sigma}，noise norm={np.linalg.norm(noise):.2f}）")
        else:
            ctrl_perturb = ctrl_expert.copy()
            print("\n>>> 未指定扰动（sigma=0），使用专家 ctrl 作为起点")

    grad_comp = WarpGradComputer(str(MODEL_XML))
    nu = grad_comp.nu

    # ─── 计算三条轨迹的初始 loss ───
    info_expert  = grad_comp.grad_fd_batch(qpos0, qvel0, ctrl_expert,  eps=args.eps)
    info_perturb = grad_comp.grad_fd_batch(qpos0, qvel0, ctrl_perturb, eps=args.eps)
    loss_expert  = info_expert["loss_center"]
    loss_perturb = info_perturb["loss_center"]
    print(f"\n  专家 ctrl    loss = {loss_expert:.5f}")
    print(f"  扰动 ctrl    loss = {loss_perturb:.5f}  "
          f"（{'劣化' if loss_perturb > loss_expert else '改善'} "
          f"{abs(loss_perturb-loss_expert)/max(loss_expert,1e-8)*100:.1f}%）")

    # ─── 从扰动 ctrl 出发梯度下降 ───
    print("\n>>> 梯度下降修正（从扰动 ctrl 出发）...")
    result = grad_comp.refine_actions(
        qpos0=qpos0, qvel0=qvel0,
        ctrl_seq=ctrl_perturb,
        n_iters=args.n_iters,
        step_size=args.step_size,
        eps=args.eps,
    )
    ctrl_refined = result["ctrl_refined"]
    loss_refined = result["loss_history"][-1]
    print(f"  修正后 loss  = {loss_refined:.5f}  "
          f"（距专家 {abs(loss_refined-loss_expert)/max(loss_expert,1e-8)*100:.1f}%）")

    # ─── 构建 z_grad（可选）───
    if HAS_TORCH:
        print("\n>>> 编码 z_grad...")
        grad_info = grad_comp.grad_fd_batch(qpos0, qvel0, ctrl_orig)
        feat_input = GradientEncoder.build_input(
            grad_seq  = grad_info["grad"],
            grad_norm = grad_info["grad_norm"],
            loss_val  = grad_info["loss_center"],
            clipped   = False,
            nu        = nu,
        )
        enc = GradientEncoder(H=H, nu=nu)
        import torch
        z_grad = enc(feat_input.unsqueeze(0))
        print(f"  z_grad shape: {z_grad.shape}  ||z_grad||={z_grad.norm().item():.4f}")
    else:
        print("\n>>> 跳过 z_grad 编码（torch 不可用）")
        z_grad = None

    # ─── 三条轨迹仿真 ───
    print("\n>>> 仿真三条轨迹...")
    qpos_expert  = rollout_ctrl_trace(grad_comp, qpos0, qvel0, ctrl_expert)   # [H, nq]
    qpos_perturb = rollout_ctrl_trace(grad_comp, qpos0, qvel0, ctrl_perturb)  # [H, nq]
    qpos_refined = rollout_ctrl_trace(grad_comp, qpos0, qvel0, ctrl_refined)  # [H, nq]

    # ─── 保存所有轨迹 ───
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_DIR / "qpos_expert_p4.npy"),  qpos_expert)
    np.save(str(OUTPUT_DIR / "qpos_perturb_p4.npy"), qpos_perturb)
    np.save(str(OUTPUT_DIR / "qpos_refined_p4.npy"), qpos_refined)
    # 向后兼容：qpos_orig_p4 也保存（渲染脚本可能引用）
    np.save(str(OUTPUT_DIR / "qpos_orig_p4.npy"),    qpos_expert)

    loss_summary = np.array([loss_expert, loss_perturb] + list(result["loss_history"]))
    np.save(str(OUTPUT_DIR / "loss_history_p4.npy"), loss_summary)
    np.save(str(OUTPUT_DIR / "ctrl_refined_p4.npy"), ctrl_refined)

    print(f"\n>>> 轨迹已保存至 {OUTPUT_DIR}/")
    print(f"  qpos_expert_p4.npy  : {qpos_expert.shape}")
    print(f"  qpos_perturb_p4.npy : {qpos_perturb.shape}")
    print(f"  qpos_refined_p4.npy : {qpos_refined.shape}")

    # ─── 内联渲染（mujoco.Renderer，phyvla-p2 环境，无需切换 conda env）───
    print("\n>>> 渲染三路对比视频（mujoco.Renderer 单实例）...")
    try:
        import importlib.util, sys as _sys
        _spec = importlib.util.spec_from_file_location(
            "render_warp_comparison",
            str(PROJECT_ROOT / "scripts" / "render_warp_comparison.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.render_comparison(
            task_id=task_id,
            start_frame=args.start_frame,
        )
    except Exception as _e:
        print(f"  ⚠️  内联渲染失败（{_e}）；可手动运行：")
        print(f"      python scripts/render_warp_comparison.py --task-id {task_id}")

    # ─── P4 验收 ───
    print("\n" + "=" * 60)
    print("P4 验收")
    print("=" * 60)
    refine_improves = loss_refined < loss_perturb
    checks = [
        ("梯度计算无 NaN",        not np.any(np.isnan(result["grad_history"]))),
        ("修正后 loss < 扰动 loss", refine_improves),
        ("qpos 轨迹已保存",        (OUTPUT_DIR / "qpos_refined_p4.npy").exists()),
    ]
    all_pass = True
    for name, ok in checks:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
        if not ok:
            all_pass = False

    print(f"\n  loss 专家 = {loss_expert:.5f}")
    print(f"  loss 扰动 = {loss_perturb:.5f}  +{(loss_perturb-loss_expert)/max(loss_expert,1e-8)*100:.1f}%")
    print(f"  loss 修正 = {loss_refined:.5f}  "
          f"{'↓' if refine_improves else '→'}"
          f"{abs(loss_perturb-loss_refined)/max(loss_perturb,1e-8)*100:.1f}% 相对扰动")
    print()
    print("P4 Demo:", "✅ ALL PASS" if all_pass else "❌ 部分失败")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id",       type=int,   default=None,
                        help="任务 ID，自动推导路径")
    parser.add_argument("--suite",         default="libero_spatial")
    parser.add_argument("--start-frame",   type=int,   default=55)
    parser.add_argument("--horizon",       type=int,   default=25)
    parser.add_argument("--n-iters",       type=int,   default=5)
    parser.add_argument("--step-size",     type=float, default=5.0)
    parser.add_argument("--eps",           type=float, default=2.0)
    parser.add_argument("--perturb-sigma", type=float, default=8.0,
                        help="扰动噪声标准差（ctrl 量级~30-50，8≈15%%噪声）")
    parser.add_argument("--perturb-mode",  default="scale",
                        choices=["gaussian", "zero", "reverse", "scale"],
                        help="扰动方式：gaussian/zero/reverse/scale")
    parser.add_argument("--perturb-scale", type=float, default=0.05,
                        help="mode=scale 时的缩放系数（默认0.05，5%%幅值）")
    parser.add_argument("--seed",          type=int,   default=42)
    main(parser.parse_args())
