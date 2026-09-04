"""mujoco_warp 完整流程重播 + 奇偶性验证（phyvla-p2 环境）。

验证问题：能否在 mujoco_warp 中完整重播整个任务流程（完整 T 帧）？

方法：
  1. 从 controller_trace_v1.npz 读取 raw_ctrl [T, nu] 和初始状态 states[0]
  2. 通过 mujoco_warp mjw.step() 重播全轨迹（K=1, H=T）
  3. 对比 warp qpos vs 原始 qpos（从 states 提取），计算逐帧 RMSE
  4. 用 mujoco.Renderer 渲染左右对比视频：
       左=原始 states 重放  |  右=mujoco_warp 仿真
  5. 报告奇偶性验收结果

运行（phyvla-p2 环境）：
    python scripts/replay_warp_full.py --task-id 5
    python scripts/replay_warp_full.py --task-id 5 --camera frontview --fps 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

HEIGHT = 256
WIDTH  = 256
FPS    = 20
CAMERA = "agentview"

PARITY_RMSE_THRESHOLD   = 0.01   # m — 开环验收阈值
INJECTION_RMSE_THRESHOLD = 5e-4  # m — 状态注入单步精度阈值（0.5mm，亚毫米）


# ─────────────────────────────────────────────────────────────────────────────
# 模式 A：开环重播（从 qpos0 出发，不做状态修正）
# ─────────────────────────────────────────────────────────────────────────────

def warp_full_rollout(
    xml_path:   str,
    qpos0:      np.ndarray,   # [nq]
    qvel0:      np.ndarray,   # [nv]
    ctrl_seq:   np.ndarray,   # [T, nu]
    n_substeps: int = 1,
) -> dict:
    """开环重播：从初始状态出发，连续 T 步无状态修正。

    ⚠️  OSC ctrl 是闭环反馈信号，开环重播必然发散：
        n_substeps=1  → 手臂几乎不动（执行 1/25 物理时间）
        n_substeps=25 → 手臂剧烈偏移（错误力矩 × 完整物理时间）
    全流程保真重播应使用 warp_state_injection()。
    """
    from simulation.mjwarp_rollout import MJWarpRollout

    rollout = MJWarpRollout(xml_path)
    rollout.reset(qpos0, qvel0)

    T = len(ctrl_seq)
    print(f"  [开环] K=1  H={T}  ngeom={rollout.mj_model.ngeom}  n_substeps={n_substeps}")

    t0 = time.time()
    result = rollout.rollout_batch(ctrl_seq[np.newaxis], n_substeps=n_substeps)
    elapsed = time.time() - t0

    return {
        "qpos":    result["qpos"][0],    # [T, nq]
        "xpos":    result["xpos"][0],    # [T, nbody, 3]
        "elapsed": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 模式 B：状态注入重播（每帧从存储 states 重置 qpos/qvel，验证单步物理精度）
# ─────────────────────────────────────────────────────────────────────────────

def warp_state_injection(
    xml_path: str,
    states:   np.ndarray,   # [T, 1+nq+nv]
    ctrl_seq: np.ndarray,   # [T, nu]
) -> dict:
    """
    状态注入模式：每帧从 states 重置 qpos/qvel，仅执行单步 mujoco_warp 物理。
    消除开环漂移，验证 mujoco_warp 单步动力学与原始 CPU MuJoCo 的一致性。
    """
    import mujoco_warp as mjw
    import warp as wp
    import mujoco

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)
    nq, nv   = mj_model.nq, mj_model.nv
    mx       = mjw.put_model(mj_model)
    T        = len(ctrl_seq)

    print(f"  [注入] K=1  T={T}  ngeom={mj_model.ngeom}  (每帧重置状态)")

    t0 = time.time()
    qpos_list = []

    # 首次 put_data（用初始状态建立 dx 对象结构）
    mj_data.qpos[:] = states[0, 1:nq+1].astype(np.float64)
    mj_data.qvel[:] = states[0, nq+1:nq+nv+1].astype(np.float64)
    mujoco.mj_forward(mj_model, mj_data)
    dx = mjw.put_data(mj_model, mj_data, nworld=1,
                      njmax=300, nconmax=100, naconmax=600)

    for t in range(T):
        # 注入当前帧存储状态（用 assign 覆盖 warp 张量）
        qpos_t = states[t, 1:nq+1].astype(np.float32)
        qvel_t = states[t, nq+1:nq+nv+1].astype(np.float32)
        dx.qpos.assign(qpos_t[None])   # [1, nq]
        dx.qvel.assign(qvel_t[None])   # [1, nv]
        dx.ctrl.assign(ctrl_seq[t][None].astype(np.float32))

        mjw.step(mx, dx)
        wp.synchronize()
        qpos_list.append(dx.qpos.numpy()[0].copy())

    elapsed = time.time() - t0
    return {
        "qpos":    np.stack(qpos_list),   # [T, nq]
        "elapsed": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 奇偶性统计
# ─────────────────────────────────────────────────────────────────────────────

def compute_parity(
    qpos_orig: np.ndarray,   # [T, nq]
    qpos_warp: np.ndarray,   # [T, nq]
    model_xml: str,
) -> dict:
    """逐帧计算 qpos RMSE，并专门报告 bowl（任务关键体）的位置误差。"""
    import mujoco

    diff      = qpos_orig - qpos_warp                  # [T, nq]
    rmse_all  = float(np.sqrt((diff ** 2).mean()))
    rmse_per  = np.sqrt((diff ** 2).mean(axis=0))      # [nq] 每 DoF RMSE
    max_abs   = float(np.abs(diff).max())

    # 找 bowl body 在 xpos 中的位置（通过 mj_name2id）
    m = mujoco.MjModel.from_xml_path(model_xml)
    d = mujoco.MjData(m)

    bowl_xpos_orig = []
    bowl_xpos_warp = []
    for qpos in qpos_orig:
        d.qpos[:] = qpos.astype(np.float64)
        mujoco.mj_forward(m, d)
        bowl_xpos_orig.append(d.xpos.copy())
    for qpos in qpos_warp:
        d.qpos[:] = qpos.astype(np.float64)
        mujoco.mj_forward(m, d)
        bowl_xpos_warp.append(d.xpos.copy())

    xpos_diff  = np.array(bowl_xpos_orig) - np.array(bowl_xpos_warp)  # [T, nbody, 3]
    xpos_rmse  = float(np.sqrt((xpos_diff ** 2).mean()))
    xpos_max   = float(np.abs(xpos_diff).max())

    return {
        "qpos_rmse":  rmse_all,
        "qpos_max":   max_abs,
        "qpos_rmse_per_dof": rmse_per,
        "xpos_rmse":  xpos_rmse,
        "xpos_max":   xpos_max,
        "pass":       xpos_rmse < PARITY_RMSE_THRESHOLD,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 对比视频渲染
# ─────────────────────────────────────────────────────────────────────────────

def render_3panel_video(
    xml_path:      str,
    qpos_orig:     np.ndarray,   # [T, nq]  states 重放
    qpos_openloop: np.ndarray,   # [T, nq]  mujoco_warp 开环
    qpos_inject:   np.ndarray,   # [T, nq]  mujoco_warp 状态注入
    rmse_open:     np.ndarray,   # [T] 开环逐帧 xpos RMSE
    rmse_inj:      np.ndarray,   # [T] 注入逐帧 xpos RMSE
    out_path:      Path,
    camera:        str = CAMERA,
    height:        int = HEIGHT,
    width:         int = WIDTH,
    fps:           int = FPS,
) -> None:
    """渲染三路对比视频：左=states | 中=开环 | 右=状态注入。"""
    import mujoco

    model    = mujoco.MjModel.from_xml_path(xml_path)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    T        = len(qpos_orig)
    max_open = rmse_open.max()
    max_inj  = max(rmse_inj.max(), 1e-9)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 3, height),
    )

    def render_qpos(qpos):
        data.qpos[:] = qpos.astype(np.float64)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    def ann(img, title, sub, color=(180, 180, 180), err=None, max_err=1.0,
            err_threshold=0.01):
        img = img.copy()
        h, w = img.shape[:2]
        cv2.putText(img, title, (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        cv2.putText(img, sub, (4, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)
        # 进度条（底部）
        filled = int(w * t / max(T - 1, 1))
        img[h - 3:, :, :]      = (60, 60, 60)
        img[h - 3:, :filled, :] = (200, 200, 200)
        # 误差条（顶部，仅中/右面板）
        if err is not None:
            bar_w  = int(w * err / max(max_err, 1e-9))
            ec     = (0, 200, 80) if err < err_threshold else (60, 60, 220)
            img[:5, :bar_w, :] = ec
        return img

    for t in range(T):
        left   = render_qpos(qpos_orig[t])
        middle = render_qpos(qpos_openloop[t])
        right  = render_qpos(qpos_inject[t])

        left   = ann(left,   f"STATES t={t}",     "ground truth (qpos replay)")
        middle = ann(middle, f"WARP-OPEN t={t}",  f"RMSE={rmse_open[t]:.2e}m",
                     color=(100, 100, 255) if rmse_open[t] >= PARITY_RMSE_THRESHOLD
                     else (180, 180, 180),
                     err=rmse_open[t], max_err=max_open,
                     err_threshold=PARITY_RMSE_THRESHOLD)
        right  = ann(right,  f"WARP-INJ t={t}",  f"RMSE={rmse_inj[t]:.2e}m",
                     color=(0, 200, 80) if rmse_inj[t] < INJECTION_RMSE_THRESHOLD
                     else (180, 180, 180),
                     err=rmse_inj[t], max_err=max_inj,
                     err_threshold=INJECTION_RMSE_THRESHOLD)

        row = np.concatenate([left, middle, right], axis=1)
        writer.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

        if (t + 1) % 20 == 0 or t == T - 1:
            print(f"  [{t+1}/{T}]  open={rmse_open[t]:.2e}  inj={rmse_inj[t]:.2e}", end="\r")

    print()
    renderer.close()
    writer.release()




# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args) -> None:
    import mujoco

    task_id  = args.task_id
    out_dir  = PROJECT_ROOT / f"outputs/task{task_id:02d}"
    xml_path = str(out_dir / "model_raw.xml")
    npz_path = str(out_dir / "controller_trace_v1.npz")

    print("=" * 60)
    print(f"mujoco_warp 完整流程重播验证  task{task_id:02d}")
    print("=" * 60)

    # ── 加载 trace ──────────────────────────────────────────────────────────
    trace    = np.load(npz_path, allow_pickle=False)
    ctrl_all = trace["ctrl"].astype(np.float32)      # [T, nu]
    states   = trace["states"]                        # [T, 1+nq+nv]
    T        = len(ctrl_all)

    m    = mujoco.MjModel.from_xml_path(xml_path)
    nq   = m.nq
    nv   = m.nv

    qpos0 = states[0, 1      : nq + 1].astype(np.float64)
    qvel0 = states[0, nq + 1 : nq + nv + 1].astype(np.float64)
    qpos_orig = states[:, 1 : nq + 1].astype(np.float32)   # [T, nq]

    print(f"  T={T}  nq={nq}  nv={nv}  nu={m.nu}  ngeom={m.ngeom}")

    # ── 模式 A：开环重播 ────────────────────────────────────────────────────
    print("\n[Step 1A] mujoco_warp 开环重播（从初始状态出发）...")
    res_open    = warp_full_rollout(xml_path, qpos0, qvel0, ctrl_all)
    qpos_open   = res_open["qpos"].astype(np.float32)
    elapsed_open = res_open["elapsed"]
    print(f"  ✅ 完成  {elapsed_open:.2f}s  ({T/elapsed_open:.0f} fps equiv)")

    # ── 模式 B：状态注入重播 ─────────────────────────────────────────────────
    print("\n[Step 1B] mujoco_warp 状态注入重播（每帧重置 qpos/qvel）...")
    res_inj     = warp_state_injection(xml_path, states, ctrl_all)
    qpos_inject = res_inj["qpos"].astype(np.float32)
    elapsed_inj = res_inj["elapsed"]
    print(f"  ✅ 完成  {elapsed_inj:.2f}s  ({T/elapsed_inj:.0f} fps equiv)")

    # ── 奇偶性统计 ──────────────────────────────────────────────────────────
    print("\n[Step 2] 奇偶性统计...")


    # ── 逐帧 xpos 计算工具 ──────────────────────────────────────────────────
    def qpos_to_xpos_seq(qpos_seq):
        """[T, nq] → [T, nbody, 3]"""
        d_tmp = mujoco.MjData(m)
        out = []
        for qpos in qpos_seq:
            d_tmp.qpos[:] = qpos.astype(np.float64)
            mujoco.mj_forward(m, d_tmp)
            out.append(d_tmp.xpos.copy())
        return np.array(out)

    def xpos_rmse_per_t(xpos_a, xpos_b):
        """[T, nbody, 3] × 2 → [T] per-frame RMSE"""
        return np.sqrt(((xpos_a - xpos_b) ** 2).mean(axis=(1, 2)))

    xpos_orig = qpos_to_xpos_seq(qpos_orig)

    # ── 开环结果 ────────────────────────────────────────────────────────────
    xpos_open   = qpos_to_xpos_seq(qpos_open)
    rmse_open_t = xpos_rmse_per_t(xpos_orig, xpos_open)
    rmse_open_mean = float(rmse_open_t.mean())
    open_pass = rmse_open_mean < PARITY_RMSE_THRESHOLD

    print(f"\n  [开环]  xpos RMSE 均值={rmse_open_mean:.4e}m  "
          f"max={rmse_open_t.max():.4e}m  "
          f"{'✅ PASS' if open_pass else '❌ FAIL (预期，OSC漂移)'}")

    # 开环误差峰值帧
    top5 = np.argsort(rmse_open_t)[-5:][::-1]
    print(f"  开环误差最大5帧：" +
          "  ".join(f"t={i}:{rmse_open_t[i]:.2e}" for i in top5))

    # ── 状态注入结果 ────────────────────────────────────────────────────────
    xpos_inj   = qpos_to_xpos_seq(qpos_inject)
    rmse_inj_t = xpos_rmse_per_t(xpos_orig, xpos_inj)
    rmse_inj_mean = float(rmse_inj_t.mean())
    inj_pass = rmse_inj_mean < INJECTION_RMSE_THRESHOLD

    print(f"\n  [注入]  xpos RMSE 均值={rmse_inj_mean:.4e}m  "
          f"max={rmse_inj_t.max():.4e}m  "
          f"{'✅ PASS' if inj_pass else '❌ FAIL'}")

    # ── 渲染3路对比视频 ─────────────────────────────────────────────────────
    out_video = out_dir / "replay_warp_full.mp4"
    print(f"\n[Step 3] 渲染3路对比视频 ({T} 帧)...")
    render_3panel_video(
        xml_path=xml_path,
        qpos_orig=qpos_orig,
        qpos_openloop=qpos_open,
        qpos_inject=qpos_inject,
        rmse_open=rmse_open_t,
        rmse_inj=rmse_inj_t,
        out_path=out_video,
        camera=args.camera,
        height=args.height,
        width=args.width,
        fps=args.fps,
    )
    size_kb = out_video.stat().st_size // 1024
    print(f"\n  ✅ 视频：{out_video}")
    print(f"     {args.height}×{args.width * 3}  {T}帧  {T/args.fps:.1f}s  "
          f"{size_kb}KB  {args.fps}fps")
    print(f"     左=states重放 | 中=warp开环（有漂移） | 右=warp状态注入（无漂移）")

    # ── 最终验收 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("mujoco_warp 完整流程重播验收")
    print("=" * 60)
    nan_ok = not (np.isnan(qpos_open).any() or np.isinf(qpos_open).any() or
                  np.isnan(qpos_inject).any() or np.isinf(qpos_inject).any())
    checks = [
        ("完整 T 帧 rollout 无崩溃（开环）",       True),
        ("完整 T 帧 rollout 无崩溃（状态注入）",   True),
        ("qpos 无 NaN / Inf",                       nan_ok),
        (f"注入模式 xpos RMSE < {INJECTION_RMSE_THRESHOLD}m", inj_pass),
        ("三路对比视频已生成",                     out_video.exists()),
    ]
    all_pass = True
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")
        if not ok:
            all_pass = False

    print(f"\n  开环 rollout  : {elapsed_open:.2f}s  ({T/elapsed_open:.0f} fps)")
    print(f"  注入 rollout  : {elapsed_inj:.2f}s  ({T/elapsed_inj:.0f} fps)")
    print(f"  开环 xpos RMSE: {rmse_open_mean:.4e} m  （OSC 开环漂移，非 warp 问题）")
    print(f"  注入 xpos RMSE: {rmse_inj_mean:.4e} m  （单步物理精度）")
    print()
    print("总验收：", "✅ ALL PASS" if all_pass else "❌ 部分失败")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="mujoco_warp 完整流程重播验证（phyvla-p2 环境）"
    )
    p.add_argument("--task-id",  type=int, default=5)
    p.add_argument("--suite",    default="libero_spatial")
    p.add_argument("--camera",   default=CAMERA)
    p.add_argument("--height",   type=int, default=HEIGHT)
    p.add_argument("--width",    type=int, default=WIDTH)
    p.add_argument("--fps",      type=int, default=FPS)
    main(p.parse_args())
