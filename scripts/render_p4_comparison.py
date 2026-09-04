"""P4 视频渲染脚本：完整103帧任务 + 梯度修正窗口对比。

读取完整专家轨迹（103帧）+ 梯度修正窗口（25帧），
用 LIBERO 正确渲染完整任务过程，在修正窗口内展示对比。

运行（libero_replay 环境）：
    conda activate libero_replay
    python scripts/render_p4_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

INPUT_DIR  = PROJECT_ROOT / "outputs/task02"
OUTPUT_DIR = PROJECT_ROOT / "outputs/task02"

QPOS_EXPERT_FILE  = INPUT_DIR / "qpos_expert_p4.npy"
QPOS_PERTURB_FILE = INPUT_DIR / "qpos_perturb_p4.npy"
QPOS_REFINED_FILE = INPUT_DIR / "qpos_refined_p4.npy"
LOSS_HIST_FILE    = INPUT_DIR / "loss_history_p4.npy"
TRACE_NPZ         = INPUT_DIR / "controller_trace_v1.npz"
MODEL_XML         = PROJECT_ROOT / "outputs/task02/model_raw.xml"

SUITE       = "libero_spatial"
TASK_ID     = 2
START_FRAME = 55    # 梯度修正窗口起始帧

HEIGHT = 256
WIDTH  = 256
FPS    = 20
WINDOW_REPEAT = 4   # 窗口内每帧重复次数（慢放4x，25帧→5s@20fps）


def render_qpos_with_libero(
    qpos_seq: np.ndarray,   # [T, nq]
    model_xml: str,
    bddl_path: str,
    camera: str = "agentview",
    height: int = HEIGHT,
    width:  int = WIDTH,
    seed:   int = 0,
) -> np.ndarray:
    """使用 LIBERO OffScreenRenderEnv 渲染 qpos 序列，返回 [T, H, W, 3]。"""
    from libero_replay_common import make_env, restore_recorded_model, transform_frame
    import mujoco

    env = make_env(bddl_path, camera, height, width, seed)
    xml_content = Path(model_xml).read_text(encoding="utf-8")
    restore_recorded_model(env, xml_content)

    mj_model = mujoco.MjModel.from_xml_path(str(model_xml))
    nq = mj_model.nq
    nv = mj_model.nv

    frames = []
    for t, qpos in enumerate(qpos_seq):
        state = np.zeros(1 + nq + nv, dtype=np.float64)
        state[0] = 0.0
        state[1:nq+1] = qpos

        obs = env.regenerate_obs_from_state(state)
        frame = transform_frame(obs, camera, flip="rot180")
        frames.append(frame)

        if (t + 1) % 20 == 0 or t == len(qpos_seq) - 1:
            print(f"  渲染 [{t+1}/{len(qpos_seq)}]", end="\r")

    env.close()
    print()
    return np.stack(frames)


def main() -> None:
    from libero_replay_common import resolve_task
    import mujoco

    print("=" * 70)
    print("P4 三路对比视频（专家 / 扰动 / 梯度修正）")
    print("=" * 70)

    task_suite, task, bddl_path, _ = resolve_task(suite_name=SUITE, task_id=TASK_ID)
    print(f"任务：{SUITE} task {TASK_ID}  {task.language}")

    # 加载完整专家轨迹（来自 states）
    trace  = np.load(str(TRACE_NPZ), allow_pickle=False)
    states = trace["states"]
    T_full = len(states)
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    nq = mj_model.nq
    qpos_expert_full = states[:, 1:nq+1].astype(np.float32)

    # 加载三条窗口轨迹（仅修正窗口）
    has_perturb = QPOS_PERTURB_FILE.exists()
    qpos_refined = np.load(str(QPOS_REFINED_FILE))
    loss_hist    = np.load(str(LOSS_HIST_FILE))
    H_refine     = len(qpos_refined)
    end_frame    = START_FRAME + H_refine

    if has_perturb:
        qpos_perturb = np.load(str(QPOS_PERTURB_FILE))
        # loss_hist: [loss_expert, loss_perturb, refine_iter0..N]
        loss_expert  = float(loss_hist[0]) if len(loss_hist) > 2 else float(loss_hist[0])
        loss_perturb = float(loss_hist[1]) if len(loss_hist) > 2 else float(loss_hist[0])
        loss_refined = float(loss_hist[-1])
        print(f"  loss 专家  = {loss_expert:.5f}")
        print(f"  loss 扰动  = {loss_perturb:.5f}  +{(loss_perturb-loss_expert)/max(loss_expert,1e-8)*100:.1f}%")
        print(f"  loss 修正  = {loss_refined:.5f}  ↓{(loss_perturb-loss_refined)/max(loss_perturb,1e-8)*100:.1f}% 相对扰动")
    else:
        print("  ⚠️  无扰动轨迹，退化为双路对比（专家 | 修正）")

    # 渲染轨迹
    print(f"\n>>> 渲染专家完整轨迹（{T_full} 帧）...")
    frames_expert_full = render_qpos_with_libero(
        qpos_expert_full, str(MODEL_XML), str(bddl_path))

    print(f">>> 渲染梯度修正轨迹（{H_refine} 帧）...")
    frames_refined = render_qpos_with_libero(
        qpos_refined, str(MODEL_XML), str(bddl_path))

    if has_perturb:
        print(f">>> 渲染扰动轨迹（{H_refine} 帧）...")
        frames_perturb = render_qpos_with_libero(
            qpos_perturb, str(MODEL_XML), str(bddl_path))

    # 合成视频
    n_panels = 3 if has_perturb else 2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "gradient_refinement_full.mp4"

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS, (WIDTH * n_panels, HEIGHT),
    )

    def add_label(img, text, color=(200, 200, 200), y=18):
        f = img.copy()
        cv2.putText(f, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        return f

    def add_border(img, color, thickness=3):
        f = img.copy()
        f[:thickness, :]  = color  # top
        f[-thickness:, :] = color  # bottom
        f[:, :thickness]  = color  # left
        f[:, -thickness:] = color  # right
        return f

    print(f"\n>>> 合成 {n_panels} 路对比视频（{T_full} 帧，窗口慢放{WINDOW_REPEAT}x）...")
    for t in range(T_full):
        in_window = START_FRAME <= t < end_frame
        seg_t = t - START_FRAME if in_window else 0

        # 左：专家（白色边框标注ground truth）
        left = frames_expert_full[t].copy()
        if in_window:
            left = add_border(left, (200, 200, 200), 2)
        left = add_label(left, f"EXPERT t={t}", color=(200, 200, 200))
        if in_window and has_perturb:
            left = add_label(left, f"loss={loss_expert:.3f}", color=(200, 200, 200), y=34)

        if has_perturb:
            # 中：扰动（红色边框标注次优起点）
            if in_window:
                mid = frames_perturb[seg_t].copy()
                mid = add_border(mid, (60, 60, 220), 3)  # 红色
                mid = add_label(mid, f"PERTURB t={t}", color=(100, 100, 255))
                mid = add_label(mid, f"loss={loss_perturb:.3f}", color=(100, 100, 255), y=34)
            else:
                mid = frames_expert_full[t].copy()
                mid = add_label(mid, f"PERTURB  [窗口外]", color=(120, 120, 120))

            # 右：修正（绿色边框标注优化结果）
            if in_window:
                right = frames_refined[seg_t].copy()
                right = add_border(right, (0, 200, 80), 3)  # 绿色
                right = add_label(right, f"REFINED t={t}", color=(0, 220, 100))
                right = add_label(right, f"loss={loss_refined:.3f}", color=(0, 220, 100), y=34)
            else:
                right = frames_expert_full[t].copy()
                right = add_label(right, f"REFINED  [窗口外]", color=(120, 120, 120))

            row = np.concatenate([left, mid, right], axis=1)
        else:
            # 双路
            if in_window:
                right = frames_refined[seg_t].copy()
                right = add_border(right, (0, 200, 80), 3)
                right = add_label(right, f"REFINED t={t}", color=(0, 220, 100))
            else:
                right = frames_expert_full[t].copy()
                right = add_label(right, f"REFINED [窗口外]", color=(120, 120, 120))
            row = np.concatenate([left, right], axis=1)

        bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
        repeat = WINDOW_REPEAT if in_window else 1
        for _ in range(repeat):
            writer.write(bgr)

    writer.release()
    total_frames = (T_full - H_refine) + H_refine * WINDOW_REPEAT
    duration_s   = total_frames / FPS
    size_kb = out_path.stat().st_size // 1024
    print(f"\n✅ 视频：{out_path}")
    print(f"   实际帧数={total_frames}帧（源{T_full}帧 + 窗口{H_refine}帧×{WINDOW_REPEAT}倍）"
          f" · {HEIGHT}×{WIDTH*n_panels} · {size_kb}KB · {FPS}fps · {duration_s:.1f}s")
    if has_perturb:
        print(f"   窗口（帧{START_FRAME}-{end_frame-1}）：白=专家 | 红=扰动 | 绿=修正")
    print("\n" + "=" * 60)
    print("渲染完成（LIBERO 正确管道 + 完整103帧任务）")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--task-id",    type=int, default=2)
    p.add_argument("--suite",      default="libero_spatial")
    p.add_argument("--start-frame", type=int, default=55)
    args = p.parse_args()

    tid = args.task_id
    out_dir = PROJECT_ROOT / f"outputs/task{tid:02d}"

    # 动态覆盖模块级常量
    INPUT_DIR  = out_dir
    OUTPUT_DIR = out_dir
    QPOS_REFINED_FILE = out_dir / "qpos_refined_p4.npy"
    LOSS_HIST_FILE    = out_dir / "loss_history_p4.npy"
    TRACE_NPZ_        = out_dir / "controller_trace_v1.npz"
    MODEL_XML_        = out_dir / "model_raw.xml"

    import sys
    _mod = sys.modules[__name__]
    _mod.TRACE_NPZ          = out_dir / "controller_trace_v1.npz"
    _mod.MODEL_XML          = out_dir / "model_raw.xml"
    _mod.QPOS_EXPERT_FILE   = out_dir / "qpos_expert_p4.npy"
    _mod.QPOS_PERTURB_FILE  = out_dir / "qpos_perturb_p4.npy"
    _mod.QPOS_REFINED_FILE  = out_dir / "qpos_refined_p4.npy"
    _mod.LOSS_HIST_FILE     = out_dir / "loss_history_p4.npy"
    _mod.OUTPUT_DIR         = out_dir
    _mod.SUITE             = args.suite
    _mod.TASK_ID           = args.task_id
    _mod.START_FRAME       = args.start_frame

    main()
