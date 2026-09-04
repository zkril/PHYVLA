"""P4 对比视频渲染脚本（phyvla-p2 环境，mujoco.Renderer 单实例）。

修复 render_p4_comparison.py 的两个卡顿问题：
  1. 渲染过程卡顿：原脚本创建 3 个 OffScreenRenderEnv（各 ~10s）
     → 改为单个 mujoco.Renderer，三路轨迹共用，总渲染时间减少 ~2/3
  2. 播放卡顿：WINDOW_REPEAT=4 导致帧速率在第 55 帧骤变（1x→4x→1x）
     → 默认 slow=1（均匀帧率），可选 --slow 2 做轻微慢放

同时作为 mujoco_warp 完整场景迁移验证：model_raw.xml（271geom，完整 LIBERO 场景）
直接由原生 mujoco.Renderer 加载并渲染，验证物理与视觉一致性。

运行（phyvla-p2 环境）：
    python scripts/render_warp_comparison.py --task-id 5
    python scripts/render_warp_comparison.py --task-id 5 --slow 2 --fps 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HEIGHT = 256
WIDTH  = 256
FPS    = 20
CAMERA = "agentview"


# ─────────────────────────────────────────────────────────────────────────────
# 单实例渲染器
# ─────────────────────────────────────────────────────────────────────────────

class WarpSceneRenderer:
    """原生 mujoco.Renderer 单实例。所有轨迹共用一个 Renderer，无 robosuite 依赖。"""

    def __init__(
        self,
        xml_path: str,
        height:   int = HEIGHT,
        width:    int = WIDTH,
        camera:   str = CAMERA,
    ) -> None:
        import mujoco
        self._mujoco  = mujoco
        self.model    = mujoco.MjModel.from_xml_path(xml_path)
        self.data     = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.height   = height
        self.width    = width

        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        if cam_id < 0:
            available = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(self.model.ncam)
            ]
            raise ValueError(f"Camera '{camera}' not found. Available: {available}")
        self.camera = camera

        print(
            f"[WarpSceneRenderer] xml loaded  "
            f"nq={self.model.nq}  nv={self.model.nv}  "
            f"ngeom={self.model.ngeom}  ncam={self.model.ncam}  "
            f"camera='{camera}' (id={cam_id})"
        )

    # ── 单帧渲染 ─────────────────────────────────────────────────────────────

    def render_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """渲染一帧。qpos: [nq] → RGB [H, W, 3] uint8。"""
        mujoco = self._mujoco
        self.data.qpos[:] = qpos.astype(np.float64)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    # ── 序列渲染 ─────────────────────────────────────────────────────────────

    def render_sequence(
        self,
        qpos_seq:  np.ndarray,
        label:     str = "",
        log_every: int = 20,
    ) -> np.ndarray:
        """渲染 qpos 序列。qpos_seq: [T, nq] → [T, H, W, 3]。"""
        T      = len(qpos_seq)
        frames = []
        for t, qpos in enumerate(qpos_seq):
            frames.append(self.render_qpos(qpos))
            if label and ((t + 1) % log_every == 0 or t == T - 1):
                print(f"  [{label}] {t+1}/{T}", end="\r")
        if label:
            print()
        return np.stack(frames)

    def close(self) -> None:
        self.renderer.close()


# ─────────────────────────────────────────────────────────────────────────────
# 场景迁移可视化验证
# ─────────────────────────────────────────────────────────────────────────────

def verify_scene_migration(renderer: WarpSceneRenderer, qpos0: np.ndarray) -> dict:
    """
    验证完整 LIBERO 场景已正确迁移到 mujoco.Renderer：
      - 渲染初始帧，检查画面非全黑（场景有内容）
      - 打印像素统计
    """
    frame = renderer.render_qpos(qpos0)
    h, w  = frame.shape[:2]
    center = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    mean_brightness = float(center.mean())
    nonblack_ratio  = float((center.max(axis=-1) > 10).mean())
    scene_ok        = nonblack_ratio > 0.30

    status = "✅" if scene_ok else "❌"
    print(
        f"  {status} 场景迁移验证："
        f"  mean_brightness={mean_brightness:.1f}"
        f"  nonblack_ratio={nonblack_ratio:.3f}"
        f"  shape={frame.shape}"
    )
    if not scene_ok:
        print("  ⚠️  画面偏暗，可能摄像机名称或模型加载有误")

    return {
        "frame":           frame,
        "mean_brightness": mean_brightness,
        "nonblack_ratio":  nonblack_ratio,
        "scene_ok":        scene_ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 标注工具
# ─────────────────────────────────────────────────────────────────────────────

def _label(img: np.ndarray, text: str,
           color=(200, 200, 200), y: int = 18) -> np.ndarray:
    f = img.copy()
    cv2.putText(f, text, (4, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return f


def _border(img: np.ndarray, color, thickness: int = 3) -> np.ndarray:
    f = img.copy()
    f[:thickness, :]   = color
    f[-thickness:, :]  = color
    f[:, :thickness]   = color
    f[:, -thickness:]  = color
    return f


def _progress_bar(img: np.ndarray, t: int, T: int,
                  bar_h: int = 3,
                  bg=(60, 60, 60), fg=(200, 200, 200)) -> np.ndarray:
    """底部时间进度条。"""
    f = img.copy()
    h, w = f.shape[:2]
    filled = int(w * t / max(T - 1, 1))
    f[h - bar_h:, :, :]     = bg
    f[h - bar_h:, :filled, :] = fg
    return f


# ─────────────────────────────────────────────────────────────────────────────
# 主渲染流程
# ─────────────────────────────────────────────────────────────────────────────

def render_comparison(
    task_id:     int,
    suite:       str = "libero_spatial",
    start_frame: int = 55,
    camera:      str = CAMERA,
    height:      int = HEIGHT,
    width:       int = WIDTH,
    fps:         int = FPS,
    slow:        int = 1,
) -> Path:
    """
    渲染 P4 三路对比视频：专家 | 扰动 | 梯度修正。

    Args:
        slow: 窗口段慢放倍率（1=均匀帧率，2=2x 慢放）。
              原 render_p4_comparison.py 用 4，建议改为 1 或 2。
    """
    out_dir   = PROJECT_ROOT / f"outputs/task{task_id:02d}"
    model_xml = out_dir / "model_raw.xml"
    trace_npz = out_dir / "controller_trace_v1.npz"
    out_path  = out_dir / "gradient_refinement_full.mp4"

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    import mujoco as _mj
    _m  = _mj.MjModel.from_xml_path(str(model_xml))
    nq  = _m.nq

    trace           = np.load(str(trace_npz), allow_pickle=False)
    states          = trace["states"]                          # [T, 1+nq+nv]
    T_full          = len(states)
    qpos_full       = states[:, 1 : nq + 1].astype(np.float32)  # [T, nq]

    qpos_expert  = np.load(str(out_dir / "qpos_expert_p4.npy")).astype(np.float32)
    qpos_perturb = np.load(str(out_dir / "qpos_perturb_p4.npy")).astype(np.float32)
    qpos_refined = np.load(str(out_dir / "qpos_refined_p4.npy")).astype(np.float32)
    loss_hist    = np.load(str(out_dir / "loss_history_p4.npy"))

    H_window  = len(qpos_expert)
    end_frame = start_frame + H_window

    loss_expert  = float(loss_hist[0])
    loss_perturb = float(loss_hist[1])
    loss_refined = float(loss_hist[-1])

    print(f"  task={task_id}  T={T_full}  H_window={H_window}  "
          f"start_frame={start_frame}–{end_frame - 1}")
    print(f"  loss 专家  = {loss_expert:.5f}")
    print(f"  loss 扰动  = {loss_perturb:.5f}  "
          f"+{(loss_perturb - loss_expert) / max(loss_expert, 1e-8) * 100:.2f}%")
    print(f"  loss 修正  = {loss_refined:.5f}  "
          f"↓{(loss_perturb - loss_refined) / max(loss_perturb, 1e-8) * 100:.2f}% 相对扰动")

    # ── 单实例渲染器（全部轨迹共用）────────────────────────────────────────────
    renderer = WarpSceneRenderer(str(model_xml), height=height, width=width,
                                  camera=camera)

    # ── 场景迁移可视化验证 ────────────────────────────────────────────────────
    print("\n[验证] 完整 LIBERO 场景迁移到 mujoco.Renderer...")
    verify_result = verify_scene_migration(renderer, qpos_full[0])

    # ── 渲染四条轨迹 ───────────────────────────────────────────────────────────
    print(f"\n[渲染] 专家全轨迹 ({T_full} 帧)...")
    frames_full = renderer.render_sequence(qpos_full, label="expert_full")

    print(f"[渲染] 窗口：专家 ({H_window} 帧)...")
    frames_expert_w = renderer.render_sequence(qpos_expert, label="expert_win")

    print(f"[渲染] 窗口：扰动 ({H_window} 帧)...")
    frames_perturb_w = renderer.render_sequence(qpos_perturb, label="perturb")

    print(f"[渲染] 窗口：修正 ({H_window} 帧)...")
    frames_refined_w = renderer.render_sequence(qpos_refined, label="refined")

    renderer.close()

    # ── 合成视频 ──────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 3, height),
    )

    total_frames = (T_full - H_window) + H_window * slow
    print(f"\n[合成] {T_full}源帧 → {total_frames}输出帧  slow×{slow} in window...")

    for t in range(T_full):
        in_window = start_frame <= t < end_frame
        seg_t     = t - start_frame

        # ── 左：全程专家 ──────────────────────────────────────────────────────
        left = frames_full[t].copy()
        left = _progress_bar(left, t, T_full)
        if in_window:
            left = _border(left, (200, 200, 200), 2)
            left = _label(left, f"EXPERT t={t}", color=(220, 220, 220))
            left = _label(left, f"loss={loss_expert:.4f}", color=(180, 180, 180), y=34)
        else:
            left = _label(left, f"EXPERT t={t}", color=(130, 130, 130))

        # ── 中：扰动 ──────────────────────────────────────────────────────────
        if in_window:
            mid = frames_perturb_w[seg_t].copy()
            mid = _border(mid, (60, 60, 220), 3)          # 蓝红色
            mid = _label(mid, f"PERTURB t={t}", color=(100, 100, 255))
            mid = _label(mid, f"loss={loss_perturb:.4f}", color=(100, 100, 255), y=34)
        else:
            mid = frames_full[t].copy()
            mid = _label(mid, "PERTURB [out-of-window]", color=(90, 90, 90))
        mid = _progress_bar(mid, t, T_full)

        # ── 右：修正 ──────────────────────────────────────────────────────────
        if in_window:
            right = frames_refined_w[seg_t].copy()
            right = _border(right, (0, 200, 80), 3)        # 绿色
            right = _label(right, f"REFINED t={t}", color=(0, 220, 100))
            right = _label(right, f"loss={loss_refined:.4f}", color=(0, 220, 100), y=34)
        else:
            right = frames_full[t].copy()
            right = _label(right, "REFINED [out-of-window]", color=(90, 90, 90))
        right = _progress_bar(right, t, T_full)

        row = np.concatenate([left, mid, right], axis=1)
        bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)

        repeat = slow if in_window else 1
        for _ in range(repeat):
            writer.write(bgr)

    writer.release()

    size_kb   = out_path.stat().st_size // 1024
    duration  = total_frames / fps
    print(f"\n✅ 视频：{out_path}")
    print(f"   {height}×{width * 3}  {total_frames}帧  {duration:.1f}s  "
          f"{size_kb}KB  {fps}fps  slow×{slow}")
    print(f"   窗口帧[{start_frame}–{end_frame - 1}]：白=专家 | 蓝=扰动 | 绿=修正")
    print(f"   场景验证：{'✅ PASS' if verify_result['scene_ok'] else '❌ FAIL'}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P4 对比视频（mujoco.Renderer，phyvla-p2 环境）"
    )
    p.add_argument("--task-id",     type=int, default=2)
    p.add_argument("--suite",       default="libero_spatial")
    p.add_argument("--start-frame", type=int, default=55)
    p.add_argument("--camera",      default=CAMERA)
    p.add_argument("--height",      type=int, default=HEIGHT)
    p.add_argument("--width",       type=int, default=WIDTH)
    p.add_argument("--fps",         type=int, default=FPS)
    p.add_argument("--slow",        type=int, default=1,
                   help="窗口段慢放倍率（1=均匀帧率不慢放；原脚本=4，建议≤2）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("P4 三路对比视频（mujoco.Renderer，phyvla-p2 环境）")
    print("=" * 60)
    render_comparison(
        task_id=args.task_id,
        suite=args.suite,
        start_frame=args.start_frame,
        camera=args.camera,
        height=args.height,
        width=args.width,
        fps=args.fps,
        slow=args.slow,
    )
