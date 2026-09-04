"""P5-0: 导出 StarVLA 在 LIBERO 演示上的基准动作预测。

从 LIBERO HDF5 demo 逐帧读取 RGB obs + 本体感知 + 任务语言，
通过 StarVLA-LIBERO-4in1 推理，保存 a_VLA 供 P5 训练使用。

运行（starVLA conda 环境，工作目录：phyvla 根目录）：
    cd /home/jiangnan-luxitech/workspace/phyvla
    conda run -n starVLA python scripts/export_vla_actions.py --task-ids 1 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

# ── 路径设置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
STARVLA_ROOT  = PROJECT_ROOT / "third_party" / "starVLA"
STARVLA_WS    = Path("/home/jiangnan-luxitech/workspace/starVLA")

for _p in [str(STARVLA_ROOT), str(STARVLA_WS), str(PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# StarVLA 使用相对路径加载 VLM 基底，必须切换到其工作目录
os.chdir(str(STARVLA_WS))

# ── 默认 checkpoint 路径 ──────────────────────────────────────────────────────
DEFAULT_CKPT = str(
    STARVLA_WS / "playground/Pretrained_models/Qwen2.5-VL-GR00T-LIBERO-4in1"
               / "checkpoints/steps_30000_pytorch_model.pt"
)
SUITE  = "libero_spatial"
DEVICE = "cuda"


def get_libero_hdf5_path(suite: str, task_id: int) -> Path:
    """从 LIBERO 库解析 HDF5 路径。"""
    # 切换回 phyvla 以运行 LIBERO 相关代码
    orig_dir = os.getcwd()
    os.chdir(str(STARVLA_WS))
    try:
        from libero.libero import benchmark, get_libero_path
        bd    = benchmark.get_benchmark_dict()
        suite_obj = bd[suite]()
        task  = suite_obj.get_task(task_id)
        demo_path = (
            Path(get_libero_path("datasets")) / suite
            / f"{task.problem_folder}.hdf5"
        )
        return demo_path, task.language
    finally:
        os.chdir(orig_dir)


def load_demo_frames(hdf5_path: Path, demo_index: int = 0) -> dict:
    """从 LIBERO HDF5 读取单个 demo 的 obs 和动作。

    Returns dict:
        "images"      : [T, H, W, 3] uint8   agentview_image
        "joint_pos"   : [T, 7] float32        robot0_joint_pos（stata_dim=7）
        "actions"     : [T, 7] float32        专家动作（7D EE，用于参考）
        "T"           : int
    """
    key = f"data/demo_{demo_index}"
    with h5py.File(str(hdf5_path), "r") as f:
        obs   = f[f"{key}/obs"]
        imgs  = obs["agentview_image"][:]             # [T, H, W, 3]
        jpos  = obs["robot0_joint_pos"][:].astype(np.float32)  # [T, 7]
        acts  = f[f"{key}/actions"][:].astype(np.float32)       # [T, 7]
    return {"images": imgs, "joint_pos": jpos, "actions": acts, "T": len(imgs)}


def export_task(
    task_id:    int,
    ckpt_path:  str,
    suite:      str   = SUITE,
    demo_index: int   = 0,
    device:     str   = DEVICE,
    batch_size: int   = 4,
) -> Path:
    """导出单任务的 StarVLA 动作预测，返回保存路径。"""
    out_dir  = PROJECT_ROOT / f"outputs/task{task_id:02d}"
    out_path = out_dir / f"vla_actions_demo{demo_index}.npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 StarVLA（仅需一次） ───────────────────────────────────────────
    from starVLA.model.framework.base_framework import baseframework

    print(f"\n[P5-0] task{task_id:02d}  加载 StarVLA...")
    vla = baseframework.from_pretrained(ckpt_path)
    vla = vla.to(device).eval()
    print(f"       action_dim={vla.config.framework.action_model.action_dim}  "
          f"horizon={vla.action_horizon}")

    # ── 读取 LIBERO demo ───────────────────────────────────────────────────
    hdf5_path, task_lang = get_libero_hdf5_path(suite, task_id)
    demo = load_demo_frames(hdf5_path, demo_index)
    T    = demo["T"]
    print(f"       task: {task_lang}")
    print(f"       demo: {T} 帧  hdf5={hdf5_path.name}")

    # ── 逐批推理 ─────────────────────────────────────────────────────────
    all_preds = []   # list of [B, horizon, 7]
    t0 = time.time()

    for start in range(0, T, batch_size):
        end     = min(start + batch_size, T)
        indices = list(range(start, end))

        examples = []
        for idx in indices:
            img  = Image.fromarray(demo["images"][idx])
            jpos = demo["joint_pos"][idx].tolist()
            examples.append({
                "image": [img],
                "lang":  task_lang,
                "state": np.array(jpos, dtype=np.float32),
            })

        with torch.inference_mode():
            out = vla.predict_action(examples)
        pred_np = out["normalized_actions"]   # [B, horizon, 7]
        all_preds.append(pred_np)

        if (start // batch_size) % 5 == 0 or end == T:
            print(f"       [{end}/{T}]", end="\r")

    print()
    a_vla = np.concatenate(all_preds, axis=0)   # [T, horizon, 7]
    elapsed = time.time() - t0

    # ── 反归一化（franka 统计参数）────────────────────────────────────────
    try:
        from deployment.model_server.policy_norm_processor import PolicyNormProcessor
        from starVLA.model.framework.share_tools import read_mode_config
        _, norm_stats = read_mode_config(ckpt_path)
        proc = PolicyNormProcessor(norm_stats, unnorm_key="franka")
        a_vla_unnorm = proc.unnormalize_actions(a_vla)
    except Exception as e:
        print(f"       ⚠️  反归一化失败（{e}），保存归一化版本")
        a_vla_unnorm = a_vla

    # ── 保存 ─────────────────────────────────────────────────────────────
    np.savez(
        str(out_path),
        a_vla_norm   = a_vla,                 # [T, horizon, 7] 归一化
        a_vla_unnorm = a_vla_unnorm,           # [T, horizon, 7] 反归一化
        a_vla_step0  = a_vla_unnorm[:, 0, :], # [T, 7] 每 chunk 首步
        a_gt         = demo["actions"],        # [T, 7] LIBERO 专家动作
        task_lang    = np.array(task_lang),
    )
    print(f"\n[P5-0] ✅ task{task_id:02d}  {elapsed:.1f}s  → {out_path}")
    print(f"       a_vla shape: {a_vla.shape}  a_gt shape: {demo['actions'].shape}")
    return out_path


def main() -> None:
    import torch  # noqa: F401

    p = argparse.ArgumentParser(description="导出 StarVLA LIBERO 基准动作")
    p.add_argument("--task-ids",   type=int, nargs="+", default=[1, 5])
    p.add_argument("--suite",      default=SUITE)
    p.add_argument("--demo-index", type=int, default=0)
    p.add_argument("--ckpt",       default=DEFAULT_CKPT)
    p.add_argument("--device",     default=DEVICE)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    import torch   # ensure torch is available
    for tid in args.task_ids:
        export_task(
            task_id=tid,
            ckpt_path=args.ckpt,
            suite=args.suite,
            demo_index=args.demo_index,
            device=args.device,
            batch_size=args.batch_size,
        )

    print(f"\n所有任务导出完成：{args.task_ids}")


if __name__ == "__main__":
    import torch
    main()
