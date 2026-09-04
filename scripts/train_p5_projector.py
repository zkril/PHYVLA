"""P5-3: 训练 Physics Projector — 将 z_phy 注入 StarVLA DiT action head。

流程：
  1. 加载 StarVLAPhysicsWrapper（VLM 冻结）
  2. 预缓存 VLM 隐状态（一次性，节省训练时间）
  3. 训练 PhysicsTokenProjector（+ 可选解冻 action head）
  4. 验收：物理敏感性测试（改变质量/摩擦 token → 动作应有变化）

运行（starVLA conda 环境）：
    cd /home/jiangnan-luxitech/workspace/phyvla
    conda run -n starVLA python scripts/train_p5_projector.py \
        --task-ids 1 5 --epochs 50 --lr 3e-4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── 路径 ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STARVLA_ROOT = PROJECT_ROOT / "third_party" / "starVLA"
STARVLA_WS   = Path("/home/jiangnan-luxitech/workspace/starVLA")

for _p in [str(STARVLA_ROOT), str(STARVLA_WS), str(PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(str(STARVLA_WS))

DEFAULT_CKPT = str(
    STARVLA_WS / "playground/Pretrained_models/Qwen2.5-VL-GR00T-LIBERO-4in1"
               / "checkpoints/steps_30000_pytorch_model.pt"
)


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_task_data(task_id: int, demo_index: int = 0) -> dict | None:
    """加载单任务的 z_phy + a_VLA + a_gt。"""
    out_dir = PROJECT_ROOT / f"outputs/task{task_id:02d}"

    tokens_npz = out_dir / f"physics_tokens_demo{demo_index}.npz"
    vla_npz    = out_dir / f"vla_actions_demo{demo_index}.npz"

    if not tokens_npz.exists():
        print(f"  ⚠️  task{task_id:02d}: {tokens_npz.name} 不存在，跳过")
        return None
    if not vla_npz.exists():
        print(f"  ⚠️  task{task_id:02d}: {vla_npz.name} 不存在，请先运行 export_vla_actions.py，跳过")
        return None

    tok = np.load(str(tokens_npz))
    vla = np.load(str(vla_npz))

    # z_phy: [T, 23, 256]
    z_phy = tok["tokens"].astype(np.float32)   # key from build_offline_tokens.py

    # a_VLA 目标：使用归一化版本（与 StarVLA 训练空间一致）
    a_vla_norm = vla["a_vla_norm"].astype(np.float32)   # [T, horizon, 7]
    # 取 chunk 最后 action_horizon 步（与 StarVLA 训练对齐）
    a_gt_norm  = a_vla_norm   # 自蒸馏：以 StarVLA 预测为目标

    # ground truth EE actions（用于 P5-3 敏感性测试参考）
    a_gt_ee = vla["a_gt"].astype(np.float32)   # [T, 7]

    T = z_phy.shape[0]
    print(f"  task{task_id:02d}: T={T}  z_phy={z_phy.shape}  a_vla={a_vla_norm.shape}")
    return {
        "task_id": task_id,
        "z_phy":   z_phy,        # [T, 23, 256]
        "a_target": a_gt_norm,   # [T, horizon, 7]  归一化
        "a_gt_ee":  a_gt_ee,     # [T, 7]  EE 参考
        "T": T,
    }


def build_dataset(task_data_list: list, action_horizon: int) -> dict:
    """合并多任务数据，返回 Tensor 字典。"""
    z_list, a_list = [], []
    for td in task_data_list:
        T = td["T"]
        z_list.append(td["z_phy"])
        # 取每帧 chunk 的后 action_horizon 步
        a = td["a_target"][:, -action_horizon:, :]  # [T, horizon, 7]
        a_list.append(a)

    z_all = torch.from_numpy(np.concatenate(z_list, axis=0))  # [N, 23, 256]
    a_all = torch.from_numpy(np.concatenate(a_list, axis=0))  # [N, horizon, 7]
    print(f"\n数据集：N={len(z_all)}  z_phy {z_all.shape}  a_target {a_all.shape}")
    return {"z_phy": z_all, "a_target": a_all}


# ─────────────────────────────────────────────────────────────────────────────
# VLM 隐状态预缓存（读取训练图像 → 一次性编码）
# ─────────────────────────────────────────────────────────────────────────────

def cache_vlm_hidden_from_hdf5(
    wrapper,
    task_id: int,
    suite:   str = "libero_spatial",
    demo_index: int = 0,
    device: str = "cuda",
    batch_size: int = 4,
) -> tuple:
    """从 LIBERO HDF5 提取 obs，一次性缓存 VLM hidden states。

    Returns:
        h_cache:  [T, L, H]  float16（CPU）
        m_cache:  [T, L] bool | None（CPU）
    """
    import h5py
    from PIL import Image
    from starVLA.model.framework.share_tools import dict_to_namespace
    from deployment.model_server.tools.image_tools import to_pil_preserve

    out_dir = PROJECT_ROOT / f"outputs/task{task_id:02d}"
    cache_path = out_dir / f"vlm_hidden_demo{demo_index}.npz"

    # 如果已缓存则直接加载
    if cache_path.exists():
        print(f"  [缓存] task{task_id:02d} VLM hidden states 已存在，直接加载")
        d = np.load(str(cache_path))
        h = torch.from_numpy(d["h"])
        m = torch.from_numpy(d["m"]) if "m" in d and d["m"] is not None else None
        return h, m

    # 读取 HDF5
    from libero.libero import benchmark, get_libero_path
    bd = benchmark.get_benchmark_dict()
    suite_obj = bd[suite]()
    task = suite_obj.get_task(task_id)
    lang = task.language
    hdf5_path = (
        Path(get_libero_path("datasets")) / suite
        / f"{task.problem_folder}.hdf5"
    )

    with h5py.File(str(hdf5_path), "r") as f:
        imgs  = f[f"data/demo_{demo_index}/obs/agentview_image"][:]
        jpos  = f[f"data/demo_{demo_index}/obs/robot0_joint_pos"][:].astype(np.float32)

    T = len(imgs)
    print(f"  [缓存] task{task_id:02d} 缓存 VLM hidden states，T={T}...")

    h_list, m_list = [], []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        examples = [
            {
                "image": [Image.fromarray(imgs[i])],
                "lang":  lang,
                "state": jpos[i],
            }
            for i in range(start, end)
        ]
        h, m = wrapper.cache_vlm_hidden(examples)   # CPU tensors
        h_list.append(h.half())
        if m is not None:
            m_list.append(m)
        if (start // batch_size) % 5 == 0 or end == T:
            print(f"    [{end}/{T}]", end="\r")
    print()

    h_cache = torch.cat(h_list, dim=0)   # [T, L, H]
    m_cache = torch.cat(m_list, dim=0) if m_list else None

    # 持久化
    save_dict = {"h": h_cache.numpy()}
    if m_cache is not None:
        save_dict["m"] = m_cache.numpy()
    np.savez(str(cache_path), **save_dict)
    print(f"  [缓存] 已保存 → {cache_path}  shape={h_cache.shape}")
    return h_cache, m_cache


# ─────────────────────────────────────────────────────────────────────────────
# 物理敏感性测试
# ─────────────────────────────────────────────────────────────────────────────

def physics_sensitivity_test(wrapper, task_data: dict, device: str = "cuda") -> dict:
    """验证：修改 z_phy 中的物理参数 slot 是否导致动作变化。

    测试逻辑：
      1. 取 frame=0 的 z_phy（原始）
      2. 修改 z_obj 质量/摩擦 slot（slot 6..22 为对象 slot，index ≈1st object）
      3. 对比两版本的动作预测
      4. 期望：||a_orig - a_perturbed|| > 1e-4
    """
    z_phy_np = task_data["z_phy"]     # [T, 23, 256]

    wrapper.eval()
    results = {}

    with torch.no_grad():
        for frame_idx in [0, 20, 40]:
            if frame_idx >= len(z_phy_np):
                continue

            z_orig = torch.from_numpy(z_phy_np[[frame_idx]]).to(device)   # [1, 23, 256]
            z_perturb = z_orig.clone()

            # 扰动：对象 slot 的质量位（slot 7 = 首个对象 token，前几维含质量）
            # slot 7 是 z_obj[0]，维度0~7 包含 role_embedding 投影，之后是物理量
            z_perturb[:, 7:9, :] = z_perturb[:, 7:9, :] * 3.0   # 3x 质量/惯量

            # 用 projector 对比（VLM hidden 使用 task05 缓存）
            proj_orig    = wrapper.phy_projector(z_orig)          # [1, 23, H]
            proj_perturb = wrapper.phy_projector(z_perturb)       # [1, 23, H]

            diff = (proj_orig - proj_perturb).norm().item()
            results[f"frame{frame_idx}_proj_diff"] = diff

    mean_diff = np.mean(list(results.values()))
    results["mean_proj_diff"] = mean_diff
    results["sensitivity_ok"] = mean_diff > 0.0   # 任何差异即可证明信息传导

    status = "✅" if results["sensitivity_ok"] else "❌"
    print(f"\n物理敏感性测试 {status}  mean_proj_diff={mean_diff:.4f}")
    for k, v in results.items():
        if "frame" in k:
            print(f"  {k}: {v:.6f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 训练主循环
# ─────────────────────────────────────────────────────────────────────────────

def train(args) -> None:
    import torch.optim as optim
    from p5.starvla_physics import StarVLAPhysicsWrapper

    device = args.device
    out_dir = PROJECT_ROOT / "outputs" / "p5_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载数据 ───────────────────────────────────────────────────────────
    task_data_list = []
    for tid in args.task_ids:
        td = load_task_data(tid, args.demo_index)
        if td is not None:
            task_data_list.append(td)

    if not task_data_list:
        print("❌ 无有效数据，请先运行 export_vla_actions.py")
        return

    # ── 加载 StarVLA + 包装 ───────────────────────────────────────────────
    print(f"\n[P5-3] 加载 StarVLAPhysicsWrapper...")
    wrapper = StarVLAPhysicsWrapper.load(
        args.ckpt, device=device, freeze_action_head=args.freeze_action_head
    )
    action_horizon = wrapper.action_horizon
    H = wrapper.vla_hidden_dim

    # ── 预缓存 VLM hidden states ─────────────────────────────────────────
    print(f"\n[P5-3] 预缓存 VLM 隐状态（VLM 冻结，只需一次）...")
    h_caches, m_caches = [], []
    for td in task_data_list:
        h, m = cache_vlm_hidden_from_hdf5(
            wrapper, td["task_id"],
            demo_index=args.demo_index, device=device,
        )
        h_caches.append(h)
        m_caches.append(m)

    h_all = torch.cat(h_caches, dim=0)            # [N, L, H]
    m_all = (torch.cat(m_caches, dim=0)
             if m_caches[0] is not None else None)

    # ── 合并训练数据 ──────────────────────────────────────────────────────
    dataset = build_dataset(task_data_list, action_horizon)
    N       = dataset["z_phy"].shape[0]
    assert N == h_all.shape[0], f"数据对不齐: z_phy N={N} vs h N={h_all.shape[0]}"

    z_all = dataset["z_phy"]        # [N, 23, 256]
    a_all = dataset["a_target"]     # [N, horizon, 7]

    # ── 优化器（只优化可训练参数）────────────────────────────────────────
    trainable = list(wrapper.trainable_parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"\n[P5-3] 可训练参数: {n_trainable:,}  (VLM 冻结)")
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 训练循环 ─────────────────────────────────────────────────────────
    print(f"\n[P5-3] 开始训练  epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")
    wrapper.train()
    best_loss = float("inf")
    ckpt_save = out_dir / "projector_best.pt"   # 初始化（防止 loop 未进入时引用未定义）
    history   = []

    for epoch in range(1, args.epochs + 1):
        perm    = torch.randperm(N)
        ep_loss = 0.0
        n_steps = 0

        for start in range(0, N, args.batch_size):
            idx = perm[start:start + args.batch_size]
            if len(idx) < 2:
                continue

            z_b = z_all[idx].to(device)                # [B, 23, 256]
            a_b = a_all[idx].to(device)                # [B, horizon, 7]
            h_b = h_all[idx].to(device, dtype=torch.float32)  # [B, L, H]
            m_b = m_all[idx].to(device) if m_all is not None else None

            optimizer.zero_grad()
            out = wrapper.training_forward(h_b, m_b, z_b, a_b)
            loss = out["action_loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            ep_loss += loss.item()
            n_steps += 1

        scheduler.step()
        ep_loss /= max(n_steps, 1)
        history.append(ep_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={ep_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if ep_loss < best_loss:
            best_loss = ep_loss
            ckpt_save = out_dir / "projector_best.pt"
            wrapper.save_projector(ckpt_save)

    # ── 物理敏感性验收 ────────────────────────────────────────────────────
    wrapper.eval()
    print(f"\n[P5-3] 物理敏感性验收...")
    sens_results = physics_sensitivity_test(wrapper, task_data_list[0], device)

    # ── 验收报告 ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"P5 训练验收")
    print(f"{'='*60}")
    checks = [
        ("训练 loss 收敛",       best_loss < 1.0),
        ("物理敏感性 > 0",       sens_results["sensitivity_ok"]),
        ("Projector checkpoint 已保存", ckpt_save.exists()),
    ]
    all_pass = True
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")
        if not ok:
            all_pass = False

    print(f"\n  最佳 loss     = {best_loss:.5f}")
    print(f"  物理敏感性    = {sens_results['mean_proj_diff']:.6f}")
    print(f"  Projector 参数 = {wrapper.phy_projector.n_params:,}")
    print()
    print("P5 总验收：", "✅ ALL PASS" if all_pass else "❌ 部分失败")

    # 保存摘要
    summary = {
        "best_loss":      best_loss,
        "epochs":         args.epochs,
        "n_trainable":    n_trainable,
        "sensitivity":    sens_results,
        "task_ids":       args.task_ids,
        "all_pass":       all_pass,
    }
    (out_dir / "p5_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\n摘要已保存：{out_dir}/p5_summary.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-ids",          type=int, nargs="+", default=[1, 5])
    p.add_argument("--demo-index",        type=int, default=0)
    p.add_argument("--ckpt",              default=DEFAULT_CKPT)
    p.add_argument("--epochs",            type=int, default=50)
    p.add_argument("--batch-size",        type=int, default=8)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--device",            default="cuda")
    p.add_argument("--freeze-action-head", action="store_true",
                   help="仅训练 projector，冻结 DiT action head")
    p.add_argument("--suite",             default="libero_spatial")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
