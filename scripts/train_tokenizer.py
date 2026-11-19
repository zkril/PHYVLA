"""Train PhysicalTokenizer decoder heads and report acceptance metrics.

Supervised losses:
  z_robot    → MSE reconstruction  (robot_vec)
  z_obj[i]   → MSE pose, MSE params, CE role classification
  z_contact  → BCE relation flags (7 binary)
  z_afford   → CE phase classification

Usage:
    python scripts/train_tokenizer.py --epochs 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from datasets.physics_dataset import PhysicsDataset, MAX_N_OBJECTS   # noqa: E402
from models.physics_tokenizer import PhysicalTokenizer                # noqa: E402
from utils.task_role_parser import ROLE_TARGET                        # noqa: E402

LIBERO_TASK02_LANGUAGE = (
    "pick up the black bowl from table center and place it on the plate"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--physics", type=Path,
                   default=Path("outputs/physics/task02_demo00_contact_force_data.npz"))
    p.add_argument("--events",  type=Path,
                   default=Path("outputs/contact_events_v1.npz"))
    p.add_argument("--stats",   type=Path,
                   default=Path("outputs/normalization_v1.json"))
    p.add_argument("--rollout",  type=Path, default=None,
                   help="P3-2: MJX rollout 特征 NPZ（替换 teacher-forcing z_dyn 和规则 z_risk）")
    p.add_argument("--epochs",     type=int, default=500)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--token-dim",  type=int, default=256)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--log-every",  type=int, default=50)
    p.add_argument("--language", default=LIBERO_TASK02_LANGUAGE)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("outputs/tokenizer/tokenizer_last.pt"))
    return p.parse_args()


# ------------------------------------------------------------------ #
# Loss computation
# ------------------------------------------------------------------ #

def compute_loss(
    tokenizer: PhysicalTokenizer,
    batch: dict,
    device: str,
) -> tuple:
    batch_dev = {k: v.to(device) for k, v in batch.items()
                 if isinstance(v, torch.Tensor)}

    z_phy = tokenizer(batch_dev)   # [B, 7+N, D]

    losses = {}

    # 1. z_robot reconstruction
    pred_robot = tokenizer.decode_robot(z_phy)
    losses["robot_mse"] = F.mse_loss(pred_robot, batch_dev["robot_vec"])

    # 2. z_afford phase classification
    phase_logits = tokenizer.classify_phase(z_phy)
    losses["phase_ce"] = F.cross_entropy(phase_logits, batch_dev["phase"])

    # 3. z_contact relation BCE (7 binary flags from contacts)
    # Use max of relation flags across valid contacts as target
    contacts = batch_dev["contacts"]          # [B, C, 28]
    cmask    = batch_dev["contact_mask"] > 0.5
    rel_target = (contacts[:, :, 21:28].masked_fill(~cmask.unsqueeze(-1), 0).max(dim=1).values > 0.5).float()
    rel_pred   = tokenizer.decode_contact_relations(z_phy)
    losses["contact_bce"] = F.binary_cross_entropy_with_logits(rel_pred, rel_target)

    # 4. Per-object losses (for valid objects)
    obj_mask  = batch_dev["obj_mask"]          # [B, N]
    obj_cont  = batch_dev["obj_cont"]          # [B, N, 25]
    obj_roles = batch_dev["obj_roles"]         # [B, N] long
    N = obj_cont.shape[1]

    role_loss_list, pose_loss_list, param_loss_list = [], [], []
    for i in range(min(N, MAX_N_OBJECTS)):
        valid = obj_mask[:, i] > 0.5
        if not valid.any():
            continue
        idx = i  # slot in z_phy is 7+i

        # role classification
        role_logits = tokenizer.classify_obj_role(z_phy, idx)
        role_loss_list.append(
            F.cross_entropy(role_logits[valid], obj_roles[valid, i])
        )

        # pose reconstruction (position[3] + rot_6d[6] = indices 0:9)
        pred_pose = tokenizer.decode_obj_pose(z_phy, idx)
        role_loss_list  # noqa — just extend list above
        pose_loss_list.append(
            F.mse_loss(pred_pose[valid], obj_cont[valid, i, :9])
        )

        # param regression (mass[1]+inertia[3]+friction[3] = indices 15:22)
        pred_param = tokenizer.decode_obj_params(z_phy, idx)
        param_loss_list.append(
            F.mse_loss(pred_param[valid], obj_cont[valid, i, 15:22])
        )

    if role_loss_list:
        losses["obj_role_ce"]  = torch.stack(role_loss_list).mean()
    if pose_loss_list:
        losses["obj_pose_mse"] = torch.stack(pose_loss_list).mean()
    if param_loss_list:
        losses["obj_param_mse"]= torch.stack(param_loss_list).mean()

    # Weighted total
    total = (
        losses.get("robot_mse",    torch.tensor(0.0)) * 1.0 +
        losses.get("phase_ce",     torch.tensor(0.0)) * 1.0 +
        losses.get("contact_bce",  torch.tensor(0.0)) * 1.0 +
        losses.get("obj_role_ce",  torch.tensor(0.0)) * 2.0 +
        losses.get("obj_pose_mse", torch.tensor(0.0)) * 1.0 +
        losses.get("obj_param_mse",torch.tensor(0.0)) * 2.0
    )
    return total, losses


# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate(tokenizer: PhysicalTokenizer, loader: DataLoader, device: str) -> dict:
    tokenizer.eval()
    metrics = {k: [] for k in [
        "robot_rmse", "phase_acc", "contact_rel_f1",
        "obj0_role_acc", "obj0_param_rmse", "obj0_pose_rmse",
    ]}

    for batch in loader:
        bd = {k: v.to(device) for k, v in batch.items()
              if isinstance(v, torch.Tensor)}
        z_phy = tokenizer(bd)
        B     = z_phy.shape[0]

        # robot RMSE
        pred_r = tokenizer.decode_robot(z_phy).cpu().numpy()
        true_r = bd["robot_vec"].cpu().numpy()
        metrics["robot_rmse"].append(float(np.sqrt(np.mean((pred_r - true_r)**2))))

        # phase accuracy
        logits = tokenizer.classify_phase(z_phy).cpu().numpy()
        pred_p = np.argmax(logits, axis=-1)
        true_p = bd["phase"].cpu().numpy()
        metrics["phase_acc"].append(float((pred_p == true_p).mean()))

        # contact relation F1 (macro, 7 flags)
        contacts = bd["contacts"]
        rel_tgt  = ((contacts[:, :, 21:28] * bd["contact_mask"].unsqueeze(-1)).max(dim=1).values > 0.5).float().cpu().numpy()
        rel_pred = (tokenizer.decode_contact_relations(z_phy).sigmoid() > 0.5).cpu().numpy()
        tp = (rel_pred * rel_tgt).sum()
        fp = (rel_pred * (1 - rel_tgt)).sum()
        fn = ((1 - rel_pred) * rel_tgt).sum()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        metrics["contact_rel_f1"].append(float(2 * prec * rec / max(prec + rec, 1e-6)))

        # obj[0] metrics
        obj_cont  = bd["obj_cont"].cpu().numpy()
        obj_roles = bd["obj_roles"].cpu().numpy()
        valid     = bd["obj_mask"][:, 0].cpu().numpy() > 0.5
        if valid.any():
            pred_role = np.argmax(tokenizer.classify_obj_role(z_phy, 0).cpu().numpy(), -1)
            metrics["obj0_role_acc"].append(float((pred_role[valid] == obj_roles[valid, 0]).mean()))

            pred_par = tokenizer.decode_obj_params(z_phy, 0).cpu().numpy()
            true_par = obj_cont[:, 0, 15:22]
            metrics["obj0_param_rmse"].append(float(np.sqrt(np.mean((pred_par[valid] - true_par[valid])**2))))

            pred_pos = tokenizer.decode_obj_pose(z_phy, 0).cpu().numpy()
            true_pos = obj_cont[:, 0, :9]
            metrics["obj0_pose_rmse"].append(float(np.sqrt(np.mean((pred_pos[valid] - true_pos[valid])**2))))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    args   = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device
    stats  = args.stats if args.stats.exists() else None

    rollout_path = args.rollout if (args.rollout and args.rollout.exists()) else None
    if rollout_path:
        print(f"[train] 使用 MJX rollout 特征: {rollout_path}")
    else:
        print("[train] 使用 teacher-forcing / 规则特征（未提供 --rollout）")

    dataset = PhysicsDataset(
        args.physics, args.events,
        task_language=args.language,
        stats_path=stats,
        rollout_path=rollout_path,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    tokenizer = PhysicalTokenizer(token_dim=args.token_dim).to(device)
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print(f"Training {args.epochs} epochs, {len(dataset)} samples, device={device}")
    print("-" * 60)

    best_phase_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tokenizer.train()
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            total, _ = compute_loss(tokenizer, batch, device)
            total.backward()
            nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total.item()
        scheduler.step()

        if epoch % args.log_every == 0 or epoch == args.epochs:
            m = evaluate(tokenizer, eval_loader, device)
            best_phase_acc = max(best_phase_acc, m["phase_acc"])
            print(
                f"epoch {epoch:4d}/{args.epochs} | loss={epoch_loss/len(loader):.4f} | "
                f"robot_rmse={m['robot_rmse']:.4f} | "
                f"phase_acc={m['phase_acc']:.3f} | "
                f"obj0_role_acc={m['obj0_role_acc']:.3f} | "
                f"obj0_param_rmse={m['obj0_param_rmse']:.4f} | "
                f"contact_f1={m['contact_rel_f1']:.3f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokenizer_state_dict": tokenizer.state_dict(),
        "feature_schema": "rollout_v1" if rollout_path else "teacher_forcing_v1",
        "tokenizer_config": {"token_dim": args.token_dim, "max_n_objects": MAX_N_OBJECTS},
        "seed": args.seed, "epochs": args.epochs, "language": args.language,
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "metrics_scope": "training_set_only",
    }
    temporary = args.output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(f"Saved tokenizer checkpoint: {args.output}")

    print("\n" + "=" * 60)
    print("FINAL METRICS vs ACCEPTANCE CRITERIA")
    print("=" * 60)
    m = evaluate(tokenizer, eval_loader, device)

    criteria = [
        ("z_robot   reconstruction RMSE", m["robot_rmse"],       "< 0.10",   m["robot_rmse"] < 0.10),
        ("z_afford  phase accuracy",       m["phase_acc"],        "> 0.85",   m["phase_acc"] > 0.85),
        ("z_contact relation F1",          m["contact_rel_f1"],   "> 0.85",   m["contact_rel_f1"] > 0.85),
        ("z_obj[0]  role accuracy",        m["obj0_role_acc"],    "> 0.85",   m["obj0_role_acc"] > 0.85),
        ("z_obj[0]  param RMSE",           m["obj0_param_rmse"],  "< 0.05",   m["obj0_param_rmse"] < 0.05),
        ("z_obj[0]  pose RMSE",            m["obj0_pose_rmse"],   "< 0.10",   m["obj0_pose_rmse"] < 0.10),
    ]
    all_pass = True
    for name, val, threshold, passed in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name:<38s} = {val:.4f}  (threshold {threshold})")
        if not passed:
            all_pass = False

    print()
    print("P1 ACCEPTANCE:", "✅ ALL PASS" if all_pass else "❌ NOT YET — see failures above")


if __name__ == "__main__":
    main()
