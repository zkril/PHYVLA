"""Build offline physical tokens (v2) — generalized N-object pipeline.

Usage:
    python scripts/build_offline_tokens.py \
        --physics  outputs/physics/task02_demo00_contact_force_data.npz \
        --events   outputs/contact_events_v1.npz \
        --output   outputs/physics_tokens_demo0.npz \
        --language "pick up the black bowl from table center and place it on the plate"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from datasets.physics_dataset import PhysicsDataset, MAX_N_OBJECTS   # noqa: E402
from models.physics_tokenizer import PhysicalTokenizer, NUM_BASE_TOKENS  # noqa: E402


LIBERO_TASK02_LANGUAGE = (
    "pick up the black bowl from table center and place it on the plate"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--physics",  type=Path,
                   default=Path("outputs/physics/task02_demo00_contact_force_data.npz"))
    p.add_argument("--events",   type=Path,
                   default=Path("outputs/contact_events_v1.npz"))
    p.add_argument("--stats",    type=Path,
                   default=Path("outputs/normalization_v1.json"))
    p.add_argument("--output",   type=Path,
                   default=Path("outputs/physics_tokens_demo0.npz"))
    p.add_argument("--language", type=str, default=LIBERO_TASK02_LANGUAGE)
    p.add_argument("--token-dim",   type=int, default=256)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--device",      type=str, default="cpu")
    return p.parse_args()


def _verify(tokens: np.ndarray, tokenizer: PhysicalTokenizer,
            dataset: PhysicsDataset, device: str) -> None:
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch  = {k: v.to(device) for k, v in next(iter(loader)).items()
              if isinstance(v, torch.Tensor)}

    tokenizer.eval()
    with torch.no_grad():
        z_phy = tokenizer(batch)   # [B, 7+N, D]

        # z_robot reconstruction
        pred_robot = tokenizer.decode_robot(z_phy).cpu().numpy()
        true_robot = batch["robot_vec"].cpu().numpy()
        rmse_robot = float(np.sqrt(np.mean((pred_robot - true_robot)**2)))

        # z_contact relation decode
        rel_logits = tokenizer.decode_contact_relations(z_phy).cpu().numpy()

        # phase classification
        phase_logits = tokenizer.classify_phase(z_phy).cpu().numpy()
        pred_phase   = np.argmax(phase_logits, axis=-1)
        true_phase   = batch["phase"].cpu().numpy()
        phase_acc    = float((pred_phase == true_phase).mean())

        # first object token: role classification
        role_logits  = tokenizer.classify_obj_role(z_phy, 0).cpu().numpy()
        pred_role    = np.argmax(role_logits, axis=-1)
        true_role    = batch["obj_roles"][:, 0].cpu().numpy()
        role_acc     = float((pred_role == true_role).mean())

        # first object token: param decode
        pred_params  = tokenizer.decode_obj_params(z_phy, 0).cpu().numpy()
        true_cont    = batch["obj_cont"][:, 0].cpu().numpy()
        # params are indices 9:16 in OBJ_CONT_DIM (mass[1]+inertia[3]+friction[3])
        rmse_params  = float(np.sqrt(np.mean(
            (pred_params - true_cont[:, 9:16])**2
        )))

    print("\n[verify] Decoder-head check (untrained — expect random/high error):")
    print(f"  z_robot  reconstruction RMSE : {rmse_robot:.4f}")
    print(f"  z_contact relation logits    : {rel_logits.shape}")
    print(f"  z_afford phase accuracy      : {phase_acc:.3f}  (random={1/8:.3f})")
    print(f"  z_obj[0] role accuracy       : {role_acc:.3f}  (random={1/5:.3f})")
    print(f"  z_obj[0] param RMSE          : {rmse_params:.4f}")
    print("  [note] high error expected before training]")


def main() -> None:
    args = parse_args()

    for p in (args.physics, args.events):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    stats_path = args.stats if args.stats.exists() else None
    device     = args.device

    print("[tokens] loading dataset ...")
    dataset = PhysicsDataset(
        args.physics, args.events,
        task_language=args.language,
        stats_path=stats_path,
    )
    print(f"[tokens] T={len(dataset)}, objects={dataset.obj_keys}, "
          f"roles={dataset.obj_roles}")

    tokenizer = PhysicalTokenizer(
        token_dim=args.token_dim,
        max_n_objects=MAX_N_OBJECTS,
    ).to(device)
    tokenizer.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    all_tokens, all_phases = [], []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}
            z_phy = tokenizer(batch_dev)   # [B, 7+N, D]
            all_tokens.append(z_phy.cpu().numpy())
            all_phases.append(batch["phase"].numpy())

    tokens_arr = np.concatenate(all_tokens, axis=0).astype(np.float32)
    phases_arr = np.concatenate(all_phases, axis=0)

    expected_slots = NUM_BASE_TOKENS + MAX_N_OBJECTS
    assert tokens_arr.shape == (len(dataset), expected_slots, args.token_dim), \
        f"Expected {(len(dataset), expected_slots, args.token_dim)}, got {tokens_arr.shape}"
    assert not np.any(np.isnan(tokens_arr)), "NaN in output!"
    assert not np.any(np.isinf(tokens_arr)), "Inf in output!"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output), tokens=tokens_arr, phases=phases_arr)

    _verify(tokens_arr, tokenizer, dataset, device)

    token_names = (
        ["robot", "contact", "dyn", "risk", "afford", "grad", "scene_ctx"]
        + [f"obj_{i}" for i in range(MAX_N_OBJECTS)]
    )
    summary = {
        "schema_version": "physics-tokens-v2",
        "shape":          list(tokens_arr.shape),
        "token_dim":      args.token_dim,
        "num_base":       NUM_BASE_TOKENS,
        "max_n_objects":  MAX_N_OBJECTS,
        "token_names":    token_names,
        "objects":        dataset.obj_keys,
        "object_roles":   dataset.obj_roles,
        "has_nan":        bool(np.any(np.isnan(tokens_arr))),
        "has_inf":        bool(np.any(np.isinf(tokens_arr))),
        "task_language":  args.language,
    }
    summary_path = args.output.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[tokens] shape   : {tokens_arr.shape}")
    print(f"[tokens] NaN     : {np.any(np.isnan(tokens_arr))}")
    print(f"[tokens] Inf     : {np.any(np.isinf(tokens_arr))}")
    print(f"[tokens] saved   : {args.output}")
    print(f"[tokens] summary : {summary_path}")


if __name__ == "__main__":
    main()
