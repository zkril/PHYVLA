"""Step 6 smoke test: extract canonical physics state from a LIBERO demo.

Place this file in the project's scripts directory together with
replay_libero_actions.py, and place libero_physics_adapter.py in src/adapters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

for candidate in (str(SCRIPTS_DIR), str(SRC_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from replay_libero_actions import (  # noqa: E402
    check_success,
    load_demonstration,
    make_environment,
    resolve_task_and_demo,
    restore_recorded_model,
)
from adapters.libero_physics_adapter import (  # noqa: E402
    CONTACT_FEATURE_NAMES,
    LiberoPhysicsAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Step-6 canonical physics features from a LIBERO demo."
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-contacts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/physics/task02_demo00_step6.npz"),
    )
    return parser.parse_args()


def _stack(records: List[Dict[str, Any]], path: List[str]) -> np.ndarray:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(np.asarray(value))
    return np.stack(values, axis=0)


def _finite_or_raise(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        indices = np.argwhere(~np.isfinite(array))
        raise FloatingPointError(
            "Non-finite values in {} at {}".format(name, indices[:10].tolist())
        )


def main() -> None:
    args = parse_args()

    if args.stride <= 0:
        raise ValueError("stride must be positive")

    _, task, bddl_path, demo_path = resolve_task_and_demo(
        suite_name=args.suite,
        task_id=args.task_id,
    )
    demo = load_demonstration(
        demo_path=demo_path,
        demo_index=args.demo_index,
    )

    states = np.asarray(demo["states"], dtype=np.float64)
    actions = np.asarray(demo["actions"], dtype=np.float64)

    stop = len(states)
    if args.max_steps >= 0:
        stop = min(stop, args.max_steps)
    timesteps = np.arange(0, stop, args.stride, dtype=np.int32)

    env = make_environment(
        bddl_path=bddl_path,
        camera_name=args.camera,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    try:
        restore_recorded_model(env, demo["model_xml"])

        # Restore the first state before resolving model names.
        env.regenerate_obs_from_state(states[int(timesteps[0])])
        env.sim.forward()

        adapter = LiberoPhysicsAdapter(
            env=env,
            object_patterns={
                "bowl": ("black_bowl", "black bowl", "bowl"),
                "plate": ("plate",),
            },
            max_contacts=args.max_contacts,
        )

        description = adapter.describe()
        print(json.dumps(description, indent=2, ensure_ascii=False))

        records: List[Dict[str, Any]] = []
        success = []
        selected_actions = []

        for item_index, timestep in enumerate(timesteps.tolist()):
            env.regenerate_obs_from_state(states[timestep])
            env.sim.forward()

            snapshot = adapter.extract()
            records.append(snapshot)
            success.append(float(check_success(env)))

            if timestep < len(actions):
                selected_actions.append(actions[timestep])
            else:
                selected_actions.append(np.zeros(actions.shape[1], dtype=np.float64))

            if item_index % 20 == 0 or item_index == len(timesteps) - 1:
                bowl_position = snapshot["objects"]["bowl"]["position"]
                plate_position = snapshot["objects"]["plate"]["position"]
                goal_distance = float(np.linalg.norm(bowl_position - plate_position))
                print(
                    "[extract] {:4d}/{:4d} t={:4d} contacts={:2d} "
                    "bowl_z={:.4f} goal_dist={:.4f} success={}".format(
                        item_index,
                        len(timesteps) - 1,
                        timestep,
                        int(snapshot["contact_count"]),
                        float(bowl_position[2]),
                        goal_distance,
                        bool(success[-1]),
                    )
                )

        arrays: Dict[str, np.ndarray] = {
            "timesteps": timesteps,
            "actions": np.asarray(selected_actions, dtype=np.float32),
            "success": np.asarray(success, dtype=np.float32),
            "robot_qpos": _stack(records, ["robot", "qpos"]).astype(np.float32),
            "robot_qvel": _stack(records, ["robot", "qvel"]).astype(np.float32),
            "ee_pos": _stack(records, ["robot", "ee_pos"]).astype(np.float32),
            "ee_quat_xyzw": _stack(records, ["robot", "ee_quat_xyzw"]).astype(np.float32),
            "ee_lin_vel": _stack(records, ["robot", "ee_lin_vel"]).astype(np.float32),
            "ee_ang_vel": _stack(records, ["robot", "ee_ang_vel"]).astype(np.float32),
            "gripper_qpos": _stack(records, ["robot", "gripper_qpos"]).astype(np.float32),
            "gripper_qvel": _stack(records, ["robot", "gripper_qvel"]).astype(np.float32),
            "bowl_position": _stack(records, ["objects", "bowl", "position"]).astype(np.float32),
            "bowl_quat_xyzw": _stack(records, ["objects", "bowl", "quat_xyzw"]).astype(np.float32),
            "bowl_linear_velocity": _stack(records, ["objects", "bowl", "linear_velocity"]).astype(np.float32),
            "bowl_angular_velocity": _stack(records, ["objects", "bowl", "angular_velocity"]).astype(np.float32),
            "bowl_mass": _stack(records, ["objects", "bowl", "mass"]).astype(np.float32),
            "bowl_inertia": _stack(records, ["objects", "bowl", "inertia"]).astype(np.float32),
            "bowl_friction": _stack(records, ["objects", "bowl", "friction"]).astype(np.float32),
            "plate_position": _stack(records, ["objects", "plate", "position"]).astype(np.float32),
            "plate_quat_xyzw": _stack(records, ["objects", "plate", "quat_xyzw"]).astype(np.float32),
            "plate_linear_velocity": _stack(records, ["objects", "plate", "linear_velocity"]).astype(np.float32),
            "plate_angular_velocity": _stack(records, ["objects", "plate", "angular_velocity"]).astype(np.float32),
            "plate_mass": _stack(records, ["objects", "plate", "mass"]).astype(np.float32),
            "plate_inertia": _stack(records, ["objects", "plate", "inertia"]).astype(np.float32),
            "plate_friction": _stack(records, ["objects", "plate", "friction"]).astype(np.float32),
            "contacts": _stack(records, ["contacts"]).astype(np.float32),
            "contact_mask": _stack(records, ["contact_mask"]).astype(np.float32),
            "contact_count": _stack(records, ["contact_count"]).astype(np.int32),
        }

        for name, array in arrays.items():
            if np.issubdtype(array.dtype, np.floating):
                _finite_or_raise(name, array)

        bowl_to_plate = np.linalg.norm(
            arrays["bowl_position"] - arrays["plate_position"],
            axis=-1,
        )
        bowl_z = arrays["bowl_position"][:, 2]
        active_contacts = arrays["contact_mask"].sum(axis=1)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(args.output), **arrays)

        summary_path = args.output.with_suffix(".json")
        summary = {
            "suite": args.suite,
            "task_id": int(args.task_id),
            "task_name": task.name,
            "language": task.language,
            "demo_key": demo["demo_key"],
            "demo_path": str(demo_path),
            "num_source_states": int(len(states)),
            "num_extracted_frames": int(len(timesteps)),
            "stride": int(args.stride),
            "max_contacts": int(args.max_contacts),
            "resolved": description,
            "contact_feature_names": list(CONTACT_FEATURE_NAMES),
            "metrics": {
                "bowl_z_min": float(bowl_z.min()),
                "bowl_z_max": float(bowl_z.max()),
                "bowl_plate_distance_start": float(bowl_to_plate[0]),
                "bowl_plate_distance_end": float(bowl_to_plate[-1]),
                "active_contacts_mean": float(active_contacts.mean()),
                "active_contacts_max": float(active_contacts.max()),
                "success_ever": bool(np.any(arrays["success"] > 0.5)),
                "success_final": bool(arrays["success"][-1] > 0.5),
            },
            "array_shapes": {
                name: list(array.shape)
                for name, array in arrays.items()
            },
            "output_npz": str(args.output.resolve()),
        }

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

        print("\nStep 6 extraction completed")
        print("NPZ    :", args.output.resolve())
        print("Summary:", summary_path.resolve())
        print(json.dumps(summary["metrics"], indent=2))

    finally:
        env.close()


if __name__ == "__main__":
    main()
