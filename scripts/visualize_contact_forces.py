from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
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
from robosuite.utils.camera_utils import get_camera_transform_matrix  # noqa: E402


SCHEMA_VERSION = "dpvla-contact-video-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize LIBERO contact forces and export frame-aligned NPZ/JSON data."
        )
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-contacts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--flip",
        default="rot180",
        choices=("none", "vertical", "horizontal", "rot180"),
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--label-top-k", type=int, default=6)
    parser.add_argument("--min-force", type=float, default=0.5)
    parser.add_argument("--force-scale", type=float, default=0.0025)
    parser.add_argument("--force-cap-m", type=float, default=0.08)
    parser.add_argument(
        "--task-related-only",
        action="store_true",
        help=(
            "Only draw bowl-gripper, bowl-plate, bowl-table, and "
            "robot-environment contacts. NPZ still stores all saved contacts."
        ),
    )
    parser.add_argument(
        "--draw-landmarks",
        action="store_true",
        help="Draw projected bowl, plate, and end-effector centers.",
    )
    parser.add_argument(
        "--pixel-offset-x",
        type=int,
        default=0,
        help="Constant horizontal correction after calibrated projection.",
    )
    parser.add_argument(
        "--pixel-offset-y",
        type=int,
        default=0,
        help="Constant vertical correction after calibrated projection.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("outputs/physics/task02_demo00_contact_force_vis_v5.mp4"),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help=(
            "Frame-aligned data output. Default: <output-video-stem>_data.npz"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Metadata output. Default: same path as output NPZ with .json suffix"
        ),
    )
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    video_path = args.output_video
    npz_path = args.output_npz
    if npz_path is None:
        npz_path = video_path.with_name(f"{video_path.stem}_data.npz")
    json_path = args.output_json
    if json_path is None:
        json_path = npz_path.with_suffix(".json")
    return video_path, npz_path, json_path


def render_synced_rgb(
    env: Any,
    camera_name: str,
    height: int,
    width: int,
) -> np.ndarray:
    """Render RGB from the current state without an implicit flip."""
    rendered = env.sim.render(
        camera_name=camera_name,
        height=height,
        width=width,
    )
    if isinstance(rendered, tuple):
        rendered = rendered[0]
    frame = np.asarray(rendered)
    if frame.dtype != np.uint8:
        if frame.size and frame.max() <= 1.0:
            frame = np.clip(frame * 255.0, 0.0, 255.0)
        else:
            frame = np.clip(frame, 0.0, 255.0)
        frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame)


def apply_image_transform(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        result = frame
    elif mode == "vertical":
        result = cv2.flip(frame, 0)
    elif mode == "horizontal":
        result = cv2.flip(frame, 1)
    elif mode == "rot180":
        result = cv2.rotate(frame, cv2.ROTATE_180)
    else:
        raise ValueError(f"Unsupported transform: {mode}")
    return np.ascontiguousarray(result)


def transform_pixel_rc(
    row: float,
    col: float,
    height: int,
    width: int,
    mode: str,
) -> Tuple[float, float]:
    if mode == "none":
        return row, col
    if mode == "vertical":
        return height - 1 - row, col
    if mode == "horizontal":
        return row, width - 1 - col
    if mode == "rot180":
        return height - 1 - row, width - 1 - col
    raise ValueError(f"Unsupported transform: {mode}")


def projection_transform_for_raw_image_transform(raw_transform: str) -> str:
    """Map robosuite camera-utils pixels into the final displayed frame."""
    mapping = {
        "none": "vertical",
        "vertical": "none",
        "horizontal": "rot180",
        "rot180": "horizontal",
    }
    try:
        return mapping[raw_transform]
    except KeyError as exc:
        raise ValueError(f"Unsupported transform: {raw_transform}") from exc


def make_world_to_pixel(
    env: Any,
    camera_name: str,
    height: int,
    width: int,
) -> np.ndarray:
    return np.asarray(
        get_camera_transform_matrix(
            sim=env.sim,
            camera_name=camera_name,
            camera_height=height,
            camera_width=width,
        ),
        dtype=np.float64,
    )


def project_world_point(
    point_world: np.ndarray,
    world_to_pixel: np.ndarray,
    height: int,
    width: int,
    image_transform: str,
    pixel_offset_x: int = 0,
    pixel_offset_y: int = 0,
) -> Optional[Tuple[int, int]]:
    point = np.asarray(point_world, dtype=np.float64).reshape(3)
    homogeneous = np.concatenate((point, np.ones(1, dtype=np.float64)))
    projected = world_to_pixel @ homogeneous
    depth = float(projected[2])
    if not np.isfinite(depth) or depth <= 1e-10:
        return None

    col = float(projected[0] / depth)
    row = float(projected[1] / depth)
    if not np.isfinite(row) or not np.isfinite(col):
        return None

    projection_transform = projection_transform_for_raw_image_transform(
        image_transform
    )
    row, col = transform_pixel_rc(
        row=row,
        col=col,
        height=height,
        width=width,
        mode=projection_transform,
    )
    x = int(round(col)) + int(pixel_offset_x)
    y = int(round(row)) + int(pixel_offset_y)
    return x, y


def feature_indices() -> Dict[str, int]:
    return {name: index for index, name in enumerate(CONTACT_FEATURE_NAMES)}


def category(
    row: np.ndarray,
    index: Dict[str, int],
) -> Tuple[str, Tuple[int, int, int]]:
    if row[index["is_bowl_gripper"]] > 0.5:
        return "bowl-gripper", (0, 0, 255)
    if row[index["is_bowl_plate"]] > 0.5:
        return "bowl-plate", (0, 255, 0)
    if row[index["is_bowl_table"]] > 0.5:
        return "bowl-table", (255, 0, 0)
    if row[index["is_robot_environment"]] > 0.5:
        return "robot-env", (0, 255, 255)
    if row[index["touches_bowl"]] > 0.5:
        return "bowl-other", (255, 0, 255)
    if row[index["touches_plate"]] > 0.5:
        return "plate-other", (255, 255, 0)
    return "other", (180, 180, 180)


def is_task_related(row: np.ndarray, index: Dict[str, int]) -> bool:
    return bool(
        max(
            row[index["is_bowl_gripper"]],
            row[index["is_bowl_plate"]],
            row[index["is_bowl_table"]],
            row[index["is_robot_environment"]],
        )
        > 0.5
    )


def draw_landmark(
    frame: np.ndarray,
    point_world: np.ndarray,
    text: str,
    color: Tuple[int, int, int],
    world_to_pixel: np.ndarray,
    args: argparse.Namespace,
) -> None:
    pixel = project_world_point(
        point_world=point_world,
        world_to_pixel=world_to_pixel,
        height=args.height,
        width=args.width,
        image_transform=args.flip,
        pixel_offset_x=args.pixel_offset_x,
        pixel_offset_y=args.pixel_offset_y,
    )
    if pixel is None:
        return
    x, y = pixel
    if not (0 <= x < args.width and 0 <= y < args.height):
        return
    cv2.drawMarker(
        frame,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x + 6, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color,
        1,
        cv2.LINE_AA,
    )


def compute_contact_projection_arrays(
    contacts: np.ndarray,
    mask: np.ndarray,
    world_to_pixel: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project every saved contact into the final video coordinate system.

    Returns:
        start_xy: [C, 2] int32. Contact point pixel; invalid entries are [-1, -1].
        end_xy: [C, 2] int32. Force-arrow endpoint pixel; invalid entries are [-1, -1].
        valid: [C] uint8. Both start and endpoint are projectable.

    These arrays are not limited by top-k/min-force/task-related drawing filters.
    They preserve one-to-one alignment with ``contacts``.
    """
    contacts = np.asarray(contacts)
    mask = np.asarray(mask) > 0.5
    count = int(contacts.shape[0])
    start_xy = np.full((count, 2), -1, dtype=np.int32)
    end_xy = np.full((count, 2), -1, dtype=np.int32)
    valid = np.zeros((count,), dtype=np.uint8)

    index = feature_indices()
    force_norm_index = index["force_norm"]

    for contact_index in np.flatnonzero(mask):
        row = contacts[contact_index]
        position = row[0:3].astype(np.float64)
        force = row[6:9].astype(np.float64)
        magnitude = float(row[force_norm_index])

        start = project_world_point(
            point_world=position,
            world_to_pixel=world_to_pixel,
            height=args.height,
            width=args.width,
            image_transform=args.flip,
            pixel_offset_x=args.pixel_offset_x,
            pixel_offset_y=args.pixel_offset_y,
        )
        if start is None:
            continue

        if magnitude > 1e-10:
            direction = force / (np.linalg.norm(force) + 1e-12)
            arrow_length = min(
                args.force_cap_m,
                args.force_scale * magnitude,
            )
            end_world = position + direction * arrow_length
        else:
            end_world = position

        end = project_world_point(
            point_world=end_world,
            world_to_pixel=world_to_pixel,
            height=args.height,
            width=args.width,
            image_transform=args.flip,
            pixel_offset_x=args.pixel_offset_x,
            pixel_offset_y=args.pixel_offset_y,
        )
        if end is None:
            continue

        start_xy[contact_index] = start
        end_xy[contact_index] = end
        valid[contact_index] = 1

    return start_xy, end_xy, valid


def draw_contacts(
    frame: np.ndarray,
    contacts: np.ndarray,
    mask: np.ndarray,
    world_to_pixel: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, float]:
    index = feature_indices()
    rows = np.asarray(contacts)[np.asarray(mask) > 0.5]
    saved_count = int(len(rows))
    if args.task_related_only and len(rows):
        rows = np.asarray([row for row in rows if is_task_related(row, index)])
    if not len(rows):
        return {"saved": saved_count, "drawn": 0, "max_force": 0.0}

    rows = rows[rows[:, index["force_norm"]] >= args.min_force]
    if not len(rows):
        return {"saved": saved_count, "drawn": 0, "max_force": 0.0}

    order = np.argsort(-rows[:, index["force_norm"]])
    rows = rows[order[: args.top_k]]
    max_force = float(rows[:, index["force_norm"]].max())
    drawn = 0

    for rank, row in enumerate(rows):
        position = row[0:3].astype(np.float64)
        force = row[6:9].astype(np.float64)
        magnitude = float(row[index["force_norm"]])
        if magnitude <= 1e-10:
            continue
        direction = force / (np.linalg.norm(force) + 1e-12)
        end_world = position + direction * min(
            args.force_cap_m,
            args.force_scale * magnitude,
        )

        start = project_world_point(
            position,
            world_to_pixel,
            args.height,
            args.width,
            args.flip,
            args.pixel_offset_x,
            args.pixel_offset_y,
        )
        end = project_world_point(
            end_world,
            world_to_pixel,
            args.height,
            args.width,
            args.flip,
            args.pixel_offset_x,
            args.pixel_offset_y,
        )
        if start is None or end is None:
            continue
        x0, y0 = start
        x1, y1 = end
        if not (-32 <= x0 < args.width + 32 and -32 <= y0 < args.height + 32):
            continue

        label, color = category(row, index)
        cv2.circle(frame, (x0, y0), 4, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(
            frame,
            (x0, y0),
            (x1, y1),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )
        if rank < args.label_top_k:
            cv2.putText(
                frame,
                f"{label}:{magnitude:.1f}N",
                (x0 + 5, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        drawn += 1

    return {"saved": saved_count, "drawn": drawn, "max_force": max_force}


def draw_status(
    frame: np.ndarray,
    frame_id: int,
    timestep: int,
    raw_contacts: int,
    statistics: Dict[str, float],
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (480, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0.0, frame)
    lines = (
        f"frame={frame_id} timestep={timestep}",
        (
            f"contacts raw={raw_contacts} "
            f"saved={int(statistics['saved'])} "
            f"drawn={int(statistics['drawn'])}"
        ),
        f"max_force={statistics['max_force']:.2f} N",
    )
    for line_index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (8, 20 + 24 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def stack_path(records: Sequence[Dict[str, Any]], path: Sequence[str]) -> np.ndarray:
    values: List[np.ndarray] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(np.asarray(value))
    return np.stack(values, axis=0)


def finite_or_raise(name: str, array: np.ndarray) -> None:
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        bad = np.argwhere(~np.isfinite(array))
        raise FloatingPointError(
            f"Non-finite values in {name} at {bad[:10].tolist()}"
        )


def build_arrays(
    records: Sequence[Dict[str, Any]],
    timesteps: np.ndarray,
    selected_actions: Sequence[np.ndarray],
    success: Sequence[float],
    world_to_pixel_matrices: Sequence[np.ndarray],
    contact_start_pixels: Sequence[np.ndarray],
    contact_end_pixels: Sequence[np.ndarray],
    contact_projection_masks: Sequence[np.ndarray],
    drawn_counts: Sequence[int],
    displayed_max_forces: Sequence[float],
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        "video_frame_index": np.arange(len(records), dtype=np.int32),
        "timesteps": np.asarray(timesteps, dtype=np.int32),
        "actions": np.asarray(selected_actions, dtype=np.float32),
        "success": np.asarray(success, dtype=np.float32),
        "robot_qpos": stack_path(records, ("robot", "qpos")).astype(np.float32),
        "robot_qvel": stack_path(records, ("robot", "qvel")).astype(np.float32),
        "ee_pos": stack_path(records, ("robot", "ee_pos")).astype(np.float32),
        "ee_quat_xyzw": stack_path(records, ("robot", "ee_quat_xyzw")).astype(np.float32),
        "ee_lin_vel": stack_path(records, ("robot", "ee_lin_vel")).astype(np.float32),
        "ee_ang_vel": stack_path(records, ("robot", "ee_ang_vel")).astype(np.float32),
        "gripper_qpos": stack_path(records, ("robot", "gripper_qpos")).astype(np.float32),
        "gripper_qvel": stack_path(records, ("robot", "gripper_qvel")).astype(np.float32),
        "bowl_position": stack_path(records, ("objects", "bowl", "position")).astype(np.float32),
        "bowl_quat_xyzw": stack_path(records, ("objects", "bowl", "quat_xyzw")).astype(np.float32),
        "bowl_linear_velocity": stack_path(records, ("objects", "bowl", "linear_velocity")).astype(np.float32),
        "bowl_angular_velocity": stack_path(records, ("objects", "bowl", "angular_velocity")).astype(np.float32),
        "bowl_mass": stack_path(records, ("objects", "bowl", "mass")).astype(np.float32),
        "bowl_inertia": stack_path(records, ("objects", "bowl", "inertia")).astype(np.float32),
        "bowl_friction": stack_path(records, ("objects", "bowl", "friction")).astype(np.float32),
        "plate_position": stack_path(records, ("objects", "plate", "position")).astype(np.float32),
        "plate_quat_xyzw": stack_path(records, ("objects", "plate", "quat_xyzw")).astype(np.float32),
        "plate_linear_velocity": stack_path(records, ("objects", "plate", "linear_velocity")).astype(np.float32),
        "plate_angular_velocity": stack_path(records, ("objects", "plate", "angular_velocity")).astype(np.float32),
        "plate_mass": stack_path(records, ("objects", "plate", "mass")).astype(np.float32),
        "plate_inertia": stack_path(records, ("objects", "plate", "inertia")).astype(np.float32),
        "plate_friction": stack_path(records, ("objects", "plate", "friction")).astype(np.float32),
        "contacts": stack_path(records, ("contacts",)).astype(np.float32),
        "contact_mask": stack_path(records, ("contact_mask",)).astype(np.float32),
        "contact_count": stack_path(records, ("contact_count",)).astype(np.int32),
        # Camera matrix produced for the exact state rendered in each video frame.
        "world_to_pixel": np.asarray(world_to_pixel_matrices, dtype=np.float64),
        # Pixel coordinates are in the final output-video coordinate system.
        "contact_start_xy": np.asarray(contact_start_pixels, dtype=np.int32),
        "contact_end_xy": np.asarray(contact_end_pixels, dtype=np.int32),
        "contact_projection_mask": np.asarray(contact_projection_masks, dtype=np.uint8),
        "drawn_contact_count": np.asarray(drawn_counts, dtype=np.int32),
        "displayed_max_force": np.asarray(displayed_max_forces, dtype=np.float32),
    }

    for name, array in arrays.items():
        finite_or_raise(name, array)
    return arrays


def masked_values(
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(values)[np.asarray(mask) > 0.5]
    return np.asarray(selected)


def build_summary(
    args: argparse.Namespace,
    task: Any,
    demo: Dict[str, Any],
    demo_path: Path,
    adapter_description: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    video_path: Path,
    npz_path: Path,
    json_path: Path,
    num_source_states: int,
    num_source_actions: int,
) -> Dict[str, Any]:
    index = feature_indices()
    contact_mask = arrays["contact_mask"] > 0.5
    force_norms = masked_values(
        arrays["contacts"][..., index["force_norm"]],
        contact_mask,
    )
    normal_forces = masked_values(
        arrays["contacts"][..., index["normal_force"]],
        contact_mask,
    )
    projection_mask = arrays["contact_projection_mask"] > 0
    active_contacts = arrays["contact_mask"].sum(axis=1)
    raw_contact_count = arrays["contact_count"]
    bowl_to_plate = np.linalg.norm(
        arrays["bowl_position"] - arrays["plate_position"],
        axis=-1,
    )
    bowl_z = arrays["bowl_position"][:, 2]

    valid_contact_total = int(contact_mask.sum())
    projected_contact_total = int((projection_mask & contact_mask).sum())

    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "suite": args.suite,
        "task_id": int(args.task_id),
        "task_name": task.name,
        "language": task.language,
        "demo_key": demo["demo_key"],
        "demo_path": str(demo_path),
        "num_source_states": int(num_source_states),
        "num_source_actions": int(num_source_actions),
        "num_video_frames": int(len(arrays["video_frame_index"])),
        "frame_alignment": (
            "Video frame i corresponds exactly to row i of every time-major NPZ "
            "array. timesteps[i] is the source demonstration state index."
        ),
        "outputs": {
            "video": str(video_path.resolve()),
            "npz": str(npz_path.resolve()),
            "json": str(json_path.resolve()),
        },
        "video": {
            "camera": args.camera,
            "width": int(args.width),
            "height": int(args.height),
            "fps": float(args.fps),
            "stride": int(args.stride),
            "flip": args.flip,
            "projection_pixel_transform": (
                projection_transform_for_raw_image_transform(args.flip)
            ),
            "pixel_offset_x": int(args.pixel_offset_x),
            "pixel_offset_y": int(args.pixel_offset_y),
            "top_k": int(args.top_k),
            "label_top_k": int(args.label_top_k),
            "min_force": float(args.min_force),
            "force_scale_m_per_newton": float(args.force_scale),
            "force_cap_m": float(args.force_cap_m),
            "task_related_only_for_drawing": bool(args.task_related_only),
            "draw_landmarks": bool(args.draw_landmarks),
        },
        "contact_schema": {
            "max_contacts": int(args.max_contacts),
            "feature_names": list(CONTACT_FEATURE_NAMES),
            "contact_start_xy": (
                "Projected contact point in final video pixel coordinates [x, y]. "
                "Invalid entries are [-1, -1]."
            ),
            "contact_end_xy": (
                "Projected force-arrow endpoint in final video pixel coordinates "
                "using the configured force scale/cap. Invalid entries are [-1, -1]."
            ),
            "contact_projection_mask": (
                "1 when both contact_start_xy and contact_end_xy are projectable."
            ),
        },
        "resolved": adapter_description,
        "metrics": {
            "bowl_z_min": float(bowl_z.min()),
            "bowl_z_max": float(bowl_z.max()),
            "bowl_plate_distance_start": float(bowl_to_plate[0]),
            "bowl_plate_distance_end": float(bowl_to_plate[-1]),
            "active_contacts_mean": float(active_contacts.mean()),
            "active_contacts_max": float(active_contacts.max()),
            "raw_contacts_min": int(raw_contact_count.min()),
            "raw_contacts_mean": float(raw_contact_count.mean()),
            "raw_contacts_max": int(raw_contact_count.max()),
            "truncated_frame_count": int(
                np.sum(raw_contact_count > int(args.max_contacts))
            ),
            "valid_contact_total": valid_contact_total,
            "projected_contact_total": projected_contact_total,
            "projection_success_ratio": (
                float(projected_contact_total / valid_contact_total)
                if valid_contact_total
                else 0.0
            ),
            "normal_force_max": (
                float(normal_forces.max()) if normal_forces.size else 0.0
            ),
            "normal_force_mean": (
                float(normal_forces.mean()) if normal_forces.size else 0.0
            ),
            "force_norm_max": (
                float(force_norms.max()) if force_norms.size else 0.0
            ),
            "force_norm_mean": (
                float(force_norms.mean()) if force_norms.size else 0.0
            ),
            "force_nonzero_ratio": (
                float(np.mean(force_norms > 1e-8)) if force_norms.size else 0.0
            ),
            "drawn_contacts_mean": float(arrays["drawn_contact_count"].mean()),
            "drawn_contacts_max": int(arrays["drawn_contact_count"].max()),
            "success_ever": bool(np.any(arrays["success"] > 0.5)),
            "success_final": bool(arrays["success"][-1] > 0.5),
        },
        "array_shapes": {
            name: list(array.shape) for name, array in arrays.items()
        },
        "array_dtypes": {
            name: str(array.dtype) for name, array in arrays.items()
        },
    }
    return summary


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    if args.max_contacts <= 0:
        raise ValueError("max_contacts must be positive")

    video_path, npz_path, json_path = resolve_output_paths(args)

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
    stop = len(states) if args.max_steps < 0 else min(len(states), args.max_steps)
    timesteps = np.arange(0, stop, args.stride, dtype=np.int32)
    if timesteps.size == 0:
        raise ValueError("No frames selected. Check max_steps and stride.")

    env = make_environment(
        bddl_path=bddl_path,
        camera_name=args.camera,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    writer: Optional[cv2.VideoWriter] = None
    records: List[Dict[str, Any]] = []
    selected_actions: List[np.ndarray] = []
    success: List[float] = []
    world_to_pixel_matrices: List[np.ndarray] = []
    contact_start_pixels: List[np.ndarray] = []
    contact_end_pixels: List[np.ndarray] = []
    contact_projection_masks: List[np.ndarray] = []
    drawn_counts: List[int] = []
    displayed_max_forces: List[float] = []

    try:
        restore_recorded_model(env, demo["model_xml"])
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
        adapter_description = adapter.describe()

        for path in (video_path, npz_path, json_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.width, args.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {video_path}")

        print(f"[info] task: {task.name}")
        projection_transform = projection_transform_for_raw_image_transform(args.flip)
        print("[info] projection: robosuite camera_utils calibrated matrix")
        print(f"[info] raw image transform: {args.flip} (applied exactly once)")
        print(
            "[info] projected-pixel transform: "
            f"{projection_transform} (camera_utils pixels -> output frame)"
        )
        print(f"[info] video: {video_path.resolve()}")
        print(f"[info] npz  : {npz_path.resolve()}")
        print(f"[info] json : {json_path.resolve()}")

        zero_action = np.zeros(
            actions.shape[1] if actions.ndim == 2 else 0,
            dtype=np.float64,
        )

        for frame_id, timestep in enumerate(timesteps.tolist()):
            env.regenerate_obs_from_state(states[timestep])
            env.sim.forward()

            # Rendering, state/contact extraction, and camera matrix all correspond
            # to this exact forward-solved state.
            frame_rgb = render_synced_rgb(
                env,
                args.camera,
                args.height,
                args.width,
            )
            snapshot = adapter.extract()
            world_to_pixel = make_world_to_pixel(
                env,
                args.camera,
                args.height,
                args.width,
            )

            start_xy, end_xy, projection_mask = compute_contact_projection_arrays(
                contacts=snapshot["contacts"],
                mask=snapshot["contact_mask"],
                world_to_pixel=world_to_pixel,
                args=args,
            )

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = apply_image_transform(frame_bgr, args.flip)

            statistics = draw_contacts(
                frame=frame_bgr,
                contacts=snapshot["contacts"],
                mask=snapshot["contact_mask"],
                world_to_pixel=world_to_pixel,
                args=args,
            )

            if args.draw_landmarks:
                draw_landmark(
                    frame_bgr,
                    snapshot["objects"]["bowl"]["position"],
                    "BOWL-CENTER",
                    (255, 0, 255),
                    world_to_pixel,
                    args,
                )
                draw_landmark(
                    frame_bgr,
                    snapshot["objects"]["plate"]["position"],
                    "PLATE-CENTER",
                    (0, 255, 0),
                    world_to_pixel,
                    args,
                )
                draw_landmark(
                    frame_bgr,
                    snapshot["robot"]["ee_pos"],
                    "EE",
                    (0, 165, 255),
                    world_to_pixel,
                    args,
                )

            draw_status(
                frame_bgr,
                frame_id,
                int(timestep),
                int(snapshot["contact_count"]),
                statistics,
            )
            writer.write(frame_bgr)

            records.append(snapshot)
            selected_actions.append(
                actions[timestep].copy() if timestep < len(actions) else zero_action.copy()
            )
            success.append(float(check_success(env)))
            world_to_pixel_matrices.append(world_to_pixel.copy())
            contact_start_pixels.append(start_xy)
            contact_end_pixels.append(end_xy)
            contact_projection_masks.append(projection_mask)
            drawn_counts.append(int(statistics["drawn"]))
            displayed_max_forces.append(float(statistics["max_force"]))

            if frame_id % 20 == 0 or frame_id == len(timesteps) - 1:
                print(
                    "[vis] {}/{} t={} raw={} drawn={} max={:.2f}N".format(
                        frame_id,
                        len(timesteps) - 1,
                        timestep,
                        int(snapshot["contact_count"]),
                        int(statistics["drawn"]),
                        statistics["max_force"],
                    )
                )

        # Close the writer before reporting successful output.
        writer.release()
        writer = None

        arrays = build_arrays(
            records=records,
            timesteps=timesteps,
            selected_actions=selected_actions,
            success=success,
            world_to_pixel_matrices=world_to_pixel_matrices,
            contact_start_pixels=contact_start_pixels,
            contact_end_pixels=contact_end_pixels,
            contact_projection_masks=contact_projection_masks,
            drawn_counts=drawn_counts,
            displayed_max_forces=displayed_max_forces,
        )
        np.savez_compressed(str(npz_path), **arrays)

        summary = build_summary(
            args=args,
            task=task,
            demo=demo,
            demo_path=demo_path,
            adapter_description=adapter_description,
            arrays=arrays,
            video_path=video_path,
            npz_path=npz_path,
            json_path=json_path,
            num_source_states=len(states),
            num_source_actions=len(actions),
        )
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

        print("\n[done] synchronized outputs generated")
        print(f"video: {video_path.resolve()}")
        print(f"npz  : {npz_path.resolve()}")
        print(f"json : {json_path.resolve()}")
        print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))

    finally:
        if writer is not None:
            writer.release()
        env.close()


if __name__ == "__main__":
    main()
