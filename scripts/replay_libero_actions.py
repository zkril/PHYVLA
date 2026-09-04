"""
Replay a LIBERO demonstration by executing recorded actions.

Example:
python scripts/replay_libero_actions.py \
    --suite libero_spatial \
    --task-id 2 \
    --demo-index 0 \
    --output outputs/replay/task02_demo00_actions.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import imageio.v2 as imageio
import numpy as np
import robosuite

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay LIBERO expert actions."
    )

    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
    )

    parser.add_argument(
        "--task-id",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--demo-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--video-stride",
        type=int,
        default=1,
        help="Save one video frame every N actions.",
    )

    parser.add_argument(
        "--max-actions",
        type=int,
        default=-1,
        help="-1 means execute the entire demonstration.",
    )

    parser.add_argument(
        "--error-threshold",
        type=float,
        default=1e-2,
        help="Threshold used to flag state replay drift.",
    )

    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "rot180"],
        default="none",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/replay/task02_demo00_actions.mp4"
        ),
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# HDF5 and task utilities
# ---------------------------------------------------------------------

def decode_attr(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")

    return str(value)


def demo_sort_key(name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", name)

    if match is None:
        return 10**9, name

    return int(match.group(1)), name


def resolve_task_and_demo(
    suite_name: str,
    task_id: int,
):
    benchmark_dict = benchmark.get_benchmark_dict()

    if suite_name not in benchmark_dict:
        raise ValueError(
            "Unknown suite '{}'. Available suites: {}".format(
                suite_name,
                sorted(benchmark_dict.keys()),
            )
        )

    suite = benchmark_dict[suite_name]()

    if not 0 <= task_id < suite.n_tasks:
        raise ValueError(
            "task_id {} is outside [0, {})".format(
                task_id,
                suite.n_tasks,
            )
        )

    task = suite.get_task(task_id)

    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )

    relative_demo_path = suite.get_task_demonstration(
        task_id
    )

    demo_path = (
        Path(get_libero_path("datasets"))
        / relative_demo_path
    )

    if not bddl_path.exists():
        raise FileNotFoundError(
            "BDDL file not found: {}".format(bddl_path)
        )

    if not demo_path.exists():
        raise FileNotFoundError(
            "Demonstration file not found: {}".format(
                demo_path
            )
        )

    return suite, task, bddl_path, demo_path


def load_demonstration(
    demo_path: Path,
    demo_index: int,
) -> Dict[str, Any]:
    with h5py.File(str(demo_path), "r") as file:
        if "data" not in file:
            raise KeyError(
                "HDF5 file does not contain group 'data'"
            )

        data_group = file["data"]

        demo_keys = sorted(
            list(data_group.keys()),
            key=demo_sort_key,
        )

        if not demo_keys:
            raise RuntimeError(
                "No demonstrations found in {}".format(
                    demo_path
                )
            )

        if not 0 <= demo_index < len(demo_keys):
            raise IndexError(
                "demo_index {} outside [0, {})".format(
                    demo_index,
                    len(demo_keys),
                )
            )

        demo_key = demo_keys[demo_index]
        episode = data_group[demo_key]

        states = np.asarray(
            episode["states"][()],
            dtype=np.float64,
        )

        actions = np.asarray(
            episode["actions"][()],
            dtype=np.float64,
        )

        model_xml = decode_attr(
            episode.attrs.get("model_file")
        )

        return {
            "demo_key": demo_key,
            "states": states,
            "actions": actions,
            "model_xml": model_xml,
            "num_demos": len(demo_keys),
        }


# ---------------------------------------------------------------------
# XML compatibility patch
# ---------------------------------------------------------------------

def safe_edit_model_xml(xml_string: str) -> str:
    """
    Fix asset paths without calling old robosuite edit_model_xml(),
    which may fail when a path contains no 'robosuite' component.
    """

    root = ET.fromstring(xml_string)
    asset_node = root.find("asset")

    if asset_node is None:
        return xml_string

    robosuite_root = Path(
        robosuite.__file__
    ).resolve().parent

    try:
        libero_assets_root = Path(
            get_libero_path("assets")
        ).resolve()
    except Exception:
        libero_assets_root = None

    asset_elements = (
        list(asset_node.findall("mesh"))
        + list(asset_node.findall("texture"))
        + list(asset_node.findall("hfield"))
    )

    rewritten_count = 0
    unresolved_paths = []

    for element in asset_elements:
        old_file = element.get("file")

        if not old_file:
            continue

        old_path = Path(old_file)

        # Path already valid.
        if old_path.exists():
            continue

        normalized = old_file.replace("\\", "/")
        path_parts = normalized.split("/")

        candidate = None

        # Map an old robosuite installation to the current one.
        robosuite_indices = [
            index
            for index, part in enumerate(path_parts)
            if part == "robosuite"
        ]

        if robosuite_indices:
            index = robosuite_indices[-1]

            possible_path = robosuite_root.joinpath(
                *path_parts[index + 1:]
            )

            if possible_path.exists():
                candidate = possible_path

        # Map an old LIBERO asset root to the current asset root.
        if (
            candidate is None
            and libero_assets_root is not None
            and "assets" in path_parts
        ):
            assets_index = max(
                index
                for index, part in enumerate(path_parts)
                if part == "assets"
            )

            possible_path = libero_assets_root.joinpath(
                *path_parts[assets_index + 1:]
            )

            if possible_path.exists():
                candidate = possible_path

        if candidate is not None:
            element.set(
                "file",
                str(candidate.resolve()),
            )
            rewritten_count += 1
        else:
            unresolved_paths.append(old_file)

    print(
        "[info] XML asset paths rewritten: {}".format(
            rewritten_count
        )
    )

    if unresolved_paths:
        print(
            "[warning] unresolved XML asset paths: {}".format(
                len(unresolved_paths)
            )
        )

        for path in unresolved_paths[:10]:
            print("  -", path)

    return ET.tostring(
        root,
        encoding="unicode",
    )


# ---------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------

def make_environment(
    bddl_path: Path,
    camera_name: str,
    height: int,
    width: int,
    seed: int,
) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=[camera_name],
        camera_heights=height,
        camera_widths=width,
        camera_depths=False,
        use_camera_obs=True,
        ignore_done=True,
        control_freq=20,
    )

    env.seed(seed)
    env.reset()

    return env


def restore_recorded_model(
    env: OffScreenRenderEnv,
    model_xml: Optional[str],
) -> None:
    if model_xml is None:
        print(
            "[warning] Demonstration contains no model_file. "
            "Using the model generated from the BDDL file."
        )
        return

    edited_xml = safe_edit_model_xml(model_xml)

    env.reset_from_xml_string(edited_xml)
    env.sim.reset()
    env.sim.forward()

    print("[info] recorded MuJoCo model restored")


def get_action_dimension(env: OffScreenRenderEnv) -> int:
    current = env

    while hasattr(current, "env"):
        current = current.env

    if hasattr(current, "action_dim"):
        return int(current.action_dim)

    if hasattr(env, "action_dim"):
        return int(env.action_dim)

    raise AttributeError(
        "Unable to determine environment action dimension"
    )


def get_sim_state(env: OffScreenRenderEnv) -> np.ndarray:
    if hasattr(env, "get_sim_state"):
        return np.asarray(
            env.get_sim_state(),
            dtype=np.float64,
        )

    return np.asarray(
        env.sim.get_state().flatten(),
        dtype=np.float64,
    )


def check_success(env: OffScreenRenderEnv) -> bool:
    if hasattr(env, "check_success"):
        return bool(env.check_success())

    current = env

    while hasattr(current, "env"):
        current = current.env

    if hasattr(current, "_check_success"):
        return bool(current._check_success())

    return False


def step_environment(
    env: OffScreenRenderEnv,
    action: np.ndarray,
):
    result = env.step(action)

    # Old Gym interface.
    if len(result) == 4:
        observation, reward, done, info = result
        return observation, reward, done, info

    # Gymnasium-style interface.
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        return observation, reward, done, info

    raise RuntimeError(
        "Unexpected env.step return length: {}".format(
            len(result)
        )
    )


def extract_frame(
    observation: Dict[str, np.ndarray],
    camera_name: str,
    flip: str,
) -> np.ndarray:
    image_key = "{}_image".format(camera_name)

    if image_key not in observation:
        raise KeyError(
            "Observation does not contain '{}'. Keys:\n{}".format(
                image_key,
                sorted(observation.keys()),
            )
        )

    frame = np.asarray(observation[image_key])

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0

        frame = np.clip(
            frame,
            0,
            255,
        ).astype(np.uint8)

    if flip == "vertical":
        frame = frame[::-1]

    elif flip == "horizontal":
        frame = frame[:, ::-1]

    elif flip == "rot180":
        frame = frame[::-1, ::-1]

    return np.ascontiguousarray(frame)


# ---------------------------------------------------------------------
# Main replay logic
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    suite, task, bddl_path, demo_path = (
        resolve_task_and_demo(
            suite_name=args.suite,
            task_id=args.task_id,
        )
    )

    demo = load_demonstration(
        demo_path=demo_path,
        demo_index=args.demo_index,
    )

    states = demo["states"]
    actions = demo["actions"]
    model_xml = demo["model_xml"]

    print("=" * 80)
    print("LIBERO action replay")
    print("Suite       :", args.suite)
    print("Task ID     :", args.task_id)
    print("Task name   :", task.name)
    print("Language    :", task.language)
    print("BDDL        :", bddl_path)
    print("HDF5        :", demo_path)
    print("Episode     :", demo["demo_key"])
    print("States      :", states.shape)
    print("Actions     :", actions.shape)
    print("Model XML   :", model_xml is not None)
    print("=" * 80)

    if states.ndim != 2:
        raise ValueError(
            "Expected states [T, D], got {}".format(
                states.shape
            )
        )

    if actions.ndim != 2:
        raise ValueError(
            "Expected actions [T, A], got {}".format(
                actions.shape
            )
        )

    env = make_environment(
        bddl_path=bddl_path,
        camera_name=args.camera,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    writer = None

    try:
        restore_recorded_model(
            env=env,
            model_xml=model_xml,
        )

        # Restore the initial state recorded in the demonstration.
        initial_observation = env.regenerate_obs_from_state(
            states[0]
        )

        current_state = get_sim_state(env)

        if current_state.shape[0] != states.shape[1]:
            raise RuntimeError(
                "State dimension mismatch:\n"
                "  HDF5 state dim: {}\n"
                "  Environment:    {}".format(
                    states.shape[1],
                    current_state.shape[0],
                )
            )

        action_dimension = get_action_dimension(env)

        if actions.shape[1] != action_dimension:
            raise RuntimeError(
                "Action dimension mismatch:\n"
                "  HDF5 action dim: {}\n"
                "  Environment:     {}".format(
                    actions.shape[1],
                    action_dimension,
                )
            )

        num_actions = len(actions)

        if args.max_actions >= 0:
            num_actions = min(
                num_actions,
                args.max_actions,
            )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path = args.summary

        if summary_path is None:
            summary_path = args.output.with_suffix(
                ".json"
            )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer = imageio.get_writer(
            str(args.output),
            fps=args.fps,
            codec="libx264",
            quality=8,
            ffmpeg_params=[
                "-pix_fmt",
                "yuv420p",
            ],
        )

        writer.append_data(
            extract_frame(
                observation=initial_observation,
                camera_name=args.camera,
                flip=args.flip,
            )
        )

        state_errors = []
        rewards = []
        success_steps = []
        first_large_error = None

        success_ever = check_success(env)

        for action_index in range(num_actions):
            action = np.asarray(
                actions[action_index],
                dtype=np.float64,
            )

            observation, reward, done, info = (
                step_environment(
                    env=env,
                    action=action,
                )
            )

            reward_value = float(reward)
            rewards.append(reward_value)

            success_now = check_success(env)

            if success_now:
                success_ever = True
                success_steps.append(action_index)

            # actions[t] is applied at states[t].
            # Its result should approximately match states[t + 1].
            if action_index + 1 < len(states):
                actual_state = get_sim_state(env)
                expected_state = states[
                    action_index + 1
                ]

                difference = (
                    actual_state - expected_state
                )

                l2_error = float(
                    np.linalg.norm(difference)
                )

                mean_absolute_error = float(
                    np.mean(np.abs(difference))
                )

                max_absolute_error = float(
                    np.max(np.abs(difference))
                )

                state_errors.append(
                    {
                        "step": int(action_index),
                        "l2": l2_error,
                        "mae": mean_absolute_error,
                        "max_abs": max_absolute_error,
                    }
                )

                if (
                    l2_error > args.error_threshold
                    and first_large_error is None
                ):
                    first_large_error = {
                        "step": int(action_index),
                        "l2": l2_error,
                        "mae": mean_absolute_error,
                        "max_abs": max_absolute_error,
                    }

            if (
                action_index % args.video_stride
                == 0
            ):
                frame = extract_frame(
                    observation=observation,
                    camera_name=args.camera,
                    flip=args.flip,
                )

                writer.append_data(frame)

            if action_index % 10 == 0:
                latest_error = (
                    state_errors[-1]["l2"]
                    if state_errors
                    else float("nan")
                )

                print(
                    "[action] {:4d}/{:4d} | "
                    "reward={:.3f} | "
                    "success={} | "
                    "state_l2={:.6f}".format(
                        action_index,
                        num_actions - 1,
                        reward_value,
                        success_now,
                        latest_error,
                    )
                )

        l2_errors = np.asarray(
            [
                item["l2"]
                for item in state_errors
            ],
            dtype=np.float64,
        )

        mae_errors = np.asarray(
            [
                item["mae"]
                for item in state_errors
            ],
            dtype=np.float64,
        )

        if l2_errors.size > 0:
            error_statistics = {
                "l2_mean": float(
                    l2_errors.mean()
                ),
                "l2_median": float(
                    np.median(l2_errors)
                ),
                "l2_p95": float(
                    np.percentile(
                        l2_errors,
                        95,
                    )
                ),
                "l2_max": float(
                    l2_errors.max()
                ),
                "mae_mean": float(
                    mae_errors.mean()
                ),
                "num_above_threshold": int(
                    np.sum(
                        l2_errors
                        > args.error_threshold
                    )
                ),
            }
        else:
            error_statistics = {
                "l2_mean": None,
                "l2_median": None,
                "l2_p95": None,
                "l2_max": None,
                "mae_mean": None,
                "num_above_threshold": 0,
            }

        final_success = check_success(env)

        summary = {
            "suite": args.suite,
            "task_id": int(args.task_id),
            "task_name": task.name,
            "language": task.language,
            "demo_key": demo["demo_key"],
            "demo_path": str(demo_path),
            "states_shape": list(
                states.shape
            ),
            "actions_shape": list(
                actions.shape
            ),
            "num_actions_executed": int(
                num_actions
            ),
            "success_ever": bool(
                success_ever
            ),
            "final_success": bool(
                final_success
            ),
            "success_steps": success_steps,
            "max_reward": (
                float(max(rewards))
                if rewards
                else None
            ),
            "error_threshold": float(
                args.error_threshold
            ),
            "state_error": error_statistics,
            "first_large_error": (
                first_large_error
            ),
            "video_path": str(
                args.output.resolve()
            ),
        }

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print("=" * 80)
        print("Action replay completed")
        print("Success ever :", success_ever)
        print("Final success:", final_success)
        print("Error stats  :", error_statistics)
        print("Video        :", args.output.resolve())
        print("Summary      :", summary_path.resolve())
        print("=" * 80)

    finally:
        if writer is not None:
            writer.close()

        env.close()


if __name__ == "__main__":
    main()
