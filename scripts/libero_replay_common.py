"""
LIBERO demonstration replay utilities.

Compatible with Python 3.8.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import os
import xml.etree.ElementTree as ET

import robosuite

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def decode_hdf5_attr(value: Any) -> Optional[str]:
    """Convert a HDF5 string/bytes attribute into a Python string."""
    if value is None:
        return None

    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")

    return str(value)


def numeric_demo_sort_key(name: str) -> Tuple[int, str]:
    """
    Sort demo_0, demo_1, ..., demo_10 numerically rather than alphabetically.
    """
    match = re.search(r"(\d+)$", name)

    if match is None:
        return 10**9, name

    return int(match.group(1)), name


def resolve_task(
    suite_name: str,
    task_id: int,
) -> Tuple[Any, Any, Path, Path]:
    """
    Resolve a LIBERO task, BDDL file, and demonstration HDF5 path.
    """
    benchmark_dict = benchmark.get_benchmark_dict()

    if suite_name not in benchmark_dict:
        raise ValueError(
            "Unknown suite '{}'. Available suites: {}".format(
                suite_name,
                sorted(benchmark_dict.keys()),
            )
        )

    task_suite = benchmark_dict[suite_name]()

    if not 0 <= task_id < task_suite.n_tasks:
        raise ValueError(
            "task_id {} is outside valid range [0, {})".format(
                task_id,
                task_suite.n_tasks,
            )
        )

    task = task_suite.get_task(task_id)

    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )

    relative_demo_path = task_suite.get_task_demonstration(task_id)
    demo_path = Path(get_libero_path("datasets")) / relative_demo_path

    if not bddl_path.exists():
        raise FileNotFoundError(
            "BDDL file does not exist: {}".format(bddl_path)
        )

    if not demo_path.exists():
        dataset_root = Path(get_libero_path("datasets"))

        possible_matches = list(
            dataset_root.rglob("*{}*.hdf5".format(task.name))
        )

        match_text = "\n".join(str(path) for path in possible_matches)

        raise FileNotFoundError(
            "Demonstration file does not exist:\n{}\n"
            "Possible matching files:\n{}".format(
                demo_path,
                match_text or "<none>",
            )
        )

    return task_suite, task, bddl_path, demo_path


def load_demo(
    demo_path: Path,
    demo_index: int,
) -> Dict[str, Any]:
    """
    Load one demonstration into memory.

    demo_index refers to the index in the numerically sorted episode list,
    rather than assuming that demo_0 necessarily exists.
    """
    with h5py.File(str(demo_path), "r") as file:
        if "data" not in file:
            raise KeyError(
                "HDF5 file does not contain root group 'data': {}".format(
                    demo_path
                )
            )

        data_group = file["data"]

        demo_keys = sorted(
            list(data_group.keys()),
            key=numeric_demo_sort_key,
        )

        if not demo_keys:
            raise RuntimeError(
                "No demonstration episodes found in {}".format(demo_path)
            )

        if not 0 <= demo_index < len(demo_keys):
            raise IndexError(
                "demo_index {} is outside [0, {}). Episodes: {}".format(
                    demo_index,
                    len(demo_keys),
                    demo_keys,
                )
            )

        demo_key = demo_keys[demo_index]
        episode = data_group[demo_key]

        if "states" not in episode:
            raise KeyError(
                "{} does not contain 'states'".format(demo_key)
            )

        if "actions" not in episode:
            raise KeyError(
                "{} does not contain 'actions'".format(demo_key)
            )

        states = np.asarray(
            episode["states"][()],
            dtype=np.float64,
        )

        actions = np.asarray(
            episode["actions"][()],
            dtype=np.float64,
        )

        model_xml = decode_hdf5_attr(
            episode.attrs.get("model_file")
        )

        metadata = {
            "demo_key": demo_key,
            "num_demos": len(demo_keys),
            "states_shape": tuple(states.shape),
            "actions_shape": tuple(actions.shape),
            "env": decode_hdf5_attr(data_group.attrs.get("env")),
            "repository_version": decode_hdf5_attr(
                data_group.attrs.get("repository_version")
            ),
        }

    return {
        "states": states,
        "actions": actions,
        "model_xml": model_xml,
        "metadata": metadata,
    }


def make_env(
    bddl_path: Path,
    camera_name: str,
    image_height: int,
    image_width: int,
    seed: int,
) -> OffScreenRenderEnv:
    """Create a LIBERO offscreen environment."""
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=[camera_name],
        camera_heights=image_height,
        camera_widths=image_width,
        camera_depths=False,
        use_camera_obs=True,
        ignore_done=True,
        control_freq=20,
    )

    env.seed(seed)
    env.reset()

    return env
def safe_edit_model_xml(xml_str: str) -> str:
    """
    Safely repair asset paths in a recorded MuJoCo XML.

    This is compatible with old robosuite versions whose edit_model_xml()
    assumes that every mesh / texture path contains a 'robosuite' segment.
    """

    root = ET.fromstring(xml_str)
    asset = root.find("asset")

    if asset is None:
        return xml_str

    robosuite_root = Path(robosuite.__file__).resolve().parent

    try:
        libero_assets_root = Path(
            get_libero_path("assets")
        ).resolve()
    except Exception:
        libero_assets_root = None

    elements = (
        list(asset.findall("mesh"))
        + list(asset.findall("texture"))
        + list(asset.findall("hfield"))
    )

    rewritten_count = 0

    for element in elements:
        old_file = element.get("file")

        if not old_file:
            continue

        normalized = old_file.replace("\\", "/")
        path_parts = normalized.split("/")

        new_path = None

        # Case 1:
        # Recorded path points into a robosuite installation.
        robosuite_indices = [
            index
            for index, part in enumerate(path_parts)
            if part == "robosuite"
        ]

        if robosuite_indices:
            index = robosuite_indices[-1]

            candidate = robosuite_root.joinpath(
                *path_parts[index + 1:]
            )

            if candidate.exists():
                new_path = candidate

        # Case 2:
        # Path is already valid on the current machine.
        original_path = Path(old_file)

        if new_path is None and original_path.exists():
            new_path = original_path

        # Case 3:
        # Recorded path points into an old LIBERO assets directory.
        # Keep the path segment after the final "assets".
        if (
            new_path is None
            and libero_assets_root is not None
            and "assets" in path_parts
        ):
            assets_index = max(
                index
                for index, part in enumerate(path_parts)
                if part == "assets"
            )

            candidate = libero_assets_root.joinpath(
                *path_parts[assets_index + 1:]
            )

            if candidate.exists():
                new_path = candidate

        if new_path is not None:
            resolved = str(new_path.resolve())

            if resolved != old_file:
                element.set("file", resolved)
                rewritten_count += 1

    print(
        "[info] safely processed recorded XML assets; "
        "rewritten paths: {}".format(rewritten_count)
    )

    return ET.tostring(
        root,
        encoding="utf8",
    ).decode("utf8")


def restore_recorded_model(
    env: OffScreenRenderEnv,
    model_xml: Optional[str],
) -> None:
    """
    Restore the exact MuJoCo XML stored in a LIBERO demonstration.

    Uses a local compatibility implementation instead of the old
    robosuite edit_model_xml(), which may crash on LIBERO asset paths.
    """

    if model_xml is None:
        print(
            "[warning] No model_file attribute found. "
            "Using the model generated from the BDDL file."
        )
        return

    edited_xml = safe_edit_model_xml(model_xml)

    env.reset_from_xml_string(edited_xml)
    env.sim.reset()
    env.sim.forward()

    print("[info] recorded MuJoCo XML restored successfully")

def transform_frame(
    observation: Dict[str, np.ndarray],
    camera_name: str,
    flip: str,
) -> np.ndarray:
    """Extract and optionally flip an RGB camera frame."""
    image_key = "{}_image".format(camera_name)

    if image_key not in observation:
        raise KeyError(
            "Observation does not contain '{}'. Available keys:\n{}".format(
                image_key,
                sorted(observation.keys()),
            )
        )

    frame = np.asarray(observation[image_key])

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = np.clip(frame * 255.0, 0, 255)
        else:
            frame = np.clip(frame, 0, 255)

        frame = frame.astype(np.uint8)

    if flip == "vertical":
        frame = frame[::-1]

    elif flip == "horizontal":
        frame = frame[:, ::-1]

    elif flip == "rot180":
        frame = frame[::-1, ::-1]

    elif flip != "none":
        raise ValueError("Unsupported flip mode: {}".format(flip))

    return np.ascontiguousarray(frame)


def validate_state_dimension(
    env: OffScreenRenderEnv,
    states: np.ndarray,
) -> None:
    """Verify that dataset states match the active MuJoCo model."""
    current_state = np.asarray(env.get_sim_state())

    if states.ndim != 2:
        raise ValueError(
            "Expected states to have shape [T, D], got {}".format(
                states.shape
            )
        )

    if states.shape[1] != current_state.shape[0]:
        raise RuntimeError(
            "State dimension mismatch:\n"
            "  dataset state dimension: {}\n"
            "  environment dimension:   {}\n"
            "This usually means the recorded model XML, LIBERO version, "
            "robosuite version, or MuJoCo version does not match.".format(
                states.shape[1],
                current_state.shape[0],
            )
        )
