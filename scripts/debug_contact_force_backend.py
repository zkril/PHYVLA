"""Diagnose MuJoCo contact-force extraction in a LIBERO environment.

Run this script from the phyvla project environment, not system Python:
    conda activate libero_replay
    python scripts/debug_contact_force_backend.py --suite libero_spatial --task-id 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from replay_libero_actions import (  # noqa: E402
    load_demonstration,
    make_environment,
    resolve_task_and_demo,
    restore_recorded_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-print", type=int, default=12)
    return parser.parse_args()


def try_mujoco_py(model: object, data: object, contact_id: int) -> np.ndarray:
    from mujoco_py import functions as mj_functions

    result = np.zeros(6, dtype=np.float64)
    mj_functions.mj_contactForce(model, data, int(contact_id), result)
    return result


def try_native_mujoco(model: object, data: object, contact_id: int) -> np.ndarray:
    import mujoco

    result = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, int(contact_id), result)
    return result


def inspect_contacts(label: str, env: object, max_print: int) -> None:
    model = env.sim.model
    data = env.sim.data
    print(f"\n=== {label} ===")
    print("model type:", type(model), "module:", type(model).__module__)
    print("data type :", type(data), "module:", type(data).__module__)
    print("ncon      :", int(data.ncon))

    backends: list[tuple[str, Callable[[object, object, int], np.ndarray]]] = [
        ("mujoco_py", try_mujoco_py),
        ("native_mujoco", try_native_mujoco),
    ]

    active = 0
    nonzero = {name: 0 for name, _ in backends}
    successes = {name: 0 for name, _ in backends}
    errors: dict[str, str] = {}

    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        efc_address = int(getattr(contact, "efc_address", -999))
        dim = int(getattr(contact, "dim", -1))
        exclude = int(getattr(contact, "exclude", -1))
        if efc_address >= 0:
            active += 1

        details = []
        for backend_name, backend in backends:
            try:
                wrench = backend(model, data, contact_id)
                successes[backend_name] += 1
                if np.linalg.norm(wrench) > 1e-10:
                    nonzero[backend_name] += 1
                details.append(
                    f"{backend_name}=ok norm={np.linalg.norm(wrench):.6g} "
                    f"wrench={np.array2string(wrench, precision=4)}"
                )
            except Exception as exc:  # diagnostic: print exact backend error
                errors.setdefault(
                    backend_name,
                    f"{type(exc).__name__}: {exc}",
                )
                details.append(f"{backend_name}=ERROR {type(exc).__name__}: {exc}")

        if contact_id < max_print:
            print(
                f"contact={contact_id:3d} efc_address={efc_address:4d} "
                f"dim={dim} exclude={exclude} dist={float(contact.dist): .6g}"
            )
            for detail in details:
                print("   ", detail)

    print("active contacts (efc_address >= 0):", active)
    for backend_name, _ in backends:
        print(
            f"{backend_name}: successful_calls={successes[backend_name]} "
            f"nonzero_calls={nonzero[backend_name]} "
            f"first_error={errors.get(backend_name)}"
        )


def main() -> None:
    args = parse_args()
    _, _, bddl_path, demo_path = resolve_task_and_demo(
        suite_name=args.suite,
        task_id=args.task_id,
    )
    demo = load_demonstration(demo_path=demo_path, demo_index=args.demo_index)
    states = np.asarray(demo["states"], dtype=np.float64)
    actions = np.asarray(demo["actions"], dtype=np.float64)

    print("python executable:", sys.executable)
    print("python version   :", sys.version.replace("\n", " "))

    env = make_environment(
        bddl_path=bddl_path,
        camera_name=args.camera,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    try:
        restore_recorded_model(env, demo["model_xml"])

        env.regenerate_obs_from_state(states[0])
        env.sim.forward()
        inspect_contacts("state restore + sim.forward", env, args.max_print)

        env.regenerate_obs_from_state(states[0])
        env.sim.forward()
        env.step(actions[0])
        env.sim.forward()
        inspect_contacts("state restore + one expert env.step", env, args.max_print)
    finally:
        env.close()


if __name__ == "__main__":
    main()
