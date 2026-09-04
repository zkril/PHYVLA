"""Build contact event labels from the physics NPZ and save to NPZ.

Usage:
    python scripts/build_contact_events.py \
        --input outputs/physics/task02_demo00_contact_force_data.npz \
        --output outputs/contact_events_v1.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.contact_events import detect_contact_events, PHASE_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build contact event labels.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/physics/task02_demo00_contact_force_data.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/contact_events_v1.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError("Input NPZ not found: {}".format(args.input))

    print("[events] loading", args.input)
    npz = np.load(str(args.input), allow_pickle=False)

    bowl_plate_dist = np.linalg.norm(
        npz["bowl_position"] - npz["plate_position"], axis=-1
    )

    result = detect_contact_events(
        contacts=npz["contacts"],
        contact_mask=npz["contact_mask"],
        bowl_position=npz["bowl_position"],
        bowl_plate_dist=bowl_plate_dist,
        gripper_qpos=npz["gripper_qpos"],
        success=npz["success"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(args.output),
        phase=result.phase,
        events=result.events,
    )

    print("[events] saved to", args.output)
    print("[events] phase distribution:")
    for code, name in enumerate(PHASE_NAMES):
        count = int((result.phase == code).sum())
        bar = "#" * count
        print("  {:12s} ({:d}): {:3d} frames  {}".format(name, code, count, bar))
    print("[events] event totals:")
    for i, name in enumerate(result.event_names):
        print("  {:20s}: {:.0f} frames".format(name, result.events[:, i].sum()))


if __name__ == "__main__":
    main()
