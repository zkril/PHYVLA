"""Compute normalization statistics from the physics NPZ and save to JSON.

Usage:
    python scripts/compute_normalization.py \
        --input outputs/physics/task02_demo00_contact_force_data.npz \
        --output outputs/normalization_v1.json
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

from utils.normalization import compute_stats, save_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute normalization stats.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/physics/task02_demo00_contact_force_data.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/normalization_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError("Input NPZ not found: {}".format(args.input))

    print("[normalization] loading", args.input)
    npz = dict(np.load(str(args.input), allow_pickle=False))
    print("[normalization] arrays:", sorted(npz.keys()))

    stats = compute_stats(npz)

    save_stats(args.output, stats)

    print("[normalization] fields computed:", len(stats["fields"]))
    for key, s in list(stats["fields"].items())[:8]:
        print("  {:<35s}  mean={:+.4f}  std={:.4f}".format(key, s["mean"], s["std"]))
    print("  ...")


if __name__ == "__main__":
    main()
