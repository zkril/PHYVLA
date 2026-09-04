"""Normalization utilities for Physical Tokenizer.

Conventions:
  - Position / velocity / joint state: z-score (mean/std).
  - Force / torque: sign(x) * log(1 + |x| / scale) then z-score.
  - Mass / inertia / friction: log1p then z-score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

_FORCE_SCALE = 1.0   # N  — tune to dataset range
_TORQUE_SCALE = 0.1  # N·m


def force_transform(x: np.ndarray, scale: float = _FORCE_SCALE) -> np.ndarray:
    """sign(x) * log(1 + |x| / scale)."""
    return np.sign(x) * np.log1p(np.abs(x) / scale)


def force_inverse(y: np.ndarray, scale: float = _FORCE_SCALE) -> np.ndarray:
    return np.sign(y) * scale * (np.expm1(np.abs(y)))


def _compute_field_stats(values: np.ndarray) -> Dict[str, Any]:
    flat = values.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"mean": 0.0, "std": 1.0, "p01": 0.0, "p50": 0.0, "p99": 1.0}
    return {
        "mean": float(np.mean(finite)),
        "std":  float(np.std(finite) + 1e-8),
        "p01":  float(np.percentile(finite, 1)),
        "p50":  float(np.percentile(finite, 50)),
        "p99":  float(np.percentile(finite, 99)),
    }


def compute_stats(npz: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute per-field normalization statistics from a physics NPZ dict."""

    force_fields = {
        "force_world_x", "force_world_y", "force_world_z",
        "torque_world_x", "torque_world_y", "torque_world_z",
        "normal_force", "tangential_force", "force_norm", "torque_norm",
    }
    log_fields = {
        "bowl_mass", "plate_mass",
        "bowl_inertia", "plate_inertia",
        "bowl_friction", "plate_friction",
    }

    stats: Dict[str, Any] = {
        "schema_version": "normalization-v1",
        "force_scale": _FORCE_SCALE,
        "torque_scale": _TORQUE_SCALE,
        "fields": {},
    }

    for key, arr in npz.items():
        arr = np.asarray(arr, dtype=np.float64)
        if not np.issubdtype(arr.dtype, np.floating):
            continue

        if key in log_fields:
            transformed = np.log1p(np.abs(arr))
        elif key in force_fields:
            scale = _TORQUE_SCALE if "torque" in key else _FORCE_SCALE
            transformed = force_transform(arr, scale)
        else:
            transformed = arr

        stats["fields"][key] = _compute_field_stats(transformed)

    # Per-contact feature stats from contacts array [T, C, 28]
    if "contacts" in npz and "contact_mask" in npz:
        contacts = np.asarray(npz["contacts"], dtype=np.float64)
        mask = np.asarray(npz["contact_mask"], dtype=np.float64)
        valid = mask > 0.5

        from adapters.libero_physics_adapter import CONTACT_FEATURE_NAMES
        for feat_idx, feat_name in enumerate(CONTACT_FEATURE_NAMES):
            values = contacts[..., feat_idx][valid]
            if feat_name in force_fields:
                scale = _TORQUE_SCALE if "torque" in feat_name else _FORCE_SCALE
                values = force_transform(values, scale)
            stats["fields"][f"contact_{feat_name}"] = _compute_field_stats(values)

    return stats


def save_stats(path: Path, stats: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("[normalization] saved stats to", path)


def load_stats(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(x: np.ndarray, field_stats: Dict[str, Any]) -> np.ndarray:
    """Z-score normalize using precomputed field stats."""
    return (x - field_stats["mean"]) / field_stats["std"]


def denormalize(x: np.ndarray, field_stats: Dict[str, Any]) -> np.ndarray:
    return x * field_stats["std"] + field_stats["mean"]
