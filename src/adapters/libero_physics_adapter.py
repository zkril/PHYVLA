"""Canonical physics-state extraction for LIBERO / robosuite environments.

Designed for the LIBERO + robosuite 1.4.x stack, with light compatibility
fallbacks for newer MuJoCo Python bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CONTACT_FEATURE_NAMES: Tuple[str, ...] = (
    "position_x",
    "position_y",
    "position_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "force_world_x",
    "force_world_y",
    "force_world_z",
    "torque_world_x",
    "torque_world_y",
    "torque_world_z",
    "distance",
    "normal_force",
    "tangential_force",
    "force_norm",
    "torque_norm",
    "geom1_id",
    "geom2_id",
    "body1_id",
    "body2_id",
    "touches_bowl",
    "touches_plate",
    "touches_gripper",
    "is_bowl_plate",
    "is_bowl_gripper",
    "is_bowl_table",
    "is_robot_environment",
)


@dataclass(frozen=True)
class ResolvedObject:
    key: str
    body_name: str
    body_id: int


def _decode_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized xyzw quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm > 0.0:
        quaternion /= norm
    return quaternion


def _wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return quaternion[[1, 2, 3, 0]]


def _contains_any(name: str, patterns: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


class LiberoPhysicsAdapter:
    """Extract fixed-shape robot, object, contact, and material features."""

    def __init__(
        self,
        env: Any,
        object_patterns: Optional[Mapping[str, Sequence[str]]] = None,
        max_contacts: int = 16,
    ) -> None:
        self.env = env
        self.sim = env.sim
        self.model = self.sim.model
        self.data = self.sim.data
        self.max_contacts = int(max_contacts)

        if self.max_contacts <= 0:
            raise ValueError("max_contacts must be positive")

        self.body_names = self._all_names("body")
        self.geom_names = self._all_names("geom")
        self.site_names = self._all_names("site")
        self.joint_names = self._all_names("joint")

        # Auto-detect objects if no patterns provided
        if object_patterns is None:
            object_patterns = self._auto_detect_objects()

        self.objects: Dict[str, ResolvedObject] = {}
        for key, patterns in object_patterns.items():
            body_name = self._resolve_best_name(self.body_names, patterns)
            body_id = self._name_to_id("body", body_name)
            self.objects[key] = ResolvedObject(key, body_name, body_id)

        self.robot_joint_names = [
            name
            for name in self.joint_names
            if _contains_any(name, ("robot0_joint", "panda_joint", "joint"))
            and not _contains_any(name, ("finger", "gripper"))
        ]

        # Prefer canonical 7 arm joints if many generic joints were matched.
        canonical_robot_joints = [
            name
            for name in self.joint_names
            if _contains_any(name, ("robot0_joint", "panda_joint"))
            and not _contains_any(name, ("finger", "gripper"))
        ]
        if canonical_robot_joints:
            self.robot_joint_names = canonical_robot_joints

        self.gripper_joint_names = [
            name
            for name in self.joint_names
            if _contains_any(name, ("finger_joint", "gripper"))
        ]

        self.eef_site_name = self._resolve_optional_name(
            self.site_names,
            ("gripper0_grip_site", "grip_site", "eef", "end_effector"),
        )

    # ------------------------------------------------------------------
    # Object detection
    # ------------------------------------------------------------------

    def _auto_detect_objects(self) -> Dict[str, Tuple[str, ...]]:
        """Auto-detect manipulable objects in the scene.

        Returns a dict mapping object keys to name patterns.
        Excludes robot, gripper, floor, mount, and world bodies.
        """
        excluded_patterns = (
            "world", "floor", "robot", "gripper", "mount",
            "base", "link", "hand", "finger", "eef"
        )

        object_patterns = {}
        for body_name in self.body_names:
            # Skip if body name contains any excluded pattern
            if any(pat in body_name.lower() for pat in excluded_patterns):
                continue

            # Extract a clean key from the body name
            # e.g., "alphabet_soup_1_main" -> "alphabet_soup"
            clean_name = body_name.lower()
            # Remove common suffixes
            for suffix in ("_main", "_1", "_2", "_3", "_4", "_5", "_base", "_body"):
                clean_name = clean_name.replace(suffix, "")

            # Use the clean name as the key and the full body name as the pattern
            key = clean_name.strip("_")
            if key and key not in object_patterns:
                object_patterns[key] = (body_name,)

        return object_patterns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        return {
            "objects": {
                key: {
                    "body_name": value.body_name,
                    "body_id": value.body_id,
                }
                for key, value in self.objects.items()
            },
            "eef_site_name": self.eef_site_name,
            "robot_joint_names": list(self.robot_joint_names),
            "gripper_joint_names": list(self.gripper_joint_names),
            "max_contacts": self.max_contacts,
            "contact_feature_names": list(CONTACT_FEATURE_NAMES),
        }

    def extract(self) -> Dict[str, Any]:
        robot = self.extract_robot_state()
        objects = {
            key: self.extract_body_state(resolved.body_name)
            for key, resolved in self.objects.items()
        }
        contacts, contact_mask, contact_count = self.extract_contacts()

        return {
            "robot": robot,
            "objects": objects,
            "contacts": contacts,
            "contact_mask": contact_mask,
            "contact_count": np.int32(contact_count),
        }

    def extract_robot_state(self) -> Dict[str, np.ndarray]:
        robot_qpos = self._joint_values(self.robot_joint_names, velocity=False)
        robot_qvel = self._joint_values(self.robot_joint_names, velocity=True)
        gripper_qpos = self._joint_values(self.gripper_joint_names, velocity=False)
        gripper_qvel = self._joint_values(self.gripper_joint_names, velocity=True)

        ee_pos = np.zeros(3, dtype=np.float64)
        ee_quat_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        ee_lin_vel = np.zeros(3, dtype=np.float64)
        ee_ang_vel = np.zeros(3, dtype=np.float64)

        if self.eef_site_name:
            site_id = self._name_to_id("site", self.eef_site_name)
            ee_pos = np.asarray(self.data.site_xpos[site_id], dtype=np.float64).copy()
            site_matrix = np.asarray(self.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
            ee_quat_xyzw = _matrix_to_quat_xyzw(site_matrix)

            if hasattr(self.data, "get_site_xvelp"):
                ee_lin_vel = np.asarray(
                    self.data.get_site_xvelp(self.eef_site_name),
                    dtype=np.float64,
                ).copy()
            if hasattr(self.data, "get_site_xvelr"):
                ee_ang_vel = np.asarray(
                    self.data.get_site_xvelr(self.eef_site_name),
                    dtype=np.float64,
                ).copy()

        return {
            "qpos": robot_qpos,
            "qvel": robot_qvel,
            "ee_pos": ee_pos,
            "ee_quat_xyzw": ee_quat_xyzw,
            "ee_lin_vel": ee_lin_vel,
            "ee_ang_vel": ee_ang_vel,
            "gripper_qpos": gripper_qpos,
            "gripper_qvel": gripper_qvel,
        }

    def extract_body_state(self, body_name: str) -> Dict[str, np.ndarray]:
        body_id = self._name_to_id("body", body_name)

        if hasattr(self.data, "get_body_xpos"):
            position = np.asarray(self.data.get_body_xpos(body_name), dtype=np.float64).copy()
            quat_wxyz = np.asarray(self.data.get_body_xquat(body_name), dtype=np.float64).copy()
            linear_velocity = np.asarray(
                self.data.get_body_xvelp(body_name),
                dtype=np.float64,
            ).copy()
            angular_velocity = np.asarray(
                self.data.get_body_xvelr(body_name),
                dtype=np.float64,
            ).copy()
        else:
            position = np.asarray(self.data.xpos[body_id], dtype=np.float64).copy()
            quat_wxyz = np.asarray(self.data.xquat[body_id], dtype=np.float64).copy()
            linear_velocity = np.zeros(3, dtype=np.float64)
            angular_velocity = np.zeros(3, dtype=np.float64)

        mass = np.asarray([float(self.model.body_mass[body_id])], dtype=np.float64)
        inertia = np.asarray(self.model.body_inertia[body_id], dtype=np.float64).copy()
        friction = self._body_friction(body_id)

        return {
            "position": position,
            "quat_xyzw": _wxyz_to_xyzw(quat_wxyz),
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "mass": mass,
            "inertia": inertia,
            "friction": friction,
        }

    def extract_contacts(self) -> Tuple[np.ndarray, np.ndarray, int]:
        features: List[np.ndarray] = []
        ncon = int(self.data.ncon)

        for contact_id in range(ncon):
            contact = self.data.contact[contact_id]
            feature = self._contact_feature(contact_id, contact)
            features.append(feature)

        # Keep task-relevant contacts first, then strongest contacts.
        features.sort(
            key=lambda row: (
                -float(max(row[24], row[25], row[26], row[27])),
                -float(row[15]),
            )
        )

        kept = features[: self.max_contacts]
        tensor = np.zeros(
            (self.max_contacts, len(CONTACT_FEATURE_NAMES)),
            dtype=np.float32,
        )
        mask = np.zeros((self.max_contacts,), dtype=np.float32)

        if kept:
            tensor[: len(kept)] = np.stack(kept, axis=0).astype(np.float32)
            mask[: len(kept)] = 1.0

        return tensor, mask, ncon

    # ------------------------------------------------------------------
    # Contact and material helpers
    # ------------------------------------------------------------------

    def _contact_feature(self, contact_id: int, contact: Any) -> np.ndarray:
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        geom1_name = self._id_to_name("geom", geom1_id)
        geom2_name = self._id_to_name("geom", geom2_id)

        body1_id = int(self.model.geom_bodyid[geom1_id])
        body2_id = int(self.model.geom_bodyid[geom2_id])
        body1_name = self._id_to_name("body", body1_id)
        body2_name = self._id_to_name("body", body2_id)

        pair_text = " ".join(
            (geom1_name, geom2_name, body1_name, body2_name)
        ).lower()

        position = np.asarray(contact.pos, dtype=np.float64).reshape(3)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        normal = frame[0].copy()

        wrench_local = self._contact_wrench(contact_id)
        force_world = frame.T.dot(wrench_local[:3])
        torque_world = frame.T.dot(wrench_local[3:])

        normal_force = abs(float(wrench_local[0]))
        tangential_force = float(np.linalg.norm(wrench_local[1:3]))
        force_norm = float(np.linalg.norm(force_world))
        torque_norm = float(np.linalg.norm(torque_world))
        distance = float(contact.dist)

        touches_bowl = float("bowl" in pair_text)
        touches_plate = float("plate" in pair_text)
        touches_gripper = float(
            any(token in pair_text for token in ("gripper", "finger", "hand"))
        )
        touches_table = float(any(token in pair_text for token in ("table", "counter")))
        touches_robot = float(any(token in pair_text for token in ("robot0", "panda", "gripper")))

        is_bowl_plate = float(touches_bowl and touches_plate)
        is_bowl_gripper = float(touches_bowl and touches_gripper)
        is_bowl_table = float(touches_bowl and touches_table)
        is_robot_environment = float(
            touches_robot
            and not touches_bowl
            and not touches_plate
            and not touches_gripper
        )

        feature = np.asarray(
            [
                *position.tolist(),
                *normal.tolist(),
                *force_world.tolist(),
                *torque_world.tolist(),
                distance,
                normal_force,
                tangential_force,
                force_norm,
                torque_norm,
                float(geom1_id),
                float(geom2_id),
                float(body1_id),
                float(body2_id),
                touches_bowl,
                touches_plate,
                touches_gripper,
                is_bowl_plate,
                is_bowl_gripper,
                is_bowl_table,
                is_robot_environment,
            ],
            dtype=np.float64,
        )
        return feature

    def _contact_wrench(self, contact_id: int) -> np.ndarray:
        """Return the local 6D contact wrench from native MuJoCo.

        robosuite 1.4+ wraps ``mujoco.MjModel`` and ``mujoco.MjData`` in
        ``binding_utils.MjModel`` / ``binding_utils.MjData``. Native MuJoCo
        functions must receive the wrapped ``_model`` and ``_data`` objects.
        """
        contact = self.data.contact[int(contact_id)]
        efc_address = int(getattr(contact, "efc_address", -1))
        if efc_address < 0:
            return np.zeros(6, dtype=np.float64)

        try:
            import mujoco

            raw_model = getattr(self.model, "_model", self.model)
            raw_data = getattr(self.data, "_data", self.data)
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                raw_model,
                raw_data,
                int(contact_id),
                wrench,
            )
        except Exception as exc:
            raise RuntimeError(
                "mj_contactForce failed for contact {} with model type {} "
                "and data type {}".format(
                    contact_id,
                    type(self.model),
                    type(self.data),
                )
            ) from exc

        if not np.all(np.isfinite(wrench)):
            raise FloatingPointError(
                "Non-finite contact wrench for contact {}: {}".format(
                    contact_id,
                    wrench,
                )
            )
        return wrench

    def _body_friction(self, body_id: int) -> np.ndarray:
        geom_ids = np.where(
            np.asarray(self.model.geom_bodyid, dtype=np.int64) == int(body_id)
        )[0]

        if geom_ids.size == 0:
            return np.zeros(3, dtype=np.float64)

        friction = np.asarray(self.model.geom_friction[geom_ids], dtype=np.float64)
        return friction.mean(axis=0)

    # ------------------------------------------------------------------
    # MuJoCo name and joint helpers
    # ------------------------------------------------------------------

    def _all_names(self, kind: str) -> List[str]:
        plural_attr = "{}_names".format(kind)
        count_attr = "n{}".format({"joint": "jnt"}.get(kind, kind))

        if hasattr(self.model, plural_attr):
            return [_decode_name(value) for value in getattr(self.model, plural_attr)]

        count = int(getattr(self.model, count_attr))
        return [self._id_to_name(kind, index) for index in range(count)]

    def _id_to_name(self, kind: str, index: int) -> str:
        method = getattr(self.model, "{}_id2name".format(kind), None)
        if method is not None:
            return _decode_name(method(int(index)))

        try:
            import mujoco

            object_type = {
                "body": mujoco.mjtObj.mjOBJ_BODY,
                "geom": mujoco.mjtObj.mjOBJ_GEOM,
                "site": mujoco.mjtObj.mjOBJ_SITE,
                "joint": mujoco.mjtObj.mjOBJ_JOINT,
            }[kind]
            return _decode_name(mujoco.mj_id2name(self.model, object_type, int(index)))
        except Exception as exc:
            raise RuntimeError("Cannot resolve {} id {}".format(kind, index)) from exc

    def _name_to_id(self, kind: str, name: str) -> int:
        method = getattr(self.model, "{}_name2id".format(kind), None)
        if method is not None:
            return int(method(name))

        try:
            import mujoco

            object_type = {
                "body": mujoco.mjtObj.mjOBJ_BODY,
                "geom": mujoco.mjtObj.mjOBJ_GEOM,
                "site": mujoco.mjtObj.mjOBJ_SITE,
                "joint": mujoco.mjtObj.mjOBJ_JOINT,
            }[kind]
            return int(mujoco.mj_name2id(self.model, object_type, name))
        except Exception as exc:
            raise RuntimeError("Cannot resolve {} name '{}'".format(kind, name)) from exc

    @staticmethod
    def _resolve_best_name(names: Sequence[str], patterns: Sequence[str]) -> str:
        candidates: List[Tuple[int, int, str]] = []
        for name in names:
            lowered = name.lower()
            for pattern_index, pattern in enumerate(patterns):
                pattern_lower = pattern.lower()
                if lowered == pattern_lower:
                    candidates.append((0, pattern_index, name))
                elif pattern_lower in lowered:
                    candidates.append((1, pattern_index, name))

        if not candidates:
            raise KeyError(
                "No MuJoCo name matched patterns {}. Available names include: {}".format(
                    list(patterns),
                    list(names)[:100],
                )
            )

        candidates.sort(key=lambda item: (item[0], item[1], len(item[2]), item[2]))
        return candidates[0][2]

    @classmethod
    def _resolve_optional_name(
        cls,
        names: Sequence[str],
        patterns: Sequence[str],
    ) -> Optional[str]:
        try:
            return cls._resolve_best_name(names, patterns)
        except KeyError:
            return None

    def _joint_values(self, joint_names: Sequence[str], velocity: bool) -> np.ndarray:
        values: List[float] = []
        for name in joint_names:
            address = self._joint_address(name, velocity=velocity)
            source = self.data.qvel if velocity else self.data.qpos

            if isinstance(address, tuple):
                start, stop = int(address[0]), int(address[1])
                values.extend(np.asarray(source[start:stop], dtype=np.float64).tolist())
            else:
                values.append(float(source[int(address)]))

        return np.asarray(values, dtype=np.float64)

    def _joint_address(self, joint_name: str, velocity: bool) -> Any:
        method_name = "get_joint_qvel_addr" if velocity else "get_joint_qpos_addr"
        method = getattr(self.model, method_name, None)
        if method is not None:
            return method(joint_name)

        joint_id = self._name_to_id("joint", joint_name)
        if velocity:
            return int(self.model.jnt_dofadr[joint_id])
        return int(self.model.jnt_qposadr[joint_id])
