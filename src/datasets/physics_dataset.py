"""PhysicsDataset (v2): generalized N-object physics dataset.

Each item is one timestep. Returns all data needed by PhysicalTokenizer v2.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.rotations import quat_xyzw_to_6d                   # noqa: E402
from utils.normalization import load_stats, force_transform    # noqa: E402
from utils.contact_events import NUM_EVENTS                    # noqa: E402
from utils.task_role_parser import (                           # noqa: E402
    TaskRoleParser, ROLE_TARGET, ROLE_GOAL_SURFACE,
    ROLE_DISTRACTOR, ROLE_STATIC, ROLE_GRIPPER,
)
from models.state_encoder import ROBOT_INPUT_DIM               # noqa: E402
from models.object_encoder import OBJ_CONT_DIM                 # noqa: E402


MAX_N_OBJECTS = 16   # padding target


def _build_robot_vec(npz: Dict[str, np.ndarray], t: int) -> np.ndarray:
    """Build 33-dim robot-only state vector."""
    parts = [
        npz["robot_qpos"][t],
        npz["robot_qvel"][t],
        npz["ee_pos"][t],
        quat_xyzw_to_6d(npz["ee_quat_xyzw"][t]),
        npz["ee_lin_vel"][t],
        npz["ee_ang_vel"][t],
        npz["gripper_qpos"][t],
        npz["gripper_qvel"][t],
    ]
    vec = np.concatenate(parts).astype(np.float32)
    assert vec.shape[0] == ROBOT_INPUT_DIM, vec.shape
    return vec


def _build_obj_cont(
    npz: Dict[str, np.ndarray],
    t: int,
    obj_key: str,        # e.g. "bowl" or "plate"
    ee_pos: np.ndarray,
    goal_pos: Optional[np.ndarray],
    contacts: np.ndarray,
    contact_mask: np.ndarray,
    body_id: int,
) -> np.ndarray:
    """Build OBJ_CONT_DIM=25 continuous features for one object at frame t."""
    pos = npz[f"{obj_key}_position"][t]
    quat = npz[f"{obj_key}_quat_xyzw"][t]
    lv  = npz[f"{obj_key}_linear_velocity"][t]
    av  = npz[f"{obj_key}_angular_velocity"][t]
    mass    = np.log1p(np.abs(npz[f"{obj_key}_mass"][t]))
    inertia = np.log1p(np.abs(npz[f"{obj_key}_inertia"][t]))
    friction= np.log1p(np.abs(npz[f"{obj_key}_friction"][t]))

    is_grasped   = float(
        ((contacts[..., 25] > 0.5) & (contact_mask > 0.5)).any()
    )
    dist_to_ee   = float(np.linalg.norm(pos - ee_pos))
    dist_to_goal = float(np.linalg.norm(pos - goal_pos)) if goal_pos is not None else 0.0

    parts = [
        pos,
        quat_xyzw_to_6d(quat),
        lv, av,
        mass, inertia, friction,
        np.array([is_grasped, dist_to_ee, dist_to_goal], dtype=np.float32),
    ]
    vec = np.concatenate([np.atleast_1d(p) for p in parts]).astype(np.float32)
    assert vec.shape[0] == OBJ_CONT_DIM, (vec.shape, obj_key)
    return vec


def _build_risk_vec(npz: Dict[str, np.ndarray], t: int, target_key: str) -> np.ndarray:
    contacts = npz["contacts"][t]
    mask     = npz["contact_mask"][t] > 0.5
    nf = contacts[:, 13][mask]
    tf = contacts[:, 14][mask]
    max_nf   = float(nf.max()) if nf.size > 0 else 0.0
    max_slip = float((tf / np.maximum(nf, 1e-6)).max()) if nf.size > 0 else 0.0
    target_pos = npz[f"{target_key}_position"][t]

    # Dynamically find goal position - prefer any object that isn't the target
    goal_pos = None
    position_keys = [k for k in npz.keys() if k.endswith("_position") and k != f"{target_key}_position"]
    if position_keys:
        goal_pos = npz[position_keys[0]][t]
    else:
        # Fallback: use target position itself (distance will be 0)
        goal_pos = target_pos

    dist = float(np.linalg.norm(target_pos - goal_pos))
    height = float(target_pos[2])
    is_bg  = float((contacts[:, 25][mask] > 0.5).any()) if mask.any() else 0.0
    return np.array([
        force_transform(np.array([max_nf]))[0],
        np.clip(max_slip, 0.0, 5.0),
        dist, height, is_bg, float(max_nf > 5.0),
    ], dtype=np.float32)


class PhysicsDataset(Dataset):
    """Generalized N-object physics dataset."""

    def __init__(
        self,
        npz_path:      Path,
        events_path:   Path,
        task_language: str = "",
        stats_path:    Optional[Path] = None,
        rollout_path:  Optional[Path] = None,   # P3-2: MJX rollout 特征
    ) -> None:
        npz = dict(np.load(str(npz_path), allow_pickle=False))
        events = dict(np.load(str(events_path), allow_pickle=False))
        self._initialize(npz, events, task_language)
        self.stats = load_stats(stats_path) if stats_path else None
        if rollout_path and Path(rollout_path).exists():
            self.rollout = dict(np.load(str(rollout_path), allow_pickle=False))

    @classmethod
    def from_arrays(cls, npz, events, task_language=""):
        """Build the same feature schema from in-memory simulator arrays."""
        instance = cls.__new__(cls)
        instance._initialize(npz, events, task_language)
        return instance

    def _initialize(self, npz, events, task_language):
        self.npz = npz
        self.phase = events["phase"]
        self.events = events["events"]
        self.T = self.npz["contacts"].shape[0]
        self.stats = None
        self.rollout = None
        self.task_language = task_language

        # Determine available objects from NPZ keys
        # Convention: arrays named "{obj}_position" mark objects
        self.obj_keys = sorted(set(
            k.replace("_position", "")
            for k in self.npz if k.endswith("_position")
            and not k.startswith("contact") and not k.startswith("world")
        ))

        # Assign roles via TaskRoleParser
        parser = TaskRoleParser()
        name_roles = parser.parse(task_language, self.obj_keys)
        self.obj_roles = [name_roles.get(k, ROLE_DISTRACTOR) for k in self.obj_keys]

        # Find TARGET and GOAL keys for relative features
        self.target_key = next(
            (k for k, r in zip(self.obj_keys, self.obj_roles) if r == ROLE_TARGET),
            self.obj_keys[0] if self.obj_keys else None,
        )
        self.goal_key = next(
            (k for k, r in zip(self.obj_keys, self.obj_roles) if r == ROLE_GOAL_SURFACE),
            None,
        )

    def __len__(self) -> int:
        return self.T

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t      = int(idx)
        t_fut  = min(t + 1, self.T - 1)
        npz    = self.npz
        cont   = npz["contacts"][t].astype(np.float32)
        cmask  = npz["contact_mask"][t].astype(np.float32)
        ee_pos = npz["ee_pos"][t]
        goal_pos = npz[f"{self.goal_key}_position"][t] if self.goal_key else None

        # Robot vec
        robot_vec = _build_robot_vec(npz, t)

        # Per-object continuous features [N, 25]
        obj_cont_list, obj_role_list, obj_valid = [], [], []
        for key, role in zip(self.obj_keys, self.obj_roles):
            oc = _build_obj_cont(npz, t, key, ee_pos, goal_pos, cont, cmask, 0)
            obj_cont_list.append(oc)
            obj_role_list.append(role)
            obj_valid.append(1.0)

        N = len(obj_cont_list)
        # Pad to MAX_N_OBJECTS
        pad_n = MAX_N_OBJECTS - N
        obj_cont  = np.stack(obj_cont_list + [np.zeros(OBJ_CONT_DIM, np.float32)] * pad_n)
        obj_roles = np.array(obj_role_list + [ROLE_DISTRACTOR] * pad_n, dtype=np.int64)
        obj_mask  = np.array(obj_valid + [0.0] * pad_n, dtype=np.float32)

        # z_dyn + z_risk：优先使用 MJX rollout 特征（P3-2），否则退回 teacher-forcing/规则
        if self.rollout is not None:
            dyn_target  = self.rollout["dyn_target"][t].astype(np.float32)
            # rollout dyn_rel shape: [2, 36]，pad 到 MAX_N_OBJECTS
            rf_rel = self.rollout["dyn_rel_cont"][t].astype(np.float32)   # [2, 36]
            rf_mask= self.rollout["dyn_rel_mask"][t].astype(np.float32)   # [2]
            dyn_rel_np   = np.concatenate([rf_rel,
                                           np.zeros((pad_n + N - len(rf_rel), 36), np.float32)], axis=0)
            dyn_rel_mask = np.concatenate([rf_mask,
                                           np.zeros(pad_n + N - len(rf_mask), np.float32)], axis=0)
            risk_vec = self.rollout["risk_vec"][t].astype(np.float32)
        else:
            # P1 teacher-forcing
            target_pos_t   = npz[f"{self.target_key}_position"][t]
            target_pos_fut = npz[f"{self.target_key}_position"][t_fut]
            target_quat_t  = npz[f"{self.target_key}_quat_xyzw"][t]
            target_quat_fut= npz[f"{self.target_key}_quat_xyzw"][t_fut]
            target_lv_t    = npz[f"{self.target_key}_linear_velocity"][t]
            target_lv_fut  = npz[f"{self.target_key}_linear_velocity"][t_fut]
            target_av_t    = npz[f"{self.target_key}_angular_velocity"][t]
            target_av_fut  = npz[f"{self.target_key}_angular_velocity"][t_fut]
            dist_t   = float(np.linalg.norm(target_pos_t - goal_pos)) if goal_pos is not None else 0.0
            dist_fut = float(np.linalg.norm(target_pos_fut - goal_pos)) if goal_pos is not None else 0.0

            dyn_target = np.concatenate([
                target_pos_fut - target_pos_t,
                quat_xyzw_to_6d(target_quat_fut) - quat_xyzw_to_6d(target_quat_t),
                target_lv_fut - target_lv_t,
                target_av_fut - target_av_t,
                np.array([dist_fut - dist_t]),
            ]).astype(np.float32)

            role_emb_dim = 32
            dyn_rel_list = []
            for key, role in zip(self.obj_keys, self.obj_roles):
                if key == self.target_key:
                    dyn_rel_list.append(np.zeros(36, np.float32))
                    continue
                op_t   = npz[f"{key}_position"][t]
                op_fut = npz[f"{key}_position"][t_fut]
                rel_t   = op_t   - target_pos_t
                rel_fut = op_fut - target_pos_fut
                delta_rel  = rel_fut - rel_t
                delta_dist = np.array([np.linalg.norm(rel_fut) - np.linalg.norm(rel_t)])
                role_oh = np.zeros(role_emb_dim, np.float32)
                if role < role_emb_dim:
                    role_oh[role] = 1.0
                dyn_rel_list.append(np.concatenate([delta_rel, delta_dist, role_oh]))

            dyn_rel_np   = np.stack(dyn_rel_list + [np.zeros(36, np.float32)] * pad_n)
            dyn_rel_mask = np.array([1.0] * N + [0.0] * pad_n, np.float32)
            risk_vec     = _build_risk_vec(npz, t, self.target_key)

        # z_afford: phase one-hot[8] + events[5]
        phase_oh = np.zeros(8, np.float32)
        phase_oh[int(self.phase[t])] = 1.0
        # 5 generic events: grasp_onset, release_onset, high_force, slip, contact_change
        ev5 = self.events[t, :5].astype(np.float32)
        afford_vec = np.concatenate([phase_oh, ev5])   # [13]

        # Role-based contact flags (recompute from body role map for contacts)
        # For P1, use contacts as-is (role flags already in NPZ from adapter)
        contacts_out = cont   # [C, 28]

        return {
            "robot_vec":     torch.from_numpy(robot_vec),
            "contacts":      torch.from_numpy(contacts_out),
            "contact_mask":  torch.from_numpy(cmask),
            "dyn_target":    torch.from_numpy(dyn_target),
            "dyn_rel_cont":  torch.from_numpy(dyn_rel_np.astype(np.float32)),
            "dyn_rel_mask":  torch.from_numpy(dyn_rel_mask),
            "risk_vec":      torch.from_numpy(risk_vec),
            "afford_vec":    torch.from_numpy(afford_vec),
            "obj_roles":     torch.from_numpy(obj_roles),
            "obj_cont":      torch.from_numpy(obj_cont),
            "obj_mask":      torch.from_numpy(obj_mask),
            "phase":         torch.tensor(int(self.phase[t]), dtype=torch.long),
            "success":       torch.tensor(float(npz["success"][t]), dtype=torch.float32),
            "timestep":      torch.tensor(t, dtype=torch.long),
            "n_objects":     torch.tensor(N, dtype=torch.long),
        }
