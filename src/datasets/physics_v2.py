"""Causal raw-v2 features shared by offline training and live inference."""
from pathlib import Path
import hashlib
import json
import numpy as np

SCHEMA = "causal-object-set-v2"
ROBOT_KEYS = ("robot_qpos", "robot_qvel", "robot_ee_pos", "robot_ee_quat_xyzw",
              "robot_ee_lin_vel", "robot_ee_ang_vel", "robot_gripper_qpos", "robot_gripper_qvel")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def features(a):
    """All frames, variable body count; no actions, images, labels, or future state."""
    robot = np.concatenate([a[k] for k in ROBOT_KEYS], -1)
    pos = a["object_position"]
    t, n, _ = pos.shape
    ee = a["robot_ee_pos"]
    roles = np.broadcast_to(a["object_goal_roles"], (t, n, 2))
    mass = np.broadcast_to(a["object_mass"][None, :, None], (t, n, 1))
    inertia = np.broadcast_to(a["object_inertia"], (t, n, 3))
    friction = np.zeros((n, 3))
    articulation = np.zeros((t, n, 4))
    # Hinge/slide position and velocity, and separate type flags. Free joints
    # are already represented by global body poses and velocities.
    nq = (a["recorded_states"].shape[1]-1) - int(
        max(a["joint_dof_address"] + np.where(a["joint_types"] == 0, 6,
            np.where(a["joint_types"] == 1, 3, 1))))
    for j, body in enumerate(a["object_body_ids"]):
        geoms = a["geom_body_ids"] == body
        if geoms.any():
            friction[j] = a["geom_friction"][geoms].mean(0)
        joints = np.flatnonzero((a["joint_body_ids"] == body) & (a["joint_types"] >= 2))
        if len(joints) > 1:
            raise ValueError("Multiple scalar joints on one body require explicit joint tokens")
        for k in joints:
            articulation[:, j, 0] = a["recorded_states"][:, 1+a["joint_qpos_address"][k]]
            articulation[:, j, 1] = a["recorded_states"][:, 1+nq+a["joint_dof_address"][k]]
            articulation[:, j, 2+int(a["joint_types"][k] == 3)] = 1
    # Aggregate every incident contact (no top-K); body membership is retained.
    # count, sum/max force, max tangential force, min separation, contact with
    # robot, target, goal, other; all are current-state quantities.
    contact = np.zeros((t, n, 9))
    body_slot = {int(b): j for j, b in enumerate(a["object_body_ids"])}
    for frame in range(t):
        cs = a["contacts"][a["contact_offsets"][frame]:a["contact_offsets"][frame+1]]
        for j, body in enumerate(a["object_body_ids"]):
            hits = cs[(cs[:, 19] == body) | (cs[:, 20] == body)]
            if not len(hits):
                continue
            other = np.where(hits[:, 19] == body, hits[:, 20], hits[:, 19]).astype(int)
            contact[frame, j, :5] = [np.log1p(len(hits)),
                np.log1p(hits[:, 15].sum()), np.log1p(hits[:, 15].max()),
                np.log1p(hits[:, 14].max()), hits[:, 12].min()]
            for b in other:
                slot = body_slot.get(int(b))
                if slot is None:
                    # Explicit robot-body IDs supplied by exporter/online capture.
                    contact[frame, j, 5 if b in a["robot_body_ids"] else 8] = 1
                else:
                    contact[frame, j, 6:8] = np.maximum(
                        contact[frame, j, 6:8], a["object_goal_roles"][slot])
                    contact[frame, j, 8] = 1
    # Nearest explicit goal region; missing regions carry a zero validity flag.
    region = np.zeros((t, n, 4))
    sites = a["goal_site_position"]
    if sites.shape[1]:
        delta = sites[:, None] - pos[:, :, None]
        idx = np.linalg.norm(delta, axis=-1).argmin(-1)
        region[:, :, :3] = np.take_along_axis(delta, idx[..., None, None], 2)[:, :, 0]
        region[:, :, 3] = 1
    objects = np.concatenate([pos-ee[:, None], a["object_quat_xyzw"],
        a["object_linear_velocity"], a["object_angular_velocity"],
        np.log1p(mass), np.log1p(inertia),
        np.broadcast_to(friction, (t,n,3)), roles, articulation, contact, region], -1)
    out = {"robot": robot.astype(np.float32), "objects": objects.astype(np.float32)}
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError("Nonfinite causal features")
    return out

def read_episode(entry):
    with np.load(entry["path"], allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    a["robot_body_ids"] = np.array([i for i,n in enumerate(entry["body_names"])
                                    if n.startswith(("robot","gripper"))])
    return a
