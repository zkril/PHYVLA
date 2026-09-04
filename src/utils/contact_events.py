"""Contact event detection for Physical Tokenizer.

Events are derived from sequential contact data without MJX.
Each timestep is assigned one phase label and a set of binary event flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# Task phase enum (integer codes)
PHASE_FREE       = 0   # approaching bowl, no contact
PHASE_TOUCH      = 1   # gripper touching bowl, not yet grasped
PHASE_GRASP      = 2   # gripper closed around bowl
PHASE_LIFT       = 3   # bowl lifted off table
PHASE_TRANSPORT  = 4   # bowl in the air, moving toward plate
PHASE_PLACE      = 5   # bowl near plate, being lowered
PHASE_RELEASE    = 6   # gripper opening at plate
PHASE_DONE       = 7   # task complete

PHASE_NAMES = [
    "free", "touch", "grasp", "lift",
    "transport", "place", "release", "done",
]

# Binary event flags (one-hot not required, can be multiple per step)
EVENT_GRASP_ONSET   = 0   # first frame of stable bowl-gripper contact
EVENT_RELEASE_ONSET = 1   # first frame of gripper separation from bowl
EVENT_BOWL_TABLE    = 2   # bowl touching table
EVENT_BOWL_PLATE    = 3   # bowl touching plate
EVENT_HIGH_FORCE    = 4   # normal force > threshold
EVENT_SLIP          = 5   # tangential / normal force ratio high
NUM_EVENTS = 6

_BOWL_REST_HEIGHT   = 0.905  # m — bowl resting on table
_BOWL_LIFT_HEIGHT   = 0.930  # m — bowl clearly off table
_BOWL_HIGH_HEIGHT   = 0.970  # m — bowl in the air (transport)
_HIGH_FORCE_THRESHOLD = 5.0  # N
_SLIP_RATIO_THRESHOLD = 0.8  # tan/norm


@dataclass
class ContactEventData:
    phase:        np.ndarray   # [T] int32  — task phase per timestep
    phase_names:  List[str]    # lookup table
    events:       np.ndarray   # [T, NUM_EVENTS] float32 — binary flags
    event_names:  List[str]


def _bowl_gripper_contact(contacts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return per-timestep bool: any valid contact is bowl-gripper."""
    is_bg = contacts[..., 25]          # [T, C]  is_bowl_gripper (index 25)
    return ((is_bg > 0.5) & (mask > 0.5)).any(axis=-1)  # [T]


def _bowl_plate_contact(contacts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    is_bp = contacts[..., 24]          # [T, C]  is_bowl_plate (index 24)
    return ((is_bp > 0.5) & (mask > 0.5)).any(axis=-1)


def _bowl_table_contact(contacts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    is_bt = contacts[..., 26]          # [T, C]  is_bowl_table (index 26)
    return ((is_bt > 0.5) & (mask > 0.5)).any(axis=-1)


def _max_normal_force(contacts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    nf = contacts[..., 13]             # [T, C]  normal_force
    nf = nf * (mask > 0.5)
    return nf.max(axis=-1)             # [T]


def _max_slip_ratio(contacts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    nf  = np.abs(contacts[..., 13])    # normal_force
    tf  = np.abs(contacts[..., 14])    # tangential_force
    valid = (mask > 0.5) & (nf > 0.1)
    ratio = np.where(valid, tf / np.maximum(nf, 1e-6), 0.0)
    return ratio.max(axis=-1)


def detect_contact_events(
    contacts:         np.ndarray,   # [T, C, 28]
    contact_mask:     np.ndarray,   # [T, C]
    bowl_position:    np.ndarray,   # [T, 3]
    bowl_plate_dist:  np.ndarray,   # [T]
    gripper_qpos:     np.ndarray,   # [T, 2]
    success:          np.ndarray,   # [T]
) -> ContactEventData:
    T = contacts.shape[0]

    bg  = _bowl_gripper_contact(contacts, contact_mask)  # [T]
    bp  = _bowl_plate_contact(contacts, contact_mask)
    bt  = _bowl_table_contact(contacts, contact_mask)
    max_nf  = _max_normal_force(contacts, contact_mask)
    max_slip = _max_slip_ratio(contacts, contact_mask)

    bowl_z = bowl_position[:, 2]

    # ---- Phase detection (forward state machine) ----
    # Note: is_bowl_table is always True in LIBERO (arena artifact) — not used.
    # Gripper qpos mean ≈ 0 for all frames (Panda symmetric joints) — not used.
    # Phase is determined solely by bg (bowl-gripper), bp (bowl-plate), bowl_z.
    phase = np.full(T, PHASE_FREE, dtype=np.int32)
    for t in range(T):
        prev = phase[t - 1] if t > 0 else PHASE_FREE

        if success[t] > 0.5:
            phase[t] = PHASE_DONE

        elif bp[t]:
            # Bowl touching plate → placing
            phase[t] = PHASE_PLACE

        elif bg[t] and bowl_z[t] >= _BOWL_HIGH_HEIGHT:
            # Bowl clearly in the air with gripper → transport
            phase[t] = PHASE_TRANSPORT

        elif bg[t] and prev in (PHASE_TRANSPORT, PHASE_LIFT) \
                and bowl_z[t] >= _BOWL_LIFT_HEIGHT:
            # Descending from transport, still above lift threshold → keep transport
            phase[t] = PHASE_TRANSPORT

        elif bg[t] and bowl_z[t] >= _BOWL_LIFT_HEIGHT:
            # Bowl rising away from table → lift
            phase[t] = PHASE_LIFT

        elif bg[t] and bowl_z[t] >= _BOWL_REST_HEIGHT:
            # Gripper just closed around bowl, still at table height → grasp
            phase[t] = PHASE_GRASP

        elif bg[t]:
            # First gripper-bowl contact, bowl still resting → touch
            phase[t] = PHASE_TOUCH

    # Detect release: was in place/transport, gripper just lost bowl contact
    for t in range(1, T):
        if not bg[t] and bg[t - 1] and phase[t - 1] >= PHASE_GRASP \
                and phase[t] == PHASE_FREE:
            phase[t] = PHASE_RELEASE

    # ---- Event flags [T, NUM_EVENTS] ----
    events = np.zeros((T, NUM_EVENTS), dtype=np.float32)

    # Grasp onset: first step with bg contact
    for t in range(1, T):
        if bg[t] and not bg[t-1]:
            events[t, EVENT_GRASP_ONSET] = 1.0

    # Release onset: last step with bg contact
    for t in range(1, T):
        if not bg[t] and bg[t-1] and phase[t-1] >= PHASE_GRASP:
            events[t, EVENT_RELEASE_ONSET] = 1.0

    events[:, EVENT_BOWL_TABLE]  = bt.astype(np.float32)
    events[:, EVENT_BOWL_PLATE]  = bp.astype(np.float32)
    events[:, EVENT_HIGH_FORCE]  = (max_nf > _HIGH_FORCE_THRESHOLD).astype(np.float32)
    events[:, EVENT_SLIP]        = (max_slip > _SLIP_RATIO_THRESHOLD).astype(np.float32)

    event_names = [
        "grasp_onset", "release_onset", "bowl_table",
        "bowl_plate", "high_force", "slip",
    ]

    return ContactEventData(
        phase=phase,
        phase_names=PHASE_NAMES,
        events=events,
        event_names=event_names,
    )
