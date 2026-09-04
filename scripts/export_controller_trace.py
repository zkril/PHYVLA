"""Export low-level actuator ctrl trace from a LIBERO demonstration.

robosuite's OSC converts 7D high-level actions → low-level data.ctrl.
This script replays the demonstration state-by-state and records the
actual ctrl written to the simulator at each step.

Output: controller_trace_v1.npz with arrays:
  ctrl         [T, n_actuators]   actual actuator control signals
  mocap_pos    [T, n_mocap, 3]    mocap target positions (if any)
  mocap_quat   [T, n_mocap, 4]    mocap target orientations (if any)
  qfrc_applied [T, nv]            applied joint forces (if any)
  states       [T, nq+nv]         qpos+qvel snapshot for parity check
  actions      [T, 7]             original 7D actions for reference

Usage (libero env, Python 3.8):
    python scripts/export_controller_trace.py \
        --suite libero_spatial --task-id 2 --demo-index 0 \
        --output outputs/task02/controller_trace_v1.npz
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

from libero_replay_common import (   # noqa: E402
    load_demo, make_env, resolve_task, restore_recorded_model,
    validate_state_dimension,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export raw ctrl trace from LIBERO demo.")
    p.add_argument("--suite",       default="libero_spatial")
    p.add_argument("--task-id",     type=int, default=2)
    p.add_argument("--demo-index",  type=int, default=0)
    p.add_argument("--camera",      default="agentview")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--output",      type=Path,
                   default=Path("outputs/task02/controller_trace_v1.npz"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _, task, bddl_path, demo_path = resolve_task(args.suite, args.task_id)
    demo = load_demo(demo_path, args.demo_index)

    states  = demo["states"]    # [T, D]
    actions = demo["actions"]   # [T, 7]
    T       = len(states)

    env = make_env(bddl_path, args.camera, 128, 128, args.seed)

    try:
        restore_recorded_model(env, demo["model_xml"])
        validate_state_dimension(env, states)

        sim = env.sim
        n_ctrl   = sim.data.ctrl.shape[0]
        n_mocap  = sim.model.nmocap
        nv       = sim.model.nv

        ctrl_trace    = np.zeros((T, n_ctrl),    dtype=np.float64)
        mocap_pos     = np.zeros((T, max(n_mocap, 1), 3),  dtype=np.float64)
        mocap_quat    = np.zeros((T, max(n_mocap, 1), 4),  dtype=np.float64)
        qfrc_applied  = np.zeros((T, nv),        dtype=np.float64)
        states_out    = np.zeros_like(states)

        print(f"n_actuators={n_ctrl}  n_mocap={n_mocap}  nv={nv}  T={T}")
        print("Strategy: action-based replay (OSC computes ctrl on each env.step)")

        # Restore to initial state
        env.regenerate_obs_from_state(states[0])
        sim.forward()

        for t in range(T):
            # Record state BEFORE step (qpos+qvel snapshot)
            states_out[t] = np.asarray(env.get_sim_state(), dtype=np.float64).copy()

            # Apply action — OSC computes and writes data.ctrl internally
            if t < len(actions):
                action = np.asarray(actions[t], dtype=np.float64)
                result = env.step(action)

            # Record ctrl AFTER step (the ctrl that OSC just applied)
            ctrl_trace[t]   = np.asarray(sim.data.ctrl,         dtype=np.float64).copy()
            qfrc_applied[t] = np.asarray(sim.data.qfrc_applied, dtype=np.float64).copy()

            if n_mocap > 0:
                mocap_pos[t]  = np.asarray(sim.data.mocap_pos,  dtype=np.float64).copy()
                mocap_quat[t] = np.asarray(sim.data.mocap_quat, dtype=np.float64).copy()

            if t % 20 == 0:
                print(f"  [{t:3d}/{T}]  ctrl_norm={np.linalg.norm(ctrl_trace[t]):.4f}"
                      f"  qfrc_norm={np.linalg.norm(qfrc_applied[t]):.4f}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(args.output),
            ctrl=ctrl_trace,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            qfrc_applied=qfrc_applied,
            states=states_out,
            actions=actions,
        )

        print(f"\nSaved to {args.output}")
        print(f"  ctrl         : {ctrl_trace.shape}  norm_mean={np.linalg.norm(ctrl_trace, axis=1).mean():.4f}")
        print(f"  mocap_pos    : {mocap_pos.shape}")
        print(f"  qfrc_applied : {qfrc_applied.shape}  norm_mean={np.linalg.norm(qfrc_applied, axis=1).mean():.4f}")
        print(f"  states       : {states_out.shape}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
