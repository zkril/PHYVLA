"""Build causal physics features from original LIBERO demonstrations.

Uses the exact live-state adapter and recorded simulator XML. No rendering.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "thirdparty/LIBERO/libero")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "thirdparty/LIBERO")]
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from libero_replay_common import load_demo, restore_recorded_model, validate_state_dimension
from adapters.live_physics import LivePhysicsState
from datasets.physics_dataset import PhysicsDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo-indices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--task-ids", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"feature_schema": "causal_hold_v1", "benchmark": "LIBERO",
                "contact_count": 16, "episodes": [], "demo_indices": args.demo_indices}
    for suite_name in args.suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        if suite.n_tasks != 10:
            raise ValueError("Preparation requires the original 10-task suites")
        for task_id in args.task_ids:
            task = suite.get_task(task_id)
            demo_path = ROOT / "data" / suite.get_task_demonstration(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            for demo_index in args.demo_indices:
                out = args.output / suite_name / f"task_{task_id}" / f"demo_{demo_index}.npz"
                meta_path = out.with_suffix(".json")
                if out.exists() and meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    if meta["feature_schema"] != "causal_hold_v1" or meta["language"] != task.language:
                        raise ValueError(f"Incompatible cache {out}")
                else:
                    demo = load_demo(demo_path, demo_index)
                    if demo["model_xml"] is None:
                        raise ValueError(f"Missing recorded XML: {demo_path}")
                    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False,
                                     has_offscreen_renderer=False, has_renderer=False)
                    try:
                        env.seed(7)
                        env.reset()
                        restore_recorded_model(env, demo["model_xml"])
                        validate_state_dimension(env, demo["states"])
                        provider = LivePhysicsState(env, task.language)
                        samples = []
                        for state in demo["states"]:
                            env.set_state(state)
                            env.sim.forward()
                            payload = provider.capture()
                            features = PhysicsDataset.from_arrays(payload["arrays"], payload["events"], task.language)[0]
                            samples.append({k: v.numpy() for k, v in features.items()})
                        arrays = {k: np.stack([row[k] for row in samples]) for k in samples[0]}
                        actions = np.asarray(demo["actions"], dtype=np.float32)
                        if len(actions) != len(samples):
                            raise ValueError("State/action length mismatch")
                        if any(not np.isfinite(v).all() for v in arrays.values()):
                            raise ValueError("Nonfinite features")
                        out.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(out, **arrays, expert_actions=actions)
                        meta = {"suite": suite_name, "task_id": task_id, "language": task.language,
                                "demo_index": demo_index, "demo_key": demo["metadata"]["demo_key"],
                                "source": str(demo_path), "frames": len(samples), "objects": provider.keys,
                                "feature_schema": "causal_hold_v1", "path": str(out.resolve()),
                                "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
                        meta_path.write_text(json.dumps(meta, indent=2))
                    finally:
                        env.close()
                manifest["episodes"].append(meta)
                (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
                print(json.dumps({"suite": suite_name, "task_id": task_id, "demo": demo_index,
                                  "frames": meta["frames"], "objects": len(meta["objects"])}), flush=True)
    print("CAUSAL_DATA_READY", len(manifest["episodes"]), flush=True)

if __name__ == "__main__":
    main()
