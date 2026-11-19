#!/usr/bin/env python3
"""Verify LIBERO-plus imports, resources, rendering, stepping and optional policy calls."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--with-policy", action="store_true")
    parser.add_argument("--with-physics-state", action="store_true")
    parser.add_argument("--policy-mode", choices=["baseline", "phyvla"], default="baseline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9883)
    parser.add_argument("--checkpoint", default=str(ROOT / "thirdparty/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"))
    parser.add_argument("--report", type=Path, default=ROOT / "logs/liberoplus_repair/environment-smoke.json")
    args = parser.parse_args()
    libero_root = ROOT / "thirdparty/LIBERO-plus"
    starvla_root = ROOT / "thirdparty/starVLA"
    os.environ["LIBERO_HOME"] = str(libero_root)
    os.environ["LIBERO_CONFIG_PATH"] = str(libero_root / "libero")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    sys.path[:0] = [str(starvla_root / "examples/simBenchmarks/LIBERO-plus/eval_files"), str(libero_root), str(starvla_root)]
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from libero.libero import benchmark
    from eval_libero import _get_libero_env, _binarize_gripper_open, ModelClient
    client = None
    results = []
    try:
        if args.with_policy:
            client = ModelClient(policy_ckpt_path=args.checkpoint, host=args.host, port=args.port, image_size=[224, 224], require_physics=args.policy_mode == "phyvla")
        for name in args.suites:
            suite = benchmark.get_benchmark_dict()[name]()
            task = suite.get_task(0)
            env, language = _get_libero_env(task, 128, 7)
            try:
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(0)[0])
                for _ in range(10):
                    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                images = [np.ascontiguousarray(obs[k][::-1, ::-1]) for k in ("agentview_image", "robot0_eye_in_hand_image")]
                assert all(im.shape == (128, 128, 3) for im in images)
                result = {"suite": name, "task_id": 0, "task_count": suite.n_tasks, "render_and_step": True}
                if args.with_physics_state or args.policy_mode == "phyvla":
                    from adapters.live_physics import LivePhysicsState
                    from datasets.physics_dataset import PhysicsDataset
                    provider = LivePhysicsState(env, language)
                    payload = provider.capture()
                    features = PhysicsDataset.from_arrays(payload["arrays"], payload["events"], language)[0]
                    expected = {"robot_vec": (33,), "contacts": (16, 28), "obj_cont": (16, 25),
                                "dyn_target": (16,), "dyn_rel_cont": (16, 36), "afford_vec": (13,)}
                    for key, shape in expected.items():
                        value = features[key].numpy()
                        assert value.shape == shape and np.isfinite(value).all(), (key, value.shape)
                    assert not features["dyn_target"].any(), "Live features must not use future frames"
                    provider.reset()
                    assert len(provider.history) == 0
                    result["live_physics_features"] = True
                    result["objects"] = provider.keys
                if client:
                    client.reset(task_description=language)
                    response = client.step(example={"image": images, "lang": language}, step=0, **({"physics_state": payload} if args.policy_mode == "phyvla" else {}))
                    raw = response["raw_action"]
                    action = np.concatenate([np.asarray(raw["world_vector"]).reshape(-1), np.asarray(raw["rotation_delta"]).reshape(-1), _binarize_gripper_open(np.asarray(raw["open_gripper"]).reshape(-1))])
                    assert action.shape == (7,) and np.isfinite(action).all(), action
                    env.step(action)
                    result["policy_step"] = True
                    result["policy_mode"] = args.policy_mode
                results.append(result)
                print(json.dumps(result), flush=True)
            finally:
                env.close()
    finally:
        if client:
            client.client.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"python": sys.executable, "results": results, "scope": "first task of each selected suite; not a success-rate evaluation"}, indent=2) + "\n")
    assert len(results) == len(args.suites)

if __name__ == "__main__":
    main()
