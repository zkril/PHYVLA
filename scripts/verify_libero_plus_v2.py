"""Audit all plus initial-state paths and smoke-test each suite/category with v2."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from adapters.live_physics_v2 import LivePhysicsV2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files'))
from model2libero_interface import ModelClient


def main():
    root = Path(os.environ['LIBERO_HOME']).resolve()
    assert root in Path(benchmark.__file__).resolve().parents, 'Wrong LIBERO package'
    mapping = json.loads((root / 'libero/libero/benchmark/task_classification.json').read_text())
    result = {'status': 'running', 'package': str(Path(benchmark.__file__).resolve()),
              'resource_checks': {}, 'smoke': [], 'errors': []}
    out = ROOT / 'outputs/physics_v2_evaluation/plus_preflight.json'
    def save():
        out.write_text(json.dumps(result, indent=2))
    client = ModelClient(policy_ckpt_path='', host='127.0.0.1', port=6695, require_physics=True)
    result['server'] = client.server_metadata
    for name, rows in mapping.items():
        with contextlib.redirect_stdout(io.StringIO()):
            suite = benchmark.get_benchmark_dict()[name]()
        assert suite.n_tasks == len(rows)
        missing = []
        paths = set()
        load = torch.load
        def check_path(path, *args, **kwargs):
            paths.add(str(path))
            if not Path(path).is_file():
                missing.append(str(path))
            return np.zeros((1, 1))
        torch.load = check_path
        try:
            for i, row in enumerate(rows):
                assert row['id'] == i + 1 and row['name'] == suite.get_task(i).name
                suite.get_task_init_states(i)
        finally:
            torch.load = load
        result['resource_checks'][name] = {'tasks': suite.n_tasks, 'unique_init_paths': len(paths), 'missing': sorted(set(missing))}
        save()
        selected = {}
        for row in rows:
            selected.setdefault(row['category'], row['id'] - 1)
        for category, task_id in selected.items():
            env = None
            try:
                task = suite.get_task(task_id)
                states = suite.get_task_init_states(task_id)
                np.random.seed(7)
                env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
                env.seed(7)
                env.reset()
                obs = env.set_init_state(states[0])
                for _ in range(10):
                    obs, _, _, _ = env.step([0.] * 6 + [-1.])
                provider = LivePhysicsV2(env, task.language)
                before = np.array(env.sim.get_state().flatten())
                physics = provider.capture()
                np.testing.assert_array_equal(before, env.sim.get_state().flatten())
                assert all(np.isfinite(v).all() for v in physics.values())
                client.reset(task.language)
                reply = client.step({'image': [np.ascontiguousarray(obs[k][::-1, ::-1]) for k in ['agentview_image','robot0_eye_in_hand_image']], 'lang': task.language}, step=0, physics_state=physics)
                result['smoke'].append({'suite': name, 'category': category, 'task_id': task_id, 'physics_shapes': {k:list(v.shape) for k,v in physics.items()}, 'inference_ok': True})
                print('PASS', name, category, task_id, flush=True)
            except Exception as exc:
                result['errors'].append({'suite': name, 'category': category, 'task_id': task_id, 'error': repr(exc)})
                traceback.print_exc()
            finally:
                if env is not None:
                    env.close()
                save()
    client.client.close()
    result['status'] = 'passed' if not result['errors'] and not any(x['missing'] for x in result['resource_checks'].values()) else 'failed'
    save()
    print(result['status'], flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
