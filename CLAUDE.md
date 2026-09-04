# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**phyvla** is a physics analysis toolkit for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) robot learning benchmark, and the foundation for **DP-VLA** (Differentiable Physics-guided Vision-Language-Action). It extracts, replays, and visualizes physics states from LIBERO demonstrations stored in HDF5 files, using MuJoCo/robosuite as the simulation backend.

The current milestone (M0–M7, completed) is reliable physics data extraction. The next phase (M8+) is building a Physical Tokenizer, MJX-JAX differentiable rollout, and eventually a VLA action-refinement loop. See `doc/plan.md` for the full roadmap.

## Environment Setup

```bash
conda create -n libero python=3.8.13

conda activate libero
cd third_party/LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

LIBERO datasets must be downloaded before running any script. The library resolves HDF5 and BDDL paths via `get_libero_path("datasets")` and `get_libero_path("bddl_files")`.

## Common Commands

**Replay a demonstration using recorded actions:**
```bash
python scripts/replay_libero_actions.py \
    --suite libero_spatial --task-id 2 --demo-index 0 \
    --output outputs/replay/task02_demo00_actions.mp4
```

**Replay using stored MuJoCo states (no drift):**
```bash
python scripts/replay_libero_states.py \
    --suite libero_spatial --task-id 2 --demo-index 0 \
    --output outputs/replay/task02_demo00_states.mp4
```

**Extract physics data and save as NPZ + JSON (smoke test):**
```bash
python scripts/extract_task02_physics_smoke.py \
    --output outputs/physics/task02_demo00_step6.npz
```

**Visualize contact forces as an annotated video:**
```bash
python scripts/visualize_contact_forces.py \
    --suite libero_spatial --task-id 2 --demo-index 0 \
    --output outputs/physics/task02_demo00_contact_force.mp4
```

Common flags across scripts: `--camera`, `--height`, `--width`, `--fps`, `--stride`, `--max-steps`, `--flip`, `--seed`, `--max-contacts`.

## Architecture

### Data Flow

```
HDF5 demo files (LIBERO benchmark)
        ↓
replay_libero_*.py  ←  configs/libero_task02.yaml
        ↓
OffScreenRenderEnv (robosuite/MuJoCo)
        ↓
LiberoPhysicsAdapter  →  contact/robot/object feature tensors
        ↓
NPZ + JSON + MP4 output
```

### Key Source Files

**`src/adapters/libero_physics_adapter.py`** — Core physics extraction. `LiberoPhysicsAdapter` wraps a robosuite env and produces fixed-shape arrays per call to `extract()`:
- `contacts`: `[max_contacts, 28]` float32 — up to 16 contacts (default) sorted by task relevance then force magnitude. The 28 features are defined in `CONTACT_FEATURE_NAMES` (position, normal, world-frame force/torque, distance, scalar force magnitudes, geom/body IDs, boolean relation flags: `touches_bowl`, `touches_plate`, `touches_gripper`, `is_bowl_plate`, `is_bowl_gripper`, `is_bowl_table`, `is_robot_environment`).
- `contact_mask`: `[max_contacts]` — 1.0 for valid slots, 0.0 for padding.
- `robot`: joint qpos/qvel (7 DoF), EE pos + xyzw quaternion + lin/ang vel, gripper joints.
- `objects`: per-object position, xyzw quaternion, linear/angular velocity, mass, inertia, friction (averaged over geoms).

Object name resolution uses pattern lists (e.g., `("black_bowl", "bowl")`), returning the shortest match by priority. Quaternions are always **xyzw** in output; the internal MuJoCo `wxyz` convention is converted by `_wxyz_to_xyzw`.

**`scripts/libero_replay_common.py`** — Shared utilities used by `replay_libero_states.py` and `visualize_contact_forces.py`: `load_demo()`, `make_env()`, `resolve_task()`, `restore_recorded_model()`, `transform_frame()`, `validate_state_dimension()`.

**`scripts/replay_libero_actions.py`** — Self-contained replay script that also re-exports a subset of the common utilities (used directly by `extract_task02_physics_smoke.py`). Contains a local copy of `safe_edit_model_xml()` and `restore_recorded_model()` — these are *not* bugs, they are intentional self-containment. Do not deduplicate them without checking imports in all downstream scripts.

### Two Replay Strategies

- `replay_libero_states.py` — Restores full MuJoCo `qpos`/`qvel` each frame via `regenerate_obs_from_state()`; zero drift, preferred for physics extraction.
- `replay_libero_actions.py` — Re-executes recorded actions; accumulates drift but logs state L2 error per step to a JSON summary alongside the video.

### Script `sys.path` Convention

Scripts that need `src/` modules prepend both `scripts/` and `src/` to `sys.path` at startup:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
```

This is why imports like `from adapters.libero_physics_adapter import ...` work without installing the package.

### MuJoCo Binding Compatibility

The codebase targets robosuite 1.4.x, which wraps native MuJoCo objects in `binding_utils.MjModel` / `binding_utils.MjData`. When calling native MuJoCo C functions (e.g., `mujoco.mj_contactForce`), the raw underlying objects must be unwrapped:

```python
raw_model = getattr(self.model, "_model", self.model)
raw_data  = getattr(self.data,  "_data",  self.data)
```

Fallback paths for older APIs (e.g., `model.body_id2name()` vs `mujoco.mj_id2name()`) are handled throughout `LiberoPhysicsAdapter`.

### Third-party

`third_party/LIBERO/` contains the upstream LIBERO library installed in editable mode. Modify source there only for local patches; prefer upstream contributions otherwise.

## P5 — StarVLA 物理 token 集成（进行中）

**VLA 选型**：StarVLA（Qwen2.5-VL-GR00T-LIBERO-4in1），checkpoint 在
`/home/jiangnan-luxitech/workspace/starVLA/playground/Pretrained_models/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt`

**运行环境**：`starVLA` conda（torch 2.6, flash_attn 2.7.4），工作目录必须为 `/home/jiangnan-luxitech/workspace/starVLA`（config 含相对路径）。

**P5 新增文件**：

- `src/models/physics_vla_projector.py` — `PhysicsTokenProjector`：Linear(256→2048) + GELU + LayerNorm + 可学习位置编码，将 z_phy 投影到 VLM 隐藏空间（~1.6M 参数）。
- `src/p5/starvla_physics.py` — `StarVLAPhysicsWrapper`：组合 Qwen_GR00T + PhysicsTokenProjector；`predict()` 将 z_proj 拼接至 last_hidden 后传入 DiT action head；`cache_vlm_hidden()` 预缓存冻结 VLM 的隐状态。
- `scripts/export_vla_actions.py` — P5-0：从 LIBERO HDF5 逐帧推理，导出 `vla_actions_demo0.npz`（`starVLA` env）。
- `scripts/train_p5_projector.py` — P5-3：预缓存 VLM 隐状态 + 训练 projector + 物理敏感性验收（`starVLA` env）。

**P5 sys.path 约定**：P5 脚本额外添加：
```python
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "starVLA"))
sys.path.insert(0, "/home/jiangnan-luxitech/workspace/starVLA")
os.chdir("/home/jiangnan-luxitech/workspace/starVLA")  # 必须，config 用相对路径
```

### Third-party

`third_party/LIBERO/` — LIBERO 库（P0–P4 使用，libero_replay env）。
`third_party/starVLA/` — StarVLA 框架代码（P5 使用，starVLA env）。

