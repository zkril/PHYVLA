# phyvla

**Differentiable Physics-guided Vision-Language-Action (DP-VLA)** - A physics analysis toolkit for the LIBERO robot learning benchmark, serving as the foundation for physics-enhanced VLA models.

## Project Overview

phyvla extracts, replays, and visualizes physics states from LIBERO demonstrations using MuJoCo/robosuite as the simulation backend. It builds structured Physical Tokens from robot states, object dynamics, contact forces, and physics gradients, then injects them into VLA models for physics-aware action generation.

**Current Status**: P0-P4 complete (7/7 tasks validated), P5 in progress

## Quick Start

### Environment Setup

```bash
# Create libero_replay environment (P0-P1: data extraction & tokenizer training)
conda create -n libero_replay python=3.8.13
conda activate libero_replay
cd third_party/LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

### Extract Physics Data

```bash
conda activate libero_replay

# Replay demonstration and visualize contact forces
python scripts/visualize_contact_forces.py \
    --suite libero_spatial --task-id 2 --demo-index 0 \
    --output outputs/physics/task02_demo00_contact_force.mp4

# Extract physics features to NPZ
python scripts/extract_task02_physics_smoke.py \
    --output outputs/physics/task02_demo00_step6.npz
```

### Train Physical Tokenizer

```bash
# Run full P1 pipeline (single task)
python scripts/run_task_pipeline.py --task-id 2 --py-libero

# Run multi-task pipeline (7 tasks)
python scripts/run_multi_task_pipeline.py
```

## Architecture

### Physical Token Structure

Output shape: `[B, 7+N, 256]` (7 base slots + N object slots, N=16 max)

**Base slots**:
- `z_robot`: Robot kinematic state (qpos/qvel/ee/gripper)
- `z_contact`: Role-aware contact set (position/normal/force/torque)
- `z_dyn`: Dynamics trend (rollout-based, replaces teacher-forcing in P3+)
- `z_risk`: Risk signals (rollout cost terms)
- `z_afford`: Task affordance & phase (8 phases: pre-grasp/reaching/grasping/...)
- `z_grad`: Action sensitivity (FD gradient, zero-masked in P1)
- `z_scene_ctx`: Scene summary (role-weighted pooling of object tokens)

**Object slots** (per-object, shared encoder):
- Position, orientation (6D rotation), linear/angular velocity
- Mass, inertia, friction (log-transformed)
- Role embedding (TARGET/GOAL_SURFACE/DISTRACTOR/STATIC/GRIPPER)
- Task-relevant features (is_grasped, dist_to_ee, dist_to_goal)

### Data Flow

```
HDF5 demo files (LIBERO)
    ↓
replay_libero_states.py (zero-drift replay via qpos/qvel injection)
    ↓
LiberoPhysicsAdapter (extract robot/object/contact @ 271 geoms)
    ↓
PhysicalTokenizer → [B, 23, 256] tokens
    ↓
(P5) PhysicsTokenProjector → [B, 23, 2048]
    ↓
StarVLA (Qwen2.5-VL + DiT action head) → actions [B, 8, 7]
```

## Key Results

### P1 Validation (libero_spatial tasks 1-7)

| Metric | Threshold | Result | Status |
|--------|-----------|--------|--------|
| robot_rmse | < 0.10 | 0.0048-0.0077 | ✅ |
| phase_acc | > 0.85 | 1.0000 | ✅ |
| contact_f1 | > 0.85 | 0.9773-0.9955 | ✅ |
| role_acc | > 0.85 | 1.0000 | ✅ |
| param_rmse | < 0.05 | 0.0052-0.0140 | ✅ |
| pose_rmse | < 0.10 | 0.0095-0.0191 | ✅ |

**7/7 tasks passed all criteria**

### P3 Performance (RTX 3090)

- **Backend**: mujoco_warp 3.10.0.3 (CUDA Warp, replaces MJX due to OOM)
- **Model**: Full LIBERO XML (271 geoms, no mesh reduction)
- **Rollout speed**: K=8 candidates × H=30 steps → **0.3s** (87× faster than MJX minimal model)

### P4 Gradient Refinement

- **Method**: Parallel finite difference (FD) over K=1+2Hnu worlds
- **Result**: 5 iterations, loss 0.2357 → 0.2342 (↓0.64%)
- **Note**: wp.Tape autodiff returns zero gradient for ctrl→xpos path; FD is production method

## Project Structure

```
phyvla/
├── src/
│   ├── adapters/
│   │   ├── libero_physics_adapter.py    # P0: MuJoCo physics extraction
│   │   └── mjx_state_adapter.py         # P2: MJX state mapping
│   ├── models/
│   │   ├── physics_tokenizer.py         # P1: [B,23,256] encoder
│   │   ├── physics_vla_projector.py     # P5: [256→2048] projection
│   │   └── gradient_encoder.py          # P4: [H,nu]→[256]
│   ├── simulation/
│   │   ├── mjwarp_rollout.py            # P3: Warp parallel rollout
│   │   └── warp_grad.py                 # P4: FD gradient computation
│   └── p5/
│       └── starvla_physics.py           # P5: StarVLA wrapper
├── scripts/
│   ├── replay_libero_states.py          # P0: Zero-drift replay
│   ├── train_tokenizer.py               # P1: Tokenizer training
│   ├── test_p3_warp_rollout.py          # P3: Warp validation
│   ├── demo_gradient_refinement.py      # P4: FD refinement demo
│   └── train_p5_projector.py            # P5: VLA training (pending)
├── doc/
│   └── plan.md                          # Full roadmap (P0-P8)
├── CLAUDE.md                            # Development guide
├── PROJECT_ARCHIVE.md                   # Project archive & next steps
└── physicsvla（仿真）.md                # Detailed design doc
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Developer guide for Claude Code
- **[doc/plan.md](doc/plan.md)**: Complete technical roadmap (P0-P8)
- **[physicsvla（仿真）.md](physicsvla（仿真）.md)**: Detailed design & training flow
- **[PROJECT_ARCHIVE.md](PROJECT_ARCHIVE.md)**: Archive summary & P5+ plan

## Technology Stack

| Component | Solution | Version |
|-----------|----------|---------|
| Physics backend | mujoco_warp | 3.10.0.3 |
| VLA base model | StarVLA (Qwen2.5-VL-GR00T) | LIBERO-4in1 |
| Gradient method | Parallel finite difference | K=1+2Hnu worlds |
| Simulation env | LIBERO + robosuite | 1.4.x |

## Roadmap

- [x] **P0**: Physics data extraction (zero-drift replay)
- [x] **P1**: Physical Tokenizer ([B,23,256], 7/7 tasks validated)
- [x] **P2**: MuJoCo → MJX parity (RMSE=5.76e-8m)
- [x] **P3**: mujoco_warp rollout (0.3s for K=8×H=30)
- [x] **P4**: FD gradient refinement (loss ↓0.64%)
- [ ] **P5**: StarVLA physics token injection (in progress)
- [ ] **P6**: Sim-to-real deployment
- [ ] **P7**: End-to-end differentiable training
- [ ] **P8**: Multi-task scaling & dexterous manipulation

## Citation

If you use this codebase, please cite:

```bibtex
@misc{phyvla2026,
  title={phyvla: Physics-guided Vision-Language-Action for Robot Manipulation},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/phyvla}}
}
```

## License

[Specify your license here]

## Contact

[Your contact information]
