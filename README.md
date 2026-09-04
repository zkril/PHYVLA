# phyvla

**Differentiable Physics-guided Vision-Language-Action (DP-VLA)** - A physics analysis toolkit for the LIBERO robot learning benchmark, serving as the foundation for physics-enhanced VLA models.

## Project Overview

phyvla extracts, replays, and visualizes physics states from LIBERO demonstrations using MuJoCo/robosuite as the simulation backend. It builds structured Physical Tokens from robot states, object dynamics, contact forces, and physics gradients, then injects them into VLA models for physics-aware action generation.

**Current Status**: P0-P4 complete (7/7 tasks validated), P5 in progress

### What is phyvla?

phyvla is a research toolkit for integrating physics understanding into Vision-Language-Action (VLA) models. It addresses a key limitation in current VLA approaches: **they lack explicit physics awareness**. By extracting and encoding physical states (contacts, forces, object properties, dynamics gradients) into structured tokens, phyvla enables VLA models to make more physically-grounded action predictions.

### Key Features

- 🔍 **Zero-drift Physics Extraction**: Replay LIBERO demonstrations with perfect state restoration
- 🧠 **Physical Tokenizer**: Encode physics into `[B, 7+N, 256]` structured tokens (7 base + N objects)
- ⚡ **GPU-accelerated Rollout**: mujoco_warp enables K=8×H=30 rollout in 0.3s (87× faster than MJX)
- 📊 **Gradient-based Refinement**: Parallel finite difference for action sensitivity analysis
- 🤖 **VLA Integration**: Inject physics tokens into StarVLA (Qwen2.5-VL) for physics-aware control
- ✅ **Validated**: 7/7 libero_spatial tasks pass all metrics (robot RMSE, contact F1, role accuracy)

### Why Three Environments?

Different phases require incompatible dependencies:

1. **libero_replay**: Legacy PyTorch 1.11 + CUDA 11.3 (matches robosuite/LIBERO ecosystem)
2. **phyvla-p2**: JAX + CUDA 12 (for differentiable simulation, conflicts with old PyTorch)
3. **starVLA**: Modern PyTorch 2.6 + Flash Attention (VLA training, conflicts with JAX)

## Quick Start

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support (RTX 3090 or better recommended)
- **Conda**: Anaconda or Miniconda installed
- **Disk**: 10 GB+ free space

### Environment Setup

The project requires **three separate conda environments** for different development phases:

| Environment | Python | Purpose | Key Dependencies |
|-------------|--------|---------|------------------|
| **libero_replay** | 3.8.13 | P0-P1: Physics extraction & tokenizer | PyTorch 1.11, robosuite 1.4, MuJoCo 3.2 |
| **phyvla-p2** | 3.11.15 | P2-P4: Differentiable rollout & gradients | JAX 0.10, mujoco_warp 3.10 |
| **starVLA** | 3.10.19 | P5: VLA training & inference | PyTorch 2.6, Flash Attention 2.7 |

**Automated setup (recommended):**
```bash
# 1. Clone repository
git clone <your-repo-url> phyvla
cd phyvla

# 2. Create all three environments automatically
bash setup_envs.sh

# 3. Install LIBERO
conda activate libero_replay
cd third_party/LIBERO
pip install -e .
cd ../..

# 4. Download LIBERO datasets
# Follow instructions at: https://github.com/Lifelong-Robot-Learning/LIBERO
# Datasets should be placed at: ~/.libero/datasets/
```

**Manual setup:**
```bash
# Create from environment files
conda env create -f envs/libero_replay_env.yml
conda env create -f envs/phyvla-p2_env.yml
conda env create -f envs/starVLA_env.yml
```

**Verify installation:**
```bash
# Test libero_replay
conda activate libero_replay
python -c "import torch; import mujoco; import robosuite; print('✓ libero_replay OK')"

# Test phyvla-p2
conda activate phyvla-p2
python -c "import jax; import mujoco_warp as mjw; print('✓ phyvla-p2 OK')"

# Test starVLA (optional, for P5)
conda activate starVLA
python -c "import torch; import flash_attn; print('✓ starVLA OK')"
```

See **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** for detailed instructions, troubleshooting, and manual installation steps.

---

## Usage Workflow

### Phase 0-1: Physics Extraction & Tokenizer Training

```bash
conda activate libero_replay

# Step 1: Replay demonstration and visualize contact forces
python scripts/visualize_contact_forces.py \
    --suite libero_spatial --task-id 2 --demo-index 0 \
    --output outputs/physics/task02_demo00_contact_force.mp4

# Step 2: Extract physics features to NPZ
python scripts/extract_task02_physics_smoke.py \
    --output outputs/physics/task02_demo00_step6.npz

# Step 3: Run full P1 pipeline (compute normalization + train tokenizer)
python scripts/run_task_pipeline.py --task-id 2 --py-libero

# Step 4: Run multi-task pipeline (7 tasks)
python scripts/run_multi_task_pipeline.py
```

**Expected outputs:**
- `outputs/physics/`: Contact force videos and NPZ files
- `outputs_back/normalization_v1.json`: Normalization statistics
- `outputs_back/contact_events_v1.npz`: Contact event features
- `outputs_back/multi_task_results.json`: Validation metrics (7/7 tasks)

### Phase 2-4: Differentiable Rollout & Gradient Refinement

```bash
conda activate phyvla-p2

# Step 1: Test mujoco_warp rollout (K=8 candidates, H=30 steps)
python scripts/test_p3_warp_rollout.py --task-id 2

# Step 2: Demo candidate ranking
python scripts/demo_candidate_ranking.py --task-id 2

# Step 3: Demo gradient-based action refinement
python scripts/demo_gradient_refinement.py --task-id 2

# Step 4: Run multi-task P3-P4 validation
python scripts/run_multitask_p3p4.py
```

**Expected outputs:**
- Rollout visualizations with physics overlays
- Candidate ranking comparisons
- Gradient refinement loss curves

### Phase 5: VLA Integration (In Progress)

```bash
conda activate starVLA

# Change to StarVLA directory (required for config paths)
cd /home/<username>/workspace/starVLA

# Step 1: Export VLA baseline actions
python /path/to/phyvla/scripts/export_vla_actions.py \
    --suite libero_spatial --task-id 2 --demo-index 0

# Step 2: Pre-cache VLM hidden states (saves 95% training time)
python /path/to/phyvla/scripts/train_p5_projector.py \
    --task-id 2 --cache-only

# Step 3: Train PhysicsTokenProjector
python /path/to/phyvla/scripts/train_p5_projector.py \
    --task-id 2 --epochs 200 --lr 1e-4 --batch-size 16
```

**Note**: P5 requires StarVLA checkpoint at:
`/home/<username>/workspace/starVLA/playground/Pretrained_models/Qwen2.5-VL-GR00T-LIBERO-4in1/`

See **[PROJECT_ARCHIVE.md](PROJECT_ARCHIVE.md)** for detailed P5 execution plan.

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

- **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)**: Complete environment setup guide ⭐ **Start here for new installations**
- **[CLAUDE.md](CLAUDE.md)**: Developer guide for Claude Code
- **[README.md](README.md)**: This file - project overview & quick start
- **[doc/plan.md](doc/plan.md)**: Complete technical roadmap (P0-P8)
- **[physicsvla（仿真）.md](physicsvla（仿真）.md)**: Detailed design & training flow
- **[PROJECT_ARCHIVE.md](PROJECT_ARCHIVE.md)**: Archive summary & P5+ plan
- **[CLEANUP_PLAN.md](CLEANUP_PLAN.md)**: Repository cleanup guide
- **[GIT_SETUP_SUMMARY.md](GIT_SETUP_SUMMARY.md)**: Git workflow & operations

## Technology Stack

| Component | Solution | Version | Environment |
|-----------|----------|---------|-------------|
| Physics backend (P0-P1) | MuJoCo + robosuite | 3.2.3 / 1.4.x | libero_replay |
| Physics backend (P3-P4) | mujoco_warp | 3.10.0.3 | phyvla-p2 |
| VLA base model | StarVLA (Qwen2.5-VL-GR00T) | LIBERO-4in1 | starVLA |
| ML framework (P0-P1) | PyTorch + CUDA 11.3 | 1.11.0 | libero_replay |
| ML framework (P2-P4) | JAX + CUDA 12 | 0.10.2 | phyvla-p2 |
| ML framework (P5) | PyTorch 2.6 + Flash Attn | 2.6 / 2.7.4 | starVLA |
| Gradient method | Parallel finite difference | K=1+2Hnu worlds | phyvla-p2 |
| Simulation env | LIBERO benchmark | - | All |

**Three environments**: See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for setup details.

## Roadmap

### Completed Phases

- [x] **P0: Physics Data Extraction** (2026-06)
  - Zero-drift replay via qpos/qvel injection
  - LiberoPhysicsAdapter: Extract robot/object/contact states
  - Contact force validation (271 geoms, up to 16 contacts/frame)

- [x] **P1: Physical Tokenizer** (2026-07)
  - Architecture: `[B, 7+N, 256]` structured tokens
  - Role-aware object encoding (TARGET/GOAL/DISTRACTOR/STATIC/GRIPPER)
  - Multi-task validation: **7/7 libero_spatial tasks passed**
  - Metrics: robot RMSE 0.0048-0.0077, contact F1 0.9773-0.9955

- [x] **P2: MuJoCo → MJX Parity** (2026-07)
  - State mapping validation: RMSE=5.76e-8m
  - XLA OOM issue identified (271 geoms → 600+ MB per world)

- [x] **P3: mujoco_warp Rollout** (2026-08)
  - Production backend: CUDA Warp (no XLA)
  - Performance: K=8×H=30 → **0.3s** (87× faster than MJX)
  - Full LIBERO XML (271 geoms, no mesh reduction)

- [x] **P4: Gradient-based Refinement** (2026-08)
  - Parallel finite difference (K=1+2Hnu worlds)
  - 5 iterations: loss 0.2357 → 0.2342 (↓0.64%)
  - Note: wp.Tape autodiff returns zero for ctrl→xpos; FD is production method

### Current Phase

- [ ] **P5: StarVLA Physics Token Injection** (2026-09, in progress)
  - `PhysicsTokenProjector`: Linear(256→2048) + position encoding (~1.6M params)
  - `StarVLAPhysicsWrapper`: Inject z_proj into VLM hidden states
  - VLM caching: Pre-compute frozen VLM outputs (saves 95% training time)
  - Training: 200 epochs, physics sensitivity validation
  - **Status**: Code ready, training pending (see PROJECT_ARCHIVE.md)

### Future Phases

- [ ] **P6: Sim-to-Real Deployment** (2026-10+)
  - Domain randomization (mass ±30%, friction ±50%)
  - Real robot deployment on Franka Panda
  - Sim-to-real transfer validation

- [ ] **P7: End-to-End Differentiable Training** (2026-11+)
  - Backprop through physics (wp.Tape or Brax)
  - Joint VLA + tokenizer training
  - Physics-constrained loss functions

- [ ] **P8: Multi-Task Scaling** (2027+)
  - Scale to all 130 LIBERO tasks
  - Dexterous manipulation (multi-finger grippers)
  - Long-horizon task decomposition

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{phyvla2026,
  title={phyvla: Differentiable Physics-guided Vision-Language-Action for Robot Manipulation},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/phyvla}},
  note={Physics-enhanced VLA toolkit for the LIBERO benchmark}
}
```

## License

[MIT License / Apache-2.0 / GPL-3.0 - Specify your license here]

## Acknowledgments

- **LIBERO**: Lifelong Robot Learning benchmark ([GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO))
- **robosuite**: Robot simulation framework ([GitHub](https://github.com/ARISE-Initiative/robosuite))
- **MuJoCo**: Multi-Joint dynamics with Contact physics engine
- **mujoco_warp**: GPU-accelerated MuJoCo with NVIDIA Warp ([GitHub](https://github.com/Farama-Foundation/mujoco_warp))
- **StarVLA**: Vision-Language-Action model for robot control

## Contact & Support

- **Issues**: Please open an issue on GitHub for bug reports or feature requests
- **Documentation**: See [FINAL_SUMMARY.md](FINAL_SUMMARY.md) for complete project documentation index
- **Environment Setup**: See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed installation guide
- **Development Guide**: See [CLAUDE.md](CLAUDE.md) for development conventions

## Related Projects

- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): Benchmark for lifelong robot learning
- [robosuite](https://github.com/ARISE-Initiative/robosuite): Robot simulation framework
- [mujoco_warp](https://github.com/Farama-Foundation/mujoco_warp): GPU-accelerated physics simulation
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL): Vision-language foundation model
