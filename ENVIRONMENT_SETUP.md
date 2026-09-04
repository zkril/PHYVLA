# Environment Setup Guide

This document provides complete instructions for replicating the **phyvla** development environment on a new machine.

---

## Overview

The project uses **three separate conda environments** for different phases:

| Environment | Python | Key Frameworks | GPU | Purpose |
|-------------|--------|----------------|-----|---------|
| **libero_replay** | 3.8.13 | PyTorch 1.11 + CUDA 11.3, robosuite 1.4, MuJoCo 3.2 | ✅ | P0-P1: Physics extraction, tokenizer training |
| **phyvla-p2** | 3.11.15 | JAX 0.10.2, mujoco_warp 3.10.0.3, no PyTorch | ✅ | P2-P4: Differentiable rollout, gradients |
| **starVLA** | 3.10.19 | PyTorch 2.6, Flash Attention 2.7.4 | ✅ | P5: VLA integration |

**Why three environments?**
- PyTorch 1.11 (CUDA 11.3) vs PyTorch 2.6 (CUDA 12.4) are incompatible
- JAX uses XLA/CUDA 12, conflicts with older PyTorch
- mujoco_warp uses CUDA Warp, needs clean JAX environment

---

## Prerequisites

### System Requirements
- **OS**: Linux (tested on Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support (RTX 3090 or better recommended)
- **CUDA**: 11.3 + 12.4 (via conda, no system install needed)
- **RAM**: 32 GB+ recommended
- **Disk**: 10 GB+ free space

### Software Dependencies
- Anaconda or Miniconda (conda 4.10+)
- Git

### Install Conda (if not already installed)
```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Initialize (then restart terminal)
conda init bash
```

---

## Quick Setup (Automated)

### 1. Clone the Repository
```bash
git clone <your-repo-url> phyvla
cd phyvla
```

### 2. Run Setup Script
```bash
bash setup_envs.sh
```

This will create all three environments from the YAML files in `envs/`.

### 3. Install Third-party Dependencies

#### Install LIBERO
```bash
conda activate libero_replay

# Clone LIBERO (if not included in repo)
cd third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO

# Install in editable mode
pip install -e .

cd ../..
```

#### Download LIBERO Datasets
```bash
# Still in libero_replay environment
python -c "from libero.libero import get_libero_path; print(get_libero_path('datasets'))"

# Follow instructions at:
# https://github.com/Lifelong-Robot-Learning/LIBERO#download-datasets
# Datasets should be placed at: ~/.libero/datasets/
```

#### Install StarVLA (optional, for P5)
```bash
conda activate starVLA

# Clone StarVLA (if not included in repo)
cd third_party
git clone <starVLA-repo-url> starVLA
cd starVLA

# Install in editable mode
pip install -e .

cd ../..
```

**Download StarVLA Checkpoint:**
```bash
# Checkpoint should be placed at:
# /home/<username>/workspace/starVLA/playground/Pretrained_models/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
```

---

## Manual Setup (Step-by-Step)

If the automated setup fails, follow these manual steps:

### Environment 1: libero_replay (P0-P1)

```bash
# Create environment
conda create -n libero_replay python=3.8.13
conda activate libero_replay

# Install PyTorch with CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install core dependencies
pip install numpy==1.22.4 h5py==3.11.0 matplotlib==3.5.3 opencv-python==4.6.0.66

# Install MuJoCo and robosuite
pip install mujoco==3.2.3
pip install robosuite==1.4.0 robomimic==0.2.0

# Install LIBERO
cd third_party/LIBERO
pip install -r requirements.txt
pip install -e .
cd ../..

# Install other utilities
pip install wandb==0.13.1 tensorboard==2.14.0 hydra-core==1.2.0 \
    einops==0.4.1 transformers==4.21.1 easydict==1.9
```

### Environment 2: phyvla-p2 (P2-P4)

```bash
# Create environment
conda create -n phyvla-p2 python=3.11.15
conda activate phyvla-p2

# Install JAX with CUDA 12
pip install jax[cuda12]==0.10.2

# Install MuJoCo and mujoco_warp
pip install mujoco==3.10.0
pip install mujoco-mjx==3.10.0
pip install mujoco-warp==3.10.0.3

# Install other dependencies
pip install numpy matplotlib opencv-python h5py scipy
pip install warp-lang==1.15.0
```

### Environment 3: starVLA (P5)

```bash
# Create environment
conda create -n starVLA python=3.10.19
conda activate starVLA

# Install PyTorch 2.6 with CUDA 12.4
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention (requires CUDA toolkit)
pip install flash-attn==2.7.4 --no-build-isolation

# Install transformers and diffusers
pip install transformers==4.49.2 diffusers==0.36.0 accelerate==1.5.2

# Install other dependencies
pip install einops albumentations deepspeed decord pillow opencv-python h5py
pip install huggingface-hub datasets

# Install StarVLA
cd third_party/starVLA
pip install -e .
cd ../..
```

---

## Verification

### Test libero_replay
```bash
conda activate libero_replay

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
python -c "import robosuite; print(f'robosuite: {robosuite.__version__}')"
python -c "from libero.libero import get_libero_path; print('LIBERO: OK')"

# Quick test script
python scripts/inspect_libero_hdf5.py --suite libero_spatial --task-id 2
```

### Test phyvla-p2
```bash
conda activate phyvla-p2

# Test imports
python -c "import jax; print(f'JAX: {jax.__version__}, Devices: {jax.devices()}')"
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
python -c "import mujoco_warp as mjw; print('mujoco_warp: OK')"

# Quick test script
python scripts/test_p3_warp_rollout.py --task-id 2 --quick
```

### Test starVLA
```bash
conda activate starVLA

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import flash_attn; print('Flash Attention: OK')"
python -c "from transformers import Qwen2VLForConditionalGeneration; print('Transformers: OK')"

# Quick test (requires checkpoint)
cd /home/<username>/workspace/starVLA
python /path/to/phyvla/scripts/export_vla_actions.py --task-id 2 --demo-index 0 --max-steps 5
```

---

## Troubleshooting

### Issue: `conda activate` not working
**Solution:**
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate libero_replay
```

Or add to `~/.bashrc`:
```bash
eval "$(conda shell.bash hook)"
```

### Issue: CUDA out of memory
**Solution:**
- Reduce batch size in training scripts
- Use `--max-steps` flag to limit trajectory length
- Close other GPU processes

### Issue: Flash Attention installation fails
**Solution:**
```bash
# Option 1: Install with pre-built wheel
pip install flash-attn==2.7.4 --no-build-isolation

# Option 2: Install from source (requires CUDA toolkit)
pip install flash-attn --no-build-isolation

# Option 3: Skip Flash Attention (slower inference)
# Modify scripts to use standard attention
```

### Issue: MuJoCo contact forces return zeros
**Solution:** Ensure you're using the unwrapped objects:
```python
raw_model = getattr(self.model, "_model", self.model)
raw_data = getattr(self.data, "_data", self.data)
mujoco.mj_contactForce(raw_model, raw_data, contact_id, result)
```

### Issue: `libero_replay` import errors
**Solution:**
```bash
cd third_party/LIBERO
pip install -e .  # Reinstall in editable mode
```

### Issue: mujoco_warp buffer overflow
**Solution:** Increase buffer sizes in `mjw.put_data()`:
```python
dx = mjw.put_data(mj_model, mj_data, nworld=K,
                  njmax=300, nconmax=100, naconmax=600)
```

---

## Environment Files

The `envs/` directory contains full environment exports:

```
envs/
├── libero_replay_env.yml  (157 packages, ~4 KB)
├── phyvla-p2_env.yml      (82 packages, ~2 KB)
└── starVLA_env.yml        (169 packages, ~4 KB)
```

These were exported with:
```bash
conda env export --no-builds > envs/<env_name>_env.yml
```

To create from these files:
```bash
conda env create -f envs/libero_replay_env.yml
conda env create -f envs/phyvla-p2_env.yml
conda env create -f envs/starVLA_env.yml
```

---

## Minimal Environment (Quick Start)

If you only want to run specific phases, you can create minimal environments:

### P1 Only (Physics Extraction + Tokenizer)
```bash
conda create -n phyvla-minimal python=3.8.13
conda activate phyvla-minimal
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy h5py matplotlib mujoco robosuite
# Install LIBERO as above
```

### P3-P4 Only (Warp Rollout + Gradients)
```bash
conda create -n phyvla-warp python=3.11
conda activate phyvla-warp
pip install jax[cuda12] mujoco==3.10.0 mujoco-warp==3.10.0.3
pip install numpy h5py matplotlib
```

---

## Docker Alternative (Advanced)

For complete reproducibility, consider creating a Docker image:

```dockerfile
# Dockerfile (example)
FROM nvidia/cuda:12.4.0-cudnn9-devel-ubuntu22.04

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

# Copy environment files
COPY envs/ /workspace/envs/
WORKDIR /workspace

# Create environments
RUN conda env create -f envs/libero_replay_env.yml && \
    conda env create -f envs/phyvla-p2_env.yml && \
    conda env create -f envs/starVLA_env.yml

# Copy source code
COPY . /workspace/phyvla/
```

Build and run:
```bash
docker build -t phyvla:latest .
docker run --gpus all -it phyvla:latest bash
```

---

## CI/CD Integration

For automated testing, add to `.github/workflows/test.yml`:

```yaml
name: Test Environments
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Conda
        uses: conda-incubator/setup-miniconda@v2
        with:
          python-version: 3.8
      
      - name: Create libero_replay environment
        run: conda env create -f envs/libero_replay_env.yml
      
      - name: Test imports
        run: |
          conda activate libero_replay
          python -c "import torch; import mujoco; print('OK')"
```

---

## Summary

- **Three environments**: libero_replay (P0-P1), phyvla-p2 (P2-P4), starVLA (P5)
- **Automated setup**: `bash setup_envs.sh`
- **Manual setup**: Follow step-by-step instructions above
- **Verification**: Run test commands for each environment
- **Third-party**: LIBERO and StarVLA need separate installation

For detailed usage, see:
- **README.md**: Quick start commands
- **CLAUDE.md**: Development guide
- **PROJECT_ARCHIVE.md**: Complete project documentation

---

**Last Updated**: 2026-09-04  
**Tested On**: Ubuntu 20.04, CUDA 11.3/12.4, RTX 3090
