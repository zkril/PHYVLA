#!/bin/bash
# Environment setup script for phyvla project
# This script creates all three required conda environments

set -e  # Exit on error

echo "=========================================="
echo "phyvla Project Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Function to create environment
create_env() {
    local env_name=$1
    local yml_file=$2

    echo "----------------------------------------"
    echo "Creating environment: $env_name"
    echo "----------------------------------------"

    if conda env list | grep -q "^$env_name "; then
        echo "⚠️  Environment $env_name already exists. Skipping..."
    else
        echo "Creating from $yml_file..."
        conda env create -f "$yml_file"
        echo "✓ Environment $env_name created successfully"
    fi
    echo ""
}

# Create all three environments
create_env "libero_replay" "envs/libero_replay_env.yml"
create_env "phyvla-p2" "envs/phyvla-p2_env.yml"
create_env "starVLA" "envs/starVLA_env.yml"

echo "=========================================="
echo "Post-installation Steps"
echo "=========================================="
echo ""

echo "1. Install LIBERO (for libero_replay environment):"
echo "   conda activate libero_replay"
echo "   cd third_party/LIBERO"
echo "   pip install -e ."
echo ""

echo "2. Download LIBERO datasets:"
echo "   python -c 'from libero.libero import get_libero_path; print(get_libero_path(\"datasets\"))'"
echo "   # Follow instructions at: https://github.com/Lifelong-Robot-Learning/LIBERO"
echo ""

echo "3. Install StarVLA (for starVLA environment):"
echo "   conda activate starVLA"
echo "   cd third_party/starVLA"
echo "   pip install -e ."
echo "   # Ensure checkpoint exists at:"
echo "   # /home/jiangnan-luxitech/workspace/starVLA/playground/Pretrained_models/Qwen2.5-VL-GR00T-LIBERO-4in1/"
echo ""

echo "=========================================="
echo "Environment Verification"
echo "=========================================="
echo ""

echo "Test libero_replay:"
echo "  conda activate libero_replay"
echo "  python -c 'import torch; import mujoco; import robosuite; print(\"✓ libero_replay OK\")'"
echo ""

echo "Test phyvla-p2:"
echo "  conda activate phyvla-p2"
echo "  python -c 'import jax; import mujoco_warp as mjw; print(\"✓ phyvla-p2 OK\")'"
echo ""

echo "Test starVLA:"
echo "  conda activate starVLA"
echo "  python -c 'import torch; import flash_attn; print(\"✓ starVLA OK\")'"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Available environments:"
echo "  - libero_replay  (Python 3.8, PyTorch 1.11, CUDA 11.3) → P0-P1"
echo "  - phyvla-p2      (Python 3.11, JAX, mujoco_warp)       → P2-P4"
echo "  - starVLA        (Python 3.10, PyTorch 2.6, Flash Attention) → P5"
echo ""
echo "Next steps: See ENVIRONMENT_SETUP.md for detailed instructions"
