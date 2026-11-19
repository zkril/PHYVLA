#!/bin/bash
# LIBERO-plus环境安装脚本

set -euo pipefail

PHYVLA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
THIRDPARTY_DIR="${PHYVLA_DIR}/thirdparty"

echo "========================================"
echo "LIBERO-plus Environment Setup"
echo "========================================"
echo ""

# 1. Clone LIBERO-plus repository
echo "[1/4] Clone LIBERO-plus repository..."
cd "${THIRDPARTY_DIR}"

if [[ ! -d "LIBERO-plus" ]]; then
    git clone https://github.com/sylvestf/LIBERO-plus.git
    echo "  ✓ LIBERO-plus cloned"
else
    echo "  ✓ LIBERO-plus already exists"
fi

# 2. Create conda environment
echo ""
echo "[2/4] Create conda environment 'libero-plus'..."
echo "  请手动执行以下命令："
echo ""
echo "  conda create -n libero-plus python=3.9 -y"
echo "  conda activate libero-plus"
echo ""

# 3. Install LIBERO-plus dependencies
echo "[3/4] Install LIBERO-plus..."
echo "  在libero-plus环境中执行："
echo ""
echo "  cd ${THIRDPARTY_DIR}/LIBERO-plus"
echo "  pip install -e ."
echo ""

# 4. Install additional dependencies
echo "[4/4] Install additional dependencies..."
echo "  在libero-plus环境中执行："
echo ""
echo "  pip install tyro matplotlib mediapy websockets msgpack"
echo "  pip install numpy==1.24.4 mujoco==3.2.3"
echo ""

echo "========================================"
echo "Complete Installation Steps:"
echo "========================================"
echo ""
echo "# 创建环境"
echo "conda create -n libero-plus python=3.9 -y"
echo ""
echo "# 激活环境"
echo "conda activate libero-plus"
echo ""
echo "# 安装LIBERO-plus"
echo "cd ${THIRDPARTY_DIR}/LIBERO-plus"
echo "pip install -e ."
echo ""
echo "# 安装额外依赖"
echo "pip install tyro matplotlib mediapy websockets msgpack"
echo "pip install numpy==1.24.4 mujoco==3.2.3"
echo ""
echo "# 设置环境变量（每次运行评估前）"
echo "export LIBERO_HOME=${THIRDPARTY_DIR}/LIBERO-plus"
echo "export LIBERO_Python=python"
echo "export MUJOCO_GL=osmesa"
echo "export PYOPENGL_PLATFORM=osmesa"
echo "export LIBERO_CONFIG_PATH=\${LIBERO_HOME}/libero"
echo ""
echo "安装完成后，可以运行baseline评估："
echo "bash ${PHYVLA_DIR}/scripts/run_baseline_client.sh"
