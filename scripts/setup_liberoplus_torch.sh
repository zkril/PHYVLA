#!/bin/bash
# 为libero-plus环境补全torch相关依赖

set -euo pipefail

echo "========================================"
echo "补全 libero-plus 环境依赖"
echo "========================================"
echo ""

echo "请在新终端执行以下命令："
echo ""
echo "# 1. 激活环境"
echo "conda activate libero-plus"
echo ""
echo "# 2. 安装PyTorch (CUDA 11.3, 与libero_replay一致)"
echo "pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113"
echo ""
echo "# 3. 安装其他必要依赖"
echo "pip install numpy==1.24.4 mujoco==3.2.3"
echo "pip install tyro matplotlib mediapy websockets msgpack"
echo "pip install pillow opencv-python imageio"
echo "pip install tqdm termcolor"
echo ""
echo "# 4. 验证安装"
echo "python -c \"import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())\""
echo "python -c \"import libero; print('LIBERO-plus OK')\""
echo ""
echo "========================================"
echo "安装完成后可以运行baseline评估"
echo "========================================"
