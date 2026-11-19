# 环境与路径配置

[返回首页](../README.md)

## 阶段环境

所有命令在 Linux 服务器执行。先设置项目路径：

```bash
export PHYVLA_ROOT=/0604-10T-test/jiangnan/PHYVLA
cd "$PHYVLA_ROOT"
conda env list
```

| 环境 | 用途 | 配置 |
|---|---|---|
| libero_replay | HDF5 回放、物理状态提取 | envs/libero_replay_env.yml |
| phyvla-p2 | MuJoCo Warp rollout、梯度验证 | envs/phyvla-p2_env.yml |
| starVLA | VLA 推理、动作缓存、P5 训练 | envs/starVLA_env.yml |
| libero-plus | LIBERO-plus 评估 | 按第三方仓库依赖单独配置，无根目录环境 YAML |

新机器可运行 `bash setup_envs.sh` 创建前三个环境，或逐个执行 `conda env create -f envs/<文件名>`。不要将所有阶段依赖混装；GPU / CUDA 兼容性需实际验证。

## 第三方依赖

实际目录为 `thirdparty/LIBERO`、`thirdparty/LIBERO-plus`、`thirdparty/starVLA`。旧安装脚本的部分提示仍写作 `third_party/`，使用实际目录。三个独立 Git 仓库尚未作为主仓库 submodule 管理；迁移时需分别保存版本和本地修改。

LIBERO 与 LIBERO-plus 都暴露 `libero` Python 包，应在不同环境安装：

```bash
conda activate libero_replay
python -m pip install -e "$PHYVLA_ROOT/thirdparty/LIBERO"
export LIBERO_CONFIG_PATH="$PHYVLA_ROOT/thirdparty/LIBERO/libero"
python -c "import libero; print(libero.__file__)"
```

LIBERO-plus 环境采用同样流程，替换为对应目录。依赖安装详情以各第三方 README、requirements 为准。客户端还需要 StarVLA WebSocket 通信依赖。评估前检查 `import libero` 实际来源。

## 权重与渲染

代码默认 VLA 权重：

```bash
export VLA_CKPT="$PHYVLA_ROOT/thirdparty/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
test -f "$VLA_CKPT"
```

需要同时准备对应模型配置、处理器、归一化统计及底座模型资源，不能只复制单个 checkpoint。H100 等较新 GPU 的训练优先使用支持其架构的 `starVLA` 环境。

无显示器渲染可设置 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`；使用 OSMesa 时两者均改为 `osmesa` 并准备系统依赖。先验证单次环境初始化，不要立即启动全量评估。

## LIBERO-plus 修复与验证（2026-09-14）

已补齐评估 requirements 中的依赖，固定 gym 0.25.2、OpenCV 4.6.0.66 与 NumPy 1.24.4；pip check 无冲突。egl_probe 构建需要先安装 CMake：

```bash
conda activate libero-plus
cd /0604-10T-test/jiangnan/PHYVLA
python -m pip install "cmake<4"
python -m pip install -r envs/libero-plus-eval-requirements.txt
```

这是现有 Python 3.9 环境的评估依赖补充；模型服务仍使用 starVLA 环境。评估 Python 文件已加延迟类型注解以兼容 Python 3.9。

官方资产包已从 Sylvest/LIBERO-plus 的 dd2bd61b7d9a6fef1abc52d606e983b41886a149 版本下载、SHA-256 校验并解压：
- 原始包：data/downloads/liberoplus-assets.zip（6,395,849,578 字节）。
- 生效目录：thirdparty/LIBERO-plus/libero/libero/assets/（448,799 个文件）。
- config.yaml 的 datasets 已指向项目 data/。
- 安装记录、修复前备份与版本快照：logs/liberoplus_repair/。

启动脚本现已导出 LIBERO_HOME、LIBERO_CONFIG_PATH 和渲染变量，校验资源/服务，按 suite 串行运行并传播失败退出码。默认模型服务端口仍为 9883。

```bash
cd /0604-10T-test/jiangnan/PHYVLA/thirdparty/starVLA
CHECK_ONLY=1 bash examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero.sh
# 实际评估；默认四个 suite，每个变体任务一次试验
bash examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero.sh
# 只评估一个 suite，可覆盖端口、次数和 GPU
SUITES=libero_spatial num_trials_per_task=1 base_port=9883 gpu_id=0 \
  bash examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero.sh
```

默认四个 suite 合计 10,030 个变体任务，不是普通 LIBERO 的 40 个任务。本次没有跑完整成功率实验。

复查环境与真实策略调用：

```bash
cd /0604-10T-test/jiangnan/PHYVLA
python scripts/verify_liberoplus_environment.py --with-policy --port 9883
```

已通过 OSMesa 渲染、ImageMagick、WebSocket 握手，并完成四个 suite 各第一个任务的渲染、10 次模拟器步进、一次真实策略动作执行。报告为 logs/liberoplus_repair/environment-smoke.json；这仅是环境联调验收，不是成功率结果。服务端日志和启动进程 PID 分别在同目录的 policy-server.log、policy-server.pid。

### 正式入口客户端导入修正

LIBERO-plus 的 eval_libero.py 已改为导入同目录的 model2libero_interface.ModelClient，修复误用普通 LIBERO 客户端导致 policy_ckpt_path 参数不匹配的问题。环境检查脚本也改为从正式评估模块取得同一个 ModelClient。通过正式 CLI 入口运行 libero_spatial 首个任务的一次完整 episode（仅测试进程中限制任务数，未修改默认任务集合），已成功完成并写出视频与结果。日志：logs/liberoplus_repair/actual-entry-one-episode.log。该输出属于单任务联调，不是全 suite 成功率。
