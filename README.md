# PHYVLA

从 LIBERO 原始示范提取物理状态，训练共享物理 tokenizer，再将物理 token 作为残差融合到 StarVLA 的视觉动作查询中。本项目使用普通 LIBERO 的 40 个任务、全部 2,000 条 demo 训练物理模块，并在普通 LIBERO 与 LIBERO-plus 上进行闭环评测。

本文对应当前 **causal-object-set-v2** 流程。旧 projector、旧 tokenizer 和旧 `outputs/libero_*` 格式不能与 v2 混用。代码入口已核对；新机器从零安装仍需验证 CUDA、FlashAttention 和渲染依赖。

## 1. 项目结构与工作目录

```text
src/adapters/       离线/在线 MuJoCo 物理状态适配器
src/datasets/       物理特征定义、数据读取与来源校验
src/models/         PhysicsTokenizerV2、视觉查询残差融合模块
src/p5/             完整权重加载、在线策略封装
scripts/            全量数据处理、训练、服务和评测入口
thirdparty/         LIBERO、LIBERO-plus、starVLA 独立仓库
configs/、envs/     配置与环境规格
data/               原始 HDF5 和下载资源
outputs/            特征、token、权重、评测报告
logs/               运行日志和版本记录
```

以下命令默认在 Linux 项目根目录运行：

```bash
export PHYVLA_ROOT=/0604-10T-test/jiangnan/PHYVLA
cd "$PHYVLA_ROOT"
export VLA_CKPT="$PHYVLA_ROOT/thirdparty/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
export PHYSICS_CKPT="$PHYVLA_ROOT/outputs/physics_fusion_v2/policy_best.pt"
```

三个第三方仓库分别来自 Lifelong-Robot-Learning/LIBERO、sylvestf/LIBERO-plus 和 starVLA/starVLA。它们包含本项目本地修改，尚非主仓库 submodule；迁移时须连同修改一起保存，不能只重新 clone 上游。

## 2. 环境安装

### 当前使用的环境

| 环境 | 用途 | 已核对的主要版本 |
|---|---|---|
| `libero-plus` | 原始数据恢复、普通 LIBERO 和 plus 仿真/评测客户端 | Python 3.9、Torch 1.11.0+cu113、NumPy 1.24.4、MuJoCo 3.2.3、robosuite 1.4.0 |
| `starVLA` | tokenizer/融合训练、视觉缓存、模型服务 | Python 3.10、Torch 2.6.0+cu124、torchvision 0.21.0、NumPy 1.26.4 |
| `phyvla-p2` | 可选 MuJoCo Warp rollout 和梯度验证 | `envs/phyvla-p2_env.yml` |

当前服务器解释器位于 `/0604-10T-test/Eamon/miniconda3/envs/<环境>/bin/python`。客户端不承担 H100 上的模型推理；模型服务使用 `starVLA`。普通 LIBERO 和 LIBERO-plus 都提供 `libero` 包，必须核对导入来源。

### 新机器配置

需要 Linux、Conda、NVIDIA 驱动，以及 OSMesa、ImageMagick 等系统依赖。Ubuntu/Debian 示例（有系统管理权限时）：

```bash
sudo apt-get update
sudo apt-get install -y build-essential libosmesa6-dev libgl1-mesa-dev \
  libexpat1 libfontconfig1-dev libmagickwand-dev imagemagick ffmpeg unzip

conda create -n libero-plus python=3.9 pip -y
conda activate libero-plus
python -m pip install 'cmake<4'
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r envs/libero-plus-eval-requirements.txt
python -m pip install mujoco==3.2.3 h5py imageio-ffmpeg huggingface-hub
python -m pip install --no-deps -e thirdparty/LIBERO-plus
python -m pip check
```

`envs/libero-plus-eval-requirements.txt` 是评测依赖补充，不能单独代替 Torch 和系统依赖安装。不要再安装第三方旧 requirements 中相冲突的 NumPy 固定版本。

训练环境规格在 `envs/starVLA_env.yml`。其中含旧机器 `prefix` 和专用 FlashAttention wheel 版本，跨机器安装时先移除 `prefix`，根据 Torch 2.6/CUDA 12.4/Python 3.10 选择匹配的 FlashAttention 构建，再创建环境：

```bash
# 编辑后的副本保存到 /tmp/starVLA_env.yml；不要改写正在使用的环境。
conda env create -f /tmp/starVLA_env.yml
conda activate starVLA
python -m pip install --no-deps -e "$PHYVLA_ROOT/thirdparty/starVLA"
python -m pip check
python -c 'import torch, flash_attn; print(torch.__version__, torch.cuda.is_available())'
```

`setup_envs.sh` 是历史三环境安装入口，不包含当前 `libero-plus` 环境，部分提示仍为旧路径。Warp 为可选扩展：`conda env create -f envs/phyvla-p2_env.yml`；当前 v2 token 流程不依赖 Warp。

### 配置仿真路径

为两个仓库分别写入绝对路径配置，避免默认配置指向另一套 benchmark：

```bash
conda activate libero-plus
python - <<'PY'
import os, pathlib, yaml
root = pathlib.Path(os.environ['PHYVLA_ROOT'])
for name in ('LIBERO', 'LIBERO-plus'):
    package = root / 'thirdparty' / name / 'libero' / 'libero'
    paths = dict(benchmark_root=str(package), assets=str(package/'assets'),
                 bddl_files=str(package/'bddl_files'), init_states=str(package/'init_files'),
                 datasets=str(root/'data'))
    (package.parent/'config.yaml').write_text(yaml.safe_dump(paths))
PY
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=1  # 当前服务器已验证的 EGL 设备索引
```

普通 LIBERO 的 v2 导出/评测入口会显式选择 `thirdparty/LIBERO`；plus 的 shell 会选择 `thirdparty/LIBERO-plus`。手工运行脚本时同步设置 `LIBERO_HOME`、`LIBERO_CONFIG_PATH="$LIBERO_HOME/libero"` 与 `PYTHONPATH`，不能仅凭 Conda 环境名称判断 benchmark。

## 3. 原始数据、资产和基础权重

### 普通 LIBERO 示范

目标目录为 `data/<suite>/<task_name>_demo.hdf5`，四套件为 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，各 10 个任务、每任务 50 条 demo。

```bash
conda activate libero-plus
export LIBERO_HOME="$PHYVLA_ROOT/thirdparty/LIBERO"
export LIBERO_CONFIG_PATH="$LIBERO_HOME/libero"
export PYTHONPATH="$LIBERO_HOME:$PHYVLA_ROOT/src:$PHYVLA_ROOT/thirdparty/starVLA"
python thirdparty/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --download-dir "$PHYVLA_ROOT/data" --datasets all --use-huggingface
```

当前下载入口没有独立的 `--datasets libero_10` 选项；`all` 可能包含额外数据。下载后必须核对四个目标 suite、任务身份和 demo 数。根目录历史 `download_libero_10.py` 名称与实际筛选内容不符，勿作为当前入口。

### LIBERO-plus 资产

当前服务器已准备好资产，无需重复下载：

- 下载包：`data/downloads/liberoplus-assets.zip`。
- 使用目录：`thirdparty/LIBERO-plus/libero/libero/assets/`。
- 任务分类：`thirdparty/LIBERO-plus/libero/libero/benchmark/task_classification.json`。

新机器从 `Sylvest/LIBERO-plus` 下载 `assets.zip`，解压到对应 `libero/libero/`；当前已验证资产版本为 `dd2bd61b7d9a6fef1abc52d606e983b41886a149`。示例：

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Sylvest/LIBERO-plus', repo_type='dataset',
    filename='assets.zip', revision='dd2bd61b7d9a6fef1abc52d606e983b41886a149',
    local_dir='data/downloads/liberoplus')
PY
unzip data/downloads/liberoplus/assets.zip -d thirdparty/LIBERO-plus/libero/libero
```

同时保留 plus 仓库内 BDDL 和初始状态文件。仅做泛化测评不需要下载 plus 训练数据，也不应使用评测结果选择训练权重。

### StarVLA 基础权重

准备 `StarVLA/Qwen3-VL-OFT-LIBERO-4in1` 的完整模型目录，保持 `VLA_CKPT` 所示层级。除 `.pt` 外还需要对应 `config.yaml`、`dataset_statistics.json`、处理器及配置引用的 VLM 资源。

```bash
conda activate starVLA
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='StarVLA/Qwen3-VL-OFT-LIBERO-4in1',
    local_dir='thirdparty/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1')
PY
test -f "$VLA_CKPT"
```

当前 v2 权重绑定的基础 `.pt` SHA-256 为 `80d7a2ab0b033ce49d84070eb1124a2f9718e5b1f0ef8e2db6e4c44478b05df9`。重新下载的内容若变化，不能直接搭配已有物理权重；服务还会检查配置与归一化统计来源。

## 4. 数据处理：HDF5 → 物理特征

先检查现有产物；重新处理须指定新输出目录，避免覆盖当前训练和评测所依赖的文件。

```bash
cd "$PHYVLA_ROOT"
WORKERS=8 bash scripts/process_all_40tasks.sh
conda run -n libero-plus python scripts/validate_physics_dataset_v2.py \
  --root outputs/physics_reexport_v2
```

不设置 `DEMO_INDICES` 即处理全部示范。每个 `suite/task_<id>/demo_<id>/` 保存物理 NPZ、原始 XML、来源元数据和独立成功标签。按官方任务映射读取 HDF5，并逐帧恢复记录状态，避免动作重放累积漂移。

原始图像是动作后的观测，而记录状态是动作前的状态。使用已核对的 `observation_index_for_state` 对齐图像：通常 `state[t]` 对应 `obs[t-1]`，每条 demo 首帧无可用图像而被排除。专家动作仍为 `actions[t]`；禁止将动作、成功标签或未来状态作为物理输入。

全量导出：40 任务、2,000 demo、338,575 状态帧，其中 336,575 帧可用于视觉对齐。按完整 demo 分割：demo ID 尾数为 9 的为验证集，共 1,800/200 条训练/验证示范。

```bash
conda activate starVLA
python scripts/prepare_physics_v2.py --data outputs/physics_reexport_v2 \
  --output outputs/physics_features_v2
```

机器人特征 31 维、每对象 39 维，保留完整实例和活动关节，使用对象 mask；归一化只在训练集拟合。接触力是在记录状态上以零控制/零 warm-start 重建的近似量，不等同于真实力传感器读数。验收需同时满足原始导出 `manifest.json: status=completed`、`validation.json: passed=true`。

## 5. 训练与最终 token 导出

当前方式冻结原始 VLM 和 action head；物理 tokenizer 先预训练，再与视觉查询残差融合模块联合训练。监督使用真实专家动作，不使用 baseline 预测作为标签。

```bash
conda activate starVLA
# 共享 tokenizer：默认 12 epochs。
CUDA_VISIBLE_DEVICES=1 python scripts/train_physics_tokenizer_v2.py \
  --data outputs/physics_features_v2 --output outputs/physics_tokenizer_v2

# 缓存冻结模型的视觉动作查询；8 步动作 chunk，末尾使用 mask。
CUDA_VISIBLE_DEVICES=1 python scripts/cache_visual_queries_v2.py \
  --data outputs/physics_features_v2 --output outputs/visual_queries_v2 \
  --checkpoint "$VLA_CKPT" --batch-size 32

# 默认 20 epochs；使用验证集选择完整权重。
CUDA_VISIBLE_DEVICES=1 python scripts/train_physics_fusion_v2.py \
  --queries outputs/visual_queries_v2 \
  --tokenizer outputs/physics_tokenizer_v2/tokenizer_best.pt \
  --output outputs/physics_fusion_v2

# 导出最终联合训练后的 token，而非早期预训练 tokenizer 的 token。
CUDA_VISIBLE_DEVICES=1 python scripts/export_physics_tokens_v2.py \
  --data outputs/physics_features_v2 --checkpoint "$PHYSICS_CKPT" \
  --output outputs/physics_tokens_v2

conda run -n libero-plus python scripts/verify_physics_v2_live.py
```

`policy_best.pt` 保存 tokenizer、融合模块、完整 action head、归一化、特征版本和基础权重来源；服务另外加载原始 VLA 基础模型。`training_last.pt` 保存优化器状态，但当前 CLI 不提供自动续训参数。更换数据版本须重新指定整条流水线的输出/输入目录，不能混用旧缓存。

已验证线上/离线 40 任务、120 个留出状态的特征一致，并检查提取不修改模拟器。宽度 128 的最终 token 已导出全部 2,000 条 demo。详细损失、消融及检查记录见 `outputs/physics_fusion_v2/summary.json` 和 [训练说明](docs/ENCODING_TRAINING.md)。

## 6. 模型服务

两个独立终端启动服务；已有相同服务监听时直接复用，不要重复启动。

```bash
# PHYVLA：基础 VLA + 完整物理 checkpoint，GPU 1，端口 6695。
conda activate starVLA
GPU=1 PORT=6695 bash thirdparty/starVLA/examples/simBenchmarks/LIBERO/eval_files/run_phyvla_server.sh

# 匹配的 baseline：同一基础权重，GPU 1，端口 6696。
CUDA_VISIBLE_DEVICES=1 python scripts/serve_baseline_v2.py \
  --checkpoint "$VLA_CKPT" --port 6696
```

服务监听本机 `127.0.0.1`。shell 默认使用本服务器绝对解释器路径；迁移时设置 `STARVLA_PYTHON`。Baseline 和 PHYVLA 均采用 VLM BF16 autocast、FP32 action head，不能混入旧 `--use_bf16` 全模型转换服务作为严格对照。

## 7. 普通 LIBERO 测评与消融

使用仿真环境解释器；输出目录必须是新目录。下例每个任务一个初始状态、seed 7，共 40 回合/模型：

```bash
conda activate libero-plus
python scripts/evaluate_libero_v2.py --port 6696 --trials 1 --seed 7 \
  --output outputs/eval_libero_baseline_run2
python scripts/evaluate_libero_v2.py --port 6695 --phyvla --trials 1 --seed 7 \
  --output outputs/eval_libero_phyvla_run2
python scripts/compare_physics_v2.py \
  --baseline outputs/eval_libero_baseline_run2 \
  --phyvla outputs/eval_libero_phyvla_run2 \
  --output outputs/eval_libero_comparison_run2.json
```

覆盖 spatial/object/goal/libero_10，步数上限分别 220/280/300/520，稳定等待 10 步，动作 chunk 为 8。增加覆盖可将两组 `--trials` 同步设为 50（需对应初始状态足够），不是只提高某一组试验次数。

普通 LIBERO 的关闭物理残差消融可启动独立服务：

```bash
conda activate starVLA
CUDA_VISIBLE_DEVICES=1 python scripts/serve_phyvla_v2.py \
  --checkpoint "$VLA_CKPT" --physics-checkpoint "$PHYSICS_CKPT" \
  --port 6697 --mode no_physics
# 另一终端，使用仿真环境
conda run -n libero-plus python scripts/evaluate_libero_v2.py \
  --port 6697 --phyvla --trials 1 --seed 7 --output outputs/eval_libero_no_physics_run2
```

离线 `full/no_physics/shuffled` 验证指标由融合训练保存；当前 plus 客户端要求真实使用物理 token，不应直接拿它运行 no_physics 模式。

## 8. LIBERO-plus 全量/指定任务测评

plus 启动脚本默认使用 EGL GPU 离屏渲染；当前 H100 已验证。没有 EGL 支持时可显式设置 `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`。`MUJOCO_EGL_DEVICE_ID` 是 EGL 设备枚举索引，新机器需要核对。

断点续跑：保持原来的输出目录、配置和 `RUN_ID`，向原启动命令追加 `--args.resume`；脚本校验权重与评测参数，跳过已保存回合。不要同时启动两个进程写入同一结果目录。每个回合记录 `render_backend`。2026-09-15 17:39 切换前的 OSMesa 回合已保留并标注；严格对比必须同时匹配任务、初始状态和渲染后端，跨后端结果不能直接当作同协议配对。


四套件共 **10,030 个变体**：goal 2,591、spatial 2,402、object 2,518、libero_10 2,519。每个变体默认一次试验；部分变体只有一个初始状态。不要将普通 LIBERO 的 40 任务计数或 50 次试验配置直接套用。

```bash
cd "$PHYVLA_ROOT"
# 验证资源、完整物理 checkpoint 和服务。
CHECK_ONLY=1 bash thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files/eval_phyvla.sh

# PHYVLA：四套件串行。
PORT=6695 NUM_TRIALS=1 SEED=7 OUTPUT_DIR="$PHYVLA_ROOT/outputs/plus_phyvla_run2" \
  bash thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files/eval_phyvla.sh

# Baseline：同一任务顺序、图像处理、seed 与试验次数。
base_port=6696 num_trials_per_task=1 seed=7 output_dir="$PHYVLA_ROOT/outputs/plus_baseline_run2" \
  bash thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files/eval_baseline_v2.sh

# 小规模调试：ID 为该 plus suite 的零起始变体 ID。
SUITES=libero_spatial TASK_IDS='0 1' PORT=6695 NUM_TRIALS=1 \
  bash thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files/eval_phyvla.sh
```

可用 `SUITES='libero_goal libero_object'` 选择套件，baseline 同样支持 `SUITES` 和 `TASK_IDS`。迁移时通过 `LIBERO_Python=/实际环境/bin/python` 覆盖仿真解释器。

两组均使用旋转后的 agent/wrist 图像、PIL bilinear 224 像素缩放及相同动作后处理。PHYVLA 额外读取模拟器物理状态和 BDDL 目标角色，解释传感器噪声、语言扰动结果时须注明此观测差异。

### 当前全量运行与结果读取

当前任务由后台独立进程启动，断开 SSH 不会终止；下面目录对应正在运行的实验，不要再次向它们启动重复任务：

| 模型 | 端口 | 输出根目录 |
|---|---|---|
| PHYVLA v2 | 6695 | `outputs/physics_v2_evaluation/plus_full/` |
| 匹配 baseline v2 | 6696 | `outputs/physics_v2_evaluation/plus_baseline_matched/` |

```bash
tail -f outputs/physics_v2_evaluation/plus_baseline_matched/launcher.log
```

`launch.json` 保存启动 PID 与配置；`logs/<时间>/<suite>_result.json` 逐回合更新 `total`、`successes`、任务 ID、扰动类别和状态；视频在 `<时间>/<suite>/`。`status=running` 的数值只反映已完成回合。最终 `<suite>.json` 汇总扰动类别，勿把计划数量当作中途完成数量。

横向对比应取双方相同 suite/task ID/initial_state_index 的交集，并核对基础权重、图像缩放、精度、seed 和步数预算；分别汇总七类扰动。旧 9883 baseline 日志的精度和缩放不同，只能作为历史参考，不能与当前 v2 对照混算。

## 9. 已验证结果与限制

普通 LIBERO 的首轮 40 回合：baseline 39/40（97.5%），PHYVLA 36/40（90%），尚未显示闭环成功率提升。虽然留出集动作 L1 从约 0.05921 降至 0.05260，但不能以此替代成功率。该验证划分只约束新物理模块，不保证 baseline 预训练未使用这些 demo。

LIBERO-plus 准备阶段已通过四套件 × 七类扰动的 28 次接口检查，所有 10,030 个任务引用的初始状态文件存在；双方同一冒烟回合均成功。这不等于全量环境逐个实例化或完整成功率测评。动态进度以运行报告为准，不将阶段性数值写成最终结论。

## 更多文档

- [环境与路径](docs/ENVIRONMENT.md)、[原始数据](docs/DATA_PREPARATION.md)
- [物理状态提取](docs/PHYSICS_PROCESSING.md)、[物理编码训练](docs/ENCODING_TRAINING.md)
- [对比评测](docs/BASELINE_COMPARISON.md)、[脚本清单](docs/SCRIPTS.md)
- [贡献约定](AGENTS.md)

历史文档中的单 demo 和旧 projector 流程只用于复查。保留原始数据、已有权重与实验记录；新实验使用独立输出目录。
