# 原始数据准备

[返回首页](../README.md) · 前置：[环境配置](ENVIRONMENT.md)

## 数据与目录

准备四个 suite：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，每个包含 10 个任务 HDF5。一个 HDF5 可包含多个 demonstration，不等于一个 episode。2026-09-14 检查发现本机 `data/` 下共 40 个 HDF5，但文件数不能证明内容完整。

```text
data/<suite>/<task_name>_demo.hdf5
thirdparty/LIBERO/libero/libero/{assets,bddl_files,init_files}/
thirdparty/LIBERO-plus/libero/libero/{assets,bddl_files,init_files}/
```

本地 LIBERO 下载入口为 `thirdparty/LIBERO/benchmark_scripts/download_libero_datasets.py`：

```bash
conda activate libero_replay
cd "$PHYVLA_ROOT"
export LIBERO_CONFIG_PATH="$PHYVLA_ROOT/thirdparty/LIBERO/libero"
python thirdparty/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --download-dir "$PHYVLA_ROOT/data" --datasets all --use-huggingface
```

`all` 可能下载额外数据；下载后确认所需四个目录及 HDF5，而不是仅凭进程退出判断完成。此入口的 choices 没有单独的 `libero_10`。根目录 `download_libero_10.py` 实际筛选的是 `libero_object/*`，不要依据其文件名判断行为。

## 配置真实路径

`LIBERO_CONFIG_PATH` 指向**包含 config.yaml 的目录**。在对应环境配置文件中确认：

- `datasets` 指向项目的绝对路径 `data/`。
- `benchmark_root` 指向对应仓库的 `libero/libero/`。
- `assets`、`bddl_files`、`init_states` 分别指向 assets、bddl_files、init_files。

LIBERO-plus 的 datasets 已在环境修复中指向项目 data/；普通 LIBERO 的默认路径仍需检查并修改后才能依赖 `get_libero_path("datasets")` 读取。验证：

```bash
python -c 'from libero.libero import get_libero_path; from pathlib import Path; p=Path(get_libero_path("datasets")); print(p); assert p.is_dir()'
```

## 任务与 demonstration 对齐

用 LIBERO benchmark 的 task 列表确定 suite-local ID，并核对任务语言、BDDL 和 HDF5 文件。不要用文件名字母排序替代 benchmark 顺序。检查 `data/demo_0` 等组中的 states、actions、obs 和模型元数据；各序列应对应同一 demonstration。

处理脚本接收 suite-local ID 0–9。P5 的 `--suite-mode multi` 映射为：0–9 spatial、10–19 object、20–29 goal、30–39 libero_10。旧文档曾交换 object 与 goal；当前以 `train_p5_projector.py` 映射为准。

## LIBERO-plus 评估资源

按第三方 [LIBERO-plus README](../thirdparty/LIBERO-plus/README.md) 准备其 assets.zip，解压后应得到 `thirdparty/LIBERO-plus/libero/libero/assets/`。该资源目录已在 2026-09-14 环境修复中补齐并校验，详情见 [环境配置](ENVIRONMENT.md)。普通 LIBERO assets 不能被默认视为完整的 plus 资源。

仅做 plus 泛化评估时，需要对应 BDDL、初始状态和资源；不要求先用 plus 训练数据微调。若使用 plus 训练数据，需单独记录数据划分，避免将训练任务混入泛化测试。
