# LIBERO / LIBERO-plus 与 Baseline 对比实验

[返回首页](../README.md) · 前置：[数据准备](DATA_PREPARATION.md)、[编码与训练](ENCODING_TRAINING.md)

已清理无效的第一方包装器；表格中的旧 scripts/ 入口已移入 archive/，具体位置和原因见 [脚本清单](SCRIPTS.md)。当前保留的 baseline 包装器为 scripts/start_baseline_server.sh 和 scripts/run_baseline_client.sh。

## 当前 v2：先评测普通 LIBERO

先完成 `ENCODING_TRAINING.md` 中的全量训练。v2 使用同一个 baseline VLM/action head，
增加当前物理状态 token 到视觉 action queries 的残差，不再替换视觉查询。
旧 projector 不能加载到这个入口。

```bash
cd /0604-10T-test/jiangnan/PHYVLA
GPU=1 PORT=6695 bash thirdparty/starVLA/examples/simBenchmarks/LIBERO/eval_files/run_phyvla_v2_server.sh
# 另一个终端：四个 suite，初步评测每任务 1 个初始状态。
PORT=6695 NUM_TRIALS=1 SEED=7 \
  bash thirdparty/starVLA/examples/simBenchmarks/LIBERO/eval_files/eval_phyvla_v2.sh
```

baseline 也必须使用同一 runner、相同 trial 数和 seed（使用精度匹配的新 baseline 服务，端口 6696）：

```bash
conda run --no-capture-output -n libero-plus python scripts/evaluate_libero_v2.py \
  --port 6696 --output outputs/physics_v2_evaluation/baseline_matched --trials 1 --seed 7
conda run -n starVLA python scripts/compare_physics_v2.py \
  --baseline outputs/physics_v2_evaluation/baseline_matched \
  --phyvla outputs/physics_v2_evaluation/full \
  --output outputs/physics_v2_evaluation/comparison.json
```

这里的 `libero-plus` 仅为依赖环境名称；runner 强制导入原始
`thirdparty/LIBERO`，不会将 LIBERO-plus 成绩算作普通 LIBERO。
输出目录已有结果时须指定新目录，避免覆盖。
四个 suite 的步数上限为 220/280/300/520，稳定等待 10 步；
两路相机按 agentview、wrist 排列，旋转 180 度并缩放至 224，
action chunk 为 8，gripper 映射复用同一个评测器。

现有 9883 服务启用了整体 BF16 转换，不能直接用于 v2 正式对照。
新 baseline 使用 `scripts/serve_baseline_v2.py --checkpoint "$VLA_CKPT" --port 6696`，
与 v2 相同地保留权重加载精度、VLM BF16 autocast 和 FP32 action head。

训练验证集报告 full、关闭物理、打乱物理状态的动作误差；
关闭物理残差应与 baseline 一致。首轮每任务一次只能检查链路并提供探索性比较，
后续需预先固定更多初始状态/seed，报告分 suite 成功率、配对胜负及区间。
PHYVLA 使用额外的仿真物理状态和 BDDL 角色，应在实验说明中写明这一观测差异。

LIBERO-plus 的 v2 迁移和扰动评测在普通 LIBERO 验证后进行；
下面历史 plus 入口不能直接混用 v2 checkpoint。

## 实验定义

Baseline 使用固定 StarVLA checkpoint；PHYVLA 使用相同底座，加上训练并正确加载的物理编码与 projector。保持任务列表、初始状态、seed、试验次数、图像预处理、动作归一化、horizon、步数上限一致。分别报告 LIBERO 和 LIBERO-plus，不混合计分；plus 还应按其扰动类别汇总。

每次运行记录三个第三方仓库提交及本地 diff、所有权重/统计文件、配置、任务与 demo 划分。先做少量 episode 的联调，再执行每任务 50 次等预先确定的预算。

## Baseline 运行入口

完成环境、资源和 checkpoint 检查后，在服务器终端运行：

```bash
conda activate starVLA
cd "$PHYVLA_ROOT/thirdparty/starVLA"
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path "$VLA_CKPT" --port 6694 --use_bf16
```

等待服务实际就绪，再在另一终端选择 LIBERO 客户端：

```bash
conda activate libero_replay
export LIBERO_CONFIG_PATH="$PHYVLA_ROOT/thirdparty/LIBERO/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd "$PHYVLA_ROOT/thirdparty/starVLA"
python examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "$VLA_CKPT" --args.host 127.0.0.1 --args.port 6694 \
  --args.task-suite-name libero_spatial --args.num-trials-per-task 50 \
  --args.seed 7 --args.video-out-path "$PHYVLA_ROOT/outputs/comparison/libero/baseline"
```

这些是按源码参数整理的入口，未在此次整理中执行验证。环境依赖和资源未通过检查时，不应启动全量任务。

LIBERO-plus 使用独立环境和配置，并替换客户端：

```bash
conda activate libero-plus
export LIBERO_CONFIG_PATH="$PHYVLA_ROOT/thirdparty/LIBERO-plus/libero"
cd "$PHYVLA_ROOT/thirdparty/starVLA"
python examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero.py \
  --args.pretrained-path "$VLA_CKPT" --args.host 127.0.0.1 --args.port 6694 \
  --args.task-suite-name libero_spatial --args.num-trials-per-task 50 \
  --args.seed 7 \
  --args.video-out-path "$PHYVLA_ROOT/outputs/comparison/liberoplus/baseline/videos" \
  --args.log-path "$PHYVLA_ROOT/outputs/comparison/liberoplus/baseline/logs"
```

在新终端重新设置环境指南中的 PHYVLA_ROOT、VLA_CKPT 和渲染变量。依次替换 suite 为 libero_object、libero_goal、libero_10。确认客户端实际导入对应的 libero 包。

## PHYVLA 在线运行

服务默认 GPU 1、127.0.0.1:9884。已完成的配套训练目录存在时，
`start_phyvla_server.sh` 自动加载
`outputs/online_physics_checkpoints_v1/tokenizer_best.pt` 和 `projector_best.pt`，
并使用 baseline 的 VLA/action head。训练及特征约定见 [编码训练](ENCODING_TRAINING.md)。

```bash
cd "$PHYVLA_ROOT"
GPU_ID=1 PORT=9884 bash scripts/start_phyvla_server.sh
# 服务就绪后，在另一终端做一条真实在线闭环
conda activate libero-plus
SUITE=libero_spatial TASK_IDS=0 NUM_TRIALS=1 bash thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files/eval_phyvla.sh
```

默认依次运行 libero_goal、libero_spatial、libero_object、libero_10 四个 suite。
可用 SUITES（空格分隔）选择多个 suite，或用 SUITE 选择一个；SUITES 优先。
可设置 TASK_IDS、NUM_TRIALS、SEED、PORT、OUTPUT_DIR；TASK_IDS 对每个所选 suite 生效。
输出结构与 eval_libero.sh 一致：output_dir/logs/<run_id>/<suite>.log，
视频为 output_dir/<run_id>/<suite>/；默认 output_dir 为 thirdparty/starVLA/results/libero_plus_phyvla_eval。
脚本使用当前 Python，运行前 conda activate libero-plus，或设置 LIBERO_Python 为该环境的 Python 绝对路径。
配置名与 baseline 对齐：host、base_port（默认 9884）、gpu_id（默认 1）、your_ckpt、output_dir、num_trials_per_task、seed。
物理分支额外配置 projector_ckpt 和 tokenizer_ckpt。先在同目录运行 run_phyvla_server.sh 加载两组模型，再运行 eval_phyvla.sh；client 不重复加载 GPU 模型。
CHECK_ONLY=1 验证资源、服务、底座路径和物理权重哈希/配对，不运行 episode；DRY_RUN=1 仅打印命令。
旧大写参数仍兼容，设置小写参数时优先使用小写。
TASK_IDS 为零起始任务编号；不设置时运行 LIBERO-plus 的完整 suite，
不是仅 40 个原任务。四个 suite 名为 libero_spatial、libero_object、libero_goal、libero_10。
所有任务仍使用原始初始状态、步数上限、图像预处理和 gripper 规则。

client 每步从实际 MuJoCo 状态采集机器人、对象、接触及事件历史；
服务器加载保存的 tokenizer 生成 [1,23,256]，然后执行 projector 和 OFT head。
动作反归一化复用 baseline 的 PolicyNormProcessor。episode 切换时重置物理历史和动作缓存。
握手包含 `online_physics_ready=true`、`paired_projector_verified=true` 以及权重哈希。
配置错配或缺少权重时明确报错，不使用随机编码器，也不静默变为 baseline。

指定其他成对训练目录：

```bash
PAIRED_CHECKPOINT_DIR=/absolute/path/to/completed_run bash scripts/start_phyvla_server.sh
```

PROJECTOR_CKPT、TOKENIZER_CKPT 可显式覆盖；服务核对配对 JSON 中的两个 SHA-256
及 baseline checkpoint 来源。不能随意把新 tokenizer 与旧 projector 拼在一起。
旧 `outputs/p5_checkpoints/projector_40tasks_best.pt` 保留，可作无在线编码器的历史诊断，
不是默认新在线组合。

### 验证与实验限制

```bash
conda run -n libero-plus python scripts/verify_liberoplus_environment.py \
  --with-policy --policy-mode phyvla --port 9884
conda run -n libero-plus python scripts/verify_phyvla_client.py --port 9884
```

前者检查四套件首个任务的实时编码、推理和环境执行；后者检查 40 任务的成对导出 token、
动作敏感性、缓存重置和错误输入，不是闭环成功率评测。
正式评测写入实际任务列表、模型元数据、视频和分类结果；分类分母按所选任务与 trial 数计算。

当前策略沿用旧 P5 的物理 action queries 前向，没有将图像 hidden states 融合进 queries。
新训练默认每任务 demo_0 训练、demo_1 验证，只能视为完成在线接入的小规模模型。
不能根据接口跑通、离线 loss 降低或单次 episode 就宣称优于 baseline。

## 结果统计与消融

保存每个任务/扰动的 success_count、total_count、seed、失败原因和原始日志。成功率为成功次数 / 有效试验次数；同时报告总体加权成功率、任务平均值和样本数。提升用百分点表示：100 × (PHYVLA 成功率 − baseline 成功率)。失败或缺失的任务不可默认为 0 后直接比较，也不可静默排除。

建议设置 baseline、真实物理 token、置零/打乱 token 三组；若不冻结 action head，需增加相同训练预算的 baseline 对照。不要把相同训练集上的 loss 当作泛化指标。

`scripts/generate_comparison_report.py` 面向逐任务 JSON；plus 日志按类别聚合，先确认实际 schema 再用对应汇总工具。旧文档的示例成功率已移除，它们不作为真实实验结果。

## 本次接入验证记录

2026-09-14：新成对模型通过四套件实时状态/编码/动作执行检查、40 任务 token 敏感性检查，以及 80 条轨迹共 240 帧的独立权重重载对齐检查。GPU 1 的 9884 服务已加载成对权重。正式 LIBERO-plus spatial task 0 跑完一条 episode，成功数为 0/1；链路运行成功不代表策略完成任务。原始记录与视频位于 `logs/online_physics_rebuild/`，汇总为 `completion.json`。

## 旧 projector 的非配套实验

若明确要测试旧 `projector_40tasks_best.pt` 与新的 tokenizer，可在 `scripts/serve_phyvla.py` 显式传入 `--allow-unpaired-tokenizer`，指定两份权重并使用独立端口。该选项只允许缺少 pairing manifest 的旧权重进入实验模式；已有 manifest 的哈希校验仍执行。默认配对检查不变。

服务及评测元数据标记 `experimental_unpaired_tokenizer=true`、`paired_projector_verified=false`。由于旧 tokenizer 和训练后的 head 未保存，这不是旧训练模型的完整恢复，也不能据此推断其原始成功率。此次同条件 task 0 实验记录在 `logs/old_projector_eval/`。

旧 projector 的同条件实验已完成：LIBERO-plus spatial task 0、seed 7、1 个 episode，成功 0/1；新配套 projector 此前同条件也是 0/1。这只是单次任务尝试，不是完整 40 任务评测。旧权重实验服务使用 GPU 1 的 9885 端口，新配套服务保留在 9884。

## 普通 LIBERO：旧代码恢复及启动尝试

按用户要求已恢复原始 PHYVLA server/eval 启动脚本及 client，恢复文件与归档哈希一致。
修正版保存在 `thirdparty/starVLA/examples/simBenchmarks/LIBERO/eval_files/revised_backup/20260914_221406/`。
GPU 1 的旧 server 已在 6695 启动，加载原始 VLA 与 `projector_40tasks_best.pt`。
旧 eval 在 client 初始化处失败：`phyvla_client.py:51` 的 `PhysicsTokenizer` 未定义，未执行 episode。
启动日志、错误及恢复校验记录在 `logs/legacy_libero_retry/`。修正版 9886 服务已停止。

### 旧入口修复后的运行状态

已在恢复的旧入口上修复 client 类名/编码接口与 server WebSocket 接口。
client 使用 LivePhysicsState，server 复用 PhysicsPolicy 加载保存的 tokenizer、投影权重，
并使用 baseline 的动作反归一化；缺少物理分支或推理未确认 physics_used 时直接报错。
episode 在 env.reset/set_init_state 后重新绑定物理适配器。
旧重复 server 启动段已移除，修改前版本保存在 logs/legacy_libero_fix/20260914_221927/。
6695 已重启，普通 LIBERO spatial 10 任务、每任务 1 次、seed 7 正在测评。
日志 logs/legacy_libero_fix/run/eval.log；逐 episode 结果 spatial/result.json。
仍使用旧 40 任务 projector + 现有替代 tokenizer，标记 unpaired 实验。


## LIBERO-plus with the complete v2 checkpoint

The original LIBERO pilot (40 tasks, one initial state each, seed 7) completed:
baseline 39/40 (97.5%), PHYVLA 36/40 (90%). Spatial/object: both 10/10;
goal: 10/10 versus 9/10; LIBERO-10: 9/10 versus 7/10.
The physical residual did not improve this pilot's success rate. The paired
comparison is `outputs/physics_v2_evaluation/comparison.json`. Do not substitute
held-out action L1 improvement for closed-loop success.

LIBERO-plus contains 10,030 variants across four suites (spatial 2,402, object
2,518, goal 2,591, LIBERO-10 2,519). `NUM_TRIALS=1` means one rollout per
variant, not one rollout per original task. Some variants only provide one
initial state. Keep the original-LIBERO-trained checkpoint fixed; do not train
or select checkpoints on plus evaluation outcomes.

Run from the repository root:

```bash
# Start only if port 6695 is not already serving the verified v2 checkpoint.
GPU=1 PORT=6695 bash thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files/run_phyvla_server.sh

# In another terminal: validate resources, service metadata and checkpoint hash.
CHECK_ONLY=1 bash thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files/eval_phyvla.sh

# All four suites, all variants; long-running evaluation.
PORT=6695 NUM_TRIALS=1 SEED=7 bash thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files/eval_phyvla.sh

# Matched baseline: requires serve_baseline_v2.py on port 6696.
base_port=6696 num_trials_per_task=1 seed=7 bash thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files/eval_baseline_v2.sh
```

Both clients use rotated agent/wrist images with PIL bilinear 224-pixel resizing,
chunk length 8, the same step budgets and initial-state order. PHYVLA sends v2
physical features and verifies the server used the requested checkpoint.
The saved complete checkpoint is `outputs/physics_fusion_v2/policy_best.pt`.
Reports are under the printed timestamped log directory: `<suite>_result.json`
contains completed episode counts, task IDs, categories, seed and server metadata;
`<suite>.json` contains final category totals. An interrupted run remains marked
running; do not treat scheduled category denominators as completed trials.

`outputs/physics_v2_evaluation/plus_preflight.json` records the resource and
seven-category-per-suite inference checks. These are compatibility checks, not
success-rate measurements. PHYVLA has simulator-state and BDDL-goal access beyond
the image baseline; particularly report this when interpreting sensor-noise and
language-perturbation results. Summarize all seven categories, including failures.

Preparation validation completed: 28/28 suite/category inference probes passed;
all 10,030 classified tasks resolved to existing initial-state files. This does
not exhaustively instantiate every variant. The same spatial variant (task ID 0,
initial state 0) completed successfully for both policies: PHYVLA 79 policy steps,
baseline 80. These one-episode checks only verify the rollout/reporting path.
Full LIBERO-plus evaluation has not been launched.
