# 脚本入口与归档清单

[返回首页](../README.md)

## 整理范围和规则

2026-09-14 对项目根目录及 scripts/ 的 95 个 Python/Shell 文件检查后，37 个移入 archive/，58 个保留。src/、第三方仓库、数据、权重未移动。

归档路径见 [archive/LATEST](../archive/LATEST)，对应目录的 manifest.json 包含每个原路径、归档路径、SHA-256 和理由；before.tar.gz 保存整理前的全部 95 个脚本及工作流文档。归档文件用于恢复或参考，不应在归档目录直接执行，因为许多脚本根据相对位置计算项目根目录。

保护优先级：下载/仿真验证、通用复现、训练入口优先保留。默认 task02 但支持参数化的脚本不等于单 demo 临时脚本；固定数据的 MuJoCo/Warp 验证也保留。仅做 task02 路径查询、单份 NPZ 检查、固定 demo 展示的脚本归档。

## 保留的主要入口

| 用途 | 入口 |
|---|---|
| 环境安装、原始数据下载 | setup_envs.sh；download_libero_10.py；第三方 LIBERO 的 benchmark_scripts/download_libero_datasets.py |
| MuJoCo / MJX / Warp 验证 | scripts/audit_mjx_compatibility.py、verify_mujoco_mjx_parity.py、verify_parity_visual.py、test_p3_warp_rollout.py、verify_warp_gradients.py、debug_contact_force_backend.py |
| 通用回放与提取 | scripts/replay_libero_actions.py、replay_libero_states.py、replay_warp_full.py、extract_task02_physics_smoke.py、visualize_contact_forces.py |
| 通用流水线 | scripts/run_task_pipeline.py、run_multi_task_pipeline.py、run_multitask_p3p4.py、process_all_40tasks.sh |
| 物理编码 | scripts/compute_normalization.py、build_contact_events.py、build_offline_tokens.py、train_tokenizer.py |
| P5 训练 | scripts/export_vla_actions.py、train_p5_projector.py、train_40tasks_projector.sh、train_p5_all_40tasks.sh |
| 数据验证 | scripts/check_40tasks_data.py、verify_data_alignment.py |
| Baseline 服务与客户端 | scripts/start_baseline_server.sh、run_baseline_client.sh |
| 结果分析/绘图 | scripts/generate_comparison_report.py、compare_eval_results.py、analyze_eval_results.py、summarize_libero_plus_results.py、visualize_libero_plus_results.py |

完整保留名单见 manifest.json 的 retained 字段。项目使用的包名是 mujoco_warp；文件名中的 warp 与用户所述 mujoco_wrap 对应同一验证方向。

## 可核对的 baseline 入口

在项目根目录、已配置环境和权重的前提下：

```bash
# 先仅打印命令，不加载模型或启动评估
DRY_RUN=1 bash scripts/start_baseline_server.sh
DRY_RUN=1 BENCHMARK=LIBERO bash scripts/run_baseline_client.sh
DRY_RUN=1 BENCHMARK=LIBERO-plus bash scripts/run_baseline_client.sh

# 服务端与客户端在不同终端运行
bash scripts/start_baseline_server.sh
BENCHMARK=LIBERO SUITE=libero_spatial NUM_TRIALS=50 bash scripts/run_baseline_client.sh
```

LIBERO-plus 将 BENCHMARK 改为 LIBERO-plus。可设置 VLA_CKPT、HOST、PORT、SEED、OUTPUT_DIR、EVAL_ENV 和 MUJOCO_GL。客户端连接已有服务，不会自动启动/关闭其他进程。

服务端已改为真实的 server_policy.py 和下划线参数；客户端已改为真实的 eval_files/eval_libero.py 和其 Args 参数，并区分两套环境。全量处理脚本已修正 --py-libero 参数值、增加 --py-starvla，统一输出至 outputs/<suite>/task_<id>。

## 保留不代表完成运行验收

58 个保留脚本通过 Python AST / bash -n 静态语法检查；baseline 三条 dry-run 命令通过。此次未运行 GPU 仿真、训练或完整评估；资源缺失和运行时依赖仍按其他指南核对。

核心训练脚本按要求保留；tokenizer 保存/重导出问题已修复，新旧流程的区别见 [编码训练](ENCODING_TRAINING.md)。以下训练辅助包装器按保护要求保留，但仍有旧 cache 或输出参数，不能作为推荐复现入口：run_full_verification.sh、run_p5_demo_10tasks.sh、scripts/batch_cache_vlm.sh。请直接使用训练指南中的命令。verify_physics_token_impact.py 是未完成的实验编排器，已归档；它调用的真实训练实现 train_p5_projector.py 保留。fix_task02_and_train.sh 虽先修补 task02，但随后启动 40 任务训练，按训练保护规则保留。

## 对比与消融状态

首次归档时尚无已验证的 PHYVLA 在线闭环；后续新增的配套训练与在线入口已完成运行验证，见文末。此次归档的是 env 未定义、导入不存在模块、调用不存在客户端、或根本不发送物理 token 的第一方入口；没有以 baseline 输出冒充 PHYVLA 结果。

第三方仓库保留原样，包括尚有缺口的本地 PHYVLA 扩展；它们不在此次“有效入口”认证范围。有效的官方 baseline 客户端和结果汇总工具保留，待补齐真实物理编码接入后，按 [对比指南](BASELINE_COMPARISON.md) 进行 baseline / 真 token / 置零或打乱 token 对照。

汇总工具仅适用于各自预期 JSON schema，不能混用或静默忽略缺失任务。保留它们是为已存在结果及未来真实实验服务，并不说明旧评估已经成功。

## 新增环境验证入口

2026-09-14 新增 `scripts/verify_liberoplus_environment.py`，属于受保护的环境验证脚本。使用 `--with-policy --port 9883` 可检查四个 suite 各一个任务的渲染及真实模型调用；此次验证通过，详见 [环境配置](ENVIRONMENT.md)。此前的 58 个保留脚本未因本次环境修复归档。

## PHYVLA 接口新增入口

- `scripts/serve_phyvla.py` / `start_phyvla_server.sh`：加载 baseline 与 40 任务 projector，默认 GPU 1、9884 端口。
- `thirdparty/starVLA/examples/simBenchmarks/LIBERO-plus/eval_files/eval_phyvla.sh`：LIBERO-plus 在线入口，必须连接已加载兼容 tokenizer 的服务。
- `scripts/verify_phyvla_client.py`：验证全部 40 任务的已有 token 请求、动作变化、缓存重置和错误输入；不执行 demo 回放或计算成功率。
- `verify_liberoplus_environment.py --with-physics-state`：验证四套件实时物理特征。

这些入口属于通用验证/对比脚本，保留使用。历史归档 manifest 不包含后续新增文件。现已训练并保存成对 tokenizer/projector，详见 [对比指南](BASELINE_COMPARISON.md)。

## 共享在线编码器训练

`prepare_online_physics.py` 按原始 LIBERO 官方任务顺序恢复 XML/state，构建与在线相同的物理特征。`train_online_physics.py` 使用不重叠的 demo 训练/验证，保存 tokenizer、projector、配对哈希、训练状态并重新导出 token。默认覆盖全部 40 个任务，属于受保护的通用数据与训练入口。

旧 `train_tokenizer.py`、`build_offline_tokens.py`、`run_task_pipeline.py` 已修复权重保存/加载和先训练后导出顺序；旧流程仍不等同于新的在线特征契约。

- `scripts/export_physics_dataset_v2.py`：普通 LIBERO 全示范原始物理状态导出，无 token 训练。
- `scripts/validate_physics_dataset_v2.py`：独立核对来源、状态/动作、图像索引、接触和数值。
