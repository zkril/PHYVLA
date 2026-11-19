# 数据处理与物理状态提取

[返回首页](../README.md) · 前置：[原始数据](DATA_PREPARATION.md)

当前 40 任务全量训练使用本文后半的 **raw physics v2** 导出和 [v2 编码流程](ENCODING_TRAINING.md)。下述单任务入口仅用于历史复查。

## 从 HDF5 到物理状态

`scripts/visualize_contact_forces.py` 配合 `src/adapters/libero_physics_adapter.py` 恢复 demonstration 的模拟器状态，读取机器人、对象和接触信息。用记录状态恢复可避免仅重放动作导致的累计漂移；对象位姿、速度、质量/惯量/摩擦和接触力需保持坐标系、单位及帧索引一致。

## 单任务完整入口

以下命令会提取状态、构建初始 token 并训练 tokenizer；不是纯提取命令。先用单个 demo 验证：

```bash
conda activate libero_replay
cd "$PHYVLA_ROOT"
export LIBERO_CONFIG_PATH="$PHYVLA_ROOT/thirdparty/LIBERO/libero"
SUITE=libero_spatial
TASK_ID=2
OUT="$PHYVLA_ROOT/outputs/$SUITE/task_$TASK_ID"
python scripts/run_task_pipeline.py \
  --suite "$SUITE" --task-id "$TASK_ID" --demo-index 0 \
  --output-dir "$OUT" --py-libero "$(command -v python)" \
  --epochs 200 --device cuda
```

`--py-libero` 必须带 Python 可执行文件路径；省略时会跳过 XML / controller trace 导出。若回放环境的 PyTorch 不支持 GPU，补充 `--py-starvla /实际/starVLA/bin/python` 让 tokenizer 训练使用该解释器。

必须显式指定 `outputs/<suite>/task_<local_id>`，与 P5 读取路径一致。默认 `outputs/task02/` 不区分 suite，批量处理时可能冲突。多个 demo 应规划独立运行目录：统计、事件、XML、trace 等文件名不含 demo 编号，会覆盖。

## 各阶段与产物

| 脚本 | 输出或作用 |
|---|---|
| extract_task_xml.py | model_raw.xml：完整模拟器模型 |
| export_controller_trace.py | controller_trace_v1.npz：控制及状态轨迹 |
| visualize_contact_forces.py | task02_demo00_physics.npz、对应 JSON 和视频 |
| compute_normalization.py | normalization_v1.json：物理字段归一化 |
| build_contact_events.py | contact_events_v1.npz：接触事件和阶段标签 |
| build_offline_tokens.py | physics_tokens_demo0.npz：初始编码 |
| train_tokenizer.py | 训练及重建指标；当前没有保存训练权重 |

只提取物理数据可单独运行：

```bash
mkdir -p "$OUT"
python scripts/visualize_contact_forces.py \
  --suite "$SUITE" --task-id "$TASK_ID" --demo-index 0 \
  --output-npz "$OUT/task02_demo00_physics.npz" \
  --output-json "$OUT/task02_demo00_physics.json" \
  --output-video "$OUT/task02_demo00.mp4"
```

上述文件名对应示例 task 2；更换任务时同步修改。随后训练步骤见 [编码与训练](ENCODING_TRAINING.md)。

## 验证与批量处理

检查 NPZ 中有限数值、序列长度、有效对象/接触 mask，并抽查视频中的接触力与物体角色。逐项核对 JSON 的 demo_path、language 与目标任务。归一化统计应由训练划分计算，测试阶段复用，避免信息泄漏。

单任务通过后逐 suite、逐 local ID 扩展，保留每项日志。旧批处理包装器已归档；统一入口为 `bash scripts/process_all_40tasks.sh`，其参数和输出目录已修正。P5 数据生成后用 `python scripts/check_40tasks_data.py` 检查，但仍需人工确认任务身份和所有序列的帧对齐。

P2–P4 为可选扩展：`scripts/test_p3_warp_rollout.py`、`scripts/verify_warp_gradients.py` 在 `phyvla-p2` 中运行；二者当前固定读取旧 task02 产物路径，应先核对输入，不要直接套用新目录。

## 40 任务重新导出：raw physics v2

推荐入口：`WORKERS=8 bash scripts/process_all_40tasks.sh`。
默认原始 LIBERO 四套件、全部 demo，输出 `outputs/physics_reexport_v2/`；
旧 `outputs/libero_*` 保留，不作为新导出的缓存。设置 OUTPUT_DIR 可指定独立目录，
DEMO_INDICES 可指定小规模测试；不设置即读取全部示范。

旧入口曾混入其他 suite、丢失对象实例编号、只提取 bowl/plate、逐任务训练编码器，
且仅校验帧数。新入口根据 benchmark 官方映射读取 HDF5，验证 suite/task/source，
恢复每条示范原始 XML，并对每帧直接 set_state/forward（不执行动作推进）。
原始状态重载误差必须小于 1e-9；控制器动作重放不作为状态对齐依据。

每个 `<suite>/task_<id>/demo_<id>/` 包含：
- physics.npz：完整记录状态、专家动作、机器人/对象位姿与速度、所有关节地址、
  物体质量/惯量、geom 摩擦/尺寸、父子关系、无截断的接触信息。
- model_recorded.xml：原始场景模型。
- metadata.json：任务映射、对象实例名、BDDL 目标实体、字段维度和来源 SHA-256。
- labels.json：成功标签独立存放，禁止作为 token 输入。

接触按 contact_offsets[t]:contact_offsets[t+1] 索引 contacts，21 列字段定义见 metadata。
不使用 bowl/plate 特化关系，不人为补造 phase/gradient/future token。
目标角色为 target/goal 两列多标签，源于 BDDL 谓词；多个目标不合并为单目标。
对象实例编号及关节活动子节点不删除；后续编码器须支持实际对象集合，不能直接套旧 16 槽规则。

**图像对齐：** 原始 create_dataset.py 在 env.step(action) 后保存 obs，但保存动作前的 states。
因此不能把 obs[t] 与 states[t] 直接拼接。使用 observation_index_for_state[t] 找已数值验证
的前一帧图像；-1 表示缺失/未通过关节一致性检查，视觉动作训练应排除此帧或另行重渲染。
原始状态与动作不移动；专家动作监督为 expert_actions[t]。

固定按完整 demo 划分：ID 尾数为 9 的 demo 为 validation，其余 train（完整 50 demo 为 45/5）。
归一化统计及 tokenizer 必须仅在 train 拟合，随后统一保存权重、在两个 split 上导出 token。
这次仅导出原始物理数据，不训练 token，不产生 VLA 预测动作缓存。

完成标准：manifest.json 的 status=completed 且 validation.json 的 passed=true。
独立复查：`conda run -n libero-plus python scripts/validate_physics_dataset_v2.py --root outputs/physics_reexport_v2`。
新格式与旧 PhysicsDataset 的 flat NPZ 不兼容，应按以上 schema 编写/更新后续编码数据加载器，
禁止把这批数据误当作旧 tokenizer 的直接输入。

本次全量验收：40 任务、2,000 demo、338,575 帧、13,586,170 条接触记录；
训练/验证为 1,800/200 demo；336,575 帧通过图像索引核对，每条 demo 首帧标记无可匹配图像。
validation.json: passed=true, errors=[]。原始 data 和旧 outputs/libero_* 均保留。
