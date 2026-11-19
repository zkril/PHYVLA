# 物理状态编码与训练

[返回首页](../README.md) · 前置：[物理状态提取](PHYSICS_PROCESSING.md)

## 当前推荐：40 任务、全部 demo 的 v2 流程

工作目录为 `/0604-10T-test/jiangnan/PHYVLA`。输入是已验收的
`outputs/physics_reexport_v2`，40 任务、2,000 条 demo。
按完整 demo 固定划分 1,800/200 条训练/验证；不会随机拆帧。
该划分约束本次物理模块训练，不保证已有 baseline 预训练未见过这些示范。

原始 HDF5 图像是动作执行后的观测。加载器通过
`observation_index_for_state` 将上一条图像与当前状态、专家动作配对，
排除每条 demo 首帧，共 336,575 帧（训练 303,534，验证 33,041）。

## 特征与共享 tokenizer

`src/datasets/physics_v2.py` 定义 `causal-object-set-v2`：
机器人 31 维；每个物体 39 维，包括相对末端位置、姿态、速度、
质量/惯量、摩擦、BDDL 目标角色、活动关节、接触统计与目标区域。
完整保留实例和活动子节点；不同物体数量用 mask，模型没有固定物体槽上限。
接触在每个物体上聚合全部记录，不按 top-K 丢弃；接触力是在记录状态上、
零控制与零 warm-start 条件下重建的量，不是额外的真实传感器读数。

输入只依赖当前状态及静态任务信息，不含专家动作、成功标签或未来状态。
归一化均值/标准差仅从训练 demo 拟合，验证和线上复用。
`PhysicsTokenizerV2` 将机器人及物体编码为宽度 128 的 token 集合，
以机器人/物体重建损失预训练；随后与融合分支联合训练。

```bash
conda activate starVLA
python scripts/prepare_physics_v2.py --data outputs/physics_reexport_v2 \
  --output outputs/physics_features_v2
CUDA_VISIBLE_DEVICES=1 python scripts/train_physics_tokenizer_v2.py \
  --data outputs/physics_features_v2 --output outputs/physics_tokenizer_v2
```

已有完整输出时使用新目录，避免覆盖不同版本的特征和权重。

## 保留视觉语言输入的物理融合

冻结原始 LIBERO-4in1 VLM 和 action head。先缓存原始视觉 action queries，
再以这些 queries 查询物理 token，并将零初始化的残差加回原始 queries。
关闭残差时应复现 baseline；不会用物理 token 替换整套视觉输入。

缓存每条 demo 从首个可对齐帧开始、每 8 帧取一个视觉输入。
8 步专家动作 chunk 覆盖全部可用动作，末尾不足 8 步的部分用 mask 排除。
监督是真实专家动作，不是 baseline 自己的预测。缓存使用批量推理，线上使用单环境推理；两者可能有小幅浮点计算差异。
已验证相同批量下缓存一致，以及单环境下零残差与 baseline 完全一致。
两路相机、180 度旋转、
224 像素缩放、语言提示及动作归一化与评测入口一致。

```bash
export VLA_CKPT="$PWD/thirdparty/starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
CUDA_VISIBLE_DEVICES=1 python scripts/cache_visual_queries_v2.py \
  --data outputs/physics_features_v2 --output outputs/visual_queries_v2 \
  --checkpoint "$VLA_CKPT" --batch-size 32
CUDA_VISIBLE_DEVICES=1 python scripts/train_physics_fusion_v2.py \
  --queries outputs/visual_queries_v2 \
  --tokenizer outputs/physics_tokenizer_v2/tokenizer_best.pt \
  --output outputs/physics_fusion_v2
```

## 验证、产物与恢复

- `physics_tokenizer_v2/tokenizer_best.pt`：编码器、结构、归一化、数据与代码哈希。
- `physics_fusion_v2/policy_best.pt`：同一次训练的 tokenizer、融合模块、
  完整 action head、baseline 来源、归一化、特征版本与验证指标。
- `training_last.pt`：训练状态和优化器；当前 CLI 不提供自动续训参数。
- `history.json`、`summary.json`：逐 epoch 指标及 full/no-physics/shuffled 消融。

`conda run -n libero-plus python scripts/verify_physics_v2_live.py`
在四个 suite 的 40 个任务、120 个留出帧上比较线上与离线特征，
同时检查提取不修改仿真状态。重建损失、动作误差均不能替代闭环成功率。
真实有效性需要按相同任务、初始状态、步数与随机种子进行
[baseline 对比](BASELINE_COMPARISON.md)。

## 历史入口

`train_online_physics.py` 和旧 `PhysicalTokenizer` 使用不同特征契约，
旧 projector 也不兼容 v2。历史路径保留供复查，不能与新 tokenizer 混用。
旧的物理 queries 路径没有保留视觉动作查询；不得把其结果当作当前融合模型。
