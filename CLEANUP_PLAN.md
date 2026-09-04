# phyvla 仓库清理计划

**执行日期**: 2026-09-04  
**目标**: 清理冗余文件，释放磁盘空间，保持核心数据完整性

---

## 清理策略总览

### 当前磁盘占用
```
outputs/         11 MB   （当前工作目录）
outputs_back/    21 MB   （历史实验数据）
third_party/    817 MB   （第三方库，必须保留）
总计约          849 MB
```

### 清理目标
- **释放空间**: ~15-20 MB
- **保留核心数据**: P1 验收数据、归一化统计、架构图
- **删除冗余**: 中间实验视频、日志、临时模型文件

---

## 清理操作清单

### 阶段 1: 安全删除（可立即执行）

#### 1.1 删除历史任务输出目录（释放 ~18 MB）

```bash
cd /home/jiangnan-luxitech/workspace/phyvla/outputs_back

# 删除各任务的中间实验数据（视频、临时 NPZ 等）
rm -rf task01/ task02/ task03/ task04/ task05/ task06/ task07/
rm -rf physics/   # P0 早期测试输出
rm -rf replay/    # P0 早期重放视频

# 预期释放: ~15 MB
```

**保留文件**（outputs_back/ 核心数据）:
```
normalization_v1.json          # 11.7 KB - P1 归一化统计
contact_events_v1.npz          # 469 B  - P1 接触事件标签
physics_tokens_demo0.npz       # 646 KB - P1 token 矩阵示例
physics_tokens_demo0.json      # 688 B  - P1 metadata
multi_task_results.json        # 2.1 KB - P1 7任务验收汇总
task03_pipeline.log            # 10 KB  - 参考日志（可选保留）
task05_pipeline.log            # 10 KB  - 参考日志（可选保留）
```

#### 1.2 清理 Python 缓存（释放 ~1-2 MB）

```bash
cd /home/jiangnan-luxitech/workspace/phyvla

# 删除所有 __pycache__ 目录
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 删除 .pyc 字节码文件
find . -name "*.pyc" -delete

# 删除 Jupyter checkpoint
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null
```

#### 1.3 压缩大型图片（可选，释放 ~0.5 MB）

```bash
cd /home/jiangnan-luxitech/workspace/phyvla/outputs

# 方案 A: 删除 PNG，保留 PDF（推荐）
rm dp_vla_architecture.png  # 841 KB

# 方案 B: 压缩为 WebP（需要 ImageMagick）
# convert dp_vla_architecture.png -quality 85 dp_vla_architecture.webp
# rm dp_vla_architecture.png
```

### 阶段 2: 归档后删除（需先创建备份）

#### 2.1 创建归档包

```bash
cd /home/jiangnan-luxitech/workspace/phyvla

# 归档完整 outputs 和 outputs_back
tar -czf archive_phyvla_outputs_20260904.tar.gz \
    outputs/ outputs_back/ \
    --exclude='*.pyc' \
    --exclude='__pycache__'

# 验证归档完整性
tar -tzf archive_phyvla_outputs_20260904.tar.gz | wc -l

# 移动归档文件到安全位置
mkdir -p ~/phyvla_archives
mv archive_phyvla_outputs_20260904.tar.gz ~/phyvla_archives/
```

#### 2.2 删除已归档的原始文件（可选）

**⚠️ 警告**: 仅在确认归档成功后执行

```bash
cd /home/jiangnan-luxitech/workspace/phyvla

# 删除 outputs_back 整个目录（已归档）
# rm -rf outputs_back/

# 或仅删除大文件，保留核心数据
# cd outputs_back
# rm physics_tokens_demo0.npz  # 646 KB
```

### 阶段 3: 日志清理（可选）

#### 3.1 清理管道日志

```bash
cd /home/jiangnan-luxitech/workspace/phyvla

# 如果不再需要参考，可删除日志文件
# rm outputs/task01_pipeline.log        # 13 KB
# rm outputs_back/task03_pipeline.log   # 10 KB
# rm outputs_back/task05_pipeline.log   # 10 KB
```

---

## 保留文件清单

### 必须保留（核心数据与代码）

```
phyvla/
├── CLAUDE.md                         # 项目指南
├── PROJECT_ARCHIVE.md                # 归档文档（新建）
├── CLEANUP_PLAN.md                   # 本文档（新建）
├── doc/plan.md                       # 技术路线
├── physicsvla（仿真）.md             # 详细设计
├── src/                              # 所有源代码
├── scripts/                          # 所有脚本
├── configs/                          # 配置文件
├── third_party/                      # 第三方库（必须）
├── outputs_back/
│   ├── normalization_v1.json         # P1 归一化统计
│   ├── contact_events_v1.npz         # P1 接触事件
│   ├── physics_tokens_demo0.npz      # P1 token 示例
│   ├── physics_tokens_demo0.json
│   └── multi_task_results.json       # P1 验收汇总
└── outputs/
    ├── dp_vla_architecture.pdf       # 架构图
    ├── dp_vla_architecture.txt
    ├── draw_architecture.py
    ├── task01/                       # 当前实验（保留）
    ├── task05/                       # 当前实验（保留）
    └── multitask_p3p4_results.json
```

### 可删除（中间数据）

```
outputs_back/
├── task01/, task02/, ..., task07/   # 历史任务输出
├── physics/                          # P0 早期测试
└── replay/                           # P0 早期视频

outputs/
└── dp_vla_architecture.png           # 大型图片（有 PDF）
```

---

## 执行脚本

### 一键清理脚本（安全模式）

创建文件 `cleanup_safe.sh`:

```bash
#!/bin/bash
set -e

REPO_ROOT="/home/jiangnan-luxitech/workspace/phyvla"
cd "$REPO_ROOT"

echo "=== phyvla 仓库安全清理 ==="
echo "当前目录: $(pwd)"
echo ""

# 1. 清理 Python 缓存
echo "[1/4] 清理 Python 缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
echo "✓ Python 缓存清理完成"

# 2. 删除历史任务输出
echo "[2/4] 删除 outputs_back 历史任务目录..."
if [ -d "outputs_back" ]; then
    rm -rf outputs_back/task{01,02,03,04,05,06,07}/ 2>/dev/null || true
    rm -rf outputs_back/physics/ 2>/dev/null || true
    rm -rf outputs_back/replay/ 2>/dev/null || true
    echo "✓ 历史任务目录删除完成"
else
    echo "⊘ outputs_back 不存在，跳过"
fi

# 3. 压缩大型图片（可选）
echo "[3/4] 处理大型图片..."
if [ -f "outputs/dp_vla_architecture.png" ] && [ -f "outputs/dp_vla_architecture.pdf" ]; then
    echo "  删除 PNG（已有 PDF）: dp_vla_architecture.png"
    # rm outputs/dp_vla_architecture.png  # 取消注释以执行
    echo "  （当前为演习模式，未实际删除）"
else
    echo "⊘ PNG 或 PDF 不存在，跳过"
fi

# 4. 统计清理结果
echo "[4/4] 统计磁盘占用..."
du -sh outputs outputs_back third_party 2>/dev/null || true

echo ""
echo "=== 清理完成 ==="
echo "核心数据已保留在:"
echo "  - outputs_back/normalization_v1.json"
echo "  - outputs_back/contact_events_v1.npz"
echo "  - outputs_back/physics_tokens_demo0.npz"
echo "  - outputs_back/multi_task_results.json"
```

执行：
```bash
chmod +x cleanup_safe.sh
./cleanup_safe.sh
```

### 完整归档脚本（包含备份）

创建文件 `archive_and_cleanup.sh`:

```bash
#!/bin/bash
set -e

REPO_ROOT="/home/jiangnan-luxitech/workspace/phyvla"
ARCHIVE_DIR="$HOME/phyvla_archives"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_FILE="phyvla_outputs_${TIMESTAMP}.tar.gz"

cd "$REPO_ROOT"
echo "=== phyvla 完整归档与清理 ==="

# 1. 创建归档目录
mkdir -p "$ARCHIVE_DIR"
echo "[1/5] 归档目录: $ARCHIVE_DIR"

# 2. 创建归档包
echo "[2/5] 创建归档包..."
tar -czf "$ARCHIVE_DIR/$ARCHIVE_FILE" \
    outputs/ outputs_back/ \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.ipynb_checkpoints'

echo "✓ 归档包创建完成: $ARCHIVE_FILE"
ARCHIVE_SIZE=$(du -sh "$ARCHIVE_DIR/$ARCHIVE_FILE" | cut -f1)
echo "  大小: $ARCHIVE_SIZE"

# 3. 验证归档
echo "[3/5] 验证归档完整性..."
FILE_COUNT=$(tar -tzf "$ARCHIVE_DIR/$ARCHIVE_FILE" | wc -l)
echo "✓ 归档包含 $FILE_COUNT 个文件"

# 4. 执行清理（取消注释以执行）
echo "[4/5] 执行清理..."
# bash cleanup_safe.sh

# 5. 完成
echo "[5/5] 归档完成"
echo ""
echo "归档文件: $ARCHIVE_DIR/$ARCHIVE_FILE"
echo "解压命令: tar -xzf $ARCHIVE_FILE"
echo ""
echo "⚠️  清理操作已注释，请手动执行 cleanup_safe.sh"
```

---

## 验证检查清单

清理后执行以下验证：

### ✅ 核心数据完整性
```bash
cd /home/jiangnan-luxitech/workspace/phyvla

# 验证 P1 关键文件存在
ls -lh outputs_back/normalization_v1.json
ls -lh outputs_back/contact_events_v1.npz
ls -lh outputs_back/physics_tokens_demo0.npz
ls -lh outputs_back/multi_task_results.json

# 验证 P1 token 数据可加载
python -c "import numpy as np; data = np.load('outputs_back/physics_tokens_demo0.npz'); print(data['z_phy'].shape)"
# 预期输出: (103, 23, 256)
```

### ✅ 代码可运行性
```bash
# 验证 P1 pipeline 仍可运行
conda activate libero_replay
python scripts/compute_normalization.py --help
python scripts/train_tokenizer.py --help
```

### ✅ 磁盘空间释放
```bash
# 清理前后对比
du -sh outputs outputs_back third_party
```

---

## 风险评估

| 操作 | 风险等级 | 备注 |
|------|----------|------|
| 删除 Python 缓存 | 🟢 无风险 | 可自动重新生成 |
| 删除 outputs_back/task{01-07}/ | 🟡 低风险 | 历史实验数据，核心结果已在 multi_task_results.json |
| 删除 PNG 图片 | 🟢 无风险 | 已有 PDF 版本 |
| 删除日志文件 | 🟡 低风险 | 调试时可能需要参考 |
| 删除 physics_tokens_demo0.npz | 🔴 中风险 | 可重新生成，但需重跑 P1 pipeline |
| 删除 normalization_v1.json | 🔴 高风险 | **必须保留**，需 7 任务数据重新计算 |
| 删除 third_party/ | 🔴 高风险 | **禁止删除**，LIBERO/StarVLA 依赖 |

---

## 恢复方案

### 如果误删核心数据

#### 1. 从归档包恢复
```bash
cd ~/phyvla_archives
tar -xzf phyvla_outputs_YYYYMMDD_HHMMSS.tar.gz -C /home/jiangnan-luxitech/workspace/phyvla/
```

#### 2. 重新生成 P1 数据
```bash
conda activate libero_replay
cd /home/jiangnan-luxitech/workspace/phyvla

# 重新计算归一化统计（需要 7 任务完整数据）
python scripts/run_multi_task_pipeline.py --recompute-norm

# 重新生成 physics tokens
python scripts/build_offline_tokens.py --task-id 2 --demo-index 0
```

#### 3. 从 Git 恢复（如果已提交）
```bash
cd /home/jiangnan-luxitech/workspace/phyvla
git checkout outputs_back/normalization_v1.json
git checkout outputs_back/contact_events_v1.npz
```

---

## 建议执行顺序

1. **阅读本文档**，理解每个操作的影响
2. **创建归档包**（archive_and_cleanup.sh）
3. **验证归档完整性**（解压测试）
4. **执行安全清理**（cleanup_safe.sh）
5. **验证核心数据**（检查清单）
6. **测试代码运行**（跑一个 P1 脚本）
7. **确认磁盘空间释放**（du -sh）

---

**创建日期**: 2026-09-04  
**预期释放空间**: 15-20 MB  
**风险等级**: 🟡 低风险（已有归档保护）
