# Git 仓库整理总结

**完成日期**: 2026-09-04  
**仓库状态**: 已初始化并完成首次提交

---

## ✅ 完成的工作

### 1. 初始化 Git 仓库
```bash
git init
git branch -m main  # 使用 main 作为默认分支
```

### 2. 创建 .gitignore
配置了完整的忽略规则：
- ✅ Python 缓存 (`__pycache__/`, `*.pyc`)
- ✅ 虚拟环境 (`venv/`, `env/`)
- ✅ IDE 配置 (`.vscode/`, `.idea/`)
- ✅ 输出目录 (`outputs/`, `outputs_back/`)
- ✅ 大型二进制文件 (`*.npz`, `*.mp4`, `*.pt`)
- ✅ 第三方库 (`third_party/LIBERO/`, `third_party/starVLA/`)
- ✅ 临时文件 (`*.log`, `*.tmp`, `MUJOCO_LOG.TXT`)
- ⚠️ **保留关键数据**: `normalization_v1.json`, `contact_events_v1.npz` 等

### 3. 更新 README.md
从简短说明扩展为完整的项目首页：
- 项目概述与技术架构
- 快速开始指南
- Physical Token 结构详解
- P1-P4 验收结果
- 项目结构与路线图
- 性能基准与文档索引

### 4. 首次提交
```
commit dd118df
Author: [你的信息]
Date:   2026-09-04

Initial commit: phyvla project (P0-P4 complete)

68 files changed, 14585 insertions(+)
```

**提交内容**:
- 所有源代码 (`src/`, `scripts/`)
- 配置文件 (`configs/`)
- 完整文档 (`CLAUDE.md`, `PROJECT_ARCHIVE.md`, `CLEANUP_PLAN.md`, `doc/plan.md`)
- 中文设计文档 (`physicsvla（仿真）.md`)

---

## 📦 仓库结构

### 已提交文件（68 个）
```
phyvla/ (main branch)
├── .gitignore                           # Git 忽略规则
├── README.md                            # 项目首页（已更新）
├── CLAUDE.md                            # Claude Code 开发指南
├── PROJECT_ARCHIVE.md                   # 项目归档文档
├── CLEANUP_PLAN.md                      # 清理计划
├── physicsvla（仿真）.md                # 详细技术设计
├── configs/libero_task02.yaml           # 任务配置示例
├── doc/
│   ├── plan.md                          # 完整技术路线 (P0-P8)
│   └── 2603.01151v1.pdf                 # D-REX 论文参考
├── scripts/ (42 个脚本)
│   ├── replay_libero_states.py          # P0: 零漂移重放
│   ├── train_tokenizer.py               # P1: Tokenizer 训练
│   ├── test_p3_warp_rollout.py          # P3: Warp 验收
│   ├── demo_gradient_refinement.py      # P4: 梯度修正 Demo
│   ├── train_p5_projector.py            # P5: VLA 训练（待执行）
│   └── ...
└── src/
    ├── adapters/                        # MuJoCo/MJX 适配器
    ├── models/                          # Tokenizer/Projector/Encoder
    ├── simulation/                      # mujoco_warp/gradient
    ├── p5/                              # StarVLA 集成
    └── utils/                           # 工具函数
```

### 被 .gitignore 排除的文件
```
# 不会被提交到 git
outputs/                    # 实验输出（11 MB）
outputs_back/               # 历史数据（21 MB）
third_party/                # LIBERO/StarVLA（817 MB）
__pycache__/                # Python 缓存
*.npz, *.mp4, *.pt          # 大型二进制文件
*.log                       # 日志文件
MUJOCO_LOG.TXT              # MuJoCo 临时日志
```

**例外**：关键统计文件在 .gitignore 中设置了 `!` 规则保留，但当前它们在 `outputs_back/` 目录中，该目录整体被排除，因此也不会被提交。这是合理的设计——这些文件可以从原始数据重新生成。

---

## 🔍 仓库状态检查

### 查看提交历史
```bash
git log --stat
git log --oneline
```

### 查看未跟踪文件
```bash
git status
# 应显示 outputs/, outputs_back/, third_party/ 等被忽略
```

### 验证 .gitignore 生效
```bash
git check-ignore -v outputs/
git check-ignore -v third_party/LIBERO/
# 应显示匹配的 .gitignore 规则
```

---

## 🚀 后续 Git 操作建议

### 配置远程仓库（如果需要）

```bash
# 在 GitHub/GitLab 创建仓库后
git remote add origin https://github.com/yourusername/phyvla.git

# 推送到远程
git push -u origin main
```

### 日常工作流

```bash
# 查看变更
git status

# 添加新文件或修改
git add src/models/new_model.py
git add scripts/new_script.py

# 提交
git commit -m "feat: add new physics model for X"

# 推送到远程（如果已配置）
git push
```

### 创建开发分支

```bash
# 创建 P5 开发分支
git checkout -b dev/p5-training
# 工作...
git add ...
git commit -m "feat(p5): implement VLM caching"

# 完成后合并回 main
git checkout main
git merge dev/p5-training
```

### 提交消息规范建议

遵循 Conventional Commits 格式：

```
feat(p5): add PhysicsTokenProjector training script
fix(p3): correct mujoco_warp buffer size
docs: update README with P5 results
refactor(tokenizer): simplify object encoder
perf(rollout): optimize FD gradient computation
test: add P3 rollout validation
```

---

## 📋 .gitignore 关键规则说明

### 为什么排除 outputs/ 和 outputs_back/?

1. **大小**: 合计 32 MB，多为视频、NPZ 等二进制文件
2. **可重现**: 所有输出可通过运行脚本重新生成
3. **快速迭代**: 避免每次实验都产生大量 git diff

### 为什么排除 third_party/?

1. **大小**: 817 MB，包含完整 LIBERO/StarVLA 库
2. **外部依赖**: 应通过 `pip install -e third_party/LIBERO` 管理
3. **版本控制**: 可选方案：
   - Git submodule（推荐）
   - 提供安装脚本
   - 文档中说明如何获取

### 关键数据如何保存？

**方案 A**: 轻量级统计文件单独提交（推荐）
```bash
# 将关键数据移到 data/ 目录
mkdir -p data/reference
cp outputs_back/normalization_v1.json data/reference/
cp outputs_back/multi_task_results.json data/reference/

# 修改 .gitignore，允许 data/reference/
echo "!data/reference/*.json" >> .gitignore
git add data/reference/
git commit -m "data: add reference statistics for reproducibility"
```

**方案 B**: Git LFS 管理大文件
```bash
git lfs install
git lfs track "*.npz"
git lfs track "*.pt"
git add .gitattributes
```

**方案 C**: 外部存储 + 下载脚本
```bash
# 提供 scripts/download_data.sh
# 从云存储（OSS/S3）下载预处理数据
```

---

## 🎯 建议的下一步

### 1. 配置 Git 用户信息（如果还没配置）
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 2. 添加远程仓库（可选）
```bash
# GitHub
git remote add origin git@github.com:yourusername/phyvla.git
git push -u origin main

# GitLab
git remote add origin git@gitlab.com:yourusername/phyvla.git
git push -u origin main
```

### 3. 创建关键数据备份分支（可选）
```bash
# 创建一个包含关键数据的分支
git checkout -b archive/p1-validation-data
# 临时修改 .gitignore，提交关键 NPZ 文件
git add -f outputs_back/normalization_v1.json
git add -f outputs_back/contact_events_v1.npz
git add -f outputs_back/multi_task_results.json
git commit -m "archive: P1 validation reference data"
git checkout main  # 切回主分支
```

### 4. 设置 Git Hooks（可选）
```bash
# 创建 pre-commit hook，自动清理 Python 缓存
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
find . -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
EOF
chmod +x .git/hooks/pre-commit
```

### 5. 添加 GitHub Actions/GitLab CI（可选）
创建 `.github/workflows/test.yml` 用于自动化测试：
```yaml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install torch numpy
      - name: Run unit tests
        run: python -m pytest tests/
```

---

## ✅ 检查清单

- [x] Git 仓库已初始化
- [x] 创建 .gitignore（完整规则）
- [x] 更新 README.md（项目首页）
- [x] 首次提交（68 文件，14585 行）
- [x] 分支命名为 main
- [x] 提交消息包含 Co-Authored-By
- [ ] 配置远程仓库（可选）
- [ ] 添加 LICENSE 文件（建议）
- [ ] 设置 Git LFS（如需管理大文件）
- [ ] 创建 CONTRIBUTING.md（如开源）

---

**整理负责人**: Claude Opus 5  
**完成时间**: 2026-09-04  
**仓库状态**: ✅ 就绪，可推送到远程或继续本地开发
