# physicsvla（仿真）

# 可微物理引导的视觉\-语言\-动作模型阶段技术路线

## 一、目标问题

当前 VLA（Vision\-Language\-Action）模型的主流范式是将视觉观测、语言指令和机器人动作统一到一个端到端模型中，通过大规模演示数据学习：

$a_t = f_{\text{VLA}}(I_t, L)$

这种范式在语义理解和模仿学习上已经表现出较强能力，但在机器人真实执行中仍然存在一个核心瓶颈：**VLA 模型通常学到的是“视觉\-语言到动作”的统计相关性（若改变光照等视觉条件，性能大幅下降），而不是“动作会如何改变物理世界”的可验证因果机制。**

具体表现为：

1. **物理状态缺失**：机器人和物体的位姿可以通过视觉或本体状态获得，但质量、摩擦、运动趋势、稳定性等任务相关物理信息通常没有被显式建模，真实环境中还存在部分物理状态不可直接观测的问题。

2. **接触关系缺失**：抓取、推拉、插入、旋转等任务的成败往往由接触点、接触力、摩擦和滑移决定，而普通视觉 token 难以显式表达这些信息。（calvin任务中push类的任务效果普遍比较差）

3. **动作后果不可验证**：传统 VLA 直接输出动作，缺少执行前的物理后果预测、风险评估和动作修正机制。（动作输出后若失败，没有反馈修改机制）

4. **长序列动作缺乏物理一致性**：多步操作中容易出现碰撞、穿透、物体滑落、轨迹不连续、关节超限等问题。



## 二、目标工作

如何将可微物理仿真器、物理状态编码器和 VLA 策略模型统一到一个可训练、可验证的闭环框架中，使 VLA 从视觉语言动作拟合模型升级为物理可解释、物理一致、物理可执行的动作推理模型。

（TODO：RL）



## 三、相关工作

1. gradSim（2021，iclr）https://arxiv\.org/pdf/2104\.02646

详细：

创新点主要有三个：第一，把 dynamics 和 rendering 合在一起，而不是只做物理仿真或只做渲染；第二，不依赖 3D 标签，只靠 image\-space supervision；第三，可用于系统辨识和视觉运动控制。从视频像素反传到物理属性、从视频恢复多类物理参数、提供 PyTorch 框架。

实验上，验证了刚体（Newton\-Euler 刚体动力学 \+ 接触/碰撞）、可变形体（有限元）、布料（Mass\-Spring）等场景：可以估计质量、摩擦、弹性、Lamé 材料参数、布料速度等；在部分任务中，虽然只用 2D 视频监督，效果接近甚至优于需要 3D 监督的 differentiable physics baseline。

自己实现了可微物理仿真模块和可微渲染模块



2. OrbiSim（2026，arxiv）https://arxiv\.org/html/2605\.16395v1

详细：

OrbiSim目标把 world model 重新做成完全可微的神经物理引擎 分成两个模块：OrbiSim\-Dynamics（主要梯度传播） 负责预测物理状态；OrbiSim\-Vision 负责根据预测状态生成图像。可微性带来了在Real\-to\-Sim模块中通过 StaInf 从图像推断显式状态，通过 PhyInf 从短轨迹推断隐藏物理参数；且奖励可以通过可微动力学直接反传到策略，而不是完全依赖采样式 RL。

通过OrbiSim\-Dynamics代替传统MuJoCo等仿真平台，每个物体的物理状态被编码为token输入至OrbiSim\-Dynamics中

3. D\-REX（2026，iclr）https://arxiv\.org/abs/2603\.01151

详细：

抓取刚体特化，质量预测特化。利用Gaussian Splatting进行场景重建，利用推动的仿真轨迹来进行质量预测。可以借鉴刚体运动学表示。采用到了MJX 提供机器人手的运动学和动力学基础，GradSim 提供适合质量系统辨识的梯度传播机制。

经过检查后，D\-REX是明确基于gradsim代码复现并做适当修改。D\-REX 的质量系统辨识模块直接基于 GradSim：其仓库内嵌并修改了 GradSim 的刚体、外力、积分器和仿真代码，质量估计脚本也由 GradSim 的demo\_mass\_known\_shape\.py 扩展而来。 MuJoCo/MJX 机器人 rollout 提取接触条件，再利用 GradSim/PyTorch 对质量参数进行轨迹级反向传播。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MjAyZjE0NWVhMGUwOTZmNDUyNzI5MmUyYzZmMGY2M2JfY2JmMGQzYWZkMmJkZWQ3NmMxNmM0NzY2MTI1ZDY1YjNfSUQ6NzY2MDc5MDQ3NTE0NDMwMTUyNF8xNzg4NTA5MjkwOjE3ODg1OTU2OTBfVjM)

4. Splatting Physical Scenes（2025，Robotics** **）https://arxiv\.org/abs/2506\.04120

详细：

把视觉重建、物体网格几何、机器人/相机标定、物理仿真参数放进同一个可微优化系统里联合求解，real2sim中优化采集到的rgb和3d数据，并没有显示提到物理相关属性。但本工作依托于MuJoCo MJX，也有可微渲染。使用到的物理属性是关节信息和物体几何状态。



|DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control<br>|https://arxiv\.org/abs/2603\.10448|视频生成可以成为机器人策略学习的更好 scaling proxy<br>不直接依赖最终生成的未来视频帧，而是提取视频生成过程中的中间去噪特征，再用这些特征指导动作预测。<br>|
|---|---|---|
|DreamTacVLA: Learning to Feel the Future<br>|https://arxiv\.org/pdf/2512\.23864|Hierarchical Spatial Alignment, HSA，用机器人正运动学和相机标定，把触觉传感器的位置投影到腕部视角和第三人称视角中，再用 InfoNCE 对比学习把“看到的位置”和“摸到的局部触觉”对齐<br>冻结的 V\-JEPA2 <br>Think–Dream–Act 两阶段动作生成。|
|Tactile Beyond Pixels|https://arxiv\.org/html/2506\.14754v1|面向机器人操作的多模态触觉 foundation backbone：它把 Digit 360 传感器里的 tactile image、audio、IMU/motion、pressure 四类信号融合成统一触觉表征，用于识别物理属性、插头插入、手内旋转等接触密集任务。|
|VISTA: Vision\-Grounded and Physics\-Validated Adaptation of UMI data for VLA Training|https://arxiv\.org/abs/2606\.04708|UMI数据评定适配相关<br>|
||||

可微物理引擎：MuJoCo XLA / MJX、NVIDIA Newton、Brax、Dojo



5. **灵巧手相关工作**

T\-Rex：高频触觉反应型灵巧手 VLA

**时间：2026 年 6 月**

T\-Rex 是目前较完整的“视觉语言慢规划 \+ 触觉快速反射”方案。它使用低频 VLA 专家规划整体动作，同时用高频触觉专家在一个 action chunk 内进行残差修正，解决视觉模型频率太低、无法及时处理滑动、受力和形变的问题。

平台是双臂双灵巧手系统，总计 58 DoF；数据包含 RGB、关节状态、语言指令、十个指尖的视觉触觉图像和六维力信息。论文报告在 12 个接触密集任务上平均成功率为 65%，比最强对比方法高约 30 个百分点。公开仓库包含后训练、推理、硬件采集代码及约 50 小时公开数据；完整 100 小时数据并未全部公开。

- [GitHub](https://github.com/ZhuoyangLiu2005/T-Rex)

- [论文](https://arxiv.org/abs/2606.17055)

- [项目页](https://tactile-reactive-dexterous.github.io)

**适合方向：** 触觉 VLA、接触丰富操作、双手协调、实时闭环控制。

---

Dexora：双臂双灵巧手的完整开源 VLA

**时间：2026 年 5 月，ICRA 2026**

Dexora 面向双臂、双手和高维手指控制，系统控制维度为 36 DoF。其硬件为两条 AIRBOT 机械臂和两只 XHAND，通过外骨骼采集手臂运动、Apple Vision Pro 采集手指运动。

它包含 10 万条仿真轨迹以及约 1\.22 万条真实遥操作轨迹，并使用数据质量判别器给演示片段加权，以降低抖动、碰撞和低质量遥操作数据对训练的影响。代码仓库明确发布了训练、推理、数据处理和遥操作代码，是目前复现完整度最高的项目之一。

- [GitHub](https://github.com/dexoravla/Dexora)

- [论文](https://arxiv.org/abs/2605.18722)

- [项目页](https://dexoravla.github.io/)

**适合方向：** 双臂双手 VLA、长时序任务、遥操作数据清洗、Sim2Real。

---

UniDex：基于人类第一视角视频的通用灵巧手 VLA

**时间：2026 年 3 月，CVPR 2026**

UniDex 包括三个主要部分：

- **UniDex\-Dataset**：从第一视角人类操作视频转换出机器人轨迹；

- **FAAS**：Function\-Actuator\-Aligned Space，将不同灵巧手中功能相似的关节映射到共享动作空间；

- **UniDex\-VLA**：输入点云、语言和本体状态，预测 FAAS 空间中的动作块。

数据涵盖 8 种灵巧手、超过 5 万条轨迹和约 900 万组图像—点云—动作数据。真实任务包括剪袋子、使用鼠标、喷水、冲咖啡和扫桌面，还展示了 Inspire Hand 策略向 Wuji Hand、Oymotion Hand 的零样本迁移。代码覆盖数据处理、手部重定向、预训练和微调，但完整预训练数据处理可能占用约 80 TB，复现成本较高。

- [GitHub](https://github.com/unidex-ai/UniDex)

- [论文](https://arxiv.org/abs/2603.22264)

- [项目页](https://unidex-ai.github.io/)

**适合方向：** 人类视频转机器人数据、3D VLA、跨手型迁移、工具使用。

---

XL\-VLA / DexLatent：跨灵巧手的共享潜在动作空间

**时间：2026 年 3 月，CVPR 2026 Highlight**

XL\-VLA 的核心问题是：不同灵巧手的关节数、运动学和控制接口完全不同，直接输出关节角很难联合训练。

其方法先训练每种手对应的编码器和解码器，将 Ability、Inspire、X\-Hand、Paxini 等灵巧手映射到统一的 latent action space；VLA 不再预测原始关节动作，而是预测共享潜在动作块，再由各手的解码器转换成实际关节轨迹。数据包括 4 种灵巧手、10 个任务和约 200 万个状态—动作对。

目前 GitHub 主要公开的是 **DexLatent 动作表示部分、检查点和示例数据**，不一定覆盖完整的机器人训练与部署栈。

- [GitHub](https://github.com/EmptyBlueBox/DexLatent)

- [论文](https://arxiv.org/abs/2603.10158)

- [项目页](https://xl-vla.github.io/)

**适合方向：** 多种灵巧手联合训练、跨 embodiment、通用动作 tokenizer。



[灵巧手论文，数据集，仿真](https://ccn8j7d4anqr.feishu.cn/wiki/EMhIwpE3niKhS6kp1KYc5l1qnzf)

## 四、官方仿真与可微物理平台

### 4\.1 NVIDIA 

### 4\.2  Google / DeepMind

## 五、总体技术路线

传统 VLA 通常采用：

$a_t = f_{\text{VLA}}(I_t, L)$

其中，$I_t$ 是当前视觉观测，$L$ 是语言指令，$a_t$ 是模型生成的机器人动作。



~~本文希望引入物理仿真编码器和可微物理引擎，将其扩展为：~~

$a_t^* = f_{\text{PVLA}}(I_t, L, Z_t^{phy})$

~~其中：~~

$Z_t^{phy} = E_{\text{phy}}(\tilde{S}_t, a_{t:t+H}, \hat{\tilde{S}}_{t+1:t+H}, C_{t:t+H}, J_{t:t+H}, \theta)$

~~这里：~~

- $I_t$~~：当前视觉观测；~~

- $L$~~：语言指令；~~

- $\tilde{S}_t$~~：潜在物理状态；~~

- $a_{t:t+H}$~~：候选动作序列；~~

- $\hat{\tilde{S}}_{t+1:t+H}$~~：物理引擎预测的未来状态；~~

- $C_{t:t+H}$~~：接触、碰撞、摩擦、力等交互信息；~~

- $J_{t:t+H}$~~：可微物理引擎提供的动作敏感性梯度；~~

- $\theta$~~：物性参数，如质量、摩擦、弹性、刚度、阻尼等；~~

- $Z_t^{phy}$~~：物理 token；~~

- $a_t^*$~~：最终物理一致动作。~~

    

~~整体流程为：~~

$I_t, L \rightarrow a_{t:t+H}^{0} \rightarrow F_{\theta}^{diff} \rightarrow \hat{\tau}^{phy} \rightarrow E_{\text{phy}} \rightarrow Z_t^{phy} \rightarrow a_{t:t+H}^{*}$

~~即：VLA 先生成候选动作，可微物理引擎预测动作后果，物理编码器将仿真轨迹转化为物理 token，最终 VLA 根据视觉、语言和物理 token 共同生成更可靠的动作。~~

本项目在此基础上引入显式物理状态表示、短时物理预测和动作评价机制，将系统扩展为：

$RefinementObservation \rightarrow Physical\ State/Belief \rightarrow Base\ VLA \rightarrow Candidate\ Action \rightarrow Physics\ Rollout \rightarrow Physics\ Evaluation \rightarrow Action\ Refinement$

首先，根据当前机器人、对象、接触和动力学信息构建当前物理状态：

$S_t^{phy} = \{ S_t^{robot}, S_t^{object}, S_t^{contact}, S_t^{param}, S_t^{env} \}$

并编码得到当前状态 Physics Token：

$Z_t^{state} = E_{\mathrm{state}}(S_t^{phy})$

Base VLA 根据视觉、语言以及可选的 State Physics Token 生成候选动作：

$a_{t:t+H}^{base} = f_{\mathrm{VLA}} (I_t,L,Z_t^{state})$

候选动作经过 Action Adapter 映射到机器人或仿真控制空间后，由 Physics Engine 从当前物理状态执行短时 rollout：

$\hat\tau_{t:t+H}^{phy} = F_{\theta}^{phy} (S_t^{phy},a_{t:t+H}^{base})$

其中：

$\hat\tau_{t:t+H}^{phy} = \{ \hat S_{t+1:t+H}, C_{t:t+H}, F_{t:t+H} \}$

表示预测得到的未来状态、接触和力等物理结果。

进一步将候选动作相关的预测结果编码为：

$Z_t^{rollout} = E_{\mathrm{rollout}} ( \hat\tau_{t:t+H}^{phy} )$

Physics Evaluator 根据任务目标和 rollout 计算：

$Score_{phy} = E_{\mathrm{eval}} ( \hat\tau_{t:t+H}^{phy},L )$

包括任务进展、碰撞、接触稳定性、滑移和安全风险等指标。

最终系统根据实验配置选择不同动作决策方式：

$a^* = D_{\mathrm{phy}} ( a^{base}, Z^{rollout}, Score_{phy} )$

其中可以采用候选动作 Physics Reranking：

$a^* = \arg\min_{a^k} L_{phy}(a^k)$

也可以采用局部 Residual Refinement：

$a^* = a^{base} + \Delta a^{phy}$

有限差分 Physics Gradient 作为可选增强模块，用于动作敏感性分析、physics\-refined target 生成以及物理引导优化，而不作为基础决策闭环成立的必要条件。







---



# 阶段一：物理仿真编码器设计



## 1\.1 阶段目标



第一阶段的目标是设计一个**面向 VLA 的统一物理状态表示与编码模块**，将机器人状态、对象状态、接触状态、动力学参数和环境约束转换为结构化物理表示。

首先建立当前状态编码：

$E_{\mathrm{state}}: S_t^{phy} \rightarrow Z_t^{state}$

用于描述当前物理世界处于什么状态。

同时预留 rollout 编码接口：

$E_{\mathrm{rollout}}: \hat\tau_{t:t+H}^{phy} \rightarrow Z_t^{rollout}$

用于描述执行当前候选动作后可能发生什么。

第一阶段重点建立统一物理状态接口、结构化 Physics Token 和基本监督任务，不要求依赖 Physics Gradient 或完整候选动作 rollout。



## 1\.2 统一物理状态空间设计



定义潜在物理状态：

$\tilde{S}_t = \{S_t^{robot}, S_t^{object}, S_t^{contact}, S_t^{param}, S_t^{env}\}$

其中：

### 机器人状态

$S_t^{robot} = \{q_t, \dot{q}_t, x_t^{ee}, R_t^{ee}, v_t^{ee}, g_t\}$

包括关节角、关节速度、末端位姿、末端速度和夹爪状态。



### 物体状态

$S_t^{object} = \{p_i, R_i, v_i, \omega_i\}_{i=1}^{N}$

包括每个物体的位置、姿态、线速度和角速度。



### 接触状态

$S_t^{contact} = \{p_c, n_c, f_c, \tau_c, d_c, s_c\}$

包括接触点、法向、接触力、接触力矩、穿透深度和接触状态。



### 物性参数

$S_t^{param} = \theta = \{m_i, \mu_i, e_i, I_i, k_i, c_i\}_{i=1}^{N}$

包括质量、摩擦系数、恢复系数、惯量、刚度和阻尼。仿真环境中，$\theta_t$可以直接使用 simulator parameter；真实机器人中，质量、摩擦等隐藏物性不要求获得精确值，而使用估计参数、参数范围或 Dynamics Context 表示。



### 环境约束

$S_t^{env} = \{\text{joint limits}, \text{collision constraints}, \text{workspace constraints}\}$

用于描述关节限制、碰撞约束和任务空间限制。



## 1\.3 物理 token 设计

物理编码器输出结构化物理 token 矩阵，形状为$[B,\, 7{+}N,\, 256]$ ，其中 7 为固定基础槽，为场景中可操作对象数，实际使用时 N=16，合计 23 个 token 槽：

$Z_t^{phy} = \{z_t^{robot},\; z_t^{contact},\; z_t^{dyn},\; z_t^{risk},\; z_t^{afford},\; z_t^{grad},\; z_t^{scene},\; z_t^{obj_0},\;\ldots,\; z_t^{obj_{N-1}}\}
$

当前状态 Physics Token：

$Z_t^{state} = \{ z_t^{robot}, z_t^{contact}, z_t^{scene}, z_t^{obj_0},\ldots,z_t^{obj_{N-1}} \}$

用于描述当前机器人、对象和接触状态。

候选动作相关 Physics Token：

$Z_t^{rollout} = \{ z_t^{dyn}, z_t^{risk}, z_t^{afford}, z_t^{grad} \}$

用于描述执行当前候选动作之后的动力学变化、任务风险和动作敏感性。

对象角色由 `TaskRoleParser` 从任务语言描述中自动解析，分为五类：TARGET / GOAL\_SURFACE / DISTRACTOR / STATIC / GRIPPER，编码器不依赖硬编码物体名称，具备跨任务泛化能力。

这样，VLA 不再只是接收视觉 token 和语言 token，而是接收：

$\tau_{hybrid} = [\tau_{visual}(I_t) \oplus \tau_{lang}(L) \oplus \tau_{phys}(Z_t^{phy})]
$

## 1\.4 编码器结构设计

物理编码器由五个子模块组成：

$E_{\text{phy}} = \{E_{\text{robot}}, E_{\text{obj}}, E_{\text{contact}}, E_{\text{dyn}}, E_{\text{grad}}\}
$

### 1）机器人状态编码器（StateEncoder）

专用于机器人本体感知，输入 33 维向量（7 关节角 \+ 7 关节速度 \+ EE 位置\[3\] \+ EE rot6d\[6\] \+ EE 线/角速度\[6\] \+ 夹爪\[4\]），经单层 MLP 编码为 256 维 token，并附带位姿重建解码头用于监督训练：

$z_t^{robot} = E_{\text{robot}}(q_t, \dot{q}_t, x_t^{ee}, R_t^{ee6d}, v_t^{ee}, g_t) \in \mathbb{R}^{256}
$

### 2）对象级状态编码器（ObjectEncoder）

所有可操作对象共享同一个 ObjectEncoder，以角色嵌入（role\_embedding）为条件输入，实现对不同语义角色对象的差异化编码。输入 57 维：

$v_i = [\text{role\_emb}_{32},\; p_i,\; R_i^{6d},\; v_i,\; \omega_i,\; \log(m_i),\; \log(I_i),\; \log(\mu_i),\; \text{is\_grasped},\; d_{ee},\; d_{goal}]
$

通过共享 MLP 编码为 256 维，并附带三个解码头（位姿重建、角色分类、物性回归）：

$ z_t^{obj_i} = E_{\text{obj}}(v_i,\; \text{role}_i) \in \mathbb{R}^{256}
$

场景摘要 token 由 N 个对象 token 做角色加权 mean pooling 后经线性变换得到：

$z_t^{scene} = \text{Linear}\!\left(\sum_i w_{\text{role}_i} \cdot z_t^{obj_i} \Big/ \sum_i w_{\text{role}_i}\right)$

### 3）接触集合编码器（ContactSetEncoder）

接触关系标志基于对象角色对匹配（而非名称匹配），从最多 128 个接触点中提取角色感知特征，经 per\-contact MLP 后做 masked mean pooling：

$z_t^{contact} = E_{\text{contact}}(\{p_c, n_c, f_c, \tau_c, \text{role\_flags}\}_{c=1}^{C}) \in \mathbb{R}^{256}$

接触相位由前向状态机推导（8 个相位：pre\-grasp / reaching / grasping / lifting / carrying / placing / release / done），接触事件包含 6 类（grasp\_onset / release\_onset / high\_force / slip / contact\_change / drop）。

### 4）动力学演化编码器（双路 MLP）

采用双路设计，分别编码 TARGET 对象的绝对变化和其余对象相对 TARGET 的变化：

$z_t^{dyn} = \text{Linear}_{192\to256}\!\left(\text{MLP}_{16\to128}(\Delta S_t^{TARGET}) \oplus \text{MLP\_pool}_{36\to64}(\{\Delta S_t^{others \to TARGET}\})\right)
$

P1 阶段使用 teacher\-forcing（真实未来帧差分），P3 起替换为 mujoco\_warp rollout 特征。

### 5）梯度敏感性编码器（GradientEncoder）

由于 mujoco\_warp 内部 LCP 约束求解器和碰撞检测算子缺少 Warp 反向 kernel，`wp.Tape` 对 ctrl→xpos 路径返回零梯度，**无法使用自动微分获取雅可比矩阵**。实际实现采用并行有限差分（FD）数值近似：

$\frac{\partial L_{phy}}{\partial \text{ctrl}_{t,d}} \approx \frac{L(\text{ctrl}+\varepsilon e_d) - L(\text{ctrl}-\varepsilon e_d)}{2\varepsilon}
$

构造 $K = 1 + 2 \times H \times nu$ 个并行 mujoco\_warp world（H=动作 horizon，nu=控制维度），在 RTX 3090 上 \~0\.5–1\.0s 完成单次梯度计算。GradientEncoder 将梯度向量 $\nabla_{\text{ctrl}} L_{phy} \in \mathbb{R}^{H \times nu}$ 展平后经 MLP 压缩为 256 维：

$z_t^{grad} = E_{\text{grad}}(\nabla_{\text{ctrl}} L_{phy}) \in \mathbb{R}^{256}$

它用于告诉 VLA：当前动作在哪些方向上物理代价最敏感，应如何调整以降低失败风险。P1 阶段 z\_t^\{grad\} 为全零占位 token，P4 起填入真实 FD 梯度。



## 1\.5 第一阶段训练目标



第一阶段可以先单独预训练物理编码器。损失函数为：

$\mathcal{L}_{PSE} = \lambda_1 \mathcal{L}_{state} + \lambda_2 \mathcal{L}_{dyn} + \lambda_3 \mathcal{L}_{contact} + \lambda_4 \mathcal{L}_{param} + \lambda_5 \mathcal{L}_{risk} + \lambda_6 \mathcal{L}_{afford}$

其中：

$\mathcal{L}_{state} = \|\hat{\tilde{S}}_t - \tilde{S}_t^{gt}\|^2$

$\mathcal{L}_{dyn} = \|\hat{\tilde{S}}_{t+1} - \tilde{S}_{t+1}^{gt}\|^2$

$\mathcal{L}_{contact} = \text{CE}(\hat{c}_{t+1}, c_{t+1}^{gt})$

$\mathcal{L}_{param} = \|\hat{\theta} - \theta^{gt}\|^2$

$\mathcal{L}_{risk} = \text{CE}(\hat{y}_{risk}, y_{risk})$

$\mathcal{L}_{afford} = \text{CE}(\hat{y}_{success}, y_{success})$

第一阶段最终产物是：

$Z_t^{phy} = E_{\text{phy}}(\tilde{S}_t, a_{t:t+H}, \hat{\tilde{S}}_{t+1:t+H}, C_{t:t+H}, J_{t:t+H}, \theta)$

即一个可插入 VLA 的物理 token 生成模块。



---



# 阶段二：物理编码器与可微物理引擎集成



## 2\.1 阶段目标



第二阶段的目标是将第一阶段建立的物理状态表示与具体 Physics Engine 集成，使 simulator 从离线数据提取工具转变为 VLA 决策过程中的短时物理预测与动作评价模块。



整体流程为：

$S_t^{phy}, a_{t:t+H}^{base} \rightarrow F_\theta^{phy} \rightarrow \hat\tau_{t:t+H}^{phy} \rightarrow E_{\mathrm{rollout}} \rightarrow Z_t^{rollout}$



其中，$F_{\theta}^{phy}$ 表示物理仿真器。



## 2\.2 物理引擎适配层设计



不同引擎的状态格式不同，因此需要一个统一的引擎适配器：

$\mathcal{A}_{engine} = \{\text{set\_state}, \text{step}, \text{rollout}, \text{get\_state}, \text{get\_contact}, \text{get\_force}, \text{get\_sensitivity}, \text{render}\}$

它负责把 MuJoCo / MJX / Brax / Warp / Newton / Isaac Lab 等不同系统的输出统一为标准物理轨迹：

$\hat{\tau}^{phy}_{t:t+H} = \{\tilde{S}_t, a_{t:t+H}, \hat{\tilde{S}}_{t+1:t+H}, C_{t:t+H}, F_{t:t+H}, J_{t:t+H}, \theta\}$

然后送入物理编码器：

$Z_t^{phy} = E_{\text{phy}}(\hat{\tau}^{phy}_{t:t+H})$

## 2\.3 实际引擎选型与集成

### 选型决策：mujoco\_warp 作为生产后端

理论上可按任务复杂度分层选用 MJX/Brax（原型）→ MuJoCo/MJX（操作任务）→ Warp/Isaac Lab（大规模）。但实际工程验证中，**MJX 在完整 LIBERO 操作场景（271 geom）存在严重 OOM 问题**：

|方案|结果|原因|
|---|---|---|
|完整 LIBERO XML → MJX|❌ OOM|XLA 静态分配约 17 GB，超出 RTX 3090（24 GB）|
|裁减版 MJCF（91 geom）→ MJX|❌ OOM|XLA 编译时内存仍不足|
|极简 MJCF（22 geom）→ MJX|✅ 但慢|K=8 H=30 需 26s，且缺少真实接触物理|
|**完整 LIBERO XML → mujoco\\\_warp**|**✅ 生产就绪**|CUDA 动态内核，无 XLA 静态分配；K=8 H=30 仅 0\.3s（87× 加速）|

因此，当前系统**直接采用 mujoco\_warp 3\.10\.0\.3 作为生产后端**，完整保留 271 geom，在 RTX 3090 上实现真实接触物理精度与 GPU 加速的统一。

### mujoco\_warp 关键配置

```Python
dx = mjw.put_data(mj_model, mj_data, nworld=K,
                  njmax=300, nconmax=100, naconmax=600)
# njmax/nconmax/naconmax 须显式传入，默认值对完整 LIBERO XML 会溢出
```

仿真频率 500Hz，控制频率 20Hz，每个控制步需执行 **n\_substeps=25** 个物理子步，这是保证动力学准确性的关键参数。

### 可扩展平台

框架设计了统一引擎适配器接口 $\mathcal{A}_{engine}$，后续可扩展接入 Isaac Lab（大规模 domain randomization）、NVIDIA Newton（完整可微接触）或 Brax（快速 JAX 原型验证）：

|平台|定位|
|---|---|
|**mujoco\_warp（当前）**|生产后端：完整操作场景，GPU 并行，0\.3s/K=8|
|Isaac Lab|大规模并行仿真、多模态数据生成|
|NVIDIA Newton|GPU 可微物理，支持自定义接触模型|
|Brax / MJX|轻量级 JAX 原型，接触稀疏任务|

## 2\.4 可微物理 rollout 过程

给定当前环境状态和 Base VLA 生成的候选动作，Physics Engine 需要从当前时刻重新初始化物理环境，并对未来短时间窗口内的动作执行结果进行预测。该过程的目标不是进行长时间世界预测，而是获得与当前候选动作直接相关的短时动力学、接触和风险信息，为后续 Physics Evaluator、Physics Reranking 和 Physics Refiner 提供依据。

首先，Base VLA 根据视觉观测、语言指令以及可选的 State Physics Token 生成一个长度为H的候选 action chunk：

$a_{t:t+H}^{base} = f_{\mathrm{VLA}} (I_t,L,Z_t^{state})$

其中，

$a_{t:t+H}^{base} = \{a_t,a_{t+1},\ldots,a_{t+H-1}\}$

表示未来H个控制步的候选动作序列。

### 1）获取当前物理状态

Physics Rollout 的起点必须与当前环境状态保持一致。

在仿真环境中，直接通过 Simulator State Provider 读取 simulator privileged state：

$S_t^{sim} = \{ S_t^{robot}, S_t^{object}, S_t^{contact}, \theta_t, S_t^{env} \}$

包括机器人 qpos/qvel、末端状态、对象位姿和速度、当前接触状态以及 simulator 中定义的质量、摩擦、惯量等参数。

因此仿真环境无需通过视觉重新估计这些物理状态。

在真实机器人环境中，则由 Physical Belief Estimator 根据一段时间内的传感器观测和历史动作估计：

$b_t^{real} = E_{\mathrm{belief}} ( O_{t-k:t}, a_{t-k:t-1} )$

其中：

$O_t = \{ RGB/RGBD, proprioception, gripper, F/T, tactile \}$

具体输入取决于机器人实际传感器配置。

Physical Belief 不仅包含状态估计值，还应包含对应的不确定性：

$b_t^{real} = \{ \hat S_t, \Sigma_{S_t}, \hat\theta_t, \Sigma_{\theta_t} \}$

随后通过 Digital Twin Synchronization 将该 belief 映射并同步到 mujoco\_warp 中。

因此仿真和真机最终统一得到：

$S_t^{interface}$

作为 Physics Engine 的初始状态。

---

### 2）候选动作到物理控制空间的映射

Base VLA 输出的动作通常位于机器人任务空间，例如当前系统中的 7D EE delta action：

$a_t^{VLA} = [ \Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z, g ]$

而 mujoco\_warp 实际执行的控制量可能位于关节控制空间，因此需要通过 Action Adapter 完成：

$u_t = A_{\mathrm{action}} ( S_t, a_t^{VLA} )$

其中：

$u_t\in\mathbb R^{n_u}$

表示 simulator 的控制输入。

当前 LIBERO 系统中，VLA 输出为 7D 末端增量动作，而 mujoco\_warp 接收 9D robot control，因此 Action Adapter 负责完成 EE action 到 joint/control space 的转换。当前工程中这一过程采用基于机器人 Jacobian 的近似映射，后续需要进一步验证其与真实机器人控制器之间的一致性。当前 Demo 也正是按照 7D VLA action 到 9D simulator control 的方式执行 rollout。

对整个 action chunk：

$a_{t:t+H}^{base}$

映射后得到：

$u_{t:t+H} = \{ u_t, u_{t+1}, \ldots, u_{t+H-1} \}$

作为 Physics Engine 的实际输入。

---

### 3）短时物理 rollout

将当前状态写入 mujoco\_warp：

$S_t^{warp} \leftarrow S_t^{interface}$

随后依次执行候选控制序列：

$u_t, u_{t+1}, \ldots, u_{t+H-1}$

得到：

$\hat S_{t+1} = F_{\theta}^{phy} ( S_t,u_t )$

以及递归过程：

$\hat S_{t+h+1} = F_{\theta}^{phy} ( \hat S_{t+h}, u_{t+h} )$

最终得到完整短时预测轨迹：

$\hat\tau_{t:t+H}^{phy} = \{ S_t, u_{t:t+H}, \hat S_{t+1:t+H} \}$

当前 mujoco\_warp 配置下，仿真频率为 500 Hz，而 VLA 控制频率为 20 Hz，因此一个控制周期内部需要执行：

$n_{\mathrm{substeps}}=25$

个物理子步：

$\Delta t_{control} = 25\Delta t_{physics}$

该设置用于保证每个 VLA action 在高频动力学环境中得到充分展开，而不是将一个 20 Hz 动作简单对应到单次 simulator step。

---

### 4）提取 rollout 物理结果

在每个预测时间步中，除了保存机器人和对象状态，还需要从 Physics Engine 中提取与任务执行相关的物理结果。

定义：

$\hat\tau_{t:t+H}^{phy} = \{ \hat S_{t+1:t+H}, C_{t:t+H}, F_{t:t+H}, E_{t:t+H} \}$

其中：

$\hat S_{t+1:t+H}$

表示机器人和对象的未来状态，包括：

- 机器人 qpos/qvel；

- EE pose；

- 对象 position/orientation；

- 对象 linear/angular velocity。

$C_{t:t+H}$

表示未来接触信息，包括：

- 是否发生接触；

- 接触对象；

- 接触位置；

- 接触法向；

- 接触数量；

- 接触关系变化。

$F_{t:t+H}$

表示可获得的接触力、力矩等物理反馈。

另外可以从连续状态变化中进一步计算事件：

$E_{t:t+H} = \{ collision, grasp, release, slip, drop, contact\_change \}$

这些量共同描述如果当前 Base VLA 的候选动作真正执行，未来短时间内物理世界可能发生什么。

---

### 5）生成 Rollout Physics Token

Rollout 结果不直接全部输入 VLA，而是由 Physical Tokenizer 压缩为结构化的候选动作相关表示：

$Z_t^{rollout} = E_{\mathrm{rollout}} ( \hat\tau_{t:t+H}^{phy} )$

其中可以包括：

$Z_t^{rollout} = \{ z_t^{dyn}, z_t^{risk}, z_t^{afford} \}$

分别描述：

- $z^{dyn}$：候选动作导致的动力学变化；

- $z^{risk}$：碰撞、滑移、掉落、失稳等风险；

- $z^{afford}$：动作对当前任务目标的物理可执行性。

需要特别区分$Z_t^{state}$和$Z_t^{rollout}$前者在候选动作生成前即可获得，描述当前物理世界；后者必须依赖：

$a_{t:t+H}^{base}$

---

### 6）Physics Evaluator

Physics Engine 本身只负责预测：“执行这个动作会发生什么。”

而 Physics Evaluator 负责判断：“这个结果对当前任务而言是否合理。”

因此对 rollout 轨迹计算：

$L_{phy} = \lambda_pL_{progress} + \lambda_cL_{collision} + \lambda_sL_{stability} + \lambda_{contact}L_{contact} + \lambda_dL_{deviation}$

其中：

$L_{progress}$评价候选动作是否推动任务向目标状态发展；

$L_{collision}$评价机器人和环境中的非法碰撞；

$L_{stability}$评价物体掉落、倾倒或异常运动；

$L_{contact}$评价抓取、推动或放置等任务所需的接触状态是否合理；

$L_{deviation}$限制 Physics Refiner 对 Base VLA 动作进行过大的修改。

特别需要保留$L_{progress}$，否则如果物理代价仅包含碰撞、漂移和平滑性，则：a=0

可能成为一个“非常安全但完全不完成任务”的低损失解。

最终得到候选动作物理评分：

$Score_{phy} = E_{\mathrm{eval}} ( \hat\tau_{t:t+H}^{phy}, L )$



~~首先通过视觉状态估计器获得潜在物理状态：~~

$\tilde{S}_t = E_{\text{state}}(I_{t-k:t})$

~~然后送入可微物理引擎：~~

$\hat{\tilde{S}}_{t+1:t+H} = F_{\theta}^{diff}(\tilde{S}_t, a_{t:t+H}^{0})$

~~同时提取：~~

$C_{t:t+H} = \text{ContactExtractor}(F_{\theta}^{diff})$

$J_{t:t+H} \approx \frac{L(\text{ctrl}+\varepsilon e_d) - L(\text{ctrl}-\varepsilon e_d)}{2\varepsilon} \quad \text{（有限差分，并行 } K{=}1{+}2H{\cdot}nu \text{ worlds）}$

> **~~注~~**~~：mujoco\_warp 内部 LCP 求解器和碰撞检测缺少 Warp 反向 kernel，~~~~`wp.Tape`~~~~ 对 ctrl→xpos 路径返回零梯度，因此梯度通过~~**~~并行有限差分~~**~~数值计算，而非自动微分。~~
> 
> 

~~最后得到物理 token：~~

$Z_t^{phy} = E_{\text{phy}}(\tilde{S}_t, a_{t:t+H}^{0}, \hat{\tilde{S}}_{t+1:t+H}, C_{t:t+H}, J_{t:t+H}, \theta)$

## 2\.5 动作评估与修正

物理 token 不只是作为额外输入，也可以用于动作修正：

$\Delta a_{t:t+H} = R_{\omega}(Z_t^{phy}, Z_t^{vis}, Z_t^{lang})$

$a_{t:t+H}^{*} = a_{t:t+H}^{0} + \Delta a_{t:t+H}$

其中 $R_{\omega}$ 是物理约束动作修正器。

也可以使用物理代价对多个候选动作进行重排序：

$a^* = \arg\min_{a^k} \mathcal{L}_{phy}(F_{\theta}^{diff}(\tilde{S}_t, a^k))$

这使得推理阶段有三种可选模式：

1. **物理 token 增强推理**：速度最快；

2. **候选动作仿真重排序**：最稳健；

3. **梯度动作修正**：最能体现可微物理价值。

    

## 2\.6 第二阶段最终产物



第二阶段结束后，应得到：

1. **统一物理引擎适配器**

$\mathcal{A}_{engine}$

2. **可微物理 rollout 模块**

$\hat{\tau}^{phy}_{t:t+H} = F_{\theta}^{diff}(\tilde{S}_t, a_{t:t+H})$

3. **物理 token 生成模块**

$Z_t^{phy} = E_{\text{phy}}(\hat{\tau}^{phy}_{t:t+H})$

4. **动作物理评估与修正模块**

$a^* = a^0 + R_{\omega}(Z_t^{phy})$



---



# 阶段三：物理增强 VLA 训练

## 3\.1 阶段目标

第三阶段的目标是训练一个真正使用物理 token 的 VLA 模型，使其能够基于视觉、语言和物理信息共同生成动作。

传统 VLA：

$a_t = f_{\text{VLA}}(I_t, L)$

物理增强 VLA：

Base proposal：

$a^{base} = f_{VLA} (I,L,Z^{state})$

Physics decision：

$a^* = D_{phy} ( a^{base}, Z^{rollout}, Score_{phy} )$



## 3\.2 训练数据构建



训练数据应包含：

$\mathcal{D} = \{I_t, L, a_t^{gt}, \tilde{S}_t, \tilde{S}_{t+1}, C_t, \theta, y_{success}, y_{risk}\}$

其中：

- $I_t$：图像或视频；

- $L$：语言指令；

- $a_t^{gt}$：专家动作；

- $\tilde{S}_t$：物理状态；

- $\tilde{S}_{t+1}$：下一物理状态；

- $C_t$：接触状态；

- $\theta$：物性参数；

- $y_{success}$：任务成功标签；

- $y_{risk}$：风险标签。

    

数据来源分为三类：

|数据来源|作用|
|---|---|
|仿真数据|提供完整物理状态、接触、力和物性标注|
|真实机器人数据|提供真实视觉、真实动作和真实执行结果|
|仿真\-真实对齐数据|用于 sim\-to\-real 对齐和系统辨识|

## 3\.3 训练流程

### 第一步：物理编码器预训练

先固定物理引擎，训练物理编码器：

$Z_t^{state} = E_{\text{phy}}(\hat{\tau}^{phy}_{t:t+H})$

训练目标是让物理 token 能预测：

- 未来状态；

- 接触事件；

- 动作成功率；

- 风险程度；

- 物性参数；

- 动作可供性。

损失函数：

$\mathcal{L}_{PSE} = \lambda_1 \mathcal{L}_{dyn} + \lambda_2 \mathcal{L}_{contact} + \lambda_3 \mathcal{L}_{param} + \lambda_4 \mathcal{L}_{risk} + \lambda_5 \mathcal{L}_{afford}$

这一步的目标是让 $Z_t^{phy}$ 具备明确物理语义，而不是普通 latent vector。



### 第二步：VLA 物理 token 融合训练

将视觉 token、语言 token 和物理 token 融合：

$\tau_{hybrid} = [\tau_{visual} \oplus \tau_{lang} \oplus \tau_{phys}]$

输入 VLA：

$a_t = f_{\text{VLA}}(\tau_{hybrid})$

行为克隆损失：

$\mathcal{L}_{BC} = \|a_t - a_t^{gt}\|^2$

连续动作形式：

$\mathcal{L}_{BC} = \|a_t - a_t^{gt}\|^2$

离散 action token 形式：

$\mathcal{L}_{BC} = - \sum_i y_i \log p(a_i)$

这一步的目标是让 VLA 学会在相同视觉和语言条件下，根据物理 token 生成更可执行、更安全、更稳定的动作。



### 第三步：可微物理闭环优化

VLA 生成动作：

$a_{t:t+H}^{0} = f_{\text{VLA}}(I_t, L, Z_t^{phy})$

送入可微物理引擎：

$\hat{\tilde{S}}_{t+1:t+H} = F_{\theta}^{diff}(\tilde{S}_t, a_{t:t+H}^{0})$

计算物理损失：

$\mathcal{L}_{phy} = \mathcal{L}_{task} + \mathcal{L}_{safe} + \mathcal{L}_{contact} + \mathcal{L}_{smooth} + \mathcal{L}_{energy}$

梯度通过可微物理引擎反传回 VLA：

$\frac{\partial \mathcal{L}_{phy}}{\partial \psi} = \frac{\partial \mathcal{L}_{phy}}{\partial \hat{\tilde{S}}} \frac{\partial \hat{\tilde{S}}}{\partial a} \frac{\partial a}{\partial \psi}$

其中 $\psi$ 是 VLA 参数。

如果 VLA 生成的动作在物理仿真中失败，失败信号可以通过可微物理引擎反向传播，使模型学会避免物理不可行动作。



## 3\.4 总训练目标

最终训练损失为：

$\mathcal{L} = \mathcal{L}_{BC} + \alpha \mathcal{L}_{task} + \beta \mathcal{L}_{dyn} + \gamma \mathcal{L}_{contact} + \delta \mathcal{L}_{safe} + \eta \mathcal{L}_{align} + \rho \mathcal{L}_{smooth} + \xi \mathcal{L}_{energy}$

### 行为克隆损失

$\mathcal{L}_{BC} = \|a_t - a_t^{gt}\|^2$

用于模仿专家动作。

### 任务完成损失

$\mathcal{L}_{task} = \|\hat{x}_{T} - x_{goal}\|^2$

用于约束最终任务目标。

### 动力学一致性损失

$\mathcal{L}_{dyn} = \|\hat{\tilde{S}}_{t+1} - \tilde{S}_{t+1}^{gt}\|^2$

用于约束仿真预测和真实状态一致。

### 接触一致性损失

$\mathcal{L}_{contact} = \text{CE}(\hat{c}_{t+1}, c_{t+1}^{gt})$

用于约束接触事件合理。

### 安全约束损失

$\mathcal{L}_{safe} = \lambda_c \mathcal{L}_{collision} + \lambda_f \mathcal{L}_{force} + \lambda_j \mathcal{L}_{joint} + \lambda_o \mathcal{L}_{object}$

用于约束碰撞、过大接触力、关节超限、物体倾倒或滑落。

### 视觉\-物理对齐损失

$\mathcal{L}_{align} = \|Z_t^{vis} - Z_t^{phy}\|^2$

用于让视觉表征和物理表征对齐。

### 动作平滑损失

$\mathcal{L}_{smooth} = \sum_t \|a_t - a_{t-1}\|^2$

用于生成平滑动作轨迹。



---



# ~~三阶段总结~~

## ~~第一阶段：构建物理仿真编码器~~

~~首先，建立统一物理状态空间，显式表示机器人状态、物体状态、接触关系、物性参数和环境约束。进一步设计物理仿真编码器 ~~$E_{\text{phy}}$~~，将连续物理状态、接触事件、动力学演化和动作敏感性编码为结构化物理 token（\[B, 23, 256\]，7 基础槽 \+ 16 对象槽）：~~

$Z_t^{phy} = E_{\text{phy}}(\tilde{S}_t, a_{t:t+H}, \hat{\tilde{S}}_{t+1:t+H}, C_{t:t+H}, J_{t:t+H}^{FD}, \theta)$

~~其中 ~~$J_{t:t+H}^{FD}$~~ 表示通过并行有限差分计算的动作敏感性梯度（非自动微分）。该阶段产物是可与视觉 token、语言 token 融合的结构化物理 token 矩阵，7 任务验收全部通过（phase\\\_acc=1\.0，contact\\\_f1\>0\.97，role\\\_acc=1\.0）。~~

## ~~第二阶段：接入可微物理仿真引擎~~

~~然后，以 ~~**~~mujoco\\\_warp 为生产后端~~**~~（MJX 因 OOM 放弃），构建物理引擎适配层，将仿真平台的状态、接触、力和梯度接口统一到标准物理轨迹表示中。给定 VLA 生成的候选动作，mujoco\\\_warp 执行短时域并行 rollout（271 geom，K=8 H=30 → 0\.3s），预测动作后果并提取物理反馈：~~

$\hat{\tau}^{phy}_{t:t+H} = F_{\text{warp}}(\tilde{S}_t, a_{t:t+H}^{0},\; n\_substeps{=}25)$

~~梯度 ~~$J_{t:t+H}$~~ 通过并行有限差分（K=1\+2H·nu 个 world）数值计算，再由物理编码器生成物理 token：~~

$Z_t^{phy} = E_{\text{phy}}(\hat{\tau}^{phy}_{t:t+H})$

~~该阶段使物理引擎从外部数据生成工具转变为 VLA 动作生成闭环中的物理反馈模块（梯度通过 FD \+ Surrogate Loss 注入神经网络）。~~

## ~~第三阶段：训练物理增强 VLA~~

~~最后，将视觉 token、语言 token 与物理 token 融合，训练物理增强 VLA（选定 StarVLA / Qwen2\.5\-VL\-GR00T\-LIBERO\-4in1）：~~

$a_t^* = f_{\text{PVLA}}(I_t, L, Z_t^{phy})$

~~训练包含三步：① 物理编码器预训练（已完成，7任务ALL PASS）；② VLA 行为克隆训练——冻结 Qwen2\.5\-VL\-3B backbone，通过 PhysicsTokenProjector（\~4\.77M参数）将 z\\\_phy 注入 DiT action head，以自蒸馏目标训练（P5 进行中）；③ 可微物理闭环优化——通过 FD 梯度 \+ Surrogate Loss 将 mujoco\\\_warp 物理代价反传至神经网络参数（P5\+A/B 待实现）。联合损失包括行为克隆损失（~~$\mathcal{L}_{VLA}$~~）、物理代价代理损失（~~$\mathcal{L}_{surrogate}$~~）以及后续的动力学一致性、安全约束、视觉\-物理对齐等扩展项。~~



---

# Demo

## 当前完整训练流程

### 原始数据与离线准备

整个系统的训练起点是 LIBERO 基准测试中的专家演示数据，以 HDF5格式存储。每一条演示包含机器人完成特定操作任务（例如"从桌面中央拿起黑碗并放到盘子上"）的完整轨迹，具体内容包括：每一帧的 RGB相机图像、关节位置与速度、末端执行器的状态，以及专家控制信号序列。除了 HDF5 文件，系统还需要与之对应的 MuJoCo XML 场景描述文件，其中定义了 271个几何体、所有关节参数、碰撞属性、质量与摩擦系数。

在进入任何神经网络之前，必须先完成一系列离线的物理数据提取工作。这一步由 LiberoPhysicsAdapter 完成，核心做法是将 HDF5 中存储的 MuJoCo状态（qpos、qvel）逐帧注入仿真环境，使其精确复现演示时的物理状态，而不依赖动作回放（否则会积累漂移误差）。每一帧提取三类物理量：机器人状态（七个关节的角度与速度、末端执行器在世界坐标系下的位置与姿态、夹爪开合状态，共 33 维）；场景中所有可操作对象的状态（位姿、速度、质量、惯量、摩擦系数）；以及当前帧的接触集合（最多 128个接触点，每个接触点包含空间位置、法向量、世界坐标系下的力与力矩，以及标识接触双方角色关系的布尔标志，共 28 维）。这些数据帧对齐地保存为 NPZ文件，构成整个下游管线的物理真相来源。

在物理数据准备好之后，下一步是将其编码为结构化的物理 token。系统首先通过 TaskRoleParser解析任务的自然语言描述，将场景中的每个物体分配为五种语义角色之一：TARGET（需要操作的主目标）、GOAL\_SURFACE（放置目标面）、DISTRACTOR（干扰物）、STATIC（静态环境）或GRIPPER（机器人末端）。这个角色分配使得编码器不再依赖特定物体名称，具备跨任务的泛化能力。随后，PhysicalTokenizer 将原始物理量转换为固定形状的 token 矩阵，输出维度为 \[T, 23,256\]，其中 T 是时间步数，23 是 token槽位数（7 个基础槽加上最多 16 个对象槽），256 是每个 token的特征维度。七个基础槽分别承载机器人运动学（z\_robot）、角色感知接触集合（z\_contact）、动态趋势（z\_dyn）、风险信号（z\_risk）、任务可供性与阶段（z\_afford）、动作敏感方向（z\_grad，训练早期用零向量占位）以及场景摘要（z\_scene\_ctx）；其余对象槽则由共享的 ObjectEncoder 以角色嵌入为条件，独立处理每个物体的完整物理状态。

在神经网络训练正式开始之前，还需要完成两项一次性的预处理工作。第一项是运行原始 StarVLA（即没有物理 token 输入的基础模型）对每一帧进行推理，将其输出的动作预测 a\_vla \[T, 8, 7\] 保存下来，作为后续训练的自蒸馏目标。使用这个目标而非原始专家动作的原因在于：StarVLA 已经在LIBERO 上微调，其预测本身质量较高，且与 DiT action head 的训练空间（归一化的7D EE delta分布）完全对齐，避免了直接使用专家动作时需要处理的归一化不匹配问题。第二项预处理是让Qwen2\.5\-VL\-3B视觉语言编码器（backbone永远冻结）对所有演示帧的图像与语言输入进行一次性编码，将输出的最后一层隐藏状态 last\_hidden \[T, L, 2048\] 以float16 精度存储到磁盘。由于 VLM权重在整个训练过程中不会更新，相同的图像和语言输入必然产生相同的隐藏状态，因此完全没有必要在每个epoch 重新计算。这一缓存策略将训练时间从数小时压缩到数分钟。

### 训练前向传播

离线准备阶段结束后，进入正式的在线训练循环。每个 batch 从三个来源加载数据：从physics\_tokens\_demo0\.npz 中读取当前帧的物理 token z\_phy \[B, 23, 256\]，从 vlm\_hidden\_demo0\.npz中读取预缓存的 VLM 隐藏状态 h\_cache \[B, L, 2048\]，从 vla\_actions\_demo0\.npz 中读取训练目标动作 a\_target \[B, 8, 7\]。此外还需要加载当前帧的关节状态 qpos 和qvel，供后续的物理仿真使用。

前向传播的第一步是物理 token 投影。PhysicsTokenProjector 是整个系统中参数量最小但位置最关键的模块，它接收 z\_phy \[B, 23, 256\]，通过两层全连接网络（中间使用 GELU 激活）和LayerNorm，将每个 token 的维度从 256 提升到 2048，与Qwen2\.5\-VL\-3B 的隐藏维度对齐，输出 z\_proj \[B, 23, 2048\]。每个 token 还加上一个可学习的位置编码，使 DiT action head能够通过 cross\-attention 区分各个物理槽位的语义含义——它能学会slot 0 代表机器人运动状态，slot 7以后代表各个对象的物理属性，等等。这一步是整个训练过程中梯度可以流经的核心通道，projector 的参数正是通过这里被更新。

前向传播的第二步是上下文拼接。将预缓存的视觉语言隐藏状态与投影后的物理 token 在序列维度上拼接：context = concat\(\[h\_cache, z\_proj\], dim=1\)，得到形状为 \[B, L\+23, 2048\]的增强上下文。注意力掩码也做同样的扩展，物理 token 对应的位置全部标记为有效，确保 DiT action head 的cross\-attention 不会忽略物理 token。由于 h\_cache 来自冻结的VLM，其对应的参数不进入计算图，梯度只会通过 z\_proj 部分反传到 projector，而不会流向 VLM backbone，这正是冻结设计的意图所在。

前向传播的第三步，也是生成动作预测的一步，是将增强上下文送入 DiT flow\-matching action head。DiT 通过对增强上下文做cross\-attention，结合当前关节状态条件，以流匹配（flow\-matching）的去噪过程生成动作序列预测 a\_pred \[B, 8, 7\]，即一个包含 8 步的动作 chunk，每步 7 维（3 维位置增量、3维姿态增量、1 维夹爪控制）。同时，action head 在训练模式下计算 flow\-matching 损失，即预测的速度场与目标方向之间的均方误差，这个损失直接衡量 action head预测动作的质量，构成总损失的第一部分。

### 物理损失计算

仅有行为克隆损失是不够的——它只能让模型模仿已有的VLA 预测，却无法保证动作在物理上是有效的。为此，系统将a\_pred 送入 mujoco\_warp 仿真环境，获取物理反馈信号。

具体过程是：将 a\_pred 的前几步（或全部 8 步，取决于 horizon设置）通过 action mapper 从 VLA 的 7D 末端执行器增量空间映射到 mujoco\_warp 需要的 9D关节控制空间。这一映射当前使用简化的伪逆雅可比方法近似，完整的可微映射（DiffActionMapper）是后续待完成的工作。得到控制序列 ctrl \[B, H, 9\] 之后，以当前帧的qpos 和 qvel作为初始状态，让 mujoco\_warp 对 B 个world 并行运行 H 步物理仿真，每个控制步内部以 25 个物理子步长运行（仿真频率 500Hz，控制频率20Hz），输出各帧的物体位置轨迹。基于这条rollout 轨迹，计算物理代价函数：平滑损失惩罚控制序列的抖动，漂移损失要求场景干扰物保持静止。（当前仍需继续优化）



### 梯度计算与反传

计算 L\_phy 之后，最核心的问题浮现：如何将这个物理代价信号传回神经网络的参数？

直接的方法是调用 wp\.Tape\(\)\.backward\(L\_phy\) 进行自动微分，但这条路在实践中走不通。mujoco\_warp 内部的 LCP 约束求解器（负责计算接触力）、碰撞检测以及 Cholesky分解等核心算子都没有注册 Warp 框架所需的反向 kernel，导致梯度在这些算子处完全截断为零。因此，系统采用有限差分（FD）方法来数值计算 ∂L\_phy/∂ctrl：构造K = 1 \+ 2×H×nu 个并行world，分别对控制序列的每一维施加正向和负向微小扰动，然后用中心差分公式近似梯度。mujoco\_warp 的批量并行能力使这一计算可以在 GPU 上同时完成，在 RTX 3090 上花费约 0\.1 到 0\.3秒。

通过 FD 得到的是关节控制空间中的梯度 grad\_ctrl \[B, H, 9\]，还需要通过雅可比转置将其映射回 VLA 输出的 7D EE 空间，得到 grad\_ee \[B, H, 7\]。这个向量的物理意义非常明确：对于batch 中每一帧，它指明了预测动作应该沿哪个方向调整，才能降低 mujoco\_warp 仿真中的物理代价。

然而，grad\_ee 是通过数值方法（FD）计算的，它不存在于 PyTorch 的计算图中，不能直接用于backward\(\)。解决方案是构造一个 Surrogate Loss（代理损失）：将 a\_pred 与grad\_ee\.detach\(\) 做点积求和，得到 L\_surrogate = \(a\_pred \* grad\_ee\.detach\(\)\)\.sum\(\)。当对这个 surrogate loss 做反向传播时，grad\_ee 作为固定的外部信号，而 ∂L\_surrogate/∂θ =\(∂a\_pred/∂θ\) · grad\_ee，等价于把 grad\_ee 作为"来自物理环境的上游梯度"注入到 action head 和 projector 的参数更新中。这是 Straight\-Through Estimator 思想的一个应用：用 FD梯度"绕过"不可微的仿真步骤，将物理信息以梯度的形式注入可微的神经网络路径。

最终的总损失是行为克隆损失与代理物理损失的加权和，L\_total = L\_VLA \+ λ · L\_surrogate。对L\_total 执行 backward\(\) 后，梯度沿着 a\_pred → DiT action head → concat → z\_proj →PhysicsTokenProjector 这条路径反传。优化器只收到 requires\_grad=True 的参数（projector 的权重，以及若解冻则包含 DiT 的权重），VLM backbone的参数不会被触动。参数更新完成后，进入下一个batch 的循环。

### z\_grad 反馈闭环

除了在当前帧的反传中更新参数外，FD 梯度还有另一个用途：编码为 z\_grad token，作为下一帧物理 token 的第5 个槽位。GradientEncoder 接收 grad\_ctrl \[H, nu\]，将其展平后通过 MLP压缩为 256 维向量，填入 z\_phy\[:, 5, :\]。这意味着在时间步 t\+1 的前向传播中，action head 通过 cross\-attention 可以"读到"上一帧的动作梯度方向——它隐式地学习到"当前动作在哪个方向上物理效果欠佳，应该如何调整"。这构成了一个跨帧的物理信息反馈闭环，使模型不仅在参数层面被物理梯度引导，还在输入表示层面感知到物理反馈历史。

### 推理时在线精炼

训练完成后，在部署推理阶段还可以利用同一套mujoco\_warp \+ FD 机制对VLA 预测进行在线精炼。对于给定帧的初始动作预测 a\_pred，执行若干轮梯度下降：计算 L\_phy，通过 FD 得到grad，然后以归一化步长更新动作 a\_pred ← a\_pred \- s · grad / \|\|grad\|\|，直到物理代价不再显著下降或达到最大迭代次数。P4 阶段已在 task05 上验证了这一机制，5 次迭代后 loss 从0\.2357 降至 0\.2342，下降 0\.64%。这个幅度看起来不大，但其来源是专家轨迹的微调——若从次优的初始预测出发，改进空间会显著增大，这正是将VLA（负责提供语义合理的动作）与可微物理仿真（负责在物理上精炼动作）结合的核心价值所在。

## mujoco与mujoco\_wrap环境

当前完整项目均位于/home/jiangnan\-luxitech/workspace/phyvla中

\[replay\_warp\_full\.mp4\]



