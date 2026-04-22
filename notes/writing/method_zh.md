# 方法（中文草稿）

> 状态：初稿 v1 — 待用户审阅
> 最后更新：2026-04-22

---

## III. ProtoBasis-Net

本节介绍 ProtoBasis-Net 的完整架构（见图1）。给定场景内所有飞机的历史轨迹，模型为目标飞机生成 $K=20$ 条多模态候选轨迹及对应置信度分数。整体流程分为五个阶段：坐标归一化与特征提取、时序编码、社会交互聚合、原型路由与基分解解码、时序残差精炼。

### A. 问题定义

设场景内共有 $l$ 架飞机（含目标机），第 $i$ 架飞机的历史轨迹为 $\mathbf{X}_i \in \mathbb{R}^{T_h \times 3}$，其中 $T_h = 40$（40秒，1 Hz 采样）。目标是预测目标飞机（$i=1$）未来 $T_f = 120$ 步的 $K$ 条候选轨迹 $\hat{\mathbf{Y}} \in \mathbb{R}^{K \times T_f \times 3}$ 及对应概率分数 $\mathbf{s} \in \mathbb{R}^K$。场景内飞机数量 $l$ 在 1 到 7 之间，不足时以零填充。

### B. 坐标归一化

飞机轨迹以静态 ADS-B 接收机为参考系记录，直接使用全局坐标会引入位置偏差，不利于学习局部运动模式（如转弯、爬升）。受 ASCENT [cite:ascent] 启发，我们对每架飞机的历史轨迹进行位姿归一化。

具体地，以目标飞机最后一个观测时刻的位置为原点，利用最后两步的位移方向估计飞机的偏航角 $\gamma$ 和俯仰角 $\theta$：

$$\gamma = \text{atan2}(\Delta y, \Delta x), \quad \theta = \text{atan2}(\Delta z, \sqrt{\Delta x^2 + \Delta y^2})$$

随后构造旋转矩阵 $\mathbf{R} \in \mathbb{R}^{3 \times 3}$，将所有飞机的轨迹平移至原点并旋转至目标机朝向，得到局部坐标系下的轨迹 $\mathbf{X}^{\text{local}}$。同时保留全局坐标 $\mathbf{X}^{\text{global}}$ 用于提取全局位置特征。

在局部坐标系下，我们进一步提取运动特征：逐步速度向量、速度大小、偏航角和俯仰角的正弦/余弦值，构成 8 维局部特征 $\mathbf{f}^{\text{local}} \in \mathbb{R}^{T_h \times 8}$；在全局坐标系下提取 7 维全局特征 $\mathbf{f}^{\text{global}} \in \mathbb{R}^{T_h \times 7}$（含 3D 位置和角度正余弦）。

### C. 时序编码器

时序编码器将每架飞机的历史运动特征压缩为固定维度的特征向量。局部特征和全局特征分别经线性投影至 48 维后拼接，加入可学习的时间位置编码，送入 3 层 Transformer 编码器（$d_{\text{model}}=96$，4 头注意力，Pre-LN 结构）。对序列输出取最大池化与最后时刻特征拼接，经线性层和 LayerNorm 得到每架飞机的特征向量 $\mathbf{a}_i \in \mathbb{R}^{96}$。

### D. 社会交互聚合

飞机在终端空域内并非独立运动——进近、离场和盘旋的飞机之间存在隐式的交通规则约束。为此，我们设计了目标-邻机交叉注意力聚合模块（TargetSocialAggregator）。

以目标机特征 $\mathbf{a}_1$ 为查询（Query），以场景内所有飞机特征 $\{\mathbf{a}_i\}$ 为键值（Key/Value），经 2 层多头交叉注意力（带 padding mask 屏蔽缺失飞机）聚合邻机信息，得到社会感知的目标机特征 $\mathbf{c}_{\text{target}} \in \mathbb{R}^{96}$。同时对所有飞机特征取掩码均值，得到场景级上下文 $\mathbf{c}_{\text{scene}} \in \mathbb{R}^{96}$。

这一设计使模型能够感知周围飞机的运动意图，而 ASCENT [cite:ascent] 和 GooDFlight [cite:goodflight] 均未对社会交互进行显式建模。

### E. 原型路由

飞机在终端空域的运动模式具有强结构性：进近、离场、左/右盘旋等模式在轨迹空间中形成明显的聚类。ProtoBasis-Net 利用这一先验，在训练前对所有训练轨迹进行 K-Means 聚类，得到 $N_p = 64$ 个原型轨迹，并为每个原型计算 5 维摘要特征（终点坐标 $x, y, z$、轨迹长度、频率）。

在推理时，原型路由器（PrototypeRouter）将目标机特征 $\mathbf{c}_{\text{target}}$ 与场景上下文 $\mathbf{c}_{\text{scene}}$ 融合，通过 MLP 计算对所有 64 个原型的 logit 分数，选取 top-$K_p = 5$ 个最相关原型。每个原型的 5 维摘要经线性投影后与目标机特征相加，得到原型条件化的查询特征 $\mathbf{q} \in \mathbb{R}^{5 \times 96}$。同时，路由器预测每个原型对应的终点残差 $\Delta \mathbf{e} \in \mathbb{R}^{5 \times 3}$，加到原型中心终点上，得到精细化的终点预测 $\mathbf{e} \in \mathbb{R}^{5 \times 3}$。

### F. 微模式扩展与基分解解码

仅依赖 5 个原型无法充分覆盖轨迹的多样性。为此，我们在每个原型下进一步扩展出 $N_m = 4$ 个微模式，共生成 $K = N_p \times N_m = 20$ 条候选轨迹。

**SVD 基分解**：对每个原型的训练轨迹集合进行奇异值分解（SVD），提取前 $M = 16$ 个主成分作为基向量，构成基向量库 $\mathbf{B} \in \mathbb{R}^{M \times T_f \times 3}$。任意轨迹均可表示为基向量的线性组合加上锚点轨迹：

$$\hat{\mathbf{y}} = \mathbf{a}_{\text{anchor}} + \sum_{m=1}^{M} c_m \mathbf{b}_m$$

其中锚点轨迹 $\mathbf{a}_{\text{anchor}} \in \mathbb{R}^{T_f \times 3}$ 由预测终点线性插值得到（从原点到终点的匀速直线），系数 $\mathbf{c} \in \mathbb{R}^M$ 由解码器预测。

**解码器**：原型条件化查询解码器（PrototypeConditionedQueryDecoder）以 $K=20$ 个查询向量为输入，经 2 层自注意力（捕捉模式间关系）和 2 层交叉注意力（融合目标机特征）后，输出每个候选的基系数 $\mathbf{c} \in \mathbb{R}^{K \times M}$、置信度分数 $s \in \mathbb{R}^K$ 和精炼门控值 $g \in \mathbb{R}^K$。

微模式终点由原型中心终点加上可学习的微模式偏移量得到，偏移量在训练前由各原型内轨迹终点的 K-Means 子聚类初始化。

### G. 时序残差精炼器

基分解解码得到的粗轨迹 $\hat{\mathbf{y}}^{\text{coarse}}$ 在局部细节上可能存在不平滑或不准确的问题。时序残差精炼器（TemporalResidualRefiner）以粗轨迹和查询特征为输入，通过深度可分离卷积（kernel size=5）提取局部时序模式，预测残差修正量 $\Delta \hat{\mathbf{y}} \in \mathbb{R}^{K \times T_f \times 3}$。

精炼量由门控值 $g$ 加权控制：

$$\hat{\mathbf{y}}^{\text{fine}} = \hat{\mathbf{y}}^{\text{coarse}} + g \cdot \Delta \hat{\mathbf{y}}$$

门控机制使模型能够自适应地决定每条候选轨迹的精炼强度——对于已经准确的候选，门控值趋近于零，避免过度修正。

最终，所有候选轨迹经逆坐标变换还原至全局坐标系。

### H. 训练目标

训练损失由四部分组成：

**赢者通吃轨迹损失**（Winner-Takes-All, WTA）：对每个样本，选取与真实轨迹 ADE 最小的候选轨迹，计算 Smooth-L1 损失：

$$\mathcal{L}_{\text{traj}} = \text{SmoothL1}(\hat{\mathbf{y}}^*, \mathbf{y}_{\text{gt}})$$

**分数损失**：鼓励最优候选的置信度分数最高，采用软标签交叉熵（训练初期）和硬标签交叉熵（训练后期）的组合：

$$\mathcal{L}_{\text{score}} = \lambda_{\text{score}} \cdot \mathcal{L}_{\text{CE}}$$

**多样性损失**：对所有候选轨迹的终点施加排斥力，防止模式坍缩：

$$\mathcal{L}_{\text{div}} = \lambda_{\text{div}} \sum_{i \neq j} \max(0, \delta - \|\hat{\mathbf{e}}_i - \hat{\mathbf{e}}_j\|_2)$$

**平滑损失**：对预测轨迹的二阶差分施加惩罚，鼓励轨迹平滑：

$$\mathcal{L}_{\text{smooth}} = \|\Delta^2 \hat{\mathbf{y}}^*\|_1$$

总损失为：

$$\mathcal{L} = \mathcal{L}_{\text{traj}} + \mathcal{L}_{\text{score}} + \mathcal{L}_{\text{div}} + \mathcal{L}_{\text{smooth}}$$

训练分三个阶段：阶段 A（10 轮）仅训练轨迹损失和软分数损失；阶段 B（4 轮）加入多样性损失；阶段 C（86 轮）切换为硬分数损失并加入平滑损失，总计 100 轮。

---

*[TBD: 架构图 — 展示五个模块的数据流]*
