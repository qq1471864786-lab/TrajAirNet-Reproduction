# ProtoBasis-Net 方法定位简报

> 生成日期：2026-04-25
> 用途：为后续小论文写作、Method/Related Work/Experiments 统一口径。
> 范围：当前仓库 `D:\essay\fuxian\ProtoBasis-Net` 与文献库 `D:\essay\fuxian\文献库`。

## 1. 当前项目一句话

ProtoBasis-Net 是面向 TrajAir 非塔台终端空域的多模态 3D 航迹预测方法。当前主协议是 40 s 历史观测到 120 s 未来预测，1 Hz，最多 7 架飞机，输出 20 条候选轨迹。

核心写法应是：

> Prototype-routed basis decomposition for multi-modal aircraft trajectory prediction.

不要写成泛泛的 Transformer 预测器。方法特色在“原型先验 + 低维基分解 + 结构化多候选解码”。

## 2. 已确认的代码实现

主模型在 `model/proto_basis_flight_model.py`，主要模块为：

1. `PoseNormalizer`
   以目标飞机最后观测点为原点，用末两步速度估计 yaw/pitch，将全场轨迹转到目标局部坐标系。该设计与 ASCENT 的 domain-aware normalization 接近，需要引用并说明差异。

2. `TemporalEncoder`
   每架飞机独立编码。局部特征为 3D 坐标、速度、yaw/pitch 正余弦，共 8 维；全局特征为 3D 坐标、yaw/pitch 正余弦，共 7 维。二者投影后拼接，再进入 Transformer encoder。

3. `TargetSocialAggregator`
   目标飞机作为 query，所有飞机作为 key/value 做 cross-attention；同时保留场景级 masked mean。这个模块是和 ASCENT 的重要差异点之一。

4. `PrototypeRouter`
   对训练集未来轨迹的 5 维摘要做 K-Means，默认 64 个原型。摘要为局部终点 xyz、累计 yaw 变化、累计 pitch 变化。`111_days` 默认从 64 个原型中选 top-15，并预测终点残差。

5. `PrototypeConditionedQueryDecoder`
   `111_days` 默认每个路由原型生成 2 个 micro modes，再用 `candidate_dense_topk=5` 保留 20 条候选。query 由原型 token、终点 token、micro embedding、目标上下文以及可选 micro coefficient anchor 组成。输出 basis coefficient 和 candidate score。

6. `BasisBank`
   对训练集 future local trajectory 减去线性 endpoint anchor 后做 SVD，默认取 16 维基。候选轨迹为 endpoint anchor 加 basis residual。

7. 主数据集默认增强项
   当前 `train.py` 对 `111_days` 默认启用 `local_basis_dim=2`、`support_aware_local_basis=True`、`two_stage_decoder=True`、`coupled_decoder=True`、endpoint/control shape refiners。Method 草稿如果只写“全局 SVD 基”会漏掉当前正式默认实现。

## 3. 数据与评测协议

当前主协议 `trajair_40to120_best20`：

- `obs=40`
- `preds=120`
- `obs_stride=1`
- `pred_stride=1`
- `K=20`
- `topk_proto=15`
- `micro_per_proto=2`
- `candidate_dense_topk=5`
- `max_agents=7`

本地数据文件数量：

- `111_days`: train 2230 files, test 858 files
- `7days1`: train 233, test 111
- `7days2`: train 155, test 48
- `7days3`: train 115, test 62
- `7days4`: train 141, test 62

数据行格式从样本看为：

`frame_id aircraft_id x y z wind_or_context_1 wind_or_context_2`

当前 dataset 构造只使用 `data[:, 2:5]` 的 xyz 作为轨迹输入，未使用最后两列气象/上下文字段。

## 4. 训练策略

`train.py` 当前默认阶段：

- Stage A: 10 epochs, `basis_warmup`, 强制 GT prototype
- Stage B: 4 epochs, `joint_router`, 释放 router
- Stage C: `111_days` 8 epochs, `7days*` 20 epochs, `joint_polish`
- Extra tail: 当前代码对 `111_days` 默认为 0，只有显式传 `--extra_epochs` 才追加

当前推荐正式主实验命令来自 notes：

```bash
python train.py 111_days --device cuda:1
```

损失项包括：

- WTA Smooth-L1 轨迹损失
- FDE endpoint loss
- prototype classification CE
- endpoint residual loss
- quality-aware soft score loss + hard winner mix
- endpoint diversity repulsion
- coefficient regularization
- trajectory smoothness loss

## 5. 当前实验状态

现有写作 notes 中记录的正式 111_days 结果：

- ProtoBasis-Net: ADE@20 = 0.228, FDE@20 = 0.329
- ASCENT: ADE@20 = 0.19, FDE@20 = 0.26
- GooDFlight: ADE@20 = 0.29, FDE@20 = 0.39

写作结论：

- 可以说超过 GooDFlight。
- 不能说超过 ASCENT。
- 对 ASCENT 应写 `competitive with ASCENT`。

本地 `save_model/7days1` 是旧的 failed run，协议为 `trajair_7days_11s_test`，不应作为当前论文结果引用。当前可引用结果应以 notes 中记录的正式远程 run 或重新 `test.py` 验证为准。

## 6. 文献方法脉络

### TrajAirNet / TrajAir

TrajAirNet 是 TrajAir 原始基线：TCN + GAT + CVAE，建模历史轨迹、agent-agent social interaction、机场/天气环境。它提供了数据集和传统航空社交预测 baseline，但精度已明显落后。

可用于：

- 数据集来源
- 非塔台终端空域问题定义
- social interaction 背景

### ACTrajNet

ACTrajNet 强调 altitude-aware prediction：独立编码高度分支，用 channel attention 融合垂直特征。它说明 3D 航迹中高度不是附属维度，而是终端空域预测的重要信息。

可用于：

- Related Work 的航空 3D/高度建模段
- 支撑 ProtoBasis-Net 使用 xyz 联合建模和 pitch 特征

### GooDFlight

GooDFlight 是当前最接近的航空生成式竞争方法之一。核心是 goal-oriented diffusion：先估计宏观 goal/intention distribution，再用 diffusion generator 生成轨迹。它提出 GLeV 多样性指标。

与 ProtoBasis-Net 的差异：

- GooDFlight 用 diffusion 建模 micro uncertainty；ProtoBasis-Net 用 deterministic structured basis decoding。
- GooDFlight 的 goal 是显式两阶段条件；ProtoBasis-Net 的 prototype router 是数据驱动的模式路由。
- GooDFlight 生成式更强，但推理更重；ProtoBasis-Net 更轻、更可解释。

写作口径：

- `structured deterministic alternative to diffusion-based generation`
- `prototype and basis coefficients provide interpretable mode decomposition`

### ASCENT

ASCENT 是 2026 arXiv 最新强基线：lightweight transformer + coordinate normalization + parameterized prediction + query decoder。111_days 40/120 K=20 上报 ADE/FDE = 0.19/0.26。

与 ProtoBasis-Net 的差异：

- ASCENT 以 mode queries 和 flight parameters 直接生成多模态预测。
- ProtoBasis-Net 把 mode query 进一步绑定到 K-Means prototype，并在 endpoint-conditioned basis subspace 内重建轨迹。
- ProtoBasis-Net 有 social cross-attention aggregator；ASCENT 主要强调轻量单体 motion encoding/query decoding。

写作风险：

- 坐标归一化与 ASCENT 接近，必须引用。
- 不能把 query decoder 包装成全新机制；新意在 prototype-routed basis decomposition。

### EigenTrajectory / SingularTrajectory

EigenTrajectory 用 SVD/low-rank descriptors 将轨迹映射到 ET space，再预测低维系数，是 ProtoBasis-Net 的核心相邻工作。

SingularTrajectory 进一步把 SVD/Singular space 与 adaptive anchor、diffusion predictor 结合，强调跨任务 universal trajectory prediction。

与 ProtoBasis-Net 的差异：

- 这些方法主要面向行人/通用视觉轨迹。
- EigenTrajectory 是全局低秩描述；ProtoBasis-Net 是原型路由条件下的局部/结构化 basis decoding。
- SingularTrajectory 用 diffusion 增强 prototype paths；ProtoBasis-Net 用 endpoint-preserving shape/control refiners。

写法：

> ProtoBasis-Net extends low-rank trajectory descriptors from a global representation into a prototype-routed aircraft trajectory decoder.

### MTR / QCNet / DETR

MTR 用 learnable motion query pairs 同时做 global intention localization 和 local movement refinement。DETR 是 query-based decoding 的更早来源。QCNet 强调 query-centric scene encoding 和在线预测效率。

可用于：

- 说明 query-based multimodal forecasting 已成熟。
- 将 ProtoBasis-Net 的贡献收窄到 aviation + prototype/basis route，而不是声称发明 query decoder。

## 7. 论文贡献建议

建议贡献写三点，不要写太散：

1. Prototype-routed trajectory decomposition
   用 K-Means 原型建模终端空域宏观模式，并用 router 在推理时选择 top-k prototype。强调模式可解释与稀有模式覆盖。

2. Endpoint-conditioned basis reconstruction
   不逐点回归未来 120 步，而是预测低维 basis coefficients，在 endpoint anchor 上重建完整 3D trajectory。强调长时预测的结构先验和平滑性。

3. Structured multi-candidate decoding with social context
   每个 prototype 下扩展 micro modes，并结合 social cross-attention 与 endpoint-preserving shape/control refiners，提高多样性和局部细节。

不要把 `Transformer encoder`、`cross-attention`、`SVD` 单独写成新贡献。它们是支撑模块，不是核心新意。

## 8. 必须修正/核实的风险点

### GLeV 不作为横向对比指标

GooDFlight 提出了 GLeV，但原文公式、解释文字和表格量级不够清晰，直接横向对比容易不公平。当前项目不再报告 GLeV 作为主结果，也不再用它支撑相对 GooDFlight 或 ASCENT 的结论。多模态候选质量改用 ADE/FDE、消融和典型场景可视化说明。

### Method 草稿和代码默认组件有漂移

`notes/writing/method_zh.md` 主要描述 global basis + refiner，但当前 `111_days` 默认还包含 local basis、two-stage decoder、coupled decoder 和 endpoint/control shape refiners。论文 Method 必须和最终正式实验配置一致。

### 测试和训练默认组件需一致

已清理旧的 micro endpoint offset/anchor 方向；当前测试与训练只保留 micro coefficient anchors。

### 气象/上下文未使用

TrajAir 原始数据包含 METAR/风等上下文，当前 dataset 只使用 xyz。论文不能声称使用天气或环境上下文；如要提天气，只能放在数据集描述和未来工作。

## 9. 后续实验优先级

必须优先完成：

1. 111_days final result 用 `test.py` 重新确认，并保存完整命令、checkpoint、run_config。
2. 7days2/3/4 泛化实验，协议必须为 40/120 K=20。
3. 消融实验：
   - w/o router
   - w/o basis
   - w/o micro modes
   - w/o endpoint/control shape refiners
   - w/o social
4. latency: `bs=1` 和 `bs=16`，用于对比 GooDFlight diffusion 与 ASCENT lightweight claim。

消融表必须能支撑机制：

- router 证明 prototype prior 有用
- basis 证明低维重建不是装饰
- micro 证明 top-20 多样性来自结构化展开
- social 证明多机交互有收益
- endpoint/control shape refiners 证明基分解后仍需要局部细节修正

## 10. 推荐论文叙事

问题开头：

非塔台终端空域轨迹具有明显的结构化多模态性：少数宏观飞行模式叠加连续的局部形状变化。逐点回归缺乏长期几何先验；扩散模型能生成多样轨迹，但推理成本和可解释性不总是理想。

方法转折：

ProtoBasis-Net 将未来轨迹表示为 prototype-conditioned basis coefficients，而不是直接预测每个时间步坐标。prototype 负责宏观意图区域，basis coefficients 负责局部曲率与速度变化，micro modes 负责同一 prototype 下的多候选展开。

实验结论：

在 TrajAir 111_days 的统一 40/120 K=20 协议下，ProtoBasis-Net 超过 GooDFlight，接近 ASCENT，同时提供更明确的模式/基分解结构。泛化和消融结果用于证明该结构不是单纯调参收益。

## 11. 主要资料源

本地文献库：

- `D:\essay\fuxian\文献库\ASCENT.pdf`
- `D:\essay\fuxian\文献库\GooDFlight.pdf`
- `D:\essay\fuxian\文献库\TCN-CVAE.pdf`
- `D:\essay\fuxian\文献库\EigenTrajectory_ICCV2023.pdf`
- `D:\essay\fuxian\文献库\SingularTrajectory_CVPR2024.pdf`
- `D:\essay\fuxian\文献库\MTR_NeurIPS2022.pdf`

网上核对：

- ASCENT arXiv: https://arxiv.org/abs/2603.16550
- TrajAirNet arXiv: https://arxiv.org/abs/2109.15158
- GooDFlight record / IEEE TAES DOI: https://www.researchgate.net/publication/388546955_GooDFlight_Goal-Oriented_Diffusion_Model_for_Flight_Trajectory_Prediction
- EigenTrajectory arXiv: https://arxiv.org/abs/2307.09306
- SingularTrajectory CVF: https://openaccess.thecvf.com/content/CVPR2024/papers/Bae_SingularTrajectory_Universal_Trajectory_Predictor_Using_Diffusion_Model_CVPR_2024_paper.pdf
- MTR arXiv: https://arxiv.org/abs/2209.13508
- TartanAviation Scientific Data: https://www.nature.com/articles/s41597-025-04775-6
