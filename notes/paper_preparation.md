# ProtoBasis-Net 论文写作准备文档

> 本文档为论文写作的完整前期准备，供后续 AI 会话直接读取使用。
> 最后更新：2026-04-27（清扫版，统一当前默认架构与公平对照口径）

---

## 1. 论文定位与目标期刊

**方法名称**: ProtoBasis-Net — 基于原型路由与SVD基分解的飞机轨迹预测

**核心创新点**:
- 原型路由（Prototype Routing）：64个聚类原型中选 top-15，条件化生成
- SVD基分解（Basis Decomposition）：16维全局基 + 2维原型局部基重建轨迹形状
- 结构化候选扩展（Structured Candidate Expansion）：top-5 原型保留2个 micro 候选，其余路由原型保留1个候选，共20条轨迹
- 形状精炼器（Shape Refinement）：endpoint-preserving shape/control-point refiners

**目标期刊**（按优先级）:
1. Scientific Reports (Q2, IF~4.6) — 概率 60-70%
2. IEEE TAES (Q2, IF~4.4) — 概率 30-40%
3. Q3期刊备选 — 概率 80%+

**论文类型**: 短论文，~6-8页，1个月内完成

---

## 2. 评测协议（已从原论文核实）

### 2.1 统一协议确认

GooDFlight 在所有数据集（111_days + 7days1~4）上统一使用同一协议：
- 输入：40s（40步，1Hz）
- 预测：120s（120步，1Hz）
- K=20（生成30个goal，裁剪保留top-20）
- 场景内飞机数：1~7，padding到7

ASCENT Table II 也使用相同协议（明确写了 "following the setup of [GooDFlight]"）。

我们的 `trajair_40to120_best20` 协议完全对齐。

### 2.2 各论文协议汇总（从原文逐字核实）

| 论文 | 表格 | 数据集 | 输入 | 预测 | 采样率 | K | 指标 |
|---|---|---|---|---|---|---|---|
| GooDFlight Table I | 111_days | 40s/40步 | 120s/120步 | 1Hz | 20 | ADE/FDE |
| GooDFlight Table II | 7days1~4 | 40s/40步 | 120s/120步 | 1Hz | 20 | ADE/FDE |
| ASCENT Table II 上 | 111_days | 40s/40步 | 120s/120步 | 1Hz | 20 | ADE/FDE |
| ASCENT Table II 下 | 111_days | 11s/11步 | 120s/12步 | 1Hz→0.1Hz | 5 | ADE/FDE |
| ASCENT Table I 上 | 7days1~4 | 11s/11步 | 120s/12步 | 1Hz→0.1Hz | 5 | ADE/FDE |
| ASCENT Table I 下 | 7days1~4 | 16s/16步 | 120s/24步 | 0.2Hz | 5 | ADE/FDE |
| **我们** | **全部** | **40s/40步** | **120s/120步** | **1Hz** | **20** | **ADE/FDE** |

### 2.3 关键结论

- GooDFlight 没有报 K=5 的实验
- ASCENT Table I（7days, K=5）和 Table II（111days, K=20）是完全不同的协议，数字不可混比
- 我们统一用 `trajair_40to120_best20`，和 GooDFlight 全面对齐
- ASCENT Table II 上半部分也可直接对比（同协议）

---

## 3. 实验方案（定稿）

### 实验1：主表 — 111_days SOTA对比

协议：`trajair_40to120_best20`（40s→120s, 1Hz, K=20）
对标：ASCENT Table II + GooDFlight Table I

数字全部从 ASCENT Table II 引用（更全面、更新），GooDFlight 自报数字作为交叉验证。

| Method | ADE@20↓ | FDE@20↓ | 来源 |
|---|---|---|---|
| Constant Velocity | 1.85 | 4.16 | ASCENT Tab.II |
| STG-CNN | 1.37 | 2.35 | ASCENT Tab.II |
| DAG-Net | 0.77 | 1.61 | ASCENT Tab.II |
| TrajAirNet | 0.79 | 1.58 | ASCENT Tab.II |
| Social-PatteRNN-ATT | 0.67 | 1.40 | ASCENT Tab.II |
| PECNet | 0.67 | 1.14 | ASCENT Tab.II |
| MID | 0.55 | 0.87 | ASCENT Tab.II |
| Expert-Traj | 0.55 | 0.72 | ASCENT Tab.II |
| GooDFlight | 0.29 | 0.39 | ASCENT Tab.II |
| ASCENT | 0.19 | 0.26 | ASCENT Tab.II |
| **ProtoBasis-Net (Ours)** | **待填** | **待填** | — |

不使用 GLeV 做横向对比。GooDFlight 原文提出该指标，但公式、文字解释和数值量级不够清晰，当前项目只把 ADE/FDE 作为公平主指标。

注意：GooDFlight 自报 ADE=0.27/FDE=0.35，ASCENT 复现报 0.29/0.39。
论文中统一引 ASCENT 的数字（因为主表其他 baseline 也来自 ASCENT），
但需脚注说明 GooDFlight 自报值略低。

### 实验2：7days1~4 泛化实验

协议：`trajair_40to120_best20`（同上）
对标：GooDFlight Table II（唯一在 7days 上用 K=20 协议的论文）

| Method | 7days1 | 7days2 | 7days3 | 7days4 | 来源 |
|---|---|---|---|---|---|
| TrajAirNet | 0.72/1.45 | 0.80/1.59 | 0.88/1.67 | 0.70/1.44 | GooDFlight Tab.II |
| Social-PatteRNN-ATT | 0.61/1.42 | 0.76/1.67 | 0.75/1.65 | 0.67/1.51 | GooDFlight Tab.II |
| GooDFlight | 0.27/0.41 | 0.32/0.40 | 0.36/0.48 | 0.30/0.40 | GooDFlight Tab.II |
| **Ours** | **待填** | **待填** | **待填** | **待填** | — |

注意：ASCENT 没有在 7days 上用 K=20 协议，所以此表无 ASCENT。

### 实验3：消融实验

数据集：111_days，协议：`trajair_40to120_best20`

| 配置 | ADE@20↓ | FDE@20↓ |
|---|---|---|
| Full model | 待填 | 待填 |
| w/o Prototype Router (--disable_router) | 待填 | 待填 |
| w/o SVD Basis Bank | 待填 | 待填 |
| w/o Micro expansion (`--topk_proto 20 --micro_per_proto 1 --candidate_dense_topk 0`) | 待填 | 待填 |
| w/o Social Aggregator (--disable_social) | 待填 | 待填 |

### 不做的实验

| 实验 | 不做的原因 |
|---|---|
| `legacy_11_best5` (11s→12步, K=5) | GooDFlight 没用这个协议，对比价值有限 |
| `legacy_16_best5` (16s→24步, K=5) | 只有 ASCENT 一家用，且是次要表格 |
| 111_days 上报 K=5 | GooDFlight 和 ASCENT 都没报 K=5，无对标 |
| TartanAviation 数据集 | 只有 ASCENT 跑了，且我们没有该数据集 |

---

## 4. 当前实验进度

| 实验 | 数据集 | 状态 | 最佳成绩 |
|---|---|---|---|
| 主实验 seed3407 | 111_days | ✅ 已完成，需用当前默认重新确认 | ADE@20=0.228, FDE@20=0.329 |
| 泛化实验 | 7days1 | ✅ 已有旧结果，需用统一默认重新确认 | ADE@20=0.326, FDE@20=0.509 |
| 泛化实验 | 7days2 | ⏳ 未开始 | — |
| 泛化实验 | 7days3 | ⏳ 未开始 | — |
| 泛化实验 | 7days4 | ⏳ 未开始 | — |
| 消融: w/o Router | 111_days | ⏳ 未开始 | — |
| 消融: w/o Basis | 111_days | ⏳ 未开始 | — |
| 消融: w/o Micro | 111_days | ⏳ 未开始 | — |
| 消融: w/o Social | 111_days | ⏳ 未开始 | — |

### 当前成绩 vs 竞争对手（111_days）

| Method | ADE@20 | FDE@20 |
|---|---|---|
| ASCENT | 0.19 | 0.26 |
| ProtoBasis-Net (当前) | 0.228 | 0.329 |
| GooDFlight | 0.29 | 0.39 |

当前位于 ASCENT 和 GooDFlight 之间，已超过 GooDFlight。

### 当前成绩 vs 竞争对手（7days1）

| Method | ADE@20 | FDE@20 |
|---|---|---|
| GooDFlight | 0.27 | 0.41 |
| ProtoBasis-Net (当前) | 0.326 | 0.509 |

7days1 仍弱于 GooDFlight，但旧 notes 高估了差距：ADE 差距为 0.056，FDE 差距为 0.099（ADE/FDE 越低越好）。需要继续优化 scoring head、lambda_score 等。

---

## 5. 与关键竞争方法的差异化论述

### 5.1 vs EigenTrajectory (ICCV 2023, Bae et al.)
- EigenTrajectory：全局SVD基（k=6个特征向量），无原型路由，行人场景（ETH/UCY, SDD）
- 作为"plug-in"方法，可嵌入6种baseline，最高提升74.3%
- 基是固定的、全局的、无条件的：所有轨迹投影到同一组基上
- ProtoBasis-Net 三个关键差异：
  1. 原型条件化基选择（不是全局固定基）
  2. 航空3D场景（不是行人2D）
  3. 基向量可在训练中端到端学习（不是固定SVD）
- **论文叙事**：我们将EigenTrajectory的全局基思想扩展为原型条件化的局部基分解

### 5.2 vs MTR (NeurIPS 2022, Shi et al.)
- MTR：64个可学习意图查询对，6层解码器，逐点GMM回归
- 核心区别：MTR逐时间步回归(x,y)坐标，无轨迹结构分解
- ProtoBasis-Net 关键差异：
  1. 基分解替代逐点回归（预测系数而非坐标序列）
  2. 基向量隐式保证轨迹平滑性
  3. 不依赖HD地图（航空场景无车道约束）
- **论文叙事**：借鉴MTR的查询机制，但用基分解替代逐点解码

### 5.3 vs ASCENT (arXiv 2026)
- ASCENT：纯Transformer + 模式查询解码，无基分解，无多样性指标
- 我们：原型路由 + SVD基分解 + 微模式扩展 + 可解释候选结构
- ASCENT ADE@20=0.19 优于我们的 0.228
- **应对策略**：
  1. 强调结构化先验的可解释性
  2. 用 "competitive with" 描述关系，不硬碰数字

### 5.4 vs GooDFlight (2025)
- GooDFlight：扩散模型 + goal estimation + CFG意图调节
- 我们在 111_days 上 ADE@20=0.228 已超过 GooDFlight 的 0.29
- **论文叙事**：在保持多样性的同时实现更高精度和更快推理

---

## 6. 文献分类与引用规划

论文预计引用 35-45 篇。

### 6.1 必读必引（已全部下载）— 约10篇

| 论文 | 年份 | 会议/期刊 | 引用位置 |
|---|---|---|---|
| TrajAirNet (Patrikar et al.) | 2022 | ICRA/RA-L | Introduction, Experiments |
| GooDFlight (Yang et al.) | 2025 | IEEE Trans. | Related Work, Experiments, GLeV |
| ASCENT (Prutsch et al.) | 2026 | arXiv | Related Work, Experiments |
| EigenTrajectory (Bae et al.) | 2023 | ICCV | Related Work (basis decomposition) |
| MTR (Shi et al.) | 2022 | NeurIPS | Related Work (query-based) |
| ACTrajNet | 2023 | Sci. Rep. | Related Work, Experiments |
| TartanAviation (Patrikar et al.) | 2025 | Sci. Data | Dataset description |
| Social-PatteRNN (Navarro, Oh) | 2022 | IROS | Related Work (social) |
| SingularTrajectory (Bae et al.) | 2024 | CVPR | Related Work (SVD-based) |
| MID (Gu et al.) | 2022 | CVPR | Related Work (diffusion) |

### 6.2 重要参考 — 约10篇

Trajectron++, Social-STGCNN, AgentFormer, MemoNet, LED, QCNet,
Social GAN, DETR, Attention is All You Need, DDPM

### 6.3 仅引用（从GooDFlight/ASCENT参考列表借用）— 约15-20篇

经典方法、评价指标定义、航空背景文献等。

完整文献索引见：`D:\essay\fuxian\文献库\文献库索引.md`
BibTeX 文件：`D:\essay\fuxian\ProtoBasis-Net\notes\references.bib`（25条）

---

## 7. 论文结构规划

### 推荐写作顺序
Method → Experiments → Related Work → Introduction → Abstract

### 7.1 论文大纲

**Title**: ProtoBasis-Net: Prototype-Routed Basis Decomposition for Multi-Modal Aircraft Trajectory Prediction

**1. Introduction** (~1 page)
- 通用航空轨迹预测的重要性和挑战
- 现有方法的局限（逐点回归、缺乏结构先验）
- 我们的贡献（3点）

**2. Related Work** (~1 page)
- 2.1 Aircraft Trajectory Prediction
- 2.2 Multi-Modal Trajectory Forecasting
- 2.3 Basis Decomposition for Trajectories

**3. Method** (~2 pages)
- 3.1 Problem Formulation & Pose Normalization
- 3.2 Temporal Encoder
- 3.3 Social Aggregator
- 3.4 Prototype Router
- 3.5 Basis Decomposition
- 3.6 Micro-Mode Expansion & Query Decoder
- 3.7 Endpoint-Preserving Shape Refinement
- 3.8 Training Strategy（3-stage curriculum）
- Figure 1: 架构总图

**4. Experiments** (~2 pages)
- 4.1 Dataset & Setup
- 4.2 Main Results — Table 1 (111_days SOTA对比)
- 4.3 Generalization — Table 2 (7days1~4 对比)
- 4.4 Ablation Study — Table 3
- 4.5 Qualitative and Case Analysis
- Figure 2: 轨迹可视化

**5. Conclusion** (~0.5 page)

### 7.2 需要的表格（定稿）

| 表格 | 内容 | 对标论文 | 优先级 |
|---|---|---|---|
| Table 1 | 111_days SOTA对比 (ADE@20/FDE@20) | ASCENT Tab.II | 必须 |
| Table 2 | 7days1~4 泛化 (ADE@20/FDE@20) | GooDFlight Tab.II | 必须 |
| Table 3 | 消融实验 | — | 必须 |
| Table 4 | 定性案例 / 误差分解 | — | 推荐 |

### 7.3 需要的图表

- Figure 1: 架构总图（必须）
- Figure 2: 轨迹可视化，2-3个场景（必须）
- Figure 3: 典型场景可视化或误差分解图（可选）

---

## 8. 训练配置备忘

### 当前最优超参
```bash
python train.py 111_days \
  --device cuda:1
```

### 训练阶段
- Stage A (10 epochs): GT proto warmup, no refiner
- Stage B (4 epochs): Learn routing, no refiner
- Stage C (51 epochs): Full model, refiner + diversity loss
- Extra tail: 0 by current default
- Total: 65 epochs

### 损失权重
- lambda_xyz=1.0, lambda_fde=1.0, lambda_proto=0.35
- lambda_res=0.2, lambda_score=0.03
- lambda_div=0.05, stage_c_div_weight=2.0
- lambda_coeff=0.02, lambda_smooth=0.10

### 7days 注意事项
- 7days1 当前仍弱于 GooDFlight，不应单独改小数据集架构；小数据集只允许不同超参数
- batch_size: 111_days=512, 7days=48（硬编码在 apply_training_defaults）
- 7days 数据量只有 111_days 的 1/7.5，原型/基向量质量受影响

---

## 9. 已知问题与应对策略

### 9.1 ASCENT 数字更优
- ASCENT ADE@20=0.19 vs 我们 0.228
- 应对：强调结构化先验、社会交互建模和可解释候选分解
- 用 "competitive with" 而非 "state-of-the-art"

### 9.2 7days1 仍未超过 GooDFlight
- 我们 0.326/0.509 vs GooDFlight 0.27/0.41（GooDFlight Table II）
- 旧 notes 使用了错误的 GooDFlight 7days1~4 数字，导致差距判断偏大
- 当前结论：ADE/FDE 均仍弱于 GooDFlight，但 FDE 差距比旧表判断小
- 可能原因：数据量不足、scoring head 弱、lambda_score 太低
- 需要调参优化后再定论

### 9.3 K=5 不需要报
- GooDFlight 和 ASCENT Table II 都只报 K=20
- K=5 差是 scoring head 问题，但不影响论文主表

### 9.4 GooDFlight vs ASCENT 数字不一致
- GooDFlight 自报 ADE=0.27，ASCENT 复现报 0.29
- 论文中统一引 ASCENT 的数字（主表 baseline 来源一致）
- 脚注说明 GooDFlight 自报值

---

## 10. 写作注意事项

### 10.1 语言风格
- 目标 Scientific Reports：语言直白，不过度学术化
- 避免过度claim，用 "competitive with" 描述与ASCENT的关系
- 强调 "structured diversity" 和 "interpretable mode decomposition"

### 10.2 引用策略
- 主表 baseline 数字统一引 ASCENT Table II
- 不引用 GLeV 数字作横向比较
- 7days 数字引 GooDFlight Table II

### 10.3 图表制作
- 架构图：draw.io 或 TikZ
- 轨迹可视化：matplotlib 矢量图
- 表格：LaTeX booktabs 风格

---

## 11. 时间规划（1个月）

| 周 | 任务 |
|---|---|
| Week 1 | 跑完 7days2/3/4 + 消融实验，同时写 Method |
| Week 2 | 整理实验结果，写 Experiments，画架构图和可视化 |
| Week 3 | 写 Related Work + Introduction，补齐引用 |
| Week 4 | 写 Abstract + Conclusion，全文润色，投稿 |
