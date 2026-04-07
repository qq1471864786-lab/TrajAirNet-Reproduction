# HAINet 项目 AI 接手上下文（完整版）

> 把这个文件内容粘贴到新 AI 窗口的第一条消息里，它就能立刻接上干活。

---

## 一、用户背景

- 研究生，研究方向：飞机轨迹预测（aircraft trajectory prediction）
- 数据集：TrajAir（CMU AirLab，Pittsburgh-Butler Regional Airport）
- 创新点：高度（altitude）与交互（interaction）的双向耦合建模
- 目标：ADE < 0.5，发 SCI 三区（80%概率）或二区（50%概率）
- 硬件：本地 i7-12700H（改代码），训练服务器 wangzhilin@10.23.66.99（RTX 2080 Ti）
- 项目路径（本地）：`D:\essay\fuxian\ACTrajNet-main`
- 项目路径（服务器）：`/home/wangzhilin/ACTrajNet-main`

## 二、自动化同步流程

### 同步命令
```bash
python deploy.py          # 增量同步（只传改过的文件）
python deploy.py --force  # 全量同步
python deploy.py --dry-run  # 预览不传
```

### deploy.py 工作原理
- 位置：项目根目录 `deploy.py`
- 目标：`wangzhilin@10.23.66.99:/home/wangzhilin/ACTrajNet-main`
- SSH 密钥：自动查找 `~/.ssh/id_rsa`，也可设环境变量 `TRAJAIR_SSH_KEY`
- 增量检测：用 SHA-256 指纹（`.deploy_manifest.json`）对比文件变化
- 同步范围：`.py .yaml .yml .sh .md .txt .json` + `dataset/social/` 和 `dataset/no_social/` 下的 `.txt`
- 排除目录：`.git __pycache__ .idea .vscode .claude dataset outputs save_model saved_models results tmp tool-outputs`
- 同步标记：`.last_sync`（时间戳），`.deploy_manifest.json`（文件指纹）
- 支持删除：本地删了的文件会从服务器上也删掉

### 服务器训练命令
```bash
# SSH 到服务器
ssh wangzhilin@10.23.66.99

# 进入项目目录
cd /home/wangzhilin/ACTrajNet-main

# 训练（默认参数，social 数据集）
python train.py --dataset_variant social --dataset_name 7days1

# 训练（禁用交互 = 纯 ACTrajNet baseline）
python train.py --disable_interaction

# 查看训练状态（本地运行）
python remote_run_status.py
```

## 三、项目文件结构

```
ACTrajNet-main/
├── train.py                 # 训练入口（50 epoch, AdamW, cosine LR, Best-of-5 评估）
├── test.py                  # 评估脚本（注意：还在用旧的 HeightInteractionPredictor，需要更新）
├── deploy.py                # 增量同步到服务器
├── remote_run_status.py     # SSH 查看服务器训练状态
├── summarize_run.py         # 打印本地训练 run 摘要
├── requirements.txt         # 依赖
├── model/
│   ├── __init__.py          # 导出：HAINet, HAINetLoss, SceneTrajectoryDataset, metrics, data utils
│   ├── scene_model.py       # 核心模型（HAINet）— 已完全重写
│   ├── losses.py            # 损失函数（RMSE + KL）
│   ├── metrics.py           # 评估指标（ADE, FDE, MDE, AADE, AFDE, AMDE）
│   ├── data.py              # 数据集加载（SceneTrajectoryDataset, scene_batch_collate）
│   └── run_logging.py       # 训练日志（JSONL epoch metrics, checkpoints, run summary）
├── dataset/
│   ├── social/              # 多机数据（有交互）
│   │   ├── 7days1/processed_data/{train,test}/*.txt
│   │   ├── 7days2/ ... 7days4/
│   └── no_social/           # 单机数据（无交互）
│       ├── 7days1_no_social/{train,test}/*.txt
│       ├── 7days2_no_social/ ... 7days4_no_social/
├── save_model/              # 训练输出（model.pt, epoch_metrics.jsonl, run_summary.json）
│   └── social/7days1/42/    # seed=42 的运行结果
└── tmp/                     # 诊断脚本和探针结果（不同步到服务器）
```

## 四、数据格式

TrajAir 数据集，每行 7 列，空格分隔：
```
frame_id  agent_id  x  y  altitude  wind_x  wind_y
```
- obs_len=11, pred_len=120, pred_step=10 → 12 个预测时间步
- social 数据集：同一时间窗口可能有多架飞机（交互场景）
- no_social 数据集：每个窗口只有一架飞机

## 五、模型架构（HAINet）

### 论文定位
- Baseline 1：ACTrajNet（Scientific Reports 2025）— 有高度 CAF，无交互，ADE≈0.7
- Baseline 2：TrajAirNet（ICRA 2022）— 有交互 GAT，无高度特殊处理，ADE≈0.7
- 我们的 HAINet：高度 + 交互双向耦合，目标 ADE < 0.5

### 架构流程
```
obs(11,N,3) → permute → TCN_traj(3→[256,256,12]) → encoded_traj(N,12,11)
                       → TCN_alt(1→[256,256,1])  → encoded_alt(N,1,11)
                       → CAF: gate=sigmoid(fc(encoded_alt)) → encoded_traj * gate
                       → flatten → h_fused_flat(N,132)

context(11,N,2) → Conv1d(2→1,k=2) → Linear(10→7) → ReLU → h_wind(N,7)

condition_base = [h_fused_flat, h_wind] = (N, 139)  ← 和 ACTrajNet 一致

[如果 use_interaction=True 且 N>1]:
  → AC-GAT(condition_base, last_pos, last_vel, adj_mask) → h_social(N, 256)
  → [如果 use_height_feedback]:
      → InteractionHeightFeedback: h_alt' = h_alt + sigmoid(W_g·h_social) * relu(W_u·h_social)
      → 第二轮 CAF: 用更新后的 h_alt' 重新 gate TCN 输出
      → 更新 condition_base

condition = [condition_base, h_social] = (N, 395)  或无交互时 (N, 139)

[训练时]:
  target → TCN_future → CAF_future → flatten → h_future_flat(N, 144)
  CVAE.forward(h_future_flat, condition) → H_yy, mu, logvar
  → reshape(N,3,32) → linear_decoder(32→12) → acc(N,3,12)
  → Verlet: pred[0] = 2*obs[-1] - obs[0] + acc[0]
  → prediction(12,N,3)

[推理时]:
  z ~ N(0,I), shape=(N, 128)
  CVAE.inference(z, condition) → H_yy → 同上 → prediction
```

### 关键组件细节

**TCN（严格复刻 ACTrajNet）**
- 3 层：[256, 256, 12]（traj）/ [256, 256, 1]（alt）
- weight_norm + Tanh 残差连接（不是 ReLU）
- kernel_size=4, dilation=2^i
- 输出 flatten 整个序列（不是取最后时间步）

**CAF（Channel Attention Fusion）**
- altitude TCN 输出 (N,1,11) → fc(11→12) → sigmoid → gate (N,12,1)
- gate 逐元素乘以 traj TCN 输出 (N,12,11)
- 在 flatten 之前做

**CVAE（严格复刻 ACTrajNet）**
- Encoder: [144+cond_dim, 128, 128] + ReLU → mu/logvar(128)
- Decoder: [128+cond_dim, 128, 128, 96] + 全部 Tanh
- linear_decoder: Linear(32→12)，无激活，无 scale 限制
- 训练时用 posterior（编码 future），推理时用 prior（z~N(0,I)）

**Verlet Integration**
- `pred[0] = 2*obs[-1] - obs[0] + acc[0]`（用 obs[0] 是正确的，已验证）
- `pred[1] = 2*pred[0] - obs[-1] + acc[1]`
- `pred[i] = 2*pred[i-1] - pred[i-2] + acc[i]`

**AC-GAT（创新模块）**
- 多头注意力（8 heads），输入 139 维，输出 256 维
- 高度条件化 bias：`alpha_ij = softmax(e_ij + phi(|dz|, dv_z, d_xy))`
- phi 是 MLP: Linear(3→32) → ReLU → Linear(32→8)
- adj_mask 基于 scene_ids（同场景内的飞机互相可见）

**InteractionHeightFeedback（创新模块）**
- `h_alt' = h_alt + sigmoid(W_g · h_social) * relu(W_u · h_social)`
- 社交信息 gate 更新高度表示
- 更新后的高度重新做 CAF（第二轮）

### 消融开关
- `--disable_interaction`：去掉 AC-GAT 和 feedback → 退化为 ACTrajNet
- `--disable_height_conditioning`：GAT 中去掉高度 bias → 普通 GAT
- `--disable_height_feedback`：去掉 feedback → 单向（高度→交互，无反馈）

## 六、损失函数

```python
# losses.py
loss = RMSE(prediction, target) + kl_weight * KL(mu, logvar)
# RMSE = sqrt(MSE(pred, target))
# KL = sum over latent dims, mean over batch
# kl_weight 由外部 cyclical/linear annealing 控制
```

当前超参数（train.py defaults）：
- kl_weight=0.1, free_bits=0.0
- kl_anneal_epochs=20, linear annealing（非 cyclical）
- lr=1e-4, AdamW, weight_decay=3e-4
- cosine LR scheduler, min_lr=2e-5
- grad_clip=5.0, batch_size=64, epochs=50
- best_of_n=5（评估时采样 5 次取最好的）

## 七、当前状态和待解决问题

### 已完成
1. scene_model.py 完全重写，严格对齐 ACTrajNet 基础组件
2. losses.py 改回 RMSE + KL（从 SmoothL1 改回来）
3. train.py 加入 KL annealing、best-of-N 评估、诊断日志
4. Verlet integration 已验证正确（数值 diff=0）

### 当前问题（关键！）
**ADE=1.53，比旧模型的 0.88 还差。**

根本原因分析：
- 训练 recon loss = 0.15，但评估 ADE = 1.5
- 如果 RMSE=0.15，ADE 应该 ≈0.26；如果 ADE=1.5，RMSE 应该 ≈0.87
- 这说明 CVAE posterior（训练时，有 future 信息）拟合很好，但 prior（推理时，z~N(0,I)）生成的预测很差
- 即 prior-posterior gap 巨大

**最新修改（尚未训练验证）：**
- KL 从 mean 改为 sum(dim=-1).mean()（增大 KL 权重，推 posterior 靠近 prior）
- kl_weight 从 1.0 降到 0.1（补偿 sum 带来的 128x 放大）
- free_bits 从 0.05 改为 0.0（去掉强制 KL 下限）
- cyclical annealing 改为 linear annealing（更温和）

### 下一步
1. 同步到服务器跑训练，验证 ADE 是否下降
2. 如果 ADE 降到 0.7-0.9（对齐 ACTrajNet baseline），说明基础对齐成功
3. 然后跑消融实验（4 组）：
   - Base（--disable_interaction）= ACTrajNet 复刻
   - +Interaction only（--disable_height_conditioning --disable_height_feedback）
   - +Height conditioning only（--disable_height_feedback）
   - Full HAINet（默认）
4. 目标：Full HAINet ADE < 0.5

## 八、调试历史（避免重复踩坑）

1. **Verlet 用 obs[0] 是正确的** — 不要质疑这个，已经反复验证过
2. **acc_scale 限制是错误的** — 旧代码用 `tanh * 0.02` 限制加速度，数据 99 分位是 0.062，被截断了。现在已去掉限制
3. **Loss 函数很关键** — 旧代码用 SmoothL1+scale[3,3,0.75]，效果差。现在用 RMSE（和 ACTrajNet 一致）
4. **KL collapse vs KL 过强** — 之前 KL→0.00008 导致 CVAE 退化为确定性模型。加了 anti-collapse 后 KL 稳定在 0.10 但 prior-posterior gap 太大。需要平衡
5. **test.py 还在用旧模型** — `test.py` 里 import 的是 `HeightInteractionPredictor`（旧架构），需要更新为 `HAINet`

## 九、用户偏好（重要！）

1. **不要催着写代码** — 先彻底理解需求，确认方案，再动手
2. **每次修改必须检查完整** — 不要引入新问题，用户原话："别又出新问题，你能不能检查完整"
3. **省 token** — PDF 内容提取用 WebSearch 搜论文名，不要反复 pdftotext
4. **用中文交流** — 用户是中文母语
5. **不要重复解释** — 简洁直接，不要啰嗦
