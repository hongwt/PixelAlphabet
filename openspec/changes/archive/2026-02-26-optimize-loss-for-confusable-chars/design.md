## Context
当前 PixelAlphabet 项目在 24x24 像素分辨率下识别 36 类字符（0-9, A-Z），核心挑战是形态极度近似的字符对——在仅有 576 个像素的空间中，7/T、8/B、0/D、5/S、2/Z 等字符对之间的视觉差异往往仅集中在 1-2 个像素上。

文档《游戏UI字符识别数据集生成》明确指出：
1. 标准 CE 损失会让简单字符的梯度淹没困难对的信号（Focal Loss 章节）
2. 合成数据和真实数据之间存在微小的统计学边缘差异（Domain Shift 章节）
3. 需要课程学习分阶段训练（Curriculum Learning 章节）

本变更聚焦于 **损失函数与训练循环** 层面的优化，暂不涉及 GAN 风格迁移等数据层面的对齐工作。

## Goals / Non-Goals
- **Goals**:
  - 将 Focal Loss 的困难样本挖掘能力与 Label Smoothing 的正则化能力融合为单一损失函数
  - 引入类别级权重，对容易混淆的类别施加更大的惩罚
  - 通过对比损失在特征空间中显式分离混淆字符对
  - 全面监控所有已知混淆对的混淆情况
- **Non-Goals**:
  - 不涉及 GAN/CycleGAN 域对齐（属于独立的数据层优化）
  - 不涉及课程学习调度器（可作为后续独立变更）
  - 不修改模型骨干架构（仅新增 `forward_features()` 访问接口）
  - 不变更数据增强策略

## Decisions

### Decision 1: 融合式 FocalLabelSmoothingLoss 而非叠加式
- **选择**: 将 $(1-p_t)^\gamma$ 因子直接乘入 Label Smoothing CE 的计算中
- **原因**: 叠加式（当前实现）中基础 CE 仍然主导梯度方向，Focal 仅是辅助修正；融合式确保每一个样本的梯度贡献都经过 focal 权重调制
- **替代方案考虑**:
  - 纯 Focal Loss（无 smoothing）: 不利于泛化，容易过拟合到特定像素模式
  - Focal + CE 叠加（当前方式）: 简单字符的 CE 梯度仍会淹没困难样本信号

### Decision 2: gamma=3.0（原 2.0）
- **选择**: 提高 gamma 到 3.0
- **原因**: 像素字符任务中简单样本（W, K, M 等独特字符）占比约 60%，需要更激进的抑制。gamma=3 时，p=0.9 的简单样本权重仅为 0.001，而 p=0.5 的困难样本权重为 0.125，差距扩大为 125 倍（gamma=2 时仅 81 倍）
- **风险**: gamma 过高可能导致训练初期不稳定 → 缓解：前 5 个 epoch 使用 warmup，gamma 从 1.0 线性增长到 3.0

### Decision 3: 对比损失使用 cosine distance + margin
- **选择**: 基于 cosine distance 的 margin-based contrastive loss
- **替代方案**:
  - Triplet Loss: 需要三元组挖掘，计算复杂度高且 batch 构造有要求
  - Center Loss: 仅拉紧类内，不直接推开混淆对
  - Cosine margin loss: 直接在归一化特征空间上操作，对嵌入尺度不敏感，适合混合了多种损失的场景

### Decision 4: 在 class-mean embedding 上计算而非 sample-pair
- **选择**: 对 batch 内同一类别的所有样本取特征均值，再计算类间距离
- **原因**: 逐样本对比的计算复杂度为 $O(n^2)$；class-mean 方式仅需 $O(K^2)$（K=混淆对数量），极其高效，且对 batch 内样本数量波动更鲁棒

## Risks / Trade-offs
- **训练速度**: 对比损失需要额外的 `forward_features()` 调用 → 实际开销极小，因为特征已在 forward 中计算，仅需暴露中间结果
- **超参敏感性**: `lambda_contrastive`, `margin`, `gamma` 等新增超参 → 缓解：设置合理默认值并在 proposal 中固定推荐值
- **Batch 构成依赖**: 对比损失要求 batch 中包含混淆对的两个类 → 缓解：36 类、batch_size=64 时概率极高；若 batch 中无混淆对则对比损失自动为 0

## Open Questions
- gamma warmup 的具体调度是否需要独立配置（线性、cosine、step），还是直接硬编码线性 warmup？
- 是否需要支持运行时动态更新混淆对列表（例如根据验证集混淆矩阵自动发现新的混淆对）？
