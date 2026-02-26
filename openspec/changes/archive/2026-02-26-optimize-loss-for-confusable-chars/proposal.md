# Change: Optimize Loss Function & Training for Confusable Character Pairs

## Why
当前训练流程中，Focal Loss 仅作为 `lambda=0.5` 的附加项叠加在 Label Smoothing CE 之上，无法有效放大困难样本（Hard Examples）的梯度信号。在 24x24 像素的极低分辨率下，形态极度近似的字符对（7/T, 8/B, 0/D, 5/S, 2/Z, Q/0, 1/L, 6/G）仅凭一两个像素的差异来区分，标准 CE 损失中大量"简单字符"（如 W, K, M）的梯度贡献会彻底淹没这些困难对的学习信号。此外，训练脚本缺少课程学习调度和针对混淆对的特征空间约束，导致模型在相似字符上的区分能力受限。

## What Changes
- **Focal Label Smoothing Loss**: 将 Focal Loss 的 $(1-p_t)^\gamma$ 衰减因子直接融入 Label Smoothing CE，替代当前的简单叠加方式，使单一损失函数同时具备抗过拟合（smoothing）和困难样本挖掘（focal weighting）能力
- **类别级 alpha 权重**: 为 Focal Loss 引入 36 维类别权重向量，对容易混淆的类别（7, T, 8, B, 0, D, 5, S, 2, Z, Q, 1, L, 6, G）赋予更高惩罚权重
- **混淆对对比损失 (Confusion Pair Contrastive Loss)**: 新增一个基于特征嵌入空间的 margin-based contrastive loss，针对预定义的混淆字符对，显式拉大同 batch 内这些类别在倒数第二层特征空间中的距离
- **扩展混淆监控**: 将当前仅监控 Q/0、8/B 两对扩展为覆盖所有关键混淆对（7/T, 8/B, 0/D, 5/S, 2/Z, Q/0, 1/L, 6/G）
- **模型特征提取接口**: PixelNet 新增 `forward_features()` 方法，返回 GAP 之后、分类头之前的 256 维嵌入向量，供对比损失使用
- **Focal gamma 参数提升**: 默认 gamma 从 2.0 提升至 3.0，更激进地抑制简单样本的梯度贡献

## Impact
- Affected specs: `recognition`
- Affected code:
  - `src/loss.py` — 新增 `FocalLabelSmoothingLoss`, `ConfusionPairContrastiveLoss`; 修改 `CombinedLoss`, `create_loss_function`
  - `src/model.py` — PixelNet 新增 `forward_features()` 方法
  - `src/train.py` — 修改训练循环以支持对比损失、扩展混淆监控
  - `tests/test_loss.py` — 新增对应测试
