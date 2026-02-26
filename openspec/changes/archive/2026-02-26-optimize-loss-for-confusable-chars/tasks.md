## 1. Loss Functions (`src/loss.py`)
- [x] 1.1 实现 `FocalLabelSmoothingLoss` 类：将 Focal 权重因子 $(1-p_t)^\gamma$ 融入 Label Smoothing CE，支持类别级 alpha 权重向量
- [x] 1.2 实现 `ConfusionPairContrastiveLoss` 类：基于 cosine distance 的 margin-based 对比损失，预定义 8 个混淆对
- [x] 1.3 修改 `CombinedLoss` 类：使用 `FocalLabelSmoothingLoss` 作为基损失，可选集成 `ConfusionPairContrastiveLoss`
- [x] 1.4 更新 `create_loss_function` 工厂函数：支持新参数传递（alpha_weights, gamma, lambda_contrastive, margin）
- [x] 1.5 定义 `CONFUSED_PAIRS` 和 `get_confusable_alpha_weights()` 工具常量/函数

## 2. Model Interface (`src/model.py`)
- [x] 2.1 为 `PixelNet` 新增 `forward_features()` 方法，返回 GAP 后 256 维嵌入向量

## 3. Training Loop (`src/train.py`)
- [x] 3.1 修改 `train_one_epoch()`：支持对比损失（调用 `model.forward_features()` + contrastive loss 计算）
- [x] 3.2 扩展 `log_confusion_matrix()`：覆盖所有 8 个混淆对 (7/T, 8/B, 0/D, 5/S, 2/Z, Q/0, 1/L, 6/G)
- [x] 3.3 更新 `main()` 中的损失函数创建逻辑，传入新参数

## 4. Tests (`tests/test_loss.py`)
- [x] 4.1 `FocalLabelSmoothingLoss` 单元测试：验证 focal 权重放大困难样本、alpha 类别权重生效
- [x] 4.2 `ConfusionPairContrastiveLoss` 单元测试：验证有/无混淆对在 batch 中的行为
- [x] 4.3 更新 `CombinedLoss` 测试：覆盖新的组合方式
- [x] 4.4 回归测试：确保旧接口 `create_loss_function('combined')` 仍正常工作
