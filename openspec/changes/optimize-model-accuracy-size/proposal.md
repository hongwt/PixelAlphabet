# Proposal: Optimize Model Accuracy and Size

## Metadata
- **Change ID**: optimize-model-accuracy-size
- **Created**: 2026-01-07
- **Status**: Proposed
- **Priority**: High

## Problem Statement

当前PixelNet模型在实际使用中存在两个关键问题：

### 问题1: 模型对字体和字体大小敏感，准确率不稳定
- **现象**: 测试时发现模型对不同字体和字体大小的识别准确率差异较大
- **影响**: 在真实场景中遇到未见过的字体或字体大小时，识别准确率显著下降
- **根本原因**: 
  - 训练数据的字体多样性不足
  - 数据增强策略未充分覆盖字体变化
  - 模型可能对特定字体特征过拟合

### 问题2: 模型参数过多，推理速度慢且容易过拟合
- **现象**: 
  - 当前模型约9-10M参数
  - 推理时间较长
  - 训练过程可能出现过拟合
- **影响**: 
  - 部署成本高
  - 推理延迟大，影响用户体验
  - 泛化能力受限
- **根本原因**: 
  - 模型架构复杂度过高（4个ResBlock，通道数512）
  - 空间注意力机制增加计算开销
  - FC层参数占比较大

## Proposed Solution

### 目标1: 提升模型准确率和鲁棒性

**策略A: 增强训练数据多样性**
- 扩展字体库：从当前3个字体增加到10+个字体（serif, sans-serif, monospace, handwriting等）
- 动态字体大小：训练时使用更大范围的字体大小（10px-24px，当前12-22px）
- 位置抖动增强：增加位置偏移范围（±6px，当前±4px）

**策略B: 优化数据增强策略**
- 增加字体变形：添加font stretch/condensed变换
- 模拟字体渲染差异：antialiasing, hinting variations
- 增强颜色/对比度变化：扩大jitter范围
- 添加轻微模糊/锐化：模拟不同图像质量

**策略C: 改进损失函数**
- 使用Label Smoothing减少过拟合
- 对易混淆字符对（0/O, 1/I/l, 5/S等）增加对比学习损失
- 添加焦点损失(Focal Loss)处理难分类样本

### 目标2: 减小模型大小和参数量

**策略A: 轻量化架构设计**
- 减少ResBlock数量：从4个减少到2-3个
- 降低通道数：512->256或128（根据性能权衡）
- 简化初始卷积层：单层Conv+BN替代双层结构
- 考虑使用深度可分离卷积(Depthwise Separable Conv)

**策略B: 高效注意力机制**
- 简化Spatial Attention：使用更小的kernel size（3->1）
- 或替换为更轻量的SENet-style Channel Attention
- 评估是否完全移除注意力模块（对比实验）

**策略C: 模型压缩技术**
- 知识蒸馏：使用当前大模型作为教师模型
- 量化感知训练(QAT)：INT8量化
- 剪枝：移除冗余通道/滤波器

**目标参数量**: 从9-10M减少到2-3M（减少70%）
**目标推理时间**: 减少50%以上
**准确率目标**: 保持或提升当前水平

## Impact Analysis

### Benefits
- ✅ 提高模型在不同字体/字体大小下的鲁棒性和准确率
- ✅ 减少推理延迟，改善用户体验
- ✅ 降低模型大小，便于部署（边缘设备、移动端）
- ✅ 减少过拟合风险，提升泛化能力
- ✅ 降低训练和推理的计算资源需求

### Risks
- ⚠️ 轻量化可能导致准确率下降（需要通过实验验证）
- ⚠️ 需要重新训练模型，可能需要较长时间
- ⚠️ 数据增强策略变化可能需要调整超参数
- ⚠️ 知识蒸馏等技术增加训练复杂度

### Breaking Changes
- ❌ 模型架构变化，需要重新训练
- ❌ 旧模型checkpoint不兼容（需要migration或重新训练）
- ✅ 推理接口保持兼容（输入输出格式不变）

### Affected Components
- `src/model.py` - 模型架构重构
- `src/train.py` - 损失函数、训练策略更新
- `src/data_generator.py` - 数据生成策略增强
- `src/dataset.py` - 数据增强pipeline更新
- `tests/test_model.py` - 模型测试更新
- `checkpoints/` - 新模型权重

## Alternatives Considered

### Alternative 1: 只优化数据增强，不改模型
- **优点**: 实施简单，不破坏现有模型
- **缺点**: 无法解决模型大小和推理速度问题
- **结论**: 不采纳，需要同时解决两个目标

### Alternative 2: 使用预训练模型（MobileNetV3, EfficientNet等）
- **优点**: 成熟的轻量化架构，参数少
- **缺点**: 为224x224设计，不适合24x24小图，需要大量修改
- **结论**: 不采纳，自定义架构更适合24x24场景

### Alternative 3: 只做模型压缩（量化、剪枝），不改架构
- **优点**: 实施相对简单
- **缺点**: 压缩效果有限（通常30-50%），无法达到70%目标
- **结论**: 作为补充手段，但不作为主要方案

## Success Metrics

### 准确率指标
- **基准**: 当前测试集准确率（需要记录baseline）
- **目标**: 
  - 总体准确率提升 2-5%
  - 易混淆字符对准确率提升 5-10%
  - 不同字体下准确率方差 < 5%

### 模型大小指标
- **基准**: 9-10M参数，~40MB模型文件
- **目标**: 
  - 参数量 ≤ 3M（减少70%）
  - 模型文件 ≤ 12MB

### 推理速度指标
- **基准**: 需要测量当前推理时间（单张图片，CPU/GPU）
- **目标**: 
  - CPU推理时间减少50%
  - GPU推理时间减少50%
  - Batch推理吞吐量提升50%

### 泛化能力指标
- **目标**: 
  - 在未见过字体的测试集上准确率 > 90%
  - 训练集vs验证集准确率差距 < 3%

## Implementation Plan

详见 [tasks.md](./tasks.md)

## Dependencies
- 依赖 `data-generation` 规范（扩展字体库）
- 依赖 `recognition` 规范（模型架构）

## Timeline
- **Phase 1**: 数据增强和字体扩展（2-3天）
- **Phase 2**: 轻量化模型设计和实验（3-5天）
- **Phase 3**: 损失函数优化和训练（2-3天）
- **Phase 4**: 模型压缩（可选，2-3天）
- **Phase 5**: 测试和验证（2天）
- **Total**: 约2周

## References
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
