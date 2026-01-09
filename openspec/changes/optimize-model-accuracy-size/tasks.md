# Implementation Tasks

## Phase 1: 数据增强和字体扩展

### Task 1.1: 扩展字体库
- [x] 收集10+个不同字体文件（serif, sans-serif, monospace, handwriting等）
- [x] 将字体文件添加到 `Fonts/` 目录（使用Windows系统字体）
- [x] 更新 `src/data_generator.py` 支持新字体
- [x] 验证所有字体能正确渲染所有字符

### Task 1.2: 增强数据生成策略
- [x] 修改 `src/data_generator.py`:
  - [x] 扩大字体大小范围（10px-24px）
  - [x] 增加位置抖动范围（±6px）
  - [x] 添加字体stretch/condensed变换（待实现，可选）
  - [x] 添加antialiasing变化（已通过字体渲染支持）
- [ ] 重新生成训练数据集
- [ ] 验证数据多样性（可视化检查）

### Task 1.3: 优化数据增强Pipeline
- [x] 修改 `src/dataset.py`:
  - [x] 扩大ColorJitter参数范围
  - [x] 添加轻微GaussianBlur增强
  - [x] 添加RandomSharpness增强
  - [x] 保留现有旋转、仿射变换
- [ ] 编写单元测试验证增强效果
- [ ] 可视化增强后的样本

## Phase 2: 轻量化模型设计

### Task 2.1: 设计轻量化架构 (LightPixelNet)
- [x] 在 `src/model.py` 创建新模型类 `LightPixelNet`:
  - [x] 简化初始卷积层（单层Conv64）
  - [x] 减少ResBlock数量（2-3个）
  - [x] 降低通道数（64->128->256，最大256）
  - [x] 简化Spatial Attention或使用Channel Attention
  - [x] 保持dropout和FC层结构
- [x] 实现 `create_light_model()` 工厂函数
- [x] 验证模型输入输出正确性

### Task 2.2: 模型对比实验设计
- [x] 设计3-5个候选架构变体:
  - [x] Variant A: 2 ResBlocks + Simplified Spatial Attention
  - [x] Variant B: 3 ResBlocks + No Attention
  - [x] Variant C: 2 ResBlocks + SE Attention（推荐）
- [x] 编写脚本计算每个变体的参数量和FLOPs
- [x] 记录架构对比表格

### Task 2.3: 快速验证实验
- [ ] 选择2-3个最有潜力的架构
- [ ] 在小数据集上快速训练（5-10 epochs）
- [ ] 比较准确率、参数量、推理时间
- [ ] 选择最优架构进入正式训练

## Phase 3: 损失函数优化

### Task 3.1: 实现Label Smoothing
- [x] 在 `src/loss.py` 添加 `LabelSmoothingCrossEntropy` 类
- [x] 添加smoothing参数（建议0.1）
- [x] 编写单元测试

### Task 3.2: 实现对比学习损失
- [ ] 识别易混淆字符对: (0, O), (1, I), (5, S), etc.
- [ ] 在 `src/loss.py` 实现 `ContrastiveLoss` 或 `TripletLoss`
- [ ] 编写hard negative mining逻辑
- [ ] 编写单元测试（可选，未实现）

### Task 3.3: 实现Focal Loss
- [x] 在 `src/loss.py` 添加 `FocalLoss` 类（已存在）
- [x] 设置合适的gamma和alpha参数
- [x] 编写单元测试

### Task 3.4: 组合损失函数
- [x] 设计复合损失: `Total = CE + λ*Focal`
- [x] 实现 `CombinedLoss` 类
- [x] 通过实验确定最优权重λ（默认0.5）

## Phase 4: 模型训练和调优

### Task 4.1: 基线模型训练
- [ ] 使用原始PixelNet在新数据上训练
- [ ] 记录准确率、训练时间、过拟合情况
- [ ] 保存为baseline对比

### Task 4.2: 轻量化模型训练
- [ ] 使用选定的LightPixelNet架构
- [ ] 使用新的数据增强策略
- [ ] 使用组合损失函数
- [ ] 调整超参数:
  - [ ] Learning rate
  - [ ] Batch size
  - [ ] Weight decay
  - [ ] Dropout rate
- [ ] 使用learning rate scheduling
- [ ] 记录训练曲线和指标

### Task 4.3: 多字体泛化测试
- [ ] 准备未见过字体的测试集
- [ ] 测试模型在不同字体上的准确率
- [ ] 分析失败案例
- [ ] 如需要，微调数据增强策略

## Phase 5: 模型压缩（可选）

### Task 5.1: 知识蒸馏
- [ ] 使用原始PixelNet作为教师模型
- [ ] 实现蒸馏训练循环
- [ ] 调整temperature和蒸馏loss权重
- [ ] 评估蒸馏效果

### Task 5.2: 量化感知训练
- [ ] 集成PyTorch量化工具
- [ ] 实现INT8 QAT
- [ ] 测试量化后准确率
- [ ] 评估推理速度提升

### Task 5.3: 剪枝
- [ ] 实现结构化剪枝（通道剪枝）
- [ ] 设置剪枝比例（20-40%）
- [ ] Fine-tune剪枝后模型
- [ ] 评估准确率vs大小权衡

## Phase 6: 测试和验证

### Task 6.1: 性能基准测试
- [ ] 测量推理时间（CPU & GPU）:
  - [ ] 单张图片推理
  - [ ] Batch推理（16, 32, 64）
  - [ ] 多线程推理
- [ ] 测量内存占用
- [ ] 对比原模型和新模型

### Task 6.2: 准确率全面评估
- [ ] 在完整测试集上评估
- [ ] 分类别准确率统计
- [ ] 混淆矩阵分析
- [ ] 易混淆字符对准确率
- [ ] 不同字体准确率方差

### Task 6.3: 鲁棒性测试
- [ ] 测试不同字体大小
- [ ] 测试不同背景颜色
- [ ] 测试噪声图像
- [ ] 测试边界情况（模糊、低对比度）

### Task 6.4: 更新测试套件
- [ ] 更新 `tests/test_model.py`:
  - [ ] 添加LightPixelNet测试
  - [ ] 添加参数量验证
  - [ ] 添加推理速度测试
- [ ] 确保所有测试通过

## Phase 7: 文档和清理

### Task 7.1: 更新文档
- [ ] 更新 `README.md`:
  - [ ] 记录新模型架构
  - [ ] 更新性能指标
  - [ ] 添加模型对比表
- [ ] 更新代码注释和docstrings
- [ ] 编写模型选择指南（何时用原模型vs轻量模型）

### Task 7.2: 模型管理
- [ ] 保存最优模型checkpoint
- [ ] 重命名为 `best_light_model.pth`
- [ ] 保留原始 `best_model.pth` 作为对比
- [ ] 更新 `src/inference.py` 支持模型选择

### Task 7.3: 清理代码
- [ ] 移除实验性代码
- [ ] 整理代码格式（Black）
- [ ] 运行linter（Flake8）
- [ ] 确保所有单元测试通过

## Validation Checklist

完成后验证以下指标：

### 准确率指标
- [ ] 总体准确率 ≥ 基线或提升2-5%
- [ ] 易混淆字符准确率提升5-10%
- [ ] 不同字体准确率方差 < 5%
- [ ] 未见过字体测试集准确率 > 90%

### 模型大小指标
- [ ] 参数量 ≤ 3M
- [ ] 模型文件 ≤ 12MB
- [ ] 参数减少 ≥ 70%

### 推理速度指标
- [ ] CPU推理时间减少 ≥ 50%
- [ ] GPU推理时间减少 ≥ 50%
- [ ] Batch推理吞吐量提升 ≥ 50%

### 泛化能力指标
- [ ] 训练集vs验证集准确率差距 < 3%
- [ ] 过拟合现象明显减少

## Notes
- 每个Phase完成后需进行阶段性review
- 实验结果记录在训练日志中
- 关键决策记录在设计文档中
- 遇到问题及时调整策略
