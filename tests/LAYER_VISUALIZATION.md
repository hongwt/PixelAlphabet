# 模型逐层可视化指南

本指南说明如何可视化 PixelNet 模型每一层的输出，以便直观了解图片在经过每个神经网络层处理后的数据变化。

## 功能说明

可视化功能会捕获并显示以下层的输出：

1. **输入层** - 原始输入图片 (3×24×24)
2. **Conv1** - 第一个卷积层输出 (64×24×24)
3. **BN1 + ReLU** - 批归一化和激活函数后 (64×24×24)
4. **ResBlock1** - 第一个残差块输出 (128×24×24)
5. **ResBlock2** - 第二个残差块输出 (256×12×12)
6. **Attention** - SE注意力机制输出 (256×12×12)
7. **Global Pool** - 全局平均池化输出 (256×1×1)
8. **FC层** - 全连接层的特征向量

## 使用方法

### 方法一：使用便捷脚本（推荐）

```bash
# 激活虚拟环境
venv\Scripts\activate

# 运行可视化脚本
python visualize_layers.py
```

### 方法二：使用命令行参数

```bash
# 激活虚拟环境
venv\Scripts\activate

# 对单张图片进行可视化
python -m src.inference ^
    --checkpoint checkpoints/best_model.pth ^
    --image data/test/A/image_0001.png ^
    --visualize ^
    --log-dir logs/layer_visualization
```

### 方法三：在Python代码中使用

```python
from src.inference import CharacterPredictor

# 创建预测器
predictor = CharacterPredictor("checkpoints/best_model.pth")

# 可视化某张图片的逐层输出
log_dir = predictor.visualize_layer_outputs(
    image="data/test/A/image_0001.png",
    log_dir="./logs/layer_visualization"
)

print(f"可视化已保存到: {log_dir}")
```

## 查看可视化结果

1. 运行可视化脚本后，打开 TensorBoard：

```bash
tensorboard --logdir logs/layer_visualization
```

2. 在浏览器中访问 `http://localhost:6006`

3. 在 TensorBoard 中查看不同标签：

   - **IMAGES** - 查看每一层的特征图（多通道可视化）
   - **GRAPHS** - 查看模型的网络结构图
   - 点击每个层的名称查看详细信息

## 可视化内容说明

### 卷积层可视化

对于卷积层（Conv1, ResBlock1, ResBlock2等），可视化会显示：

- **特征图网格**: 4×4 的网格显示前 16 个通道的特征图
- **颜色映射**: 使用 viridis 色图，暖色表示高激活值，冷色表示低激活值
- **每个通道独立归一化**: 便于观察每个通道的模式

### 注意力层可视化

SE Block（注意力机制）的可视化显示：
- 注意力加权后的特征图
- 哪些通道被强调（高激活值）
- 哪些通道被抑制（低激活值）

### 全连接层可视化

FC层输出显示为：
- 条形图展示特征向量
- 横轴是特征索引
- 纵轴是激活值

## 理解可视化结果

### 浅层（Conv1, BN1）
- 检测低级特征：边缘、纹理、简单形状
- 特征图通常显示清晰的边界和基本模式

### 中层（ResBlock1, ResBlock2）
- 检测更复杂的模式：笔画组合、字符部件
- 特征图开始变得更抽象
- 空间分辨率降低（ResBlock2 降采样到 12×12）

### 深层（Attention, Global Pool）
- 高级语义特征：整体字符模式
- 注意力机制突出重要特征
- 空间信息被聚合

### 输出层（FC）
- 特征向量表示整个字符的抽象表示
- 这些特征用于最终分类

## 技巧与建议

1. **比较不同字符**: 可视化多个不同字符，观察模型如何区分它们
   
2. **观察混淆案例**: 对于容易混淆的字符对（如 0 和 Q, 8 和 B），比较它们的特征图差异

3. **检查失败案例**: 对预测错误的图片进行可视化，找出问题所在

4. **注意力分析**: 关注 SE Block 的输出，看模型在关注什么

## 常见问题

**Q: 为什么有些特征图看起来很混乱？**
A: 深层网络的特征图通常是抽象的高级特征，不容易直接解释。浅层特征图通常更容易理解。

**Q: 可以可视化更多通道吗？**
A: 代码默认显示前 16 个通道。你可以修改 `inference.py` 中的 `num_vis_channels` 参数。

**Q: 如何比较两张图片的处理过程？**
A: 分别为两张图片生成可视化，使用不同的 `log_dir` 参数，然后在 TensorBoard 中切换查看。

## 扩展功能

如果需要更高级的可视化，可以考虑：

1. **Grad-CAM**: 显示模型关注图片的哪些区域
2. **激活图统计**: 分析每层激活值的分布
3. **层间相似度**: 比较不同样本在同一层的相似性
4. **特征重要性**: 分析哪些通道对分类最重要

## 相关文件

- `src/inference.py` - 包含 `visualize_layer_outputs()` 方法
- `visualize_layers.py` - 便捷的可视化脚本
- `src/model.py` - PixelNet 模型定义

## 参考资源

- [TensorBoard 文档](https://www.tensorflow.org/tensorboard)
- [CNN 可视化理解](https://cs231n.github.io/understanding-cnn/)
- [Feature Visualization](https://distill.pub/2017/feature-visualization/)
