# 逐层可视化功能 - 功能展示

## 🎯 核心功能

### 你可以看到什么

当你运行可视化后，在 TensorBoard 中你将看到：

```
logs/layer_visualization/
│
├── 0_Input/
│   └── original                    # 原始输入图片 (3×24×24)
│
├── 1_conv1/
│   └── channels (16个特征图)        # 第一层卷积：边缘、线条检测
│
├── 2_bn1/
│   └── channels (16个特征图)        # 批归一化后的特征
│
├── 3_relu1/
│   └── channels (16个特征图)        # 激活后的特征
│
├── 4_res_block1/
│   └── channels (16个特征图)        # 残差块1：笔画组合 (128通道)
│
├── 5_res_block2/
│   └── channels (16个特征图)        # 残差块2：字符部件 (256通道, 降采样)
│
├── 6_attention/
│   └── channels (16个特征图)        # 注意力机制：突出重要特征
│
└── 7_global_pool/
    └── feature_vector              # 全局池化：最终特征向量
```

## 📊 可视化类型

### 1. 特征图网格 (4×4)
每一层显示前16个通道的特征图，以热力图形式展示：
- 🔴 **红色/黄色** = 高激活值（该特征被强烈检测到）
- 🔵 **蓝色/紫色** = 低激活值（该特征不明显）

### 2. 通道可视化
所有通道并排显示，便于快速浏览：
- 每个通道独立归一化
- 可以看到不同通道关注不同的特征

### 3. 模型结构图
在 GRAPHS 标签中，可以看到：
- 完整的模型架构
- 各层之间的连接关系
- 数据流向和张量形状

## 🔍 如何解读可视化

### 浅层 (Conv1, BN1, ReLU1)
```
特征: 低级视觉特征
示例: 
  - 水平边缘
  - 垂直边缘
  - 对角线
  - 简单纹理
```

### 中层 (ResBlock1, ResBlock2)
```
特征: 中级语义特征
示例:
  - 笔画的组合
  - 字符的角落
  - 字符的弧度
  - 部分字符结构
```

### 深层 (Attention, Global Pool)
```
特征: 高级抽象特征
示例:
  - 整体字符形状
  - 字符类别信息
  - 关键判别特征
  - 注意力权重分布
```

## 💡 使用技巧

### 1. 分析混淆字符
比较容易混淆的字符对：
```bash
python compare_characters.py --mode confusion
```

观察点：
- **0 vs Q**: 看中间是否有贯穿线
- **8 vs B**: 看右侧是否封闭
- **1 vs I**: 看顶部和底部的形状
- **5 vs S**: 看弯曲的方向

### 2. 调试预测错误
当模型预测错误时：
```python
# 可视化预测错误的图片
predictor.visualize_layer_outputs("wrong_prediction.png")
```

检查：
- 输入图片是否清晰？
- 哪一层开始出现问题？
- 注意力是否关注了错误的区域？

### 3. 理解注意力机制
SE Block (第6层) 显示哪些通道被强调：
- 高激活 = 重要特征
- 低激活 = 不重要特征

### 4. 验证数据增强
对增强后的图片进行可视化：
```python
# 查看增强是否合理
from src.dataset import apply_augmentation
augmented_img = apply_augmentation(original_img)
predictor.visualize_layer_outputs(augmented_img)
```

## 📈 实际案例

### 案例 1: 为什么能区分 0 和 Q？

查看可视化后发现：
- **Conv1**: 检测到 Q 中间的斜线
- **ResBlock2**: 强化斜线特征
- **Attention**: 增强中间区域的权重
- **结果**: 成功区分

### 案例 2: 为什么会混淆 8 和 B？

查看可视化后发现：
- **Conv1**: 两者的轮廓相似
- **ResBlock1**: 右侧特征模糊
- **Attention**: 没有足够强调右侧差异
- **改进**: 可以增加数据增强或调整注意力机制

### 案例 3: 注意力机制的作用

比较有无注意力的特征图：
- **无注意力**: 所有特征同等权重
- **有注意力**: 关键特征被放大，噪声被抑制

## 🎨 可视化示例命令

### 基础使用
```bash
# 最简单的方式
python visualize_layers.py
```

### 指定图片
```bash
python -m src.inference \
    --checkpoint checkpoints/best_model.pth \
    --image data/test/Q/test_Q_001.png \
    --visualize \
    --log-dir logs/Q_analysis
```

### 批量分析
```bash
# 分析所有类别
python compare_characters.py --mode all

# 只分析混淆字符对
python compare_characters.py --mode confusion
```

### 查看结果
```bash
# 启动 TensorBoard
tensorboard --logdir logs/layer_visualization

# 比较多个运行
tensorboard --logdir logs/comparison

# 所有类别
tensorboard --logdir logs/all_classes
```

## 🔧 高级定制

### 修改显示的通道数
编辑 `src/inference.py` 中的 `num_vis_channels` 参数：
```python
# 默认显示前16个通道
num_vis_channels = min(16, num_channels)

# 修改为显示32个通道（需要调整网格大小）
num_vis_channels = min(32, num_channels)
```

### 添加更多层
在 `visualize_layer_outputs()` 中注册新的 hook：
```python
# 例如：添加 fc1 层
hooks.append(self.model.fc1.register_forward_hook(
    get_activation('8_fc1')
))
```

### 自定义配色方案
修改热力图的配色：
```python
# 当前使用 'viridis'
im = ax.imshow(feature_map, cmap='viridis')

# 可以改为其他配色，如：
# 'hot', 'cool', 'jet', 'rainbow', 'gray'
im = ax.imshow(feature_map, cmap='hot')
```

## 📚 延伸阅读

- [TensorBoard 官方教程](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
- [CNN 可视化详解](https://cs231n.github.io/understanding-cnn/)
- [Feature Visualization](https://distill.pub/2017/feature-visualization/)
- [注意力机制可视化](https://distill.pub/2016/attention/)

## 🐛 常见问题

**Q: 为什么我看不到图片？**
A: 确保在 TensorBoard 中选择了 IMAGES 标签，并且加载了正确的 logdir。

**Q: 特征图看起来都是蓝色的？**
A: 这可能表示激活值很小。检查模型是否正确加载，或者尝试不同的图片。

**Q: 可以同时比较多张图片吗？**
A: 可以！为每张图片使用不同的 `log_dir`，然后在 TensorBoard 中切换查看。

**Q: TensorBoard 启动很慢？**
A: 如果 logs 目录中有很多旧数据，可以清理一下。TensorBoard 会加载所有事件文件。

**Q: 能导出高分辨率图片吗？**
A: 可以！在保存图片时调整 `figsize` 参数和 DPI：
```python
fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=150)
```

## ✨ 总结

这个可视化工具可以帮助你：
- ✅ 理解模型如何"看"图片
- ✅ 调试预测错误
- ✅ 分析混淆字符
- ✅ 验证模型改进
- ✅ 教学和演示

开始探索吧！🚀
