# 模型逐层可视化功能 - 实现总结

## 已完成的功能

✅ **核心功能实现**
- 在 `src/inference.py` 中添加了 `visualize_layer_outputs()` 方法
- 可以捕获并可视化模型每一层的输出
- 支持卷积层、残差块、注意力层和全连接层的可视化
- 使用 TensorBoard 进行交互式查看

✅ **可视化的层**
1. 输入图片（原始 3×24×24）
2. Conv1 层输出（64 通道）
3. BatchNorm + ReLU 输出
4. ResBlock1 输出（128 通道）
5. ResBlock2 输出（256 通道，降采样到 12×12）
6. SE Attention 输出（注意力加权后）
7. Global Average Pooling 输出
8. 模型计算图

✅ **可视化方式**
- **特征图**: 4×4 网格显示前 16 个通道
- **热力图**: 使用 viridis 配色方案
- **归一化**: 每个通道独立归一化以便更好观察
- **颜色条**: 显示激活值的范围
- **通道网格**: 以图像形式显示所有通道

✅ **便捷工具**
- `visualize_layers.py` - 一键运行可视化
- `test_visualization.py` - 测试验证脚本
- `LAYER_VISUALIZATION.md` - 详细使用文档
- `QUICKSTART_VISUALIZATION.md` - 快速入门指南

## 使用方法

### 方法 1: 使用便捷脚本（最简单）

```bash
python visualize_layers.py
```

### 方法 2: 命令行参数

```bash
python -m src.inference \
    --checkpoint checkpoints/best_model.pth \
    --image data/test/A/image_0001.png \
    --visualize \
    --log-dir logs/layer_visualization
```

### 方法 3: Python 代码

```python
from src.inference import CharacterPredictor

predictor = CharacterPredictor("checkpoints/best_model.pth")
predictor.visualize_layer_outputs("path/to/image.png")
```

## 查看结果

1. 运行可视化后，启动 TensorBoard：
   ```bash
   tensorboard --logdir logs/layer_visualization
   ```

2. 在浏览器打开 `http://localhost:6006`

3. 在 TensorBoard 中查看：
   - **IMAGES** 标签：查看各层特征图
   - **GRAPHS** 标签：查看模型结构
   - 点击每个层名查看详细信息

## 技术细节

### Hook 机制
使用 PyTorch 的 `register_forward_hook()` 捕获每一层的输出：

```python
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

hook = model.conv1.register_forward_hook(get_activation('conv1'))
```

### 可视化流程
1. 预处理输入图片
2. 注册 forward hooks 到各个层
3. 执行前向传播
4. 从 hooks 中提取激活值
5. 为每一层创建可视化图表
6. 保存到 TensorBoard
7. 移除 hooks 释放资源

### 特征图处理
- 选择前 16 个通道（避免过多）
- 每个通道独立归一化到 [0, 1]
- 使用 matplotlib 生成 4×4 网格
- 添加颜色条显示数值范围

## 输出示例

### 观察结果
- **Conv1**: 检测边缘、线条等低级特征
- **ResBlock1**: 检测笔画组合
- **ResBlock2**: 检测字符部件（降采样后）
- **Attention**: 高亮重要特征通道
- **Global Pool**: 聚合空间信息

### 文件结构
```
logs/layer_visualization/
├── events.out.tfevents.xxx  # TensorBoard 事件文件
└── [可视化数据]
```

## 依赖要求

新增依赖：
- `matplotlib>=3.5.0` - 用于创建可视化图表

已在 `requirements.txt` 中添加。

## 测试验证

运行测试脚本验证功能：
```bash
python test_visualization.py
```

测试结果：
```
✓ 所有必要的库都已安装
✓ 找到模型检查点: checkpoints\best_model.pth
✓ 找到测试图片: data\test\0\...
✓ 预测结果: '0' (置信度: 0.9322)
✓ 可视化成功!
✓ 所有测试通过！
```

## 实际应用场景

1. **模型调试**: 查看每一层是否学到了合理的特征
2. **理解模型**: 直观了解图片如何被逐步转换
3. **特征分析**: 观察哪些特征对分类最重要
4. **错误诊断**: 分析预测错误时各层的表现
5. **模型比较**: 比较不同模型架构的特征提取

## 下一步改进

可以考虑添加：
- [ ] Grad-CAM 热力图（显示模型关注的区域）
- [ ] 激活值统计分析
- [ ] 批量图片对比可视化
- [ ] 特征相似度分析
- [ ] 导出高质量图片功能

## 相关文件

- `src/inference.py` - 主实现文件
- `src/model.py` - 模型定义
- `visualize_layers.py` - 便捷脚本
- `test_visualization.py` - 测试脚本
- `LAYER_VISUALIZATION.md` - 详细文档
- `QUICKSTART_VISUALIZATION.md` - 快速指南

## 参考资源

- [PyTorch Hooks 文档](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks)
- [TensorBoard 教程](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
- [CNN 可视化](https://cs231n.github.io/understanding-cnn/)
