# 快速开始：模型逐层可视化

## 安装依赖

```bash
# 确保已激活虚拟环境
venv\Scripts\activate

# 安装新增的依赖（如果还没安装）
pip install matplotlib
```

## 运行可视化

### 最简单的方式

```bash
python visualize_layers.py
```

这会自动：
- 加载训练好的模型
- 选择一张测试图片
- 生成所有层的可视化
- 保存到 `logs/layer_visualization/`

### 查看结果

```bash
# 启动 TensorBoard
tensorboard --logdir logs/layer_visualization

# 在浏览器中打开
# http://localhost:6006
```

### 在 TensorBoard 中你会看到

1. **IMAGES 标签**：
   - `0_Input/original` - 原始输入图片
   - `1_conv1_channels` - 第一层卷积的 16 个通道
   - `2_bn1_channels` - 批归一化后的特征
   - `3_relu1_channels` - ReLU 激活后的特征
   - `4_res_block1_channels` - 残差块1输出（128通道）
   - `5_res_block2_channels` - 残差块2输出（256通道，降采样）
   - `6_attention_channels` - SE注意力机制输出
   - `7_global_pool_channels` - 全局池化后的特征

2. **GRAPHS 标签**：
   - 完整的模型结构图

3. **各层的详细可视化图**：
   - 4×4 的特征图网格
   - 每个通道都有独立的热力图
   - 颜色条显示激活值范围

## 自定义使用

### 指定特定图片

```bash
python -m src.inference ^
    --checkpoint checkpoints/best_model.pth ^
    --image data/test/A/image_0001.png ^
    --visualize
```

### 在代码中使用

```python
from src.inference import CharacterPredictor

# 创建预测器
predictor = CharacterPredictor("checkpoints/best_model.pth")

# 对某张图片进行可视化
predictor.visualize_layer_outputs("path/to/your/image.png")
```

## 理解输出

- **浅层（Conv1）**: 看到边缘、线条等低级特征
- **中层（ResBlocks）**: 看到笔画组合、字符部件
- **深层（Attention）**: 看到注意力机制如何强调重要特征
- **输出层**: 看到最终的特征向量（用于分类）

## 快速测试

运行测试脚本验证一切正常：

```bash
python test_visualization.py
```

## 故障排除

**问题**: 找不到 matplotlib
**解决**: `pip install matplotlib`

**问题**: 找不到模型文件
**解决**: 确保 `checkpoints/best_model.pth` 存在

**问题**: TensorBoard 无法启动
**解决**: 确保已安装 tensorboard (`pip install tensorboard`)

**问题**: 可视化图片不显示
**解决**: 检查 `logs/layer_visualization/` 目录是否存在并有内容

## 更多信息

详细文档请参见 `LAYER_VISUALIZATION.md`
