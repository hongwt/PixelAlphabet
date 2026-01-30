# Design: Optimize PixelNet Architecture

## Context

PixelNet 是一个用于 24x24 字符识别的轻量级 CNN。当前架构虽然有效，但在以下方面有改进空间：
- 相似字符区分（0/O, 8/B, I/1, 6/9）
- 特征提取深度
- 注意力机制覆盖范围

## Goals / Non-Goals

### Goals
- 提升相似字符识别准确率 2-5%
- 保持模型轻量（参数量 <2M）
- 保持推理速度（<10ms per image on CPU）
- 向后兼容旧版 checkpoint

### Non-Goals
- 不改变输入分辨率（保持 24x24）
- 不改变输出类别数（保持 36 类）
- 不引入新的外部依赖

## Decisions

### Decision 1: 使用 CoordConv 添加位置编码

**选择**: 在第一层卷积使用 CoordConv

**原因**:
- 字符识别中位置信息重要（区分 6/9, b/d, p/q）
- CoordConv 仅增加 2 个输入通道，开销极小
- 可通过参数禁用以保持向后兼容

**实现**:
```python
class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
    
    def forward(self, x):
        b, _, h, w = x.shape
        # 生成归一化坐标 [-1, 1]
        xx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        yy = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x = torch.cat([x, xx, yy], dim=1)
        return self.conv(x)
```

### Decision 2: 使用 CBAM 替代 SE Block

**选择**: 在网络末端使用 CBAM（Channel + Spatial Attention）

**原因**:
- SE Block 仅提供通道注意力
- 字符定位需要空间注意力辅助
- CBAM 是成熟且轻量的方案

**实现**:
```python
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = SEBlock(channels, reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention
```

### Decision 3: 增加网络深度（3 个 ResBlock）

**选择**: 64→64→128→256 三阶段 ResBlock

**原因**:
- 更深的网络能学习更抽象的特征
- 第一阶段保持通道数，减少早期信息损失
- 仅在最后阶段下采样（stride=2）

**通道配置**:
| Stage | In Channels | Out Channels | Stride | Output Size |
|-------|-------------|--------------|--------|-------------|
| Conv1 | 3 (+2 coord) | 64 | 1 | 24x24 |
| ResBlock1 | 64 | 64 | 1 | 24x24 |
| ResBlock2 | 64 | 128 | 1 | 24x24 |
| ResBlock3 | 128 | 256 | 2 | 12x12 |

### Decision 4: 优化分类头

**选择**: 移除首个 Dropout，添加 BatchNorm

**原因**:
- 双重 Dropout 可能过度正则化
- BatchNorm 提供额外正则化效果
- 现代分类器通常只在最后 FC 前使用 Dropout

**实现**:
```python
self.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Dropout(p=dropout_rate),
    nn.Linear(128, num_classes)
)
```

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| 参数量增加导致过拟合 | 中 | 保持 Dropout，监控验证集 loss |
| 推理速度下降 | 低 | CoordConv 和 CBAM 开销很小 |
| 旧 checkpoint 不兼容 | 中 | 添加 `strict=False` 加载选项 |
| 训练不稳定 | 低 | 使用预热学习率 |

## Migration Plan

1. **阶段 1**: 实现所有新模块，添加配置参数
2. **阶段 2**: 默认启用新架构，保留旧架构配置选项
3. **阶段 3**: 重新训练模型，对比性能
4. **阶段 4**: 确认性能提升后，更新默认 checkpoint

## Open Questions

1. 是否需要支持混合精度训练（FP16）？
2. 是否需要添加模型剪枝/量化支持？
