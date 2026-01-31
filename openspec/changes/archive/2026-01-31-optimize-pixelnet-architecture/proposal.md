# Change: Optimize PixelNet Architecture

## Why
当前 PixelNet 模型存在以下问题：
1. 注意力机制仅在最后使用，缺少空间注意力
2. 特征提取深度不足（仅 2 个 ResBlock），难以区分相似字符（0/O, 8/B, I/1）
3. FC 层双重 Dropout 可能过度正则化，且缺少 BatchNorm
4. 缺少位置编码，影响区分旋转相似字符（6/9, b/d）
5. 缺少多尺度特征融合

## What Changes
- 添加 CBAM（通道+空间注意力）模块替代单纯的 SE Block
- 增加 1 个 ResBlock 以增强特征提取能力
- 添加 CoordConv 位置编码层
- 优化 FC 分类头：添加 BatchNorm，移除冗余 Dropout
- 将 SE Block 集成到每个 ResBlock 中（可选配置）
- 添加多尺度卷积模块（可选配置）

## Impact
- Affected specs: `recognition`
- Affected code: `src/model.py`
- 模型参数量预计增加约 20-30%（仍保持轻量级 <2M 参数）
- 预期准确率提升 2-5%，尤其在相似字符区分上

## Architecture Changes

### Before (Current)
```
Input (3, 24, 24)
    ↓
Conv1 (3→64) + BN + ReLU
    ↓
ResBlock1 (64→128, stride=1)  → 24x24
    ↓
ResBlock2 (128→256, stride=2) → 12x12
    ↓
SE Attention
    ↓
GAP → Dropout → FC1 → ReLU → Dropout → FC2
    ↓
Output (36)
```

### After (Optimized)
```
Input (3, 24, 24)
    ↓
CoordConv (3+2→64) + BN + ReLU      # 位置编码
    ↓
ResBlockSE (64→64, stride=1)         # 内置 SE 注意力
    ↓
ResBlockSE (64→128, stride=1)        # 24x24
    ↓
ResBlockSE (128→256, stride=2)       # 12x12
    ↓
CBAM (通道+空间注意力)
    ↓
GAP → FC1 → BN → ReLU → Dropout → FC2
    ↓
Output (36)
```

## Design Decisions
1. **CoordConv vs 不添加位置编码**: 选择 CoordConv，因为字符识别中位置信息重要
2. **CBAM vs 仅 SE Block**: CBAM 同时提供通道和空间注意力，更适合字符定位
3. **3个 ResBlock vs 2个**: 增加深度但保持每个 block 轻量，避免过拟合
4. **FC 层优化**: 参考现代分类器设计，BN 放在 ReLU 前
