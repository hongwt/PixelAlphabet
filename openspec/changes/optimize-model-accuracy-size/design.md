# Design Document: Optimize Model Accuracy and Size

## Overview
本文档记录优化PixelNet模型准确率和大小的技术设计决策。

## Architecture Design

### 1. 轻量化模型架构 (LightPixelNet)

#### 1.1 设计原则
- **保持高分辨率**: 在24x24输入上尽量延迟下采样
- **减少冗余**: 移除过深的ResBlock层
- **高效注意力**: 使用轻量级注意力机制
- **参数效率**: 优先减少通道数而非层数

#### 1.2 架构对比

| Component | Original PixelNet | LightPixelNet | Reduction |
|-----------|-------------------|---------------|-----------|
| Initial Conv | 2 layers (3->64->64) | 1 layer (3->64) | -50% layers |
| ResBlock 1 | 64->128, stride=1 | 64->128, stride=1 | Same |
| ResBlock 2 | 128->256, stride=2 | 128->256, stride=2 | Same |
| ResBlock 3 | 256->512, stride=1 | **Removed** | -100% |
| ResBlock 4 | 512->512, stride=1 | **Removed** | -100% |
| Attention | Spatial (kernel=7) | SE or Spatial (kernel=1) | Simplified |
| FC Layers | 512->256->37 | 256->128->37 | -50% channels |
| **Total Params** | **~9-10M** | **~2-3M** | **~70%** |

#### 1.3 候选架构变体

**Variant A: Simplified Spatial Attention**
```
Input (3, 24, 24)
├─ Conv(3->64, 3x3) -> BN -> ReLU
├─ ResBlock(64->128, stride=1) -> (128, 24, 24)
├─ ResBlock(128->256, stride=2) -> (256, 12, 12)
├─ Spatial Attention(kernel=1) -> (256, 12, 12)
├─ Global Avg Pool -> (256,)
├─ FC(256->128) -> Dropout -> ReLU
└─ FC(128->37)
Params: ~2.5M
```

**Variant B: No Attention**
```
Input (3, 24, 24)
├─ Conv(3->64, 3x3) -> BN -> ReLU
├─ ResBlock(64->128, stride=1) -> (128, 24, 24)
├─ ResBlock(128->256, stride=2) -> (256, 12, 12)
├─ ResBlock(256->256, stride=1) -> (256, 12, 12)  // Extra ResBlock
├─ Global Avg Pool -> (256,)
├─ FC(256->128) -> Dropout -> ReLU
└─ FC(128->37)
Params: ~2.8M
```

**Variant C: SE Channel Attention**
```
Input (3, 24, 24)
├─ Conv(3->64, 3x3) -> BN -> ReLU
├─ ResBlock(64->128, stride=1) -> (128, 24, 24)
├─ ResBlock(128->256, stride=2) -> (256, 12, 12)
├─ SE Module (r=16) -> (256, 12, 12)
├─ Global Avg Pool -> (256,)
├─ FC(256->128) -> Dropout -> ReLU
└─ FC(128->37)
Params: ~2.4M
```

**Variant D: Depthwise Separable Conv** (Experimental)
```
Input (3, 24, 24)
├─ DepthwiseConv(3->64) -> BN -> ReLU
├─ DSResBlock(64->128, stride=1) -> (128, 24, 24)
├─ DSResBlock(128->256, stride=2) -> (256, 12, 12)
├─ SE Module (r=16) -> (256, 12, 12)
├─ Global Avg Pool -> (256,)
├─ FC(256->128) -> Dropout -> ReLU
└─ FC(128->37)
Params: ~1.5M
```

**推荐选择**: 先实验 Variant A 和 Variant C，根据准确率选择最优方案。

### 2. 数据增强策略

#### 2.1 生成阶段增强
```python
# 字体多样性
fonts = [
    "Arial", "Times New Roman", "Courier New",  # 标准字体
    "Georgia", "Verdana", "Trebuchet MS",       # Web字体
    "Consolas", "Lucida Console",               # 等宽字体
    "Comic Sans MS", "Brush Script MT"          # 手写体
]

# 字体大小: 10-24px (原12-22px)
font_size = random.randint(10, 24)

# 位置抖动: ±6px (原±4px)
offset_x = random.randint(-6, 6)
offset_y = random.randint(-6, 6)

# 字体变形
stretch_factor = random.uniform(0.8, 1.2)  # 拉伸/压缩

# 渲染选项
antialiasing = random.choice([True, False])
hinting = random.choice(['none', 'normal', 'strong'])
```

#### 2.2 训练阶段增强
```python
transforms.Compose([
    # 几何变换
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    
    # 颜色增强 (扩大范围)
    transforms.ColorJitter(
        brightness=0.4,  # ±40%
        contrast=0.4,    # ±40%
        saturation=0.3,  # ±30%
        hue=0.15         # ±15%
    ),
    
    # 模糊/锐化 (新增)
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))
    ], p=0.3),
    
    transforms.RandomApply([
        transforms.RandomAdjustSharpness(sharpness_factor=2.0)
    ], p=0.2),
    
    # 噪声
    AddGaussianNoise(mean=0, std=0.02),
    
    # 归一化
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

### 3. 损失函数设计

#### 3.1 Label Smoothing Cross Entropy
```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        n_classes = pred.size(1)
        # Smooth labels
        smooth_target = (1 - self.epsilon) * F.one_hot(target, n_classes) + \
                        self.epsilon / n_classes
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smooth_target * log_prob).sum(dim=1).mean()
        return loss
```

#### 3.2 Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            focal_loss *= self.alpha[target]
        
        return focal_loss.mean()
```

#### 3.3 Contrastive Loss (可选)
```python
# 易混淆字符对
CONFUSED_PAIRS = [
    (0, 10),   # '0' vs 'O'
    (1, 8),    # '1' vs 'I'
    (5, 18),   # '5' vs 'S'
    (2, 25),   # '2' vs 'Z'
    (6, 11),   # '6' vs 'b'
    (8, 11),   # '8' vs 'B'
]

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        # Extract embeddings before final FC layer
        # Compute distance between confused pairs
        # Minimize distance for same class, maximize for different class
        pass  # Implementation details
```

#### 3.4 复合损失
```python
class CombinedLoss(nn.Module):
    def __init__(self, use_focal=True, use_contrastive=False):
        super().__init__()
        self.ce_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
        self.focal_loss = FocalLoss(gamma=2.0) if use_focal else None
        self.contrastive_loss = ContrastiveLoss() if use_contrastive else None
        
        # Loss weights
        self.lambda_focal = 0.5
        self.lambda_contrastive = 0.2
    
    def forward(self, pred, target, embeddings=None):
        loss = self.ce_loss(pred, target)
        
        if self.focal_loss:
            loss += self.lambda_focal * self.focal_loss(pred, target)
        
        if self.contrastive_loss and embeddings is not None:
            loss += self.lambda_contrastive * self.contrastive_loss(embeddings, target)
        
        return loss
```

**推荐配置**: 初期使用 Label Smoothing + Focal Loss，如效果不佳再考虑添加Contrastive Loss。

### 4. 训练策略

#### 4.1 超参数设置
```python
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,           # 初始学习率
    weight_decay=1e-4  # L2正则化
)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # 第一次重启周期
    T_mult=2,    # 周期倍增因子
    eta_min=1e-6 # 最小学习率
)

# Training
batch_size = 64  # 可根据GPU调整
epochs = 50
early_stopping_patience = 10
```

#### 4.2 正则化技术
- **Dropout**: p=0.4 (保持)
- **Weight Decay**: 1e-4
- **Label Smoothing**: ε=0.1
- **Data Augmentation**: 强化版（如上所述）
- **Early Stopping**: 防止过拟合

#### 4.3 训练监控
```python
# 监控指标
metrics = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'confused_pairs_acc': [],  # 易混淆字符准确率
    'per_font_acc': {}         # 每个字体的准确率
}

# 每个epoch记录
# 每5 epochs在多字体测试集上评估
# 保存最佳模型（基于val_acc + confused_pairs_acc加权）
```

### 5. 模型压缩（可选阶段）

#### 5.1 知识蒸馏
```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # Soft target loss (KL divergence)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

#### 5.2 量化感知训练
```python
import torch.quantization as quant

# Prepare model for QAT
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model_prepared = quant.prepare_qat(model, inplace=False)

# Train with QAT
train(model_prepared, ...)

# Convert to quantized model
model_quantized = quant.convert(model_prepared, inplace=False)

# Evaluate INT8 model
accuracy, speed = evaluate(model_quantized)
```

#### 5.3 结构化剪枝
```python
import torch.nn.utils.prune as prune

# Channel pruning on Conv layers
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.ln_structured(
            module,
            name='weight',
            amount=0.3,  # 剪枝30%通道
            n=2,         # L2 norm
            dim=0        # 输出通道维度
        )

# Fine-tune pruned model
train(model, ...)

# Make pruning permanent
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.remove(module, 'weight')
```

### 6. 评估框架

#### 6.1 准确率评估
```python
def evaluate_comprehensive(model, test_loaders):
    results = {}
    
    # Overall accuracy
    results['overall_acc'] = evaluate_accuracy(model, test_loaders['all'])
    
    # Per-font accuracy
    results['per_font'] = {}
    for font_name, loader in test_loaders['fonts'].items():
        results['per_font'][font_name] = evaluate_accuracy(model, loader)
    
    # Confused pairs accuracy
    results['confused_pairs'] = evaluate_confused_pairs(model, test_loaders['all'])
    
    # Confusion matrix
    results['confusion_matrix'] = compute_confusion_matrix(model, test_loaders['all'])
    
    # Font variance
    font_accs = list(results['per_font'].values())
    results['font_variance'] = np.std(font_accs)
    
    return results
```

#### 6.2 性能评估
```python
def benchmark_performance(model, device='cpu'):
    model.eval()
    dummy_input = torch.randn(1, 3, 24, 24).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    # Single inference
    times = []
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        times.append(time.time() - start)
    
    single_inference_time = np.mean(times) * 1000  # ms
    
    # Batch inference
    batch_sizes = [16, 32, 64]
    batch_throughput = {}
    for bs in batch_sizes:
        batch_input = torch.randn(bs, 3, 24, 24).to(device)
        start = time.time()
        with torch.no_grad():
            _ = model(batch_input)
        batch_time = time.time() - start
        batch_throughput[bs] = bs / batch_time
    
    return {
        'single_inference_ms': single_inference_time,
        'batch_throughput': batch_throughput
    }
```

#### 6.3 模型大小评估
```python
def evaluate_model_size(model):
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model file size
    torch.save(model.state_dict(), 'temp_model.pth')
    file_size_mb = os.path.getsize('temp_model.pth') / (1024 ** 2)
    os.remove('temp_model.pth')
    
    # FLOPs (using thop or fvcore)
    from thop import profile
    dummy_input = torch.randn(1, 3, 24, 24)
    flops, params = profile(model, inputs=(dummy_input,))
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'file_size_mb': file_size_mb,
        'flops': flops
    }
```

## Implementation Order

1. **Phase 1**: 数据增强 (优先级最高，影响最大)
   - 扩展字体库
   - 增强数据生成
   - 优化数据增强pipeline

2. **Phase 2**: 轻量化模型 (核心变更)
   - 实现3-4个候选架构
   - 快速实验选择最优方案
   - 实现选定的LightPixelNet

3. **Phase 3**: 损失函数优化
   - Label Smoothing + Focal Loss
   - 可选: Contrastive Loss

4. **Phase 4**: 训练和验证
   - 完整训练
   - 多字体测试
   - 性能基准测试

5. **Phase 5** (可选): 模型压缩
   - 知识蒸馏
   - 量化
   - 剪枝

## Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| 轻量化导致准确率下降 | 1. 保留原模型作为baseline<br>2. 增量减少参数，逐步验证<br>3. 加强数据增强补偿 |
| 数据增强过度导致训练不稳定 | 1. 逐步增加增强强度<br>2. 监控训练曲线<br>3. 调整learning rate |
| 损失函数权重难以调整 | 1. 先单独验证每个损失<br>2. Grid search或Optuna调参<br>3. 从简单配置开始 |
| 训练时间过长 | 1. 使用更大batch size<br>2. 混合精度训练(AMP)<br>3. 多GPU训练 |

## Success Criteria

### Must Have (P0)
- ✅ 参数量减少≥70% (≤3M)
- ✅ 推理速度提升≥50%
- ✅ 总体准确率≥baseline或提升2%

### Should Have (P1)
- ✅ 易混淆字符准确率提升5%
- ✅ 字体间准确率方差<5%
- ✅ 未见字体准确率>90%

### Nice to Have (P2)
- ✅ 模型压缩(量化/剪枝)
- ✅ 知识蒸馏
- ✅ 可解释性分析(GradCAM)

## References
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629)
