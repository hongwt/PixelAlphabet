# Tasks: Optimize PixelNet Architecture

## 1. 新增模块实现
- [x] 1.1 实现 CoordConv 模块（位置编码卷积）
- [x] 1.2 实现 SpatialAttention 模块（空间注意力）
- [x] 1.3 实现 CBAM 模块（通道+空间注意力组合）
- [x] 1.4 实现 ResBlockSE 模块（内置 SE 注意力的残差块）

## 2. 模型架构优化
- [x] 2.1 将第一层 Conv 替换为 CoordConv
- [x] 2.2 增加第三个 ResBlock (64→64→128→256)
- [x] 2.3 将末尾 SE Block 替换为 CBAM
- [x] 2.4 优化 FC 分类头（添加 BatchNorm，移除首个 Dropout）

## 3. 兼容性保持
- [x] 3.1 添加 `use_coord_conv` 参数（默认 True）
- [x] 3.2 添加 `attention_type` 参数（'cbam', 'se', 'none'）
- [x] 3.3 确保模型可通过参数配置兼容旧架构

## 4. 测试验证
- [x] 4.1 更新 `test_model.py` 测试用例
- [x] 4.2 验证模型前向传播输出形状
- [x] 4.3 验证参数量在合理范围（~1.28M）
- [ ] 4.4 运行完整训练验证性能提升（需要用户执行）

## 5. 文档更新
- [x] 5.1 更新 README.md 中的模型架构说明
- [x] 5.2 添加模块的 docstring 文档
