## 1. 前景渲染层重构
- [ ] 1.1 将 `_generate_sample` 中的文字渲染改为 RGBA 透明画布模式
- [ ] 1.2 使用 Pillow `stroke_width` / `stroke_fill` 原生 API 替代 8 方向偏移描边
- [ ] 1.3 通过 `getbbox()` 裁剪出紧凑的前景字符包围盒
- [ ] 1.4 添加描边宽度随机化支持 (1-2px)

## 2. Alpha 合成层实现
- [ ] 2.1 实现硬覆盖（Hard Overlay）Alpha 合成函数
- [ ] 2.2 重构 `_generate_sample`：先渲染前景 RGBA，再合成到背景 RGB 上
- [ ] 2.3 移除 `draw_text_outline` 旧函数，改用新的前景渲染函数

## 3. 插值模式修正
- [ ] 3.1 将 `resize_image` 默认插值改为 `NEAREST`
- [ ] 3.2 审查所有 resize/resample 调用确保一致使用 NEAREST

## 4. 图像退化管线 (第四层)
- [ ] 4.1 实现低保真空间重采样退化（先 downscale 再 upscale，NEAREST 插值）
- [ ] 4.2 实现高斯噪声注入函数
- [ ] 4.3 实现椒盐噪声注入函数
- [ ] 4.4 实现随机 Gamma 校正
- [ ] 4.5 实现 HSV 色彩空间随机漂移
- [ ] 4.6 实现 JPEG 压缩伪影模拟
- [ ] 4.7 在 `_generate_sample` 末端集成退化管线（随机应用子集）
- [ ] 4.8 添加 `--degradation` CLI 参数控制退化强度

## 5. 测试与验证
- [ ] 5.1 更新单元测试：验证新的前景渲染函数输出 RGBA
- [ ] 5.2 更新单元测试：验证 Alpha 合成正确性
- [ ] 5.3 添加退化管线单元测试
- [ ] 5.4 生成对比样本，目视确认与真实图片的相似度提升
