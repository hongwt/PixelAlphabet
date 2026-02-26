# Change: 基于文档设计的像素级定制化合成管线优化

## Why

当前的 `data_generator.py` 合成管线存在多项与文档《基于Python底层图像处理库的像素级定制化合成管线设计》所述设计原则严重偏离的问题，导致生成的合成数据与真实游戏 UI 截图之间存在显著的域偏移（Domain Shift）：

1. **描边方法落后**：使用"8方向偏移绘制"的老旧方案（Pillow 6.2.0 之前的变通手段），容易产生不对称伪影。文档明确推荐使用 Pillow 原生的 `stroke_width` / `stroke_fill` 参数。
2. **缺少 Alpha 合成层**：当前直接在 RGB 背景上绘制文字，没有前景/背景分离。文档要求以 RGBA 透明通道渲染前景字符，再通过硬覆盖（Hard Overlay）Alpha 合成到背景上，以模拟游戏引擎 HUD 渲染的 BitBLT 位块传输逻辑。
3. **使用了抗锯齿缩放**：`resize_image` 使用 `LANCZOS` 插值，会产生平滑过渡像素，破坏像素化游戏 UI 的离散高频梯度特征。文档强调必须使用 `NEAREST` 最近邻插值。
4. **无图像退化增强**：缺乏模拟真实场景中JPEG压缩伪影、分辨率拉伸模糊、噪声注入、全局光照色彩偏移等退化效果，导致合成数据过于"完美无瑕"。


## What Changes

- **MODIFIED**: 前景字符渲染改用 RGBA 4通道透明画布 + Pillow 原生 `stroke_width`/`stroke_fill` 描边 API 
- **MODIFIED**: Alpha 合成替代直接绘制，实现硬覆盖（Hard Overlay）前景/背景分离
- **MODIFIED**: 所有缩放操作改用 `NEAREST` 最近邻插值，保留像素化硬边缘特征
- **ADDED**: 图像退化管线（第四层）
  - 低保真空间重采样（下采样+上采样，NEAREST 插值）
  - 高斯噪声 / 椒盐噪声注入
  - 随机 Gamma 校正与 HSV 色彩空间漂移
  - JPEG 压缩伪影模拟
- **ADDED**: 描边宽度随机化（1-2 像素），更接近真实游戏多样性

## Impact

- Affected specs: `data-generation`
- Affected code: `src/data_generator.py`
- **非破坏性变更**：CLI 接口保持向后兼容，新增可选参数
- 生成数据质量将显著提升，预期缩小与真实图片的域偏移
