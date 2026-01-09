# Proposal: Optimize Fine-Grained Recognition

## Problem
The current model exhibits distinct failure modes that limit reliability in production contexts:
1. **High Confusion on Similar Shapes**: The model struggles with visually similar characters ('Q' vs '0', '8' vs 'B', 'I' vs 'l' vs '1').
2. **Font Size Sensitivity**: The model is sensitive to character scale. Larger font variations (e.g., a zoomed-in '3') are frequently misclassified as other characters (e.g., '2'), indicating a lack of scale invariance.
3. **Suboptimal Architecture for Low-Res Inputs**: The input image size of 24x24 is high-definition for single characters, but the current ResNet-style architecture applies standard aggressive pooling, reducing feature maps to 3x3 or 6x6. This discarding of spatial information prevents the model from capturing the pixel-perfect details needed for fine-grained distinction.

## Solution

### 1. Data Generation & Augmentation
Update `src/data_generator.py` and the training pipeline to enforce scale invariance.
- **Variable Font Sizes**: Expand the random generation range to include significantly larger font sizes. This ensures the model sees characters that fill the 24x24 frame almost entirely.
- **Random Scaling**: Introduce random scaling/zoom augmentation.
- **Center Alignment Jitter**: While currently top-right aligned, adding slight positional jitter forces the model to look for features rather than fixed pixel locations.

### 2. High-Resolution "Pixel" Architecture
Modify `src/model.py` to respect the 24x24 constraint by maintaining higher feature map resolutions.

- **Remove Pooling**: Eliminate the initial `MaxPool2d`. On 24x24, a 2x2 pooling operation destroys 75% of the spatial data immediately.
- **Preserve Spatial Dimensions**:
    - **Stage 1 & 2**: Maintain full 24x24 resolution (stride=1).
    - **Stage 3**: Optional downsampling to 12x12.
    - **Final Stage**: Avoid reducing below 12x12 before Global Average Pooling.
- **Increase Channel Width**: To address the question of "larger dimensions," since we cannot increase the input image size (fixed at 24x24 source), we will instead increase the **channel capacity** (width).
    - Current: Starts at 32 channels.
    - Proposed: Start at 64 channels. This allows the network to learn more complex filter combinations (edges, curves, textures) within the limited spatial area.

### 3. Loss Function Optimization
- **Focal Loss**: Replace `CrossEntropyLoss` with `Focal Loss` to focus learning on the "hard" examples (the 3 vs 2 cases) rather than the easy ones.

## Technical Implementation Plan

1.  **Modify `src/data_generator.py`**:
    - Adjust `font_size_range` logic to allow larger max sizes.
    - Verify padding/cropping logic handles larger characters without cutting off critical features.

2.  **Rewrite `src/model.py`**:
    - Remove `self.maxpool1`.
    - Change early `ResidualBlock` strides to 1.
    - Increase base channels: `32 -> 64`.

3.  **Update `src/train.py`**:
    - Integrate `FocalLoss`.

## Impact
This change impacts the core model accuracy and robustness. It addresses the user's specific feedback regarding font size sensitivity and leverages the "HD" nature of the 24x24 single-character inputs by avoiding unnecessary downsampling.
