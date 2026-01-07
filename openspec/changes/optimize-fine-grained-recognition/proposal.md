# Proposal: Optimize Fine-Grained Recognition

## Problem
The current model exhibits high confusion rates between visually similar characters, specifically:
- 'Q' vs '0' (distinguished only by a small tail)
- '8' vs 'B' (distinguished by curvature/corners)
- 'I' vs 'l' vs '1'

Analysis suggests the current 4-stage downsampling architecture (reducing 24x24 inputs to 3x3 feature maps) discards too much spatial information. The 3x3 resolution is insufficient to encode the fine-grained details (like the tail of 'Q') required for these edge cases. Additionally, the standard CrossEntropyLoss treats all samples equally, allowing the model to achieve high overall accuracy while failing on these difficult minority subclasses.

## Solution

### 1. High-Resolution Architecture
We will modify the `PixelNet` architecture to preserve spatial resolution deeper into the network.
- **Remove Early Downsampling**: Change the initial convolution block or early residual blocks to use `stride=1` instead of `stride=2`.
- **Target Feature Map Size**: Ensure the final feature map before Global Average Pooling is at least **6x6** (currently 3x3), providing 4x more spatial context for attention mechanisms.

### 2. Hard Negative Mining (Focal Loss)
Replace the standard `CrossEntropyLoss` with `Focal Loss`.
- This loss function dynamically down-weights easy examples (where the model is confident) and focuses training on hard negatives (difficult confusion cases).
- This directly addresses the "lazy" learning behavior where the model ignores subtle differences.

### 3. Refined Data Augmentation
Update the augmentation pipeline to be less destructive to fine details.
- **Reduce Rotation**: Limit random rotation to ±5° (or remove) to avoid alias artifacts that blur pixel-level details (like the Q tail).
- **Add Random Erasing**: Force the model to learn distributed features rather than relying on a single distinctive pixel cluster.

## Impact
- **Architecture**: `src/model.py` requires stride adjustments.
- **Training**: `src/train.py` requires Focal Loss implementation.
- **Spec**: Update `openspec/specs/recognition/spec.md` with new architecture and training requirements.
