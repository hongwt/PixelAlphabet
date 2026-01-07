# Capability: Character Recognition

## Overview
Recognize single alphanumeric characters from small image patches (24x24).

## Requirements

### ADDED Requirements

#### Requirement: Model Architecture
The system MUST use a custom deep Convolutional Neural Network (CNN) with residual connections.
- **Input Size**: Native 24x24 resolution (RGB).
- **Backbone**: 5 Convolutional blocks:
    - Block 1: Conv(3->32, 3x3) -> BN -> ReLU -> Conv(32->32, 3x3) -> BN -> ReLU -> MaxPool(2x2)
    - Block 2-5: Residual blocks with channel progression (64->128->256->512)
- **Attention**: Spatial Attention layer to focus on character regions.
- **Regularization**: Dropout (p=0.4) before FC layers.
- **Output Layer**: Fully Connected layer with 37 outputs.
- **Parameters**: Approximately 500K-1M parameters.

#### Scenario: Alphanumeric Classification
Given a 24x24 image containing a clear character 'A',
When the model processes the image,
Then it should return the label 'A' with probability > 0.8.

#### Requirement: 'NA' Class Handling
The system MUST support a 'Not Applicable' (NA) class for images that do not contain recognizable text.

#### Scenario: Non-text Image
Given a 24x24 image valid skill icon but no text overlay,
When the model processes the image,
Then it should return the label 'NA'.

#### Requirement: Input Preprocessing
The system MUST preprocess inputs consistently.
- Keep RGB channels (color information helps distinguish backgrounds).
- Normalization to [0, 1] range.
- **No Resizing**: Keep original 24x24 resolution to preserve pixel details.

#### Requirement: Data Augmentation
The system MUST apply augmentation during training to handle font/background variations.
- **Color Jitter**: Random brightness (±0.3), contrast (±0.3), saturation (±0.2), hue (±0.1).
- **Geometric**: Random rotation (±5°), affine transforms (translate ±2px, scale 0.9-1.1).
- **Noise**: Gaussian noise (σ=0.02).

#### Scenario: Robust to Background Variation
Given two images of 'A' with different background colors (blue vs red skill icons),
When the model processes both images,
Then it should return 'A' for both with confidence > 0.8.

#### Scenario: Font Variation Handling
Given images of '5' in different fonts (serif, sans-serif, bold),
When the model processes all variants,
Then it should return '5' with confidence > 0.75.

#### Scenario: Distinguishing Similar Characters
Given images of '0' (zero) and 'O' (letter),
When the model processes both,
Then it should correctly distinguish them with > 90% accuracy on test set.

```python
# Label Map Convention
# 0-9: Digits
# 10-35: Letters A-Z
# 36: NA
```
