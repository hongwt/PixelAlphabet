# Capability: Character Recognition

## Requirements

### MODIFIED Requirements

#### Requirement: Model Architecture
The system MUST use a custom deep Convolutional Neural Network (CNN) with residual connections that **preserves spatial resolution**.
- **Input Size**: Native 24x24 resolution (RGB).
- **Backbone**: Modified ResNet-style blocks to maintain **at least 6x6 feature map size** before the final pooling layer.
    - **Stride Strategy**: Reduce downsampling operations to retain fine-grained details.
    - **Channel Width**: Progressive channel increase (e.g., 32 -> 64 -> 128 -> 256 -> 512).
- **Attention**: Spatial Attention layer to focus on character regions.
- **Regularization**: Dropout (p=0.4) before FC layers.

#### Scenario: Fine-Grained Distinction
Given a 24x24 image of 'Q' (similar to '0' but with a tail),
When the model processes the image,
Then the feature map resolution must be sufficient to resolve the tail pixels, resulting in correct classification as 'Q'.

#### Requirement: Loss Function
The system MUST use a loss function that prioritizes difficult samples (Hard Negative Mining).
- **Focal Loss**: Use Focal Loss (gamma=2.0) instead of standard Cross Entropy to penalize misclassification of hard exaples (like Q vs 0) more heavily than easy ones.

#### Scenario: Training on Hard Examples
Given a batch of training data where the model correctly predicts 'A' (easy) but confuses '8' for 'B' (hard),
When the loss is calculated,
Then the gradient contribution from the '8' vs 'B' error should be significantly up-weighted.

#### Requirement: Data Augmentation
The system MUST apply augmentation that preserves character structure.
- **Conservative Geometry**: Limit rotation to **max ±5°** to prevent pixel aliasing on small images.
- **Random Erasing**: Apply Random Erasing (probability ~0.1) to force learning of global shape features.
- **Color Jitter**: Random brightness, contrast, saturation.

