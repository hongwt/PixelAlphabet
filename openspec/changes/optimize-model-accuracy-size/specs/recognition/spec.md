# recognition Specification - DELTA

本文档描述相对于当前 `recognition` 规范的变更。

## MODIFIED Requirements

### Requirement: Model Architecture
The system MUST use a **lightweight custom CNN** with residual connections optimized for efficiency.
- **Input Size**: Native 24x24 resolution (RGB) - UNCHANGED.
- **Backbone**: Lightweight architecture with reduced parameters:
    - **MODIFIED**: Block 1: Single Conv(3->64, 3x3, stride=1) -> BN -> ReLU (simplified from dual conv)
    - **MODIFIED**: Block 2: ResBlock(64->128, stride=1) - Keeps 24x24 resolution
    - **MODIFIED**: Block 3: ResBlock(128->256, stride=2) - Downsamples to 12x12
    - **REMOVED**: Block 4: ResBlock(256->512, stride=1) removed
    - **REMOVED**: Block 5: ResBlock(512->512, stride=1) removed
- **Attention**: 
    - **MODIFIED**: Simplified Spatial Attention with kernel_size=1, OR
    - **ALTERNATIVE**: Lightweight SE (Squeeze-and-Excitation) Channel Attention
    - **TO BE DECIDED**: Final choice based on experiments
- **Regularization**: Dropout (p=0.4) before FC layers - UNCHANGED.
- **Output Layer**: Fully Connected layer with 37 outputs - UNCHANGED.
- **Parameters**: 
    - **MODIFIED**: Target approximately 2-3M parameters (reduced from 9-10M, ~70% reduction).

#### Scenario: Efficient Inference
Given a batch of 32 images of 24x24 characters,
When the lightweight model processes the batch on CPU,
Then inference time should be ≤50% of the original model's inference time.

#### Scenario: Small Model Size
Given the trained lightweight model checkpoint,
When saved to disk,
Then the file size should be ≤12MB (compared to ~40MB for original model).

### Requirement: Data Generation & Augmentation
The system MUST apply **enhanced** robust generation and augmentation strategies:
- **Generation (Synthetic)**:
    - **MODIFIED**: **Variable Font Sizes**: Randomly vary font sizes in extended range (10px to 24px, previously 12-22px).
    - **MODIFIED**: **Positioning**: Random jitter (±6px, previously ±4px) from center.
    - **ADDED**: **Font Diversity**: Use 10+ different fonts covering serif, sans-serif, monospace, handwriting styles.
    - **ADDED**: **Font Variations**: Apply font stretch/condensed transforms.
    - **ADDED**: **Rendering Variations**: Vary antialiasing and hinting settings.
- **Augmentation (Training)**:
    - **MODIFIED**: **Color Jitter**: Increased range - brightness (±0.4, previously ±0.3), contrast (±0.4, previously ±0.3), saturation (±0.3, previously ±0.2), hue (±0.15, previously ±0.1).
    - **Geometric**: Random rotation (±5°), affine transforms - UNCHANGED.
    - **Noise**: Gaussian noise (σ=0.02) - UNCHANGED.
    - **ADDED**: **Blur/Sharpness**: Random GaussianBlur (σ=0.5-1.0) and RandomSharpness to simulate image quality variations.

#### Scenario: Font Size Robustness
Given images of character '7' at font sizes 10px, 15px, 20px, and 24px,
When the model processes all variants,
Then it should return '7' with confidence > 0.80 for all sizes.

#### Scenario: Cross-Font Generalization
Given an image of 'B' in a handwriting font not seen during training,
When the model processes the image,
Then it should return 'B' with confidence > 0.75.

#### Scenario: Low Variance Across Fonts
Given a test set with 10 different fonts,
When the model's accuracy is measured on each font subset,
Then the standard deviation of accuracies should be < 5%.

## ADDED Requirements

### Requirement: Advanced Loss Functions
The system MUST use advanced loss functions to improve learning on hard samples and reduce confusion between similar characters.
- **Label Smoothing**: Apply label smoothing with ε=0.1 to reduce overconfidence.
- **Focal Loss**: Use Focal Loss to focus on hard-to-classify samples (γ=2.0).
- **Contrastive Learning** (Optional): Apply contrastive loss on easily confused character pairs: (0, O), (1, I), (5, S), (2, Z), (6, b), (8, B).

#### Scenario: Hard Sample Learning
Given a training batch where 20% of samples are consistently misclassified (e.g., '0' vs 'O'),
When training with Focal Loss,
Then the model should pay more attention to these hard samples and improve accuracy by >5% on confused pairs.

#### Scenario: Reduced Overconfidence
Given a test image of character 'Q' that slightly resembles 'O',
When the model makes a prediction with label smoothing,
Then the predicted probability distribution should be smoother (less peaked) compared to standard cross-entropy.

### Requirement: Model Efficiency Targets
The system MUST meet the following efficiency targets compared to the baseline PixelNet:
- **Parameter Reduction**: ≥70% reduction (from ~9M to ≤3M).
- **Inference Speed**: ≥50% faster on both CPU and GPU (single image inference).
- **Batch Throughput**: ≥50% higher throughput for batch inference.
- **Memory Footprint**: ≤12MB model file size.

#### Scenario: Real-time Inference on CPU
Given a standard desktop CPU (e.g., Intel i5),
When processing single 24x24 character images,
Then inference latency should be <5ms per image (previously ~10ms).

#### Scenario: Edge Deployment
Given a lightweight computing device with limited memory,
When loading the model,
Then it should successfully load within 12MB memory constraint and run inference without issues.

### Requirement: Generalization and Robustness
The system MUST demonstrate strong generalization to unseen fonts and rendering conditions.
- **Unseen Font Test Set**: Accuracy >90% on fonts not present in training data.
- **Train-Val Gap**: <3% accuracy difference between training and validation sets to ensure no overfitting.
- **Background Robustness**: Consistent performance across different background colors and patterns.

#### Scenario: Zero-Shot Font Recognition
Given a test set containing 5 fonts completely absent from training data,
When the model processes images from this test set,
Then it should achieve >90% accuracy.

#### Scenario: Overfitting Prevention
Given training logs over 50 epochs,
When comparing training accuracy vs validation accuracy,
Then the gap should remain <3% throughout training, indicating good generalization.

## REMOVED Requirements

None. All previous requirements remain, but some are modified as noted above.

## RENAMED Requirements

None.

## Notes
- The lightweight architecture should be implemented as a new class `LightPixelNet` in `src/model.py`.
- The original `PixelNet` class should be retained for comparison and backward compatibility.
- A model selection mechanism should be added to `src/inference.py`.
- All changes should be validated through comprehensive experiments comparing baseline vs. optimized models.
- Performance metrics (accuracy, speed, size) should be documented in a comparison table.
