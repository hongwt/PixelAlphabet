# Change Proposal: Add Character Recognition Model

## Summary
Implement a character recognition system using PyTorch with a custom, lightweight CNN trained from scratch. The system will identify single alphanumeric characters (0-9, A-Z) from 24x24 pixel images, including an 'NA' class for invalid/empty inputs.

## Motivation
The core feature of PixelAlphabet is recognizing characters from game skill icons. Instead of using a bulky pre-trained model designed for 224x224 images, a custom CNN trained on the native 24x24 resolution will likely offer better accuracy and generalization for pixel-perfect low-resolution icons.

## Proposed Changes
- **Dependency**: Add `torch` and `torchvision` to requirements.
- **Model**: Implement a custom CNN (`PixelNet`) with enhanced architecture.
    - **Architecture**: 
        - 5 Convolutional blocks with residual connections (ResNet-style)
        - Batch Normalization after each Conv layer
        - Dropout (0.3-0.5) for regularization
        - Spatial Attention module before final FC layers
        - Progressive channel expansion: 32->64->128->256->512
    - **Input**: Native 24x24 resolution (RGB or Grayscale).
    - **Training Strategy**: Train from scratch using ~20K labeled samples with heavy augmentation.
- **Data**: Create a custom `Dataset` class to handle 24x24 images.
    - **Augmentation Pipeline**:
        - Random brightness/contrast adjustment (±30%)
        - Random hue/saturation shift (±10%)
        - Gaussian noise injection
        - Slight rotation (±5 degrees)
        - Random affine transformations (shift/scale)
    - Input Processing: Normalization to [0, 1] or z-score.
- **Classes**: 37 classes total (0-9, A-Z, NA).
    - Mapping: 0-9 indices -> '0'-'9'
    - 10-35 indices -> 'A'-'Z'
    - 36 index -> 'NA'
- **Training**: Implement a training script `train.py`.
- **Inference**: Implement a predictor class in `src/model.py`.

## Model Selection Rationale
Selected **Enhanced Custom CNN (PixelNet)** with the following design considerations:
1.  **Resolution Match**: Native 24x24 processing preserves pixel-level details crucial for distinguishing similar characters (0/O, I/l).
2.  **Challenging Scenarios**: The model addresses:
     - **Background Interference**: Residual connections help learn robust features despite varying skill icon backgrounds.
     - **Font Variations**: Deep feature extraction (5 blocks) captures diverse typographic styles.
     - **Similar Characters**: Spatial Attention focuses on discriminative regions (e.g., closure in 'O' vs '0').
3.  **Data Utilization**: ~20K samples justify a deeper architecture (~500K-1M parameters) that can learn complex patterns without severe overfitting (mitigated by dropout and augmentation).
4.  **Efficiency vs Accuracy Trade-off**: Still lighter than MobileNet (~3.5M params) but significantly more capable than a shallow 3-layer CNN.

## Scenarios
- **Valid Input**: Image with 'A' -> Output 'A', High Confidence.
- **Digit Input**: Image with '1' -> Output '1', High Confidence.
- **Empty/Icon Input**: Image with just skill icon art (no text) -> Output 'NA'.
