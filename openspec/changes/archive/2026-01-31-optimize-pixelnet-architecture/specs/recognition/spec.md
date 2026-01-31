# recognition Specification Delta

## MODIFIED Requirements

### Requirement: Model Architecture
The system MUST use a custom deep Convolutional Neural Network (CNN) with residual connections and attention mechanisms.
- **Input Size**: Native 24x24 resolution (RGB).
- **Position Encoding**: CoordConv layer adds 2D positional coordinates to input channels.
- **Backbone**: High-resolution preservation architecture:
    - Block 1: CoordConv(3+2->64, 3x3, stride=1) -> BN -> ReLU
    - Block 2: ResBlockSE(64->64, stride=1) - Keeps 24x24 resolution, with SE attention
    - Block 3: ResBlockSE(64->128, stride=1) - Keeps 24x24 resolution, with SE attention
    - Block 4: ResBlockSE(128->256, stride=2) - Downsamples to 12x12, with SE attention
- **Attention**: 
    - Per-block SE (Squeeze-and-Excitation) for channel attention
    - CBAM (Convolutional Block Attention Module) after final ResBlock for combined channel and spatial attention
- **Regularization**: Dropout (p=0.3) before final FC layer only.
- **Classifier Head**: 
    - Global Average Pooling -> Linear(256, 128) -> BatchNorm1d -> ReLU -> Dropout -> Linear(128, 36)
- **Output Layer**: Fully Connected layer with 36 outputs (Digits 0-9, Letters A-Z).
- **Parameters**: Approximately 1.2M-1.5M parameters.

#### Scenario: Alphanumeric Classification
- **GIVEN** a 24x24 image containing a clear character 'A'
- **WHEN** the model processes the image
- **THEN** it should return the label 'A' with probability > 0.8

#### Scenario: Position-Sensitive Character Recognition
- **GIVEN** images of '6' and '9' (rotational variants)
- **WHEN** the model processes both images
- **THEN** it should correctly distinguish them with > 95% accuracy due to CoordConv position encoding

#### Scenario: Similar Character Discrimination
- **GIVEN** images of '0' (zero), 'O' (letter O), and 'Q'
- **WHEN** the model processes all variants
- **THEN** it should correctly classify each with > 92% accuracy due to enhanced attention mechanisms

## ADDED Requirements

### Requirement: Configurable Architecture Components
The system MUST support configurable architecture components for flexibility and backward compatibility.
- **CoordConv**: Optional position encoding (default: enabled)
- **Attention Type**: Configurable attention mechanism ('cbam', 'se', 'none')
- **Legacy Mode**: Support loading checkpoints from previous model versions

#### Scenario: Backward Compatible Loading
- **GIVEN** a checkpoint trained with the previous model architecture
- **WHEN** loading with `strict=False` parameter
- **THEN** the model should load compatible weights and initialize new layers with defaults

#### Scenario: Attention Configuration
- **GIVEN** model initialization with `attention_type='se'`
- **WHEN** the model is created
- **THEN** it should use only SE Block attention (no spatial attention) for reduced computation
