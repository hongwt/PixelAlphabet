## MODIFIED Requirements

### Requirement: Model Architecture
The system MUST use a custom deep Convolutional Neural Network (CNN) with residual connections.
- **Input Size**: Native 24x24 resolution (RGB).
- **Backbone**: High-resolution preservation architecture:
    - Block 1: Conv(3->64, 3x3, stride=1) -> BN -> ReLU -> Conv(64->64, 3x3, stride=1) -> BN -> ReLU
    - Block 2: ResBlock(64->128, stride=1) - Keeps 24x24 resolution
    - Block 3: ResBlock(128->256, stride=2) - Downsamples to 12x12
    - Block 4: ResBlock(256->512, stride=1) - Keeps 12x12 resolution
    - Block 5: ResBlock(512->512, stride=1) - Keeps 12x12 resolution
- **Attention**: Spatial Attention layer to focus on character regions.
- **Regularization**: Dropout (p=0.4) before FC layers.
- **Output Layer**: Fully Connected layer with 36 outputs (Digits 0-9, Letters A-Z).
- **Parameters**: Approximately 9M-10M parameters.
- **Feature Extraction Interface**: The model MUST provide a `forward_features()` method that returns the 256-dimensional embedding vector after Global Average Pooling but before the classifier head, to support contrastive loss computation on the feature space.

#### Scenario: Alphanumeric Classification
Given a 24x24 image containing a clear character 'A',
When the model processes the image,
Then it should return the label 'A' with probability > 0.8.

#### Scenario: Feature Embedding Extraction
Given a batch of 24x24 images,
When `forward_features()` is called,
Then it MUST return a tensor of shape (B, 256) representing the feature embeddings before the classification head.

## ADDED Requirements

### Requirement: Focal Label Smoothing Loss
The system MUST implement a unified `FocalLabelSmoothingLoss` that integrates Focal Loss's dynamic weighting factor $(1-p_t)^\gamma$ directly into the Label Smoothing Cross Entropy computation, rather than treating them as separate additive components.
- **Formula**: $L_{FLS} = -\alpha_t (1-p_t)^\gamma \sum_c \tilde{y}_c \log(p_c)$ where $\tilde{y}_c$ is the smoothed label.
- **Default parameters**: `epsilon=0.1`, `gamma=3.0`.
- **Class-level alpha weights**: MUST accept a per-class alpha weight vector (dimension = num_classes). Confusable character classes (7, T, 8, B, 0, D, 5, S, 2, Z, Q, 1, L, 6, G) SHALL receive higher weights (default: 2.0x) compared to non-confusable classes (default: 1.0x).

#### Scenario: Hard example amplification
- **WHEN** a training sample of character '8' is predicted with low confidence (e.g. p=0.3 split between '8' and 'B')
- **THEN** the Focal weighting factor $(1-0.3)^3 = 0.343$ SHALL amplify the loss for this sample, producing a significantly higher loss compared to an easy sample of 'W' predicted with p=0.95

#### Scenario: Class-level alpha weighting
- **WHEN** computing loss for a confusable character (e.g. '0', '8', '5')
- **THEN** the alpha weight applied MUST be 2.0x the weight of non-confusable characters

### Requirement: Confusion Pair Contrastive Loss
The system MUST implement a `ConfusionPairContrastiveLoss` that operates on the model's 256-dimensional feature embeddings to explicitly push apart representations of predefined confusable character pairs.
- **Predefined confusable pairs**: (7, T), (8, B), (0, D), (5, S), (2, Z), (Q, 0), (1, L), (6, G)
- **Mechanism**: For each confusable pair (a, b) present in a training batch, compute the mean feature embedding of class a and class b, then apply a margin-based loss: $L_{contrast} = \max(0, margin - d(\mu_a, \mu_b))$ where $d$ is the cosine distance and `margin` defaults to 0.5.
- **Integration**: The contrastive loss SHALL be added to the combined loss with a configurable weight `lambda_contrastive` (default: 0.3).

#### Scenario: Pushing apart 8 and B
- **WHEN** a training batch contains samples of both '8' and 'B'
- **THEN** the contrastive loss computes the cosine distance between the mean embeddings of '8' and 'B' samples, and penalizes if the distance is below the margin threshold

#### Scenario: Batch without confusable pairs
- **WHEN** a training batch does not contain both members of any confusable pair
- **THEN** the contrastive loss contribution MUST be zero (no gradient)

### Requirement: Extended Confusion Monitoring
The training loop MUST monitor and log confusion counts for all predefined confusable character pairs during validation, not only (Q, 0) and (8, B).
- **Monitored pairs**: (7, T), (8, B), (0, D), (5, S), (2, Z), (Q, 0), (1, L), (6, G)
- **Logging**: Each pair's bidirectional confusion counts SHALL be logged to TensorBoard and printed to console.

#### Scenario: Full confusion pair logging
- **WHEN** a validation epoch completes
- **THEN** the system logs bidirectional confusion counts for all 8 predefined pairs to TensorBoard under `Confusion/` prefix and prints them to console

### Requirement: Updated Combined Loss
The `CombinedLoss` class MUST be updated to use `FocalLabelSmoothingLoss` as its base loss and optionally include `ConfusionPairContrastiveLoss`.
- **Formula**: $L_{total} = L_{FLS} + \lambda_{contrastive} \cdot L_{contrast}$
- The `create_loss_function` factory MUST support the new loss types via the existing `'combined'` mode.

#### Scenario: Combined loss with contrastive component
- **WHEN** `create_loss_function('combined')` is called with default parameters
- **THEN** it returns a `CombinedLoss` using `FocalLabelSmoothingLoss` as base and `ConfusionPairContrastiveLoss` as auxiliary loss
