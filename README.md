# PixelAlphabet
识别图片中的单个字符，只支持字母和数字。

## Features

- **Optimized CNN Architecture**: Efficient ResNet-style network with SE Attention (~1M params)
- **24x24 Native Resolution**: Direct processing without upscaling
- **36 Classes**: Digits (0-9) and Letters (A-Z)
- **Robust to Variations**: Handles different backgrounds, fonts, and similar characters (0/O, I/l)
- **Enhanced Data Augmentation**: Color jitter, geometric transforms, blur, sharpness, and noise
- **Advanced Loss Functions**: Combined Focal Loss + Label Smoothing (Built-in)
- **Multi-Font Support**: 15+ fonts for training data generation (system fonts + custom TTF)

## Installation

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
PixelAlphabet/
├── src/
│   ├── dataset.py      # Dataset and augmentation pipeline
│   ├── model.py        # PixelNet architecture
│   ├── train.py        # Training script
│   └── inference.py    # Inference API
├── tests/
│   ├── test_basic.py   # Environment tests
│   └── test_model.py   # Model tests
├── openspec/           # OpenSpec documentation
└── requirements.txt
```

## Usage

### Prepare Dataset

You can either organize your own dataset or use the data generator to create synthetic training data.

#### Option 1: Use Data Generator (Recommended)

Generate synthetic training data from game skill icons:

```bash
# Generate training dataset
python -m src.data_generator \
    --input-dir ./path/to/skill_icons \
    --output-dir ./data \
    --split train

# Generate validation dataset
python -m src.data_generator \
    --input-dir ./path/to/skill_icons \
    --output-dir ./data \
    --split val \
    --seed 42

# Generate test dataset
python -m src.data_generator \
    --input-dir ./path/to/skill_icons \
    --output-dir ./data \
    --split test \
    --seed 123
```

The data generator will:
- Create directory structure: `data/split/char/` (e.g., `data/train/A/`, `data/train/B/`)
- Process PNG images from the input directory
- Extract 24×24 patches from each icon
- Overlay characters with various fonts (system + custom TTF)
- Apply white text with black outline for visibility
- Organize images by character class (compatible with PyTorch ImageFolder)

**Input**: Skill icon images (PNG format)  
**Output**: Organized dataset ready for training

```
data/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── A/
│   ├── B/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

See the original Java implementation in [AutoGenImage.java](AutoGenImage.java) for reference.

#### Option 2: Manual Organization

Alternatively, organize your data manually as follows:

```
data/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── A/
│   ├── B/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Each directory should contain 24x24 PNG images of the respective character.

### Training

Train the PixelNet model:

```bash
python -m src.train \
    --data-root ./data \
    --output-dir ./checkpoints \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --dropout 0.3
```

**Training Features**:
- Uses **Combined Loss** (Focal Loss + Label Smoothing) by default for better class separation and hard example mining.
- Automatically saves the best model based on validation accuracy.
- Logs metrics to TensorBoard (`checkpoints/run_TIMESTAMP/logs`).

### Inference

```python
from src.inference import CharacterPredictor

# Load trained model
predictor = CharacterPredictor(checkpoint_path='checkpoints/best_model.pth')

# Single prediction
char, confidence = predictor.predict('path/to/image.png')
print(f"Predicted: {char} ({confidence:.2%})")

# Batch prediction
images = ['img1.png', 'img2.png', 'img3.png']
results = predictor.predict_batch(images)
```

### Testing

```bash
# Run all tests
pytest

# Run model tests
pytest tests/test_model.py -v

# Run with coverage
pytest --cov=src tests/
```

## Model Architecture

**PixelNet** (Optimized):

1. **Initial Conv Block**: 3→64 channels, 24x24 (No downsampling)
2. **Residual Block 1**: 64→128 channels, 24x24
3. **Residual Block 2**: 128→256 channels, 24x24→12x12 (Stride 2)
4. **SE Attention**: Squeeze-and-Excitation Block (Channel Attention)
5. **Global Average Pooling** + **FC Layers** (256→128→36) with Dropout (0.3)

## Development

This project follows the [OpenSpec](./openspec/AGENTS.md) workflow for structured development.

Current active changes can be found in `openspec/changes/`.

