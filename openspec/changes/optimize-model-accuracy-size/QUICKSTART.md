# Quick Start Guide: Optimized PixelAlphabet

This guide helps you get started with the newly optimized LightPixelNet model.

## What's New?

- 🚀 **87.7% fewer parameters** (9.7M → 1.2M)
- ⚡ **78% faster inference** (14.7ms → 3.2ms on CPU)
- 📦 **87.7% smaller model** (37MB → 4.6MB)
- 🎯 **Enhanced data augmentation** for better robustness
- 💪 **Advanced loss functions** for improved accuracy

## Installation

```bash
# Activate your environment
.\venv\Scripts\activate  # Windows

# Install dependencies (if not already done)
pip install -r requirements.txt
```

## Quick Test

### 1. Verify Implementation

```bash
# Test model architecture
python -m src.model

# Test loss functions
python -m src.loss

# Compare models
python -m src.compare_models
```

Expected output from `compare_models`:
- PixelNet: 9.7M params, 37MB, 14.7ms
- LightPixelNet (SE): 1.2M params, 4.6MB, 3.2ms ⭐
- LightPixelNet (Spatial): 1.2M params, 4.5MB, 4.8ms
- LightPixelNet (No Attn): 2.4M params, 9.1MB, 5.5ms

## Training Workflow

### Option A: Train Light Model (Recommended)

```bash
# 1. Generate training data with enhanced settings
python -m src.data_generator \
    --input-dir ./icons \
    --output-dir ./data \
    --split train \
    --font-size-range 10-24

# 2. Generate validation data
python -m src.data_generator \
    --input-dir ./icons \
    --output-dir ./data \
    --split val \
    --seed 42

# 3. Train LightPixelNet with combined loss
python -m src.train \
    --data-root ./data \
    --output-dir ./checkpoints \
    --model light \
    --attention se \
    --num-res-blocks 2 \
    --loss combined \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --dropout 0.3
```

### Option B: Train Original Model (Baseline)

```bash
# Train original PixelNet for comparison
python -m src.train \
    --data-root ./data \
    --output-dir ./checkpoints \
    --model pixelnet \
    --loss combined \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --dropout 0.4
```

## Command Line Options

### Model Selection
```bash
--model light              # LightPixelNet (1.2M params)
--model pixelnet          # Original PixelNet (9.7M params)
```

### Attention Type (for light model)
```bash
--attention se            # SE (Squeeze-Excitation) - Recommended
--attention spatial       # Simplified Spatial Attention
--attention none          # No attention mechanism
```

### Loss Function
```bash
--loss combined           # Label Smoothing + Focal - Recommended
--loss label_smoothing    # Label Smoothing only
--loss focal              # Focal Loss only
--loss ce                 # Standard Cross Entropy
```

### Number of ResBlocks (for light model)
```bash
--num-res-blocks 2        # 2 blocks (faster, 1.2M params) - Recommended
--num-res-blocks 3        # 3 blocks (more capacity, 2.4M params)
```

## Model Variants Comparison

| Configuration | Params | Size | Speed | Use Case |
|--------------|--------|------|-------|----------|
| **light + se + 2 blocks** | 1.2M | 4.6MB | 3.2ms | Production deployment ⭐ |
| light + spatial + 2 blocks | 1.2M | 4.5MB | 4.8ms | Alternative to SE |
| light + none + 3 blocks | 2.4M | 9.1MB | 5.5ms | More capacity needed |
| pixelnet | 9.7M | 37MB | 14.7ms | Maximum accuracy |

## Tips for Best Results

### 1. Data Generation
- Use `--font-size-range 10-24` for better font size diversity
- Generate enough samples (~10K per class minimum)
- Use different random seeds for train/val/test splits

### 2. Training
- Start with `--model light --attention se` (best efficiency)
- Use `--loss combined` (best generalization)
- Lower dropout for light model: `--dropout 0.3` (vs 0.4 for original)
- Monitor validation accuracy to detect overfitting

### 3. Hyperparameter Tuning
```bash
# Conservative (less overfitting)
--learning-rate 5e-4 --dropout 0.4 --batch-size 64

# Aggressive (faster convergence)
--learning-rate 2e-3 --dropout 0.2 --batch-size 128

# Balanced (recommended)
--learning-rate 1e-3 --dropout 0.3 --batch-size 64
```

## Monitoring Training

TensorBoard logs are saved in `checkpoints/run_*/logs/`:

```bash
tensorboard --logdir ./checkpoints/run_20260107_120000/logs
```

Monitor:
- Training/Validation Loss
- Training/Validation Accuracy
- Learning Rate
- Confusion cases (Q/0, 8/B)

## Inference

```python
from src.inference import CharacterPredictor

# Load trained light model
predictor = CharacterPredictor('checkpoints/run_*/best_model.pth')

# Single prediction
char, confidence = predictor.predict('image.png')
print(f"{char}: {confidence:.2%}")

# Batch prediction
results = predictor.predict_batch(['img1.png', 'img2.png', 'img3.png'])
```

## Troubleshooting

### Issue: Model not converging
- Reduce learning rate: `--learning-rate 5e-4`
- Increase dropout: `--dropout 0.4`
- Check data quality and balance

### Issue: Overfitting (train acc >> val acc)
- Increase dropout: `--dropout 0.4`
- Use more data augmentation (already enabled)
- Use `--loss combined` with label smoothing

### Issue: Low accuracy on certain fonts
- Generate more diverse training data
- Ensure `--font-size-range 10-24` is used
- Check font rendering quality

## Next Steps

1. **Train and Evaluate**: Run training with recommended settings
2. **Compare Models**: Train both light and original for comparison
3. **Test Generalization**: Test on unseen fonts and backgrounds
4. **Optimize Further**: Consider INT8 quantization for deployment
5. **Deploy**: Use the trained light model in production

## Support

For detailed information, see:
- `MODEL_COMPARISON.md` - Architecture details
- `IMPLEMENTATION_SUMMARY.md` - Implementation overview
- `tasks.md` - Task completion status
- `README.md` - Full documentation
