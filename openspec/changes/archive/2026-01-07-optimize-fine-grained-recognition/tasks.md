# Tasks: Optimize Fine-Grained Recognition

## Development
- [x] Implement `FocalLoss` class in `src/loss.py` <!-- id: 1 -->
- [x] Modify `PixelNet` in `src/model.py` to remove pooling, adjust strides, and increase channels (32->64) <!-- id: 2 -->
- [x] Update `src/data_generator.py` to refine generation (variable font sizes, random positions) <!-- id: 3 -->
- [x] Update `src/train.py` to use `FocalLoss` and add confusion matrix logging <!-- id: 4 -->

## Validation
- [x] Run training with new architecture and verify parameters (9.7M) <!-- id: 5 -->
- [x] Verify specific confusion cases (Q vs 0, 8 vs B) using the test set <!-- id: 6 -->
- [x] Ensure model parameter count is acceptable <!-- id: 7 -->
