# Tasks: Optimize Fine-Grained Recognition

## Development
- [x] Implement `FocalLoss` class in `src/train.py` or new `src/loss.py` <!-- id: 1 -->
- [x] Modify `PixelNet` in `src/model.py` to adjust strides and increase final feature map size (target 6x6) <!-- id: 2 -->
- [x] Update `src/dataset.py` to refine augmentation (limit rotation, add RandomErasing) <!-- id: 3 -->
- [x] Update `src/train.py` to use `FocalLoss` and add confusion matrix logging <!-- id: 4 -->

## Validation
- [ ] Run training with new architecture and compare validation accuracy <!-- id: 5 -->
- [ ] Verify specific confusion cases (Q vs 0, 8 vs B) using the test set <!-- id: 6 -->
- [ ] Ensure model parameter count remains within reasonable limits (< 2M) <!-- id: 7 -->
