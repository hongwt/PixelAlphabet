# Proposal: Remove 'NA' Class and Simplify Model Output

## Summary
Adjust the recognition model architecture to output 36 classes (Digits 0-9, Letters A-Z) instead of 37. Remove the explicit 'NA' (Not Applicable) class. Non-character images will be handled by low-confidence thresholds on the remaining classes.

## Background
Currently, the model predicts 37 classes, including 'NA' for invalid or non-character images. The user requested to remove this explicit 'NA' class and rely on confidence scores to filter out invalid inputs. This simplifies the training data organization and model architecture.

## Goals
- Update model architecture to utilize 36 output classes.
- Update data pipeline to exclude 'NA' class samples from training.
- Update inference logic to handle "no character" cases via low confidence scores.

## Implementation Details
1. **Model**: Change the final fully connected layer to output 36 logits.
2. **Data**: Stop generating 'NA' folders and exclude 'NA' from datasets.
3. **Training**: Train on 36 classes.
4. **Inference**: If the maximum softmax probability is below a certain threshold (e.g., 0.5), treat the result as "No Result" or "NA" logically, but the model itself won't have that class.

## Risks
- **False Positives**: The model will always predict *something* (0-9, A-Z) even for noise. We must tune the confidence threshold carefully to avoid false positives on empty icons.
