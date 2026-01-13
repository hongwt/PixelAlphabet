# recognition Specification Delta

## MODIFIED Requirements

### Requirement: Model Architecture
The system MUST use a custom deep Convolutional Neural Network (CNN) with residual connections.
- **Output Layer**: Fully Connected layer with **36** outputs (Digits 0-9, Letters A-Z).

### Scenario: Alphanumeric Classification
Given a 24x24 image containing a clear character 'A',
When the model processes the image,
Then it should return the label 'A' with probability > 0.8.

## REMOVED Requirements

### Requirement: 'NA' Class Handling
The system MUST support a 'Not Applicable' (NA) class for images that do not contain recognizable text.

## ADDED Requirements

### Requirement: Low Confidence Handling
The system MUST handle non-character images by producing low confidence scores for all valid classes.
- If the maximum probability of the output is below a threshold (e.g., 0.6), the result should be considered "No Character".

### Scenario: Non-text Image Confidence
Given a 24x24 image valid skill icon but no text overlay,
When the model processes the image,
Then the maximum confidence score for any class (0-9, A-Z) should be < 0.6.
