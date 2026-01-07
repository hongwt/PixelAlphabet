# Change Proposal: Add Training Data Generation

**Change ID**: `add-data-generation`  
**Status**: Draft  
**Author**: AI Assistant  
**Date**: 2026-01-06

## Problem Statement

Currently, the project lacks a systematic way to generate training data for the character recognition model. The original AutoGenImage.java demonstrates a working approach: overlaying alphanumeric characters with various fonts onto game skill icons, creating realistic training samples that simulate real-world usage.

To train the model effectively, we need to:
1. Process game skill icon images as backgrounds
2. Extract 24x24 pixel regions from these icons
3. Overlay characters (0-9, A-Z) with different fonts
4. Apply text styling (white text with black outline for readability)
5. Generate a ground truth file mapping images to labels
6. Save outputs at native 24x24 resolution (model input size)

The Java implementation exists but is not integrated with the current Python-based project structure.

## Proposed Solution

Create a Python-based training data generation system (`src/data_generator.py`) that reimplements the AutoGenImage.java logic using Python imaging libraries (Pillow/PIL). This will:

- Read skill icon images from a configurable directory
- Support multiple font types (system fonts and custom TTF fonts from the Fonts/ directory)
- Randomly sample 24x24 patches from 50x50 resized icons
- Overlay each character from the charset (0-9, A-Z) with randomized fonts and sizes
- Apply black outline effect around white characters for visibility
- Save processed images with structured naming: `{icon_name}_{font_id}_{position_hash}_{character}.png`
- Generate a ground truth file (`gt.txt`) with tab-separated paths and labels

This approach maintains compatibility with the existing dataset loading infrastructure while providing a pure Python solution.

## Scope

### In Scope
- New capability: **data-generation** (synthetic training data creation)
- New module: `src/data_generator.py` with main generation logic
- CLI interface for running data generation with configurable parameters
- Support for system fonts and custom TTF fonts
- Image processing: cropping, resizing, text overlay, outline effects
- Ground truth file generation
- Documentation in README for usage

### Out of Scope
- Data augmentation (handled by existing dataset.py augmentation pipeline)
- Real data collection from actual game screenshots
- Automated font downloading or management
- GUI for data generation
- Data validation or quality checks (manual inspection expected)

## Dependencies

### Affected Capabilities
- New: **data-generation** (no existing spec)
- Related: **recognition** (consumer of generated data, no changes needed)

### Technical Dependencies
- Pillow (PIL): Image manipulation, font rendering, drawing
- NumPy: Array operations (if needed)
- Python 3.10+ (existing project requirement)

### External Resources
- Game skill icon images (user-provided, not included in repo)
- Font files in `Fonts/` directory (already present)

## Impact Analysis

### Benefits
- Pure Python implementation aligns with project tech stack
- Reproducible training data generation
- Flexible configuration (paths, fonts, output directories)
- Supports experimentation with different fonts and parameters

### Risks & Mitigations
- **Risk**: Font rendering differences between Java AWT and PIL  
  **Mitigation**: Visual inspection of samples; adjust positioning/sizing parameters as needed
  
- **Risk**: Large dataset size (many icons × 35 chars × multiple fonts)  
  **Mitigation**: Make generation incremental; support filtering by character or icon subset

- **Risk**: Ground truth file corruption with concurrent writes  
  **Mitigation**: Use append mode with proper file locking or generate per-batch files

### Migration Path
- Java implementation remains as reference
- No breaking changes to existing code
- Generated data follows existing dataset conventions

## Alternatives Considered

1. **Keep Java implementation**: Rejected because it adds toolchain complexity and isn't integrated with Python workflow
2. **Use existing synthetic data libraries**: Rejected because our use case (game icons + overlaid text) is highly specific
3. **Manual data creation**: Rejected due to scale requirements (thousands of samples needed)

## Success Criteria

- [ ] `src/data_generator.py` successfully generates training images matching AutoGenImage.java output format
- [ ] Generated images have white text with black outlines on skill icon backgrounds
- [ ] Ground truth file correctly maps each image to its character label
- [ ] Script runs without errors on provided skill icon directory
- [ ] Generated data can be loaded by existing `PixelDataset` class
- [ ] README documents data generation process with examples
- [ ] At least 100 sample images generated and visually verified

## Open Questions

1. Should we support additional charsets beyond `0123456789ABCDEFGHJKLMNPQRSTUVWXYZ`? (Note: skips I, O)
2. What should be the default output directory structure? Flat or organized by character class?
3. Should font size randomization range (12-14 in Java) be configurable?
4. Do we need progress reporting for large batch generation?

## Related Work
- Original implementation: `AutoGenImage.java` (root directory)
- Consumer: `src/dataset.py` (PixelDataset class)
- Existing change: `add-recognition-model` (defines model requirements)
