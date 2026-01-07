# Implementation Tasks

**Change ID**: `add-data-generation`  
**Status**: Pending Approval

## Task Checklist

### 1. Setup and Infrastructure
- [x] Create `src/data_generator.py` module file
- [x] Add Pillow dependency to `requirements.txt` (if not present)
- [x] Create default output directory structure (`traindata/`)
- [x] Verify `Fonts/` directory contains expected TTF files

### 2. Core Image Processing Functions
- [x] Implement `resize_image(img, width, height)` using PIL
- [x] Implement `copy_image(img)` to duplicate BufferedImage behavior
- [x] Implement `extract_patch(img, x, y, width, height)` for region extraction
- [x] Test image operations with sample icon file

### 3. Font Management
- [x] Implement `load_system_fonts()` to enumerate available system fonts
- [x] Implement `load_custom_fonts(fonts_dir)` to load TTF files from directory
- [x] Implement `create_font(font_path_or_name, size)` to instantiate PIL.ImageFont
- [x] Handle missing font files with fallback to default font
- [x] Test font loading with both system and custom fonts

### 4. Text Rendering with Outline
- [x] Implement `measure_text_size(text, font)` using PIL FontMetrics
- [x] Implement `draw_text_outline(draw, text, position, font, outline_color, fill_color)`
  - Draw text at 9 positions (-1,0,+1 offsets) in outline color
  - Draw text at center position in fill color
- [x] Test outline rendering on dark and light backgrounds
- [x] Verify text positioning (bottom-right with 3px margins)

### 5. Data Generator Class
- [x] Create `DataGenerator` class with configurable parameters
  - `input_dir`: Source icon directory
  - `output_dir`: Generated image output directory
  - `fonts_dir`: Custom fonts directory
  - `gt_file`: Ground truth file path
  - `charset`: Characters to generate
  - `font_size_range`: (min, max) tuple
  - `seed`: Optional random seed
- [x] Implement `_discover_icons()` to recursively find PNG files
- [x] Implement `_initialize_fonts()` to prepare font list
- [x] Implement `_process_icon(icon_path)` for single icon processing
  - Resize to 50x50
  - Iterate through charset
  - For each character: random font, random position, render, save
- [x] Implement `_generate_sample(icon_img, char, font, font_idx, offset_x, offset_y, icon_name)`
  - Extract 24x24 patch
  - Overlay text with outline
  - Save at native 24x24 resolution
  - Save with structured filename
  - Append to ground truth
- [x] Implement `_append_ground_truth(image_path, label)` with thread-safe file append
- [x] Implement `generate_dataset()` main entry point
  - Progress reporting (X/Y icons)
  - Error handling and logging
  - Summary statistics

### 6. CLI Interface
- [x] Implement `main()` function with argparse
- [x] Define CLI arguments:
  - `--input-dir` (required)
  - `--output-dir` (default: "traindata")
  - `--fonts-dir` (default: "Fonts")
  - `--gt-file` (default: "gt.txt")
  - `--charset` (default: "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
  - `--font-size-range` (default: "12-14")
  - `--seed` (optional)
- [x] Add `if __name__ == "__main__":` block to enable `python -m src.data_generator`
- [x] Test CLI with sample arguments

### 7. Error Handling and Validation
- [x] Add input validation (directory exists, writable output)
- [x] Handle image loading errors (try-except, log, continue)
- [x] Handle font loading errors (fallback mechanism)
- [x] Add disk space check before generation (optional but recommended)
- [x] Ensure ground truth file is not corrupted on partial failure

### 8. Testing
- [x] Create `tests/test_data_generator.py`
- [x] Unit test: `test_resize_image()` - verify dimensions
- [x] Unit test: `test_extract_patch()` - check bounds and content
- [x] Unit test: `test_measure_text_size()` - validate measurements
- [x] Unit test: `test_draw_text_outline()` - visual inspection helper
- [x] Integration test: `test_generate_single_sample()` - end-to-end with mock icon
- [x] Integration test: `test_ground_truth_format()` - parse and validate gt.txt
- [x] Run tests with `pytest tests/test_data_generator.py`

### 9. Documentation
- [x] Add docstrings to all public functions and classes
- [x] Document expected directory structure in module docstring
- [x] Update [README.md](d:\vs_project\PixelAlphabet\README.md) with "Data Generation" section
  - Explain purpose
  - Show CLI usage example
  - Describe output format
  - Link to AutoGenImage.java as reference
- [x] Add example command for generating dataset from sample icons

### 10. Verification and Validation
- [x] Generate small test dataset (5 icons, full charset)
- [x] Manually inspect 10-20 samples for quality
  - Check text clarity
  - Verify outline visibility
  - Confirm positioning
- [x] Verify ground truth file correctness (spot check 20 lines)
- [x] Confirm generated images load correctly with `PixelDataset`
- [x] Run `openspec validate add-data-generation --strict` and fix any issues
- [x] Review all Open Questions in proposal.md and document decisions

## Task Dependencies

```
1 (Setup) → 2 (Image Ops) → 4 (Text Rendering)
1 (Setup) → 3 (Font Mgmt) → 4 (Text Rendering)
4 (Text Rendering) → 5 (Generator Class)
5 (Generator Class) → 6 (CLI)
6 (CLI) → 7 (Error Handling)
7 (Error Handling) → 8 (Testing)
8 (Testing) → 9 (Documentation)
9 (Documentation) → 10 (Verification)
```

**Parallelizable**:
- Tasks 2 and 3 can be done in parallel after task 1
- Task 9 (docs) can start once task 6 (CLI) is done, doesn't need to wait for testing

## Estimated Effort
- Core implementation (tasks 1-7): ~4-6 hours
- Testing (task 8): ~1-2 hours
- Documentation (task 9): ~30 minutes
- Verification (task 10): ~1 hour

**Total**: ~6-9 hours (single developer)

## Acceptance Criteria
- [x] All tasks marked complete with `[x]`
- [x] All unit and integration tests pass
- [x] `openspec validate add-data-generation --strict` passes
- [x] Sample dataset generated and visually verified
- [x] README updated with clear usage instructions
- [x] No open critical issues or questions in proposal.md
