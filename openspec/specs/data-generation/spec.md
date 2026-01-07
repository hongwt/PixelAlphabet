# data-generation Specification

## Purpose
TBD - created by archiving change add-data-generation. Update Purpose after archive.
## Requirements
### Requirement: Input Image Processing
The system MUST process game skill icon images as source backgrounds.
- Accept PNG format images from configurable directory
- Recursively discover all `.png` files in input directory tree
- Resize source icons to 50x50 pixels before patch extraction
- Handle corrupted or unsupported images gracefully (log warning, skip)

#### Scenario: Batch Icon Processing
Given a directory containing 100 skill icon PNG files,
When the data generator runs,
Then it should successfully process all valid icons and log errors for any corrupted files without stopping.

### Requirement: Random Patch Extraction
The system MUST extract random 24x24 pixel patches from skill icons.
- Extract from resized 50x50 icon images
- Random X offset: 0-26 pixels (ensuring 24px width fits)
- Random Y offset: 0-26 pixels (ensuring 24px height fits)
- Each character overlay uses a fresh random position

#### Scenario: Spatial Variation
Given a single skill icon,
When generating samples for character 'A' five times,
Then each sample should have the character overlaid at different positions within the icon.

### Requirement: Multi-Font Text Rendering
The system MUST support rendering characters with multiple font types.
- System fonts: Arial, Times New Roman, Helvetica, Verdana, Georgia, Garamond, Courier New, Tahoma, Trebuchet MS
- Custom TrueType fonts from `Fonts/` directory (ARHei, ARIAL variants, Kai fonts, etc.)
- Random font selection per character instance
- Font size randomization: 12-14 pixels

#### Scenario: Font Diversity
Given the character '5',
When generating 100 samples,
Then the output should include instances rendered in at least 15 different fonts.

### Requirement: Styled Text Overlay
The system MUST render text with black outline and white fill for visibility.
- Outline: Draw character in black at 8 surrounding positions (±1 pixel offset in x and y)
- Fill: Draw character in white at center position
- Text position: Bottom-right alignment (3px margins from edges)
- Measure text dimensions to calculate exact position

#### Scenario: High Contrast Text
Given a skill icon with dark blue background,
When overlaying character 'Z' with white text and black outline,
Then the character should be clearly readable against the background.

### Requirement: Output Format
The system MUST generate training images in a structured format.
- Final image size: 24x24 pixels (native resolution)
- File format: PNG with RGB color
- Naming convention: `{icon_name}_FONT{font_id}_{position_hash}_{character}.png`
- Output directory: Configurable (default: `traindata/`)

#### Scenario: Filename Uniqueness
Given an icon named "fireball.png" with character 'A' at position (5, 10) using font index 3,
When generating the sample,
Then the output filename should uniquely identify this combination (e.g., `fireball.png_FONT3_55_A.png`).

### Requirement: Ground Truth Generation
The system MUST produce a ground truth file mapping images to labels.
- File format: Tab-separated values (TSV)
- Line format: `{relative_path}\t{character_label}`
- File mode: Append mode to support incremental generation
- Path relativity: Relative to project root or configurable base path

#### Scenario: Ground Truth Accuracy
Given 1000 generated images,
When parsing the ground truth file,
Then every line should correctly map each image path to its corresponding character label with 100% accuracy.

### Requirement: Character Set Support
The system MUST generate samples for a configurable character set.
- Default charset: `0123456789ABCDEFGHJKLMNPQRSTUVWXYZ` (35 characters, excluding I and O to match charset)
- Numeric labels: 0-9 → '0'-'9'
- Alphabetic labels: A-Z → 'A'-'Z' (excluding I, O)
- Configurable via CLI parameter

#### Scenario: Complete Charset Coverage
Given the default charset of 35 characters and 10 skill icons,
When generating a complete dataset,
Then the output should contain at least 350 images (10 icons × 35 chars, minimum).

### Requirement: Reproducible Generation
The system MUST support deterministic output via random seed.
- Optional `--seed` parameter for reproducibility
- Same seed + same inputs → identical output images
- Seed affects: font selection, position offsets, font size

#### Scenario: Reproducibility Verification
Given a seed value of 42 and a specific skill icon,
When running data generation twice with the same seed,
Then both runs should produce byte-identical output images.

### Requirement: CLI Interface
The system MUST provide a command-line interface for data generation.
- Required argument: `--input-dir` (path to skill icons)
- Optional arguments: `--output-dir`, `--fonts-dir`, `--gt-file`, `--charset`, `--seed`
- Display progress: X/Y icons processed
- Summary report: Total images generated, errors encountered

#### Scenario: CLI Usage
Given a user runs `python -m src.data_generator --input-dir ./icons --output-dir ./output`,
When the script executes,
Then it should process all icons in `./icons`, save images to `./output`, and display progress during execution.

### Requirement: Error Resilience
The system MUST handle errors without aborting entire batch.
- Unsupported image formats: Log warning, skip file
- Missing fonts: Use fallback system font
- Disk space exhaustion: Stop gracefully with error message
- Partial ground truth: Ensure already-written data is valid

#### Scenario: Partial Failure Handling
Given a batch of 100 icons where 3 are corrupted,
When running data generation,
Then the system should successfully process 97 icons and log warnings for the 3 failures.

