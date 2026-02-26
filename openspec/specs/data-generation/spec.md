# data-generation Specification

## Purpose
TBD - created by archiving change add-data-generation. Update Purpose after archive.
## Requirements
### Requirement: Input Image Processing
The system MUST process game skill icon images as source backgrounds.
- Accept PNG format images from configurable directory
- Recursively discover all `.png` files in input directory tree
- Resize source icons to 50×50 pixels using **NEAREST neighbor interpolation** to preserve pixel-art hard edges
- Handle corrupted or unsupported images gracefully (log warning, skip)

#### Scenario: Nearest Neighbor Resize
- **WHEN** a 256×256 skill icon is resized to 50×50
- **THEN** the resized image SHALL contain only colors present in the original (no interpolation-generated intermediate colors)

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
The system MUST render foreground characters on a separate RGBA transparent canvas using Pillow native stroke API, then composite onto background via Alpha blending.

- Foreground rendering on RGBA canvas: `fill=(255,255,255,255)`, `stroke_fill=(0,0,0,255)`
- Use Pillow `stroke_width` and `stroke_fill` parameters for pixel-perfect outline rendering
- Stroke width: randomized between 1 and 2 pixels for diversity
- Crop foreground to tight bounding box via `getbbox()` before compositing
- Alpha compositing SHALL produce hard overlay (binary alpha: 0 or 255, no anti-aliased transition)
- Final output remains 24×24 pixels RGB PNG

#### Scenario: RGBA Foreground Rendering
- **WHEN** character 'A' is rendered with stroke_width=1
- **THEN** the foreground RGBA image SHALL contain only pixels with alpha=255 (text and outline) or alpha=0 (transparent), with no intermediate values

#### Scenario: Hard Overlay Compositing
- **WHEN** a white-on-black-outline character is composited onto a complex game background
- **THEN** the character pixels SHALL completely replace background pixels (no color blending or feathering at edges)

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

### Requirement: Image Degradation Pipeline
The system MUST support an optional image degradation pipeline applied after compositing to simulate real-world capture artifacts.

Each degradation operation SHALL be applied with independent random probability. The following operations are supported:
1. **Low-fidelity spatial resampling**: Downsample to 40-80% of original size then upsample back, using NEAREST interpolation only
2. **Gaussian noise injection**: Additive Gaussian noise with configurable sigma (default: 3-10)
3. **Salt-and-pepper noise injection**: Random pixel corruption with configurable density (default: 0.5-2%)
4. **Random Gamma correction**: Non-linear brightness adjustment with gamma range 0.7-1.4
5. **HSV color space drift**: Random perturbation of Hue (±10°), Saturation (±15%), Value (±10%) channels
6. **JPEG compression artifacts simulation**: Encode/decode at low quality (30-70) to introduce blockiness

- CLI parameter `--degradation` controls overall intensity: `none` (default), `light`, `medium`, `heavy`
- Each intensity level sets base probabilities for individual operations
- Individual degradation params are tunable via config

#### Scenario: Light Degradation
- **WHEN** `--degradation light` is specified
- **THEN** generated images SHOULD have subtle artifacts (e.g., minor noise or slight color shift) while characters remain clearly recognizable

#### Scenario: Heavy Degradation
- **WHEN** `--degradation heavy` is specified
- **THEN** generated images SHOULD simulate worst-case real scenarios (combined resampling blur, noise, compression artifacts, and color shifts)

#### Scenario: No Degradation
- **WHEN** `--degradation none` is specified (or omitted)
- **THEN** generated images SHALL be clean composites with no artificial degradation

