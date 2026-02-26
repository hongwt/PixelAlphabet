## MODIFIED Requirements

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

### Requirement: Input Image Processing
The system MUST process game skill icon images as source backgrounds.
- Accept PNG format images from configurable directory
- Recursively discover all `.png` files in input directory tree
- Resize source icons to 50×50 pixels using **NEAREST neighbor interpolation** to preserve pixel-art hard edges
- Handle corrupted or unsupported images gracefully (log warning, skip)

#### Scenario: Nearest Neighbor Resize
- **WHEN** a 256×256 skill icon is resized to 50×50
- **THEN** the resized image SHALL contain only colors present in the original (no interpolation-generated intermediate colors)

## ADDED Requirements

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


