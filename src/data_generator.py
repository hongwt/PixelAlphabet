"""
PixelAlphabet Data Generator

Generates synthetic training data by overlaying alphanumeric characters
onto game skill icon backgrounds.

Based on AutoGenImage.java logic, reimplemented in Python.
"""
import io
import os
import sys
import logging
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Default character set: 0-9 and A-Z (36 classes total)
DEFAULT_CHARSET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

# Required font files from Fonts directory
REQUIRED_FONTS = [
    "ARHei.ttf",
    "ARIALN.TTF",
    "ARKai_C.ttf",
    "ARKai_T.ttf",
    "bHEI00M.TTF",
    "bHEI01B.TTF",
    "bKAI00M.TTF",
    "bLEI00D.TTF",
    "FRIZQT__.TTF"
]

# Additional Windows system fonts for better diversity
SYSTEM_FONTS = [
    "arial.ttf",          # Sans-serif
    "times.ttf",          # Serif
    "cour.ttf",           # Monospace (Courier New)
    "georgia.ttf",        # Serif
    "verdana.ttf",        # Sans-serif
    "trebuc.ttf",         # Sans-serif (Trebuchet MS)
    "consola.ttf",        # Monospace (Consolas)
    "comic.ttf",          # Handwriting style
]

# Logger setup
logger = logging.getLogger(__name__)


def resize_image(img: Image.Image, width: int, height: int) -> Image.Image:
    """
    Resize image to specified dimensions using nearest-neighbor resampling.
    
    Uses NEAREST interpolation to preserve pixel-art hard edges and avoid
    anti-aliasing artifacts that break discrete high-frequency gradient features.
    
    Args:
        img: PIL Image to resize
        width: Target width in pixels
        height: Target height in pixels
        
    Returns:
        Resized PIL Image
    """
    return img.resize((width, height), Image.Resampling.NEAREST)


def copy_image(img: Image.Image) -> Image.Image:
    """
    Create a copy of the image.
    
    Args:
        img: PIL Image to copy
        
    Returns:
        Copied PIL Image
    """
    return img.copy()


def extract_patch(img: Image.Image, x: int, y: int, width: int, height: int) -> Image.Image:
    """
    Extract a rectangular region from the image.
    
    Args:
        img: Source PIL Image
        x: Left coordinate
        y: Top coordinate
        width: Patch width
        height: Patch height
        
    Returns:
        Cropped PIL Image
    """
    return img.crop((x, y, x + width, y + height))


def measure_text_size(text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    """
    Measure the dimensions of text when rendered with the given font.
    
    Args:
        text: Text string to measure
        font: PIL ImageFont to use
        
    Returns:
        Tuple of (width, height) in pixels
    """
    # Create a dummy image for measurement
    dummy_img = Image.new('RGB', (100, 100))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def render_foreground_char(
    text: str,
    font: ImageFont.FreeTypeFont,
    fill_color: Tuple[int, ...] = (255, 255, 255, 255),
    outline_color: Tuple[int, ...] = (0, 0, 0, 255),
    stroke_width: int = 1,
    shadow_offset: int = 0,
    hard_edge: bool = False
) -> Image.Image:
    """
    Render a single character as an RGBA foreground image.

    Supports two rendering modes selected by ``hard_edge``:

    **Smooth mode** (``hard_edge=False``, default)
        Uses Pillow's native ``stroke_width`` / ``stroke_fill`` API with
        standard anti-aliased font rendering.  Produces smoother edges with
        grey transition pixels.

    **Hard-edge mode** (``hard_edge=True``)
        Emulates the WoW-style HUD rendering pipeline:
        1. Anti-aliasing is disabled (``fontmode = "1"``) to produce binary
           pixel edges identical to real game screenshots.
        2. The outline is drawn via multi-offset passes (cardinal + diagonal)
           instead of native ``stroke_width`` which may lose pixels in binary
           mode.
        3. Additional right-down offsets simulate the drop-shadow bias
           visible in real game UI captures.

    Both styles appear in real game screenshots, so callers should randomise
    the choice for training-data diversity.

    Args:
        text: Character to render
        font: PIL ImageFont to use
        fill_color: RGBA fill color for the glyph (default: opaque white)
        outline_color: RGBA stroke color (default: opaque black)
        stroke_width: Stroke thickness in pixels (typically 1-2)
        shadow_offset: Extra right-down shadow offset in pixels (0 = none,
                       only used in hard-edge mode)
        hard_edge: If True, disable anti-aliasing and use multi-offset
                   outline + drop shadow.  If False, use native smooth stroke.

    Returns:
        RGBA PIL Image tightly cropped to the rendered glyph bounding box.
        Transparent where no glyph pixels exist.
    """
    # Use a generous canvas so the glyph + stroke + shadow fits
    canvas_size = (int(font.size * 3), int(font.size * 3))
    img = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    origin_x, origin_y = font.size, font.size

    if hard_edge:
        # --- Hard-edge (binary) rendering path ---
        # Disable anti-aliasing for hard binary edges
        draw.fontmode = "1"

        # Multi-offset outline (cardinal + diagonal directions)
        outline_offsets = set()
        for sw in range(1, stroke_width + 1):
            outline_offsets.update([
                (-sw, 0), (sw, 0), (0, -sw), (0, sw),       # cardinal
                (-sw, -sw), (sw, -sw), (-sw, sw), (sw, sw)   # diagonal
            ])

        for ox, oy in outline_offsets:
            draw.text((origin_x + ox, origin_y + oy), text, font=font,
                      fill=outline_color)

        # Right-down drop shadow (thicker right/bottom border)
        for s in range(1, shadow_offset + 1):
            draw.text((origin_x + s, origin_y + s), text, font=font,
                      fill=outline_color)

        # Main glyph on top
        draw.text((origin_x, origin_y), text, font=font, fill=fill_color)
    else:
        # --- Smooth (anti-aliased) rendering path ---
        draw.text(
            (origin_x, origin_y),
            text,
            font=font,
            fill=fill_color,
            stroke_width=stroke_width,
            stroke_fill=outline_color,
        )

    # Crop to tight bounding box
    bbox = img.getbbox()
    if bbox is None:
        # Nothing rendered – return a tiny transparent image
        return Image.new('RGBA', (1, 1), (0, 0, 0, 0))

    return img.crop(bbox)


def alpha_composite_hard_overlay(
    background: Image.Image,
    foreground: Image.Image,
    position: Tuple[int, int]
) -> Image.Image:
    """
    Composite an RGBA foreground onto an RGB background using hard overlay.

    Simulates the BitBLT hard-overlay behaviour of game HUD rendering:
    where the foreground alpha is > 0, the foreground pixel fully replaces
    the background pixel.

    Args:
        background: RGB background image
        foreground: RGBA foreground image
        position: (x, y) top-left position on the background

    Returns:
        RGB composite image (same size as background)
    """
    bg = background.convert('RGBA')

    # Create a full-size overlay with transparent pixels
    overlay = Image.new('RGBA', bg.size, (0, 0, 0, 0))
    overlay.paste(foreground, position)

    result = Image.alpha_composite(bg, overlay)
    return result.convert('RGB')


def load_custom_fonts(fonts_dir: Path) -> List[Path]:
    """
    Load required TrueType fonts from directory.
    
    Args:
        fonts_dir: Path to directory containing .ttf files
        
    Returns:
        List of paths to required .ttf font files
    """
    if not fonts_dir.exists():
        logger.error(f"Fonts directory not found: {fonts_dir}")
        return []
    
    font_paths = []
    missing_fonts = []
    
    for font_name in REQUIRED_FONTS:
        font_path = fonts_dir / font_name
        if font_path.exists():
            font_paths.append(font_path)
        else:
            missing_fonts.append(font_name)
    
    if missing_fonts:
        logger.warning(f"Missing fonts: {', '.join(missing_fonts)}")
    
    logger.info(f"Loaded {len(font_paths)}/{len(REQUIRED_FONTS)} required fonts from {fonts_dir}")
    return font_paths


def create_font(font_spec: str, size: int) -> Optional[ImageFont.FreeTypeFont]:
    """
    Create a PIL ImageFont from font specification.
    
    Args:
        font_spec: Either a system font name or path to .ttf file
        size: Font size in pixels
        
    Returns:
        PIL ImageFont object, or None if font cannot be loaded
    """
    try:
        # Check if it's a file path
        if os.path.exists(font_spec):
            return ImageFont.truetype(font_spec, size)
        
        # Try as system font name
        # On Windows, try common font paths
        if sys.platform == 'win32':
            font_paths = [
                f"C:\\Windows\\Fonts\\{font_spec}.ttf",
                f"C:\\Windows\\Fonts\\{font_spec.replace(' ', '')}.ttf",
                f"C:\\Windows\\Fonts\\{font_spec.lower().replace(' ', '')}.ttf",
            ]
            for path in font_paths:
                if os.path.exists(path):
                    return ImageFont.truetype(path, size)
        
        # Fallback: try to load by name (works on some systems)
        return ImageFont.truetype(font_spec, size)
    except Exception as e:
        logger.debug(f"Failed to load font '{font_spec}': {e}")
        return None


def get_fallback_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Get a fallback font when requested font fails to load.
    
    Args:
        size: Font size in pixels
        
    Returns:
        PIL ImageFont object (default font)
    """
    # Try common fallback fonts
    fallbacks = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    
    for fallback in fallbacks:
        font = create_font(fallback, size)
        if font:
            return font
    
    # Ultimate fallback: PIL default font
    logger.warning("Using PIL default font - text may not render correctly")
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Degradation pipeline (Layer 4)
# ---------------------------------------------------------------------------

# Default probabilities and parameter ranges per degradation level
DEGRADATION_PROFILES = {
    "none": {},
    "light": {
        "resample":     {"prob": 0.3, "scale": (0.5, 0.75)},
        "gaussian":     {"prob": 0.2, "sigma": (1, 5)},
        "salt_pepper":  {"prob": 0.15, "amount": (0.002, 0.01)},
        "gamma":        {"prob": 0.2, "gamma": (0.8, 1.2)},
        "hsv_drift":    {"prob": 0.2, "h": (-5, 5), "s": (-10, 10), "v": (-10, 10)},
        "jpeg":         {"prob": 0.3, "quality": (50, 85)},
    },
    "medium": {
        "resample":     {"prob": 0.5, "scale": (0.35, 0.65)},
        "gaussian":     {"prob": 0.4, "sigma": (2, 10)},
        "salt_pepper":  {"prob": 0.3, "amount": (0.005, 0.03)},
        "gamma":        {"prob": 0.4, "gamma": (0.6, 1.4)},
        "hsv_drift":    {"prob": 0.4, "h": (-10, 10), "s": (-20, 20), "v": (-20, 20)},
        "jpeg":         {"prob": 0.5, "quality": (25, 70)},
    },
    "heavy": {
        "resample":     {"prob": 0.7, "scale": (0.25, 0.5)},
        "gaussian":     {"prob": 0.6, "sigma": (5, 20)},
        "salt_pepper":  {"prob": 0.5, "amount": (0.01, 0.06)},
        "gamma":        {"prob": 0.6, "gamma": (0.4, 1.6)},
        "hsv_drift":    {"prob": 0.6, "h": (-15, 15), "s": (-30, 30), "v": (-30, 30)},
        "jpeg":         {"prob": 0.7, "quality": (10, 50)},
    },
}


def degrade_resample(img: Image.Image, scale: float) -> Image.Image:
    """Low-fidelity spatial resampling: downscale then upscale with NEAREST."""
    w, h = img.size
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    down = img.resize((small_w, small_h), Image.Resampling.NEAREST)
    return down.resize((w, h), Image.Resampling.NEAREST)


def degrade_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Add Gaussian noise to the image."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=img.mode)


def degrade_salt_pepper(img: Image.Image, amount: float) -> Image.Image:
    """Add salt-and-pepper noise."""
    arr = np.array(img)
    num_pixels = arr.shape[0] * arr.shape[1]
    num_salt = int(num_pixels * amount / 2)
    num_pepper = int(num_pixels * amount / 2)

    # Salt
    coords = [np.random.randint(0, d, num_salt) for d in arr.shape[:2]]
    arr[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, d, num_pepper) for d in arr.shape[:2]]
    arr[coords[0], coords[1]] = 0

    return Image.fromarray(arr, mode=img.mode)


def degrade_gamma(img: Image.Image, gamma: float) -> Image.Image:
    """Apply random gamma correction."""
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.power(arr, gamma)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=img.mode)


def degrade_hsv_drift(
    img: Image.Image, h_shift: float, s_shift: float, v_shift: float
) -> Image.Image:
    """Random HSV colour-space drift."""
    hsv = img.convert('HSV')
    arr = np.array(hsv, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + h_shift, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + s_shift, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + v_shift, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode='HSV').convert(img.mode)


def degrade_jpeg_artifacts(img: Image.Image, quality: int) -> Image.Image:
    """Simulate JPEG compression artifacts via encode/decode round-trip."""
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_degradation_pipeline(
    img: Image.Image, level: str = "none"
) -> Image.Image:
    """
    Apply a randomised degradation pipeline to the image.

    Each degradation operation is applied independently with a probability
    determined by the selected level (none / light / medium / heavy).

    Args:
        img: Input RGB image
        level: Degradation intensity level

    Returns:
        Degraded RGB image
    """
    profile = DEGRADATION_PROFILES.get(level, {})
    if not profile:
        return img

    result = img.copy()

    # 1. Resampling
    cfg = profile.get("resample")
    if cfg and random.random() < cfg["prob"]:
        scale = random.uniform(*cfg["scale"])
        result = degrade_resample(result, scale)

    # 2. Gaussian noise
    cfg = profile.get("gaussian")
    if cfg and random.random() < cfg["prob"]:
        sigma = random.uniform(*cfg["sigma"])
        result = degrade_gaussian_noise(result, sigma)

    # 3. Salt-pepper noise
    cfg = profile.get("salt_pepper")
    if cfg and random.random() < cfg["prob"]:
        amount = random.uniform(*cfg["amount"])
        result = degrade_salt_pepper(result, amount)

    # 4. Gamma correction
    cfg = profile.get("gamma")
    if cfg and random.random() < cfg["prob"]:
        gamma = random.uniform(*cfg["gamma"])
        result = degrade_gamma(result, gamma)

    # 5. HSV drift
    cfg = profile.get("hsv_drift")
    if cfg and random.random() < cfg["prob"]:
        h = random.uniform(*cfg["h"])
        s = random.uniform(*cfg["s"])
        v = random.uniform(*cfg["v"])
        result = degrade_hsv_drift(result, h, s, v)

    # 6. JPEG artifacts (always last – operates on final pixel values)
    cfg = profile.get("jpeg")
    if cfg and random.random() < cfg["prob"]:
        quality = random.randint(*cfg["quality"])
        result = degrade_jpeg_artifacts(result, quality)

    return result


class DataGenerator:
    """
    Generates synthetic training data from skill icon images.
    Organizes output by character class in subdirectories.
    """
    
    # Sampling rates for different dataset splits
    SPLIT_SAMPLING_RATES = {
        "train": 1.0,   # 100% - generate for all characters
        "val": 0.15,    # 15% - validation set
        "test": 0.10    # 10% - test set
    }
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        fonts_dir: Path,
        split: str = "train",
        charset: str = DEFAULT_CHARSET,
        font_size_range: Tuple[int, int] = (10, 24),
        seed: Optional[int] = None,
        variations_per_sample: int = 3,
        degradation: str = "none"
    ):
        """
        Initialize DataGenerator.
        
        Args:
            input_dir: Directory containing skill icon images
            output_dir: Base output directory (will create split/class subdirs)
            fonts_dir: Directory containing custom font files
            split: Dataset split (train/val/test)
            charset: Characters to generate samples for
            font_size_range: (min, max) font size in pixels
            seed: Random seed for reproducibility
            variations_per_sample: Number of variations to generate per icon-character pair
            degradation: Degradation level (none/light/medium/heavy)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) / split
        self.fonts_dir = Path(fonts_dir)
        self.split = split
        self.charset = charset
        self.font_size_range = font_size_range
        self.seed = seed
        self.variations_per_sample = variations_per_sample
        self.degradation = degradation
        
        # Get sampling rate for this split
        self.sampling_rate = self.SPLIT_SAMPLING_RATES.get(split, 1.0)
        
        # Statistics
        self.total_generated = 0
        self.total_errors = 0
        
        # Initialize random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create output directory structure
        self._create_directory_structure()
        
        # Initialize fonts
        self.fonts = self._initialize_fonts()
        
        logger.info(f"DataGenerator initialized with {len(self.fonts)} fonts")
        logger.info(f"Character set: {self.charset} ({len(self.charset)} chars)")
        logger.info(f"Variations per sample: {self.variations_per_sample}")
        logger.info(f"Sampling rate: {self.sampling_rate:.0%} (split={self.split})")
        logger.info(f"Degradation level: {self.degradation}")
        logger.info(f"Output structure: {self.output_dir}/<char>/")
    
    def _create_directory_structure(self) -> None:
        """
        Create output directory structure with subdirectories for each character.
        Structure: output_dir/split/char/
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a subdirectory for each character
        for char in self.charset:
            char_dir = self.output_dir / char
            char_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created {len(self.charset)} character subdirectories in {self.output_dir}")
    
    def _initialize_fonts(self) -> List[str]:
        """
        Initialize list of required fonts from Fonts directory and system fonts.
        
        Returns:
            List of font file paths
        """
        fonts = []
        
        # Load required custom fonts
        custom_fonts = load_custom_fonts(self.fonts_dir)
        fonts.extend([str(f) for f in custom_fonts])
        
        # Load Windows system fonts
        if sys.platform == 'win32':
            for font_name in SYSTEM_FONTS:
                font_path = f"C:\\Windows\\Fonts\\{font_name}"
                if os.path.exists(font_path):
                    fonts.append(font_path)
                    logger.debug(f"Added system font: {font_name}")
        
        if not fonts:
            raise ValueError(
                f"No required fonts found in {self.fonts_dir}. "
                f"Please ensure the following fonts are available: {', '.join(REQUIRED_FONTS)}"
            )
        
        logger.info(f"Loaded {len(fonts)} total fonts ({len(custom_fonts)} custom + {len(fonts) - len(custom_fonts)} system)")
        return fonts
    
    def _discover_icons(self) -> List[Path]:
        """
        Recursively discover all PNG files in input directory.
        
        Returns:
            List of paths to icon files
        """
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        # Use a set to avoid duplicates from case-insensitive file systems
        icons_set = set()
        icons_set.update(self.input_dir.rglob("*.png"))
        icons_set.update(self.input_dir.rglob("*.PNG"))
        icons = sorted(icons_set)
        
        logger.info(f"Found {len(icons)} icon files in {self.input_dir}")
        return icons
    
    def _process_icon(self, icon_path: Path) -> int:
        """
        Process a single icon file, generating samples for all characters.
        
        Args:
            icon_path: Path to icon image file
            
        Returns:
            Number of samples successfully generated
        """
        try:
            # Load and resize icon
            original = Image.open(icon_path).convert('RGB')
            resized = resize_image(original, 50, 50)
            
            samples_generated = 0
            icon_name = icon_path.name
            
            # Generate sample for each character
            for char in self.charset:
                # Apply sampling rate for val/test splits
                if self.sampling_rate < 1.0:
                    if random.random() > self.sampling_rate:
                        continue  # Skip this character based on sampling rate
                
                # Generate multiple variations for each icon-character pair
                for variation_idx in range(self.variations_per_sample):
                    try:
                        # Random font selection
                        font_idx = random.randint(0, len(self.fonts) - 1)
                        font_spec = self.fonts[font_idx]
                        font_size = random.randint(*self.font_size_range)
                        
                        # Load font
                        font = create_font(font_spec, font_size)
                        if font is None:
                            font = get_fallback_font(font_size)
                        
                        # Random position offsets
                        offset_x = random.randint(0, 26)
                        offset_y = random.randint(0, 26)
                    
                        # Generate sample
                        self._generate_sample(
                            resized, char, font, font_idx, 
                            offset_x, offset_y, icon_name, variation_idx
                        )
                        samples_generated += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate sample for '{char}' (var {variation_idx}) in {icon_name}: {e}")
                        self.total_errors += 1
            
            return samples_generated
            
        except Exception as e:
            logger.error(f"Failed to process icon {icon_path}: {e}")
            self.total_errors += 1
            return 0
    
    def _generate_sample(
        self,
        icon_img: Image.Image,
        char: str,
        font: ImageFont.FreeTypeFont,
        font_idx: int,
        offset_x: int,
        offset_y: int,
        icon_name: str,
        variation_idx: int = 0
    ) -> None:
        """
        Generate a single training sample using the 4-layer pipeline.

        Layer 1 – Foreground: render character as RGBA with native stroke.
        Layer 2 – Background: extract patch from skill icon.
        Layer 3 – Composite: Alpha-composite foreground onto background.
        Layer 4 – Degradation: optional image degradation augmentation.

        Args:
            icon_img: Source icon image (50x50)
            char: Character to overlay
            font: Font to use
            font_idx: Font index for filename
            offset_x: X offset for patch extraction
            offset_y: Y offset for patch extraction
            icon_name: Original icon filename
            variation_idx: Index of the variation (for unique filenames)
        """
        # --- Layer 2: Background patch ---
        x = icon_img.width - 24 - offset_x
        y = offset_y
        patch = extract_patch(icon_img, x, y, 24, 24)
        patch = copy_image(patch)  # detach from source

        # --- Layer 1: Foreground rendering (RGBA) ---
        # Randomly choose between hard-edge (binary) and smooth (anti-aliased)
        # rendering to match the diversity seen in real game screenshots.
        use_hard_edge = random.random() < 0.5
        stroke_w = random.randint(1, 2)
        shadow_off = random.randint(1, 2) if use_hard_edge else 0
        fg = render_foreground_char(
            text=char,
            font=font,
            fill_color=(255, 255, 255, 255),
            outline_color=(0, 0, 0, 255),
            stroke_width=stroke_w,
            shadow_offset=shadow_off,
            hard_edge=use_hard_edge,
        )

        # --- Position foreground (bias towards top-right with small jitter) ---
        fw, fh = fg.size
        pw, ph = patch.size

        # Target: top-right corner with 1px margin
        target_x = pw - fw - 1
        target_y = 1

        # Jitter away from top-right (x decreases, y increases)
        jitter_x = random.randint(0, 2)
        jitter_y = random.randint(0, 2)

        text_x = target_x - jitter_x
        text_y = target_y + jitter_y

        # Clamp to keep within patch bounds
        text_x = max(0, min(text_x, pw - fw))
        text_y = max(0, min(text_y, ph - fh))

        # --- Layer 3: Alpha composite ---
        result = alpha_composite_hard_overlay(patch, fg, (text_x, text_y))

        # --- Layer 4: Degradation augmentation ---
        result = apply_degradation_pipeline(result, self.degradation)

        # --- Save ---
        position_hash = (offset_x + 1) * (offset_y + 1)
        output_name = f"{icon_name}_FONT{font_idx}_{position_hash}_v{variation_idx}.png"
        char_dir = self.output_dir / char
        output_path = char_dir / output_name

        result.save(output_path, 'PNG')

        self.total_generated += 1
    
    def generate_dataset(self) -> None:
        """
        Main entry point: generate complete dataset.
        """
        logger.info("Starting dataset generation...")
        
        # Discover icons
        icons = self._discover_icons()
        if not icons:
            logger.error("No icon files found")
            return
        
        # Process each icon
        for idx, icon_path in enumerate(icons, 1):
            logger.info(f"Processing icon {idx}/{len(icons)}: {icon_path.name}")
            samples = self._process_icon(icon_path)
            logger.info(f"  Generated {samples} samples")
        
        # Summary
        logger.info("="*60)
        logger.info(f"Dataset generation complete!")
        logger.info(f"Split: {self.split}")
        logger.info(f"Total images generated: {self.total_generated}")
        logger.info(f"Total errors: {self.total_errors}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Images per character: ~{self.total_generated // len(self.charset)}")
        logger.info("="*60)


def main():
    """
    CLI entry point for data generator.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from skill icon images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m src.data_generator --input-dir ./icons --output-dir ./data --split train
  python -m src.data_generator --input-dir ./icons --split val --seed 42
  python -m src.data_generator --input-dir ./icons --split test --charset "0123456789"
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing skill icon PNG files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for generated images (default: data)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split (train/val/test) (default: train)"
    )
    
    parser.add_argument(
        "--fonts-dir",
        type=str,
        default="Fonts",
        help="Directory containing custom TTF fonts (default: Fonts)"
    )
    
    parser.add_argument(
        "--charset",
        type=str,
        default=DEFAULT_CHARSET,
        help=f"Characters to generate samples for (default: {DEFAULT_CHARSET})"
    )
    
    parser.add_argument(
        "--font-size-range",
        type=str,
        default="10-24",
        help="Font size range as 'min-max' (default: 10-24)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    
    parser.add_argument(
        "--variations",
        type=int,
        default=3,
        help="Number of variations to generate per icon-character pair (default: 3)"
    )
    
    parser.add_argument(
        "--degradation",
        type=str,
        default="none",
        choices=["none", "light", "medium", "heavy"],
        help="Image degradation intensity for augmentation (default: none)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse font size range
    try:
        min_size, max_size = map(int, args.font_size_range.split('-'))
        font_size_range = (min_size, max_size)
    except ValueError:
        logger.error(f"Invalid font-size-range format: {args.font_size_range}")
        logger.error("Expected format: 'min-max' (e.g., '12-14')")
        sys.exit(1)
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create generator
    try:
        generator = DataGenerator(
            input_dir=input_dir,
            output_dir=Path(args.output_dir),
            fonts_dir=Path(args.fonts_dir),
            split=args.split,
            charset=args.charset,
            font_size_range=font_size_range,
            seed=args.seed,
            variations_per_sample=args.variations,
            degradation=args.degradation
        )
        
        # Generate dataset
        generator.generate_dataset()
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
