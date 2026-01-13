"""
PixelAlphabet Data Generator

Generates synthetic training data by overlaying alphanumeric characters
onto game skill icon backgrounds.

Based on AutoGenImage.java logic, reimplemented in Python.
"""
import os
import sys
import logging
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
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
    Resize image to specified dimensions using high-quality resampling.
    
    Args:
        img: PIL Image to resize
        width: Target width in pixels
        height: Target height in pixels
        
    Returns:
        Resized PIL Image
    """
    return img.resize((width, height), Image.Resampling.LANCZOS)


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


def draw_text_outline(
    draw: ImageDraw.Draw,
    text: str,
    position: Tuple[int, int],
    font: ImageFont.FreeTypeFont,
    outline_color: str = 'black',
    fill_color: str = 'white'
) -> None:
    """
    Draw text with an outline effect (black outline, white fill).
    
    Args:
        draw: PIL ImageDraw object
        text: Text to draw
        position: (x, y) position for text
        font: PIL ImageFont to use
        outline_color: Color for outline (default: black)
        fill_color: Color for text fill (default: white)
    """
    x, y = position
    
    # Draw outline by rendering text at 9 positions (8 surrounding + center)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    
    # Draw main text on top
    draw.text((x, y), text, font=font, fill=fill_color)


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
        seed: Optional[int] = None
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
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) / split
        self.fonts_dir = Path(fonts_dir)
        self.split = split
        self.charset = charset
        self.font_size_range = font_size_range
        self.seed = seed
        
        # Get sampling rate for this split
        self.sampling_rate = self.SPLIT_SAMPLING_RATES.get(split, 1.0)
        
        # Statistics
        self.total_generated = 0
        self.total_errors = 0
        
        # Initialize random seed
        if seed is not None:
            random.seed(seed)
        
        # Create output directory structure
        self._create_directory_structure()
        
        # Initialize fonts
        self.fonts = self._initialize_fonts()
        
        logger.info(f"DataGenerator initialized with {len(self.fonts)} fonts")
        logger.info(f"Character set: {self.charset} ({len(self.charset)} chars)")
        logger.info(f"Sampling rate: {self.sampling_rate:.0%} (split={self.split})")
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
                        offset_x, offset_y, icon_name
                    )
                    samples_generated += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to generate sample for '{char}' in {icon_name}: {e}")
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
        icon_name: str
    ) -> None:
        """
        Generate a single training sample.
        
        Args:
            icon_img: Source icon image (50x50)
            char: Character to overlay
            font: Font to use
            font_idx: Font index for filename
            offset_x: X offset for patch extraction
            offset_y: Y offset for patch extraction
            icon_name: Original icon filename
        """
        # Extract 24x24 patch from icon
        # Note: icon is 50x50, we want to extract 24x24 starting at (26-offset_x, offset_y)
        # Java code: image.getSubimage(image.getWidth() - 24 - i, j, 24, 24)
        # Where i and j are offsets 0-26
        x = icon_img.width - 24 - offset_x
        y = offset_y
        patch = extract_patch(icon_img, x, y, 24, 24)
        patch = copy_image(patch)
        
        # Create drawing context
        draw = ImageDraw.Draw(patch)
        
        # Measure text size
        text_width, text_height = measure_text_size(char, font)
        
        # Calculate text position (Centered with random jitter)
        # Center of the 24x24 patch
        center_x = (patch.width - text_width) // 2
        center_y = (patch.height - text_height) // 2
        
        # Add random jitter (±6 pixels) to increase position diversity
        jitter_x = random.randint(-6, 6)
        jitter_y = random.randint(-6, 6)
        
        text_x = center_x + jitter_x
        text_y = center_y + jitter_y
        
        # Ensure extremely large fonts don't position completely off-screen
        # But allow partial overlap as that is good for robustness
        
        # Draw text with outline
        draw_text_outline(draw, char, (text_x, text_y), font)
        
        # Generate output filename and save to character-specific subdirectory
        position_hash = (offset_x + 1) * (offset_y + 1)
        output_name = f"{icon_name}_FONT{font_idx}_{position_hash}.png"
        char_dir = self.output_dir / char
        output_path = char_dir / output_name
        
        # Save image at native 24x24 resolution
        patch.save(output_path, 'PNG')
        
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
            seed=args.seed
        )
        
        # Generate dataset
        generator.generate_dataset()
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
