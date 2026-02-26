"""
Tests for data_generator module
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image, ImageFont

from src.data_generator import (
    resize_image,
    copy_image,
    extract_patch,
    measure_text_size,
    render_foreground_char,
    alpha_composite_hard_overlay,
    load_custom_fonts,
    create_font,
    get_fallback_font,
    DataGenerator,
    DEFAULT_CHARSET,
    degrade_resample,
    degrade_gaussian_noise,
    degrade_salt_pepper,
    degrade_gamma,
    degrade_hsv_drift,
    degrade_jpeg_artifacts,
    apply_degradation_pipeline,
)


class TestImageOperations:
    """Test image processing functions."""
    
    def test_resize_image(self):
        """Test image resizing."""
        img = Image.new('RGB', (100, 100), color='red')
        resized = resize_image(img, 50, 50)
        
        assert resized.size == (50, 50)
        assert resized.mode == 'RGB'
    
    def test_copy_image(self):
        """Test image copying."""
        img = Image.new('RGB', (50, 50), color='blue')
        copied = copy_image(img)
        
        assert copied.size == img.size
        assert copied.mode == img.mode
        assert copied is not img  # Different object
    
    def test_extract_patch(self):
        """Test patch extraction."""
        img = Image.new('RGB', (100, 100), color='green')
        patch = extract_patch(img, 10, 20, 24, 24)
        
        assert patch.size == (24, 24)
    
    def test_extract_patch_boundaries(self):
        """Test patch extraction at boundaries."""
        img = Image.new('RGB', (50, 50), color='yellow')
        
        # Extract from corner
        patch = extract_patch(img, 0, 0, 24, 24)
        assert patch.size == (24, 24)
        
        # Extract from opposite corner
        patch = extract_patch(img, 26, 26, 24, 24)
        assert patch.size == (24, 24)


class TestTextOperations:
    """Test text rendering functions."""
    
    def test_measure_text_size(self):
        """Test text dimension measurement."""
        font = get_fallback_font(12)
        width, height = measure_text_size("A", font)
        
        assert width > 0
        assert height > 0
    
    def test_render_foreground_char_returns_rgba(self):
        """Test that render_foreground_char outputs RGBA image."""
        font = get_fallback_font(14)
        fg = render_foreground_char("X", font)

        assert fg.mode == 'RGBA'
        assert fg.size[0] > 0
        assert fg.size[1] > 0

    def test_render_foreground_char_has_opaque_pixels(self):
        """Test that rendered foreground contains non-transparent pixels."""
        font = get_fallback_font(14)
        fg = render_foreground_char("A", font)

        import numpy as np
        arr = np.array(fg)
        alpha = arr[:, :, 3]
        assert alpha.max() == 255, "Should have fully opaque pixels"

    def test_render_foreground_char_stroke_width(self):
        """Test different stroke widths produce different-sized images."""
        font = get_fallback_font(14)
        fg1 = render_foreground_char("M", font, stroke_width=1)
        fg2 = render_foreground_char("M", font, stroke_width=2)

        # Wider stroke should generally produce a larger bounding box
        area1 = fg1.size[0] * fg1.size[1]
        area2 = fg2.size[0] * fg2.size[1]
        assert area2 >= area1


class TestAlphaComposite:
    """Test alpha compositing functions."""

    def test_basic_composite(self):
        """Test alpha composite produces correct output size and mode."""
        bg = Image.new('RGB', (24, 24), color=(128, 128, 128))
        fg = Image.new('RGBA', (10, 10), color=(255, 0, 0, 255))

        result = alpha_composite_hard_overlay(bg, fg, (5, 5))

        assert result.mode == 'RGB'
        assert result.size == (24, 24)

    def test_composite_overwrites_background(self):
        """Test that opaque foreground pixels replace background."""
        bg = Image.new('RGB', (24, 24), color=(0, 0, 0))
        fg = Image.new('RGBA', (2, 2), color=(255, 255, 255, 255))

        result = alpha_composite_hard_overlay(bg, fg, (0, 0))

        # Top-left 2x2 should be white
        assert result.getpixel((0, 0)) == (255, 255, 255)
        assert result.getpixel((1, 1)) == (255, 255, 255)
        # Outside should stay black
        assert result.getpixel((3, 3)) == (0, 0, 0)

    def test_composite_transparent_preserves_background(self):
        """Test that transparent foreground pixels leave background intact."""
        bg = Image.new('RGB', (24, 24), color=(100, 100, 100))
        fg = Image.new('RGBA', (10, 10), color=(255, 0, 0, 0))  # Fully transparent

        result = alpha_composite_hard_overlay(bg, fg, (0, 0))

        assert result.getpixel((5, 5)) == (100, 100, 100)


class TestFontManagement:
    """Test font loading and management."""
    
    def test_load_custom_fonts_missing_dir(self):
        """Test custom fonts with missing directory."""
        fonts = load_custom_fonts(Path("/nonexistent/path"))
        
        assert isinstance(fonts, list)
        assert len(fonts) == 0
    
    def test_load_custom_fonts_existing_dir(self):
        """Test custom fonts from existing directory."""
        # Use the project's Fonts directory if it exists
        fonts_dir = Path("Fonts")
        if fonts_dir.exists():
            fonts = load_custom_fonts(fonts_dir)
            assert isinstance(fonts, list)
    
    def test_get_fallback_font(self):
        """Test fallback font retrieval."""
        font = get_fallback_font(12)
        
        assert font is not None
        assert isinstance(font, (ImageFont.FreeTypeFont, ImageFont.ImageFont))


class TestDataGenerator:
    """Test DataGenerator class."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_root = tempfile.mkdtemp()
        input_dir = Path(temp_root) / "input"
        output_dir = Path(temp_root) / "output"
        fonts_dir = Path(temp_root) / "fonts"
        
        input_dir.mkdir()
        fonts_dir.mkdir()
        
        yield {
            'root': Path(temp_root),
            'input': input_dir,
            'output': output_dir,
            'fonts': fonts_dir
        }
        
        # Cleanup
        shutil.rmtree(temp_root)
    
    def test_generator_initialization(self, temp_dirs):
        """Test DataGenerator initialization."""
        generator = DataGenerator(
            input_dir=temp_dirs['input'],
            output_dir=temp_dirs['output'],
            fonts_dir=temp_dirs['fonts'],
            split="train",
            charset="ABC",
            seed=42
        )
        
        assert generator.charset == "ABC"
        assert generator.split == "train"
        assert generator.seed == 42
        assert generator.total_generated == 0
        assert generator.total_errors == 0
    
    def test_discover_icons_empty(self, temp_dirs):
        """Test icon discovery with empty directory."""
        generator = DataGenerator(
            input_dir=temp_dirs['input'],
            output_dir=temp_dirs['output'],
            fonts_dir=temp_dirs['fonts'],
            split="train"
        )
        
        icons = generator._discover_icons()
        assert len(icons) == 0
    
    def test_discover_icons_with_files(self, temp_dirs):
        """Test icon discovery with PNG files."""
        # Create test PNG files
        for i in range(3):
            img = Image.new('RGB', (50, 50), color='red')
            img.save(temp_dirs['input'] / f"icon{i}.png")
        
        generator = DataGenerator(
            input_dir=temp_dirs['input'],
            output_dir=temp_dirs['output'],
            fonts_dir=temp_dirs['fonts'],
            split="train"
        )
        
        icons = generator._discover_icons()
        assert len(icons) == 3
    
    def test_generate_single_sample(self, temp_dirs):
        """Test generating a single training sample."""
        # Create a test icon
        icon_path = temp_dirs['input'] / "test_icon.png"
        icon = Image.new('RGB', (50, 50), color='blue')
        icon.save(icon_path)
        
        generator = DataGenerator(
            input_dir=temp_dirs['input'],
            output_dir=temp_dirs['root'],
            fonts_dir=temp_dirs['fonts'],
            split="train",
            charset="A",
            seed=42
        )
        
        # Generate dataset
        generator.generate_dataset()
        
        # Check output - files should be in train/A/ subdirectory
        assert generator.total_generated > 0
        train_dir = temp_dirs['root'] / "train"
        assert train_dir.exists()
        
        char_dir = train_dir / "A"
        assert char_dir.exists()
        
        # Verify output image exists and has correct size
        output_files = list(char_dir.glob("*.png"))
        assert len(output_files) > 0
        
        sample_img = Image.open(output_files[0])
        assert sample_img.size == (24, 24)
    
    def test_directory_structure(self, temp_dirs):
        """Test directory structure format."""
        # Create a test icon
        icon_path = temp_dirs['input'] / "test.png"
        icon = Image.new('RGB', (50, 50), color='green')
        icon.save(icon_path)
        
        generator = DataGenerator(
            input_dir=temp_dirs['input'],
            output_dir=temp_dirs['root'],
            fonts_dir=temp_dirs['fonts'],
            split="val",
            charset="XY",
            seed=123
        )
        
        generator.generate_dataset()
        
        # Check directory structure
        val_dir = temp_dirs['root'] / "val"
        assert val_dir.exists()
        
        # Check character subdirectories exist
        x_dir = val_dir / "X"
        y_dir = val_dir / "Y"
        assert x_dir.exists()
        assert y_dir.exists()
        
        # Check that files are in correct directories
        x_files = list(x_dir.glob("*.png"))
        y_files = list(y_dir.glob("*.png"))
        assert len(x_files) > 0
        assert len(y_files) > 0
    
class TestDegradationPipeline:
    """Test image degradation functions."""

    def _make_img(self, size=(24, 24)):
        """Helper: create a simple test RGB image."""
        return Image.new('RGB', size, color=(128, 128, 128))

    def test_degrade_resample(self):
        img = self._make_img()
        result = degrade_resample(img, 0.5)
        assert result.size == img.size
        assert result.mode == img.mode

    def test_degrade_gaussian_noise(self):
        img = self._make_img()
        result = degrade_gaussian_noise(img, 5.0)
        assert result.size == img.size

    def test_degrade_salt_pepper(self):
        img = self._make_img()
        result = degrade_salt_pepper(img, 0.05)
        assert result.size == img.size

    def test_degrade_gamma(self):
        img = self._make_img()
        result = degrade_gamma(img, 1.5)
        assert result.size == img.size

    def test_degrade_hsv_drift(self):
        img = self._make_img()
        result = degrade_hsv_drift(img, 5.0, -10.0, 10.0)
        assert result.size == img.size
        assert result.mode == 'RGB'

    def test_degrade_jpeg_artifacts(self):
        img = self._make_img()
        result = degrade_jpeg_artifacts(img, 30)
        assert result.size == img.size

    def test_pipeline_none(self):
        img = self._make_img()
        result = apply_degradation_pipeline(img, "none")
        # With "none" the image should be identical
        import numpy as np
        assert np.array_equal(np.array(img), np.array(result))

    def test_pipeline_levels(self):
        """Ensure all levels run without errors."""
        img = self._make_img()
        for level in ("light", "medium", "heavy"):
            result = apply_degradation_pipeline(img, level)
            assert result.size == img.size
            assert result.mode == 'RGB'

    def test_resize_uses_nearest(self):
        """Verify resize_image uses NEAREST interpolation (no anti-aliasing)."""
        # Create a 2x2 checkerboard
        img = Image.new('RGB', (2, 2))
        img.putpixel((0, 0), (255, 255, 255))
        img.putpixel((1, 0), (0, 0, 0))
        img.putpixel((0, 1), (0, 0, 0))
        img.putpixel((1, 1), (255, 255, 255))

        resized = resize_image(img, 4, 4)

        # With NEAREST, each pixel should map to exactly one source pixel
        # => only pure black or pure white, no grey interpolation
        import numpy as np
        arr = np.array(resized)
        unique = set(map(tuple, arr.reshape(-1, 3).tolist()))
        assert unique == {(0, 0, 0), (255, 255, 255)}, \
            f"Expected only black and white, got {unique}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
