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
    draw_text_outline,
    load_system_fonts,
    load_custom_fonts,
    create_font,
    get_fallback_font,
    DataGenerator,
    DEFAULT_CHARSET
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
    
    def test_draw_text_outline(self):
        """Test text outline drawing."""
        img = Image.new('RGB', (50, 50), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        font = get_fallback_font(12)
        
        # Should not raise exception
        draw_text_outline(draw, "X", (10, 10), font)
        
        # Visual verification would require inspection
        # At least verify it doesn't crash
        assert True


class TestFontManagement:
    """Test font loading and management."""
    
    def test_load_system_fonts(self):
        """Test system fonts loading."""
        fonts = load_system_fonts()
        
        assert isinstance(fonts, list)
        assert len(fonts) > 0
        assert "Arial" in fonts
    
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
    
    def test_reproducibility(self, temp_dirs):
        """Test that same seed produces same results."""
        # Create a test icon
        icon_path = temp_dirs['input'] / "test.png"
        icon = Image.new('RGB', (50, 50), color='red')
        icon.save(icon_path)
        
        # Generate with seed 42
        output_dir1 = temp_dirs['root'] / "output1"
        generator1 = DataGenerator(
            input_dir=temp_dirs['input'],
            output_dir=output_dir1,
            fonts_dir=temp_dirs['fonts'],
            split="test",
            charset="Y",
            seed=42
        )
        generator1.generate_dataset()
        
        # Generate again with same seed
        output_dir2 = temp_dirs['root'] / "output2"
        generator2 = DataGenerator(
            input_dir=temp_dirs['input'],
            output_dir=output_dir2,
            fonts_dir=temp_dirs['fonts'],
            split="test",
            charset="Y",
            seed=42
        )
        generator2.generate_dataset()
        
        # Compare outputs
        dir1 = output_dir1 / "test" / "Y"
        dir2 = output_dir2 / "test" / "Y"
        
        files1 = sorted(dir1.glob("*.png"))
        files2 = sorted(dir2.glob("*.png"))
        
        assert len(files1) == len(files2)
        assert len(files1) > 0
        
        # Filenames should match (deterministic generation)
        names1 = [f.name for f in files1]
        names2 = [f.name for f in files2]
        assert names1 == names2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
