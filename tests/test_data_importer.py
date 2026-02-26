"""
Unit tests for data_importer module
"""

import tempfile
from pathlib import Path
import pytest

from src.data_importer import (
    parse_filename,
    scan_real_data,
    random_split,
    get_file_hash,
    copy_files,
)


class TestParseFilename:
    """Test filename parsing functionality."""
    
    def test_valid_single_digit(self):
        """Test parsing valid single digit labels."""
        assert parse_filename("valid_5_1746581875.7011735.png") == "5"
        assert parse_filename("valid_0_12345.png") == "0"
        assert parse_filename("valid_9_99999.png") == "9"
    
    def test_valid_single_letter(self):
        """Test parsing valid single letter labels."""
        assert parse_filename("valid_A_12345.png") == "A"
        assert parse_filename("valid_Z_99999.png") == "Z"
        assert parse_filename("valid_F_1728310933.5560274.png") == "F"
    
    def test_case_insensitive_prefix(self):
        """Test that 'valid' prefix is case-insensitive."""
        assert parse_filename("Valid_A_12345.png") == "A"
        assert parse_filename("VALID_5_12345.png") == "5"
        assert parse_filename("VaLiD_Z_12345.png") == "Z"
    
    def test_label_normalization(self):
        """Test that labels are normalized to uppercase."""
        assert parse_filename("valid_a_12345.png") == "A"
        assert parse_filename("valid_z_12345.png") == "Z"
    
    def test_invalid_prefix(self):
        """Test rejection of invalid prefixes."""
        assert parse_filename("invalid_5_12345.png") is None
        assert parse_filename("test_A_12345.png") is None
        assert parse_filename("_5_12345.png") is None
    
    def test_multi_character_label(self):
        """Test that NA label is accepted, but other multi-character labels are rejected."""
        assert parse_filename("valid_NA_12345.png") == "NA"
        assert parse_filename("valid_BI_12345.png") is None
        assert parse_filename("valid_ABC_12345.png") is None
    
    def test_na_label_variations(self):
        """Test NA label with different cases."""
        assert parse_filename("valid_NA_12345.png") == "NA"
        assert parse_filename("valid_na_12345.png") == "NA"
        assert parse_filename("Valid_NA_12345.png") == "NA"
    
    def test_malformed_filename(self):
        """Test rejection of malformed filenames."""
        assert parse_filename("valid_5.png") is None
        assert parse_filename("valid_5_12345") is None  # Missing .png
        assert parse_filename("5_12345.png") is None  # Missing prefix
        assert parse_filename("valid__12345.png") is None  # Empty label
    
    def test_special_characters(self):
        """Test rejection of special character labels."""
        assert parse_filename("valid_@_12345.png") is None
        assert parse_filename("valid_#_12345.png") is None
        assert parse_filename("valid_._12345.png") is None


class TestScanRealData:
    """Test directory scanning and grouping functionality."""
    
    def test_scan_valid_files(self, tmp_path):
        """Test scanning directory with valid files."""
        # Create test files
        (tmp_path / "valid_A_123.png").touch()
        (tmp_path / "valid_B_456.png").touch()
        (tmp_path / "valid_A_789.png").touch()
        
        result = scan_real_data(tmp_path)
        
        assert "A" in result
        assert "B" in result
        assert len(result["A"]) == 2
        assert len(result["B"]) == 1
    
    def test_scan_recursive(self, tmp_path):
        """Test recursive directory scanning."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        (tmp_path / "valid_A_123.png").touch()
        (subdir / "valid_A_456.png").touch()
        
        result = scan_real_data(tmp_path)
        
        assert "A" in result
        assert len(result["A"]) == 2
    
    def test_scan_nonexistent_directory(self):
        """Test error handling for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            scan_real_data(Path("/nonexistent/path"))
    
    def test_scan_not_a_directory(self, tmp_path):
        """Test error handling when path is not a directory."""
        file_path = tmp_path / "file.txt"
        file_path.touch()
        
        with pytest.raises(NotADirectoryError):
            scan_real_data(file_path)


class TestRandomSplit:
    """Test random splitting functionality."""
    
    def test_basic_split(self):
        """Test basic split with default ratios."""
        files = [Path(f"file{i}.png") for i in range(100)]
        train, val, test = random_split(files, (0.75, 0.15, 0.10), seed=42)
        
        assert len(train) == 75
        assert len(val) == 15
        assert len(test) == 10
        
        # Verify all files are included
        all_files = set(train + val + test)
        assert len(all_files) == 100
        assert all_files == set(files)
    
    def test_reproducibility(self):
        """Test that same seed produces same split."""
        files = [Path(f"file{i}.png") for i in range(50)]
        
        train1, val1, test1 = random_split(files, (0.75, 0.15, 0.10), seed=42)
        train2, val2, test2 = random_split(files, (0.75, 0.15, 0.10), seed=42)
        
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2
    
    def test_different_seeds(self):
        """Test that different seeds produce different splits."""
        files = [Path(f"file{i}.png") for i in range(50)]
        
        train1, _, _ = random_split(files, (0.75, 0.15, 0.10), seed=42)
        train2, _, _ = random_split(files, (0.75, 0.15, 0.10), seed=123)
        
        assert train1 != train2
    
    def test_empty_list(self):
        """Test splitting empty list."""
        train, val, test = random_split([], (0.75, 0.15, 0.10))
        
        assert train == []
        assert val == []
        assert test == []
    
    def test_small_list(self):
        """Test splitting small list (< 3 items)."""
        files = [Path(f"file{i}.png") for i in range(2)]
        train, val, test = random_split(files, (0.75, 0.15, 0.10), seed=42)
        
        # Should split as best as possible
        assert len(train) + len(val) + len(test) == 2
        assert len(set(train + val + test)) == 2
    
    def test_three_items(self):
        """Test splitting exactly 3 items."""
        files = [Path(f"file{i}.png") for i in range(3)]
        train, val, test = random_split(files, (0.75, 0.15, 0.10), seed=42)
        
        # Should have at least one in each split
        assert len(train) >= 1
        assert len(val) >= 1 or len(test) >= 1  # At least one of these
        assert len(train) + len(val) + len(test) == 3
    
    def test_invalid_ratios(self):
        """Test error handling for invalid ratios."""
        files = [Path(f"file{i}.png") for i in range(10)]
        
        with pytest.raises(ValueError):
            random_split(files, (0.5, 0.3, 0.1))  # Doesn't sum to 1.0
    
    def test_custom_ratios(self):
        """Test split with custom ratios."""
        files = [Path(f"file{i}.png") for i in range(100)]
        train, val, test = random_split(files, (0.8, 0.1, 0.1), seed=42)
        
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10


class TestGetFileHash:
    """Test file hashing functionality."""
    
    def test_hash_identical_files(self, tmp_path):
        """Test that identical files have same hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        
        content = b"test content"
        file1.write_bytes(content)
        file2.write_bytes(content)
        
        assert get_file_hash(file1) == get_file_hash(file2)
    
    def test_hash_different_files(self, tmp_path):
        """Test that different files have different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        
        file1.write_bytes(b"content1")
        file2.write_bytes(b"content2")
        
        assert get_file_hash(file1) != get_file_hash(file2)


class TestCopyFiles:
    """Test file copying functionality."""
    
    def test_basic_copy(self, tmp_path):
        """Test basic file copy operation."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        target_dir = tmp_path / "target"
        
        # Create source files
        file1 = source_dir / "valid_A_123.png"
        file1.write_text("content")
        
        files = [file1]
        successful, skipped = copy_files(files, target_dir, "A", dry_run=False)
        
        assert successful == 1
        assert skipped == 0
        assert (target_dir / "A" / "valid_A_123.png").exists()
    
    def test_dry_run(self, tmp_path):
        """Test that dry-run doesn't copy files."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        target_dir = tmp_path / "target"
        
        file1 = source_dir / "valid_A_123.png"
        file1.write_text("content")
        
        files = [file1]
        successful, skipped = copy_files(files, target_dir, "A", dry_run=True)
        
        assert successful == 1
        assert not (target_dir / "A").exists()
    
    def test_collision_handling(self, tmp_path):
        """Test filename collision handling."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        target_dir = tmp_path / "target"
        (target_dir / "A").mkdir(parents=True)
        
        # Create source file
        file1 = source_dir / "valid_A_123.png"
        file1.write_text("content1")
        
        # Create existing file with same name but different content
        existing = target_dir / "A" / "valid_A_123.png"
        existing.write_text("existing_content")
        
        files = [file1]
        successful, skipped = copy_files(files, target_dir, "A", dry_run=False, check_duplicates=False)
        
        assert successful == 1
        # Should create file with suffix
        assert (target_dir / "A" / "valid_A_123_1.png").exists()
    
    def test_duplicate_detection(self, tmp_path):
        """Test duplicate file detection."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        target_dir = tmp_path / "target"
        (target_dir / "A").mkdir(parents=True)
        
        # Create source file
        file1 = source_dir / "valid_A_123.png"
        content = "same_content"
        file1.write_text(content)
        
        # Create existing file with same content
        existing = target_dir / "A" / "valid_A_123.png"
        existing.write_text(content)
        
        files = [file1]
        successful, skipped = copy_files(files, target_dir, "A", dry_run=False, check_duplicates=True)
        
        assert successful == 0
        assert skipped == 1
    
    def test_create_target_directory(self, tmp_path):
        """Test that target directory is created if missing."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        target_dir = tmp_path / "target"
        
        file1 = source_dir / "valid_A_123.png"
        file1.write_text("content")
        
        files = [file1]
        copy_files(files, target_dir, "A", dry_run=False)
        
        assert (target_dir / "A").exists()
        assert (target_dir / "A").is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
