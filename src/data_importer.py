"""
Data Importer Module

Import real game screenshot images from real_data/ directory and distribute them
into the training dataset structure with automated label parsing and random distribution.
"""

import argparse
import hashlib
import logging
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)


def parse_filename(filename: str) -> Optional[str]:
    """
    Extract label from filename following the pattern: valid_<label>_<timestamp>.png
    
    Args:
        filename: The filename to parse (e.g., "valid_5_1746581875.7011735.png")
        
    Returns:
        The extracted label if valid (single character 0-9, A-Z, or 'NA'), None otherwise
        
    Examples:
        >>> parse_filename("valid_5_1746581875.7011735.png")
        '5'
        >>> parse_filename("Valid_A_12345.png")
        'A'
        >>> parse_filename("valid_NA_12345.png")
        'NA'
        >>> parse_filename("invalid_5_12345.png")
        None
    """
    # Pattern: <prefix>_<label>_<timestamp>.png
    # Prefix must be "valid" (case-insensitive)
    # Label can be single character (0-9, A-Z) or 'NA'
    pattern = r'^valid_(NA|[0-9A-Z])_[^_]+\.png$'
    
    match = re.match(pattern, filename, re.IGNORECASE)
    if match:
        label = match.group(1).upper()  # Normalize to uppercase
        return label
    
    return None


def scan_real_data(source_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan source directory for PNG files and group them by label.
    
    Args:
        source_dir: Path to the directory containing real data images
        
    Returns:
        Dictionary mapping labels to lists of file paths
        
    Raises:
        FileNotFoundError: If source directory doesn't exist
        PermissionError: If source directory is not readable
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")
    
    label_files: Dict[str, List[Path]] = defaultdict(list)
    skipped_files = []
    
    # Find all PNG files recursively
    for file_path in source_dir.rglob("*.png"):
        if not file_path.is_file():
            continue
            
        filename = file_path.name
        label = parse_filename(filename)
        
        if label:
            label_files[label].append(file_path)
        else:
            skipped_files.append(filename)
            logger.debug(f"Skipping file with invalid format: {filename}")
    
    # Log summary
    total_valid = sum(len(files) for files in label_files.values())
    logger.info(f"Scanned {source_dir}: found {total_valid} valid files, skipped {len(skipped_files)}")
    
    if skipped_files and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Skipped files: {', '.join(skipped_files[:10])}" + 
                    (f" (and {len(skipped_files) - 10} more)" if len(skipped_files) > 10 else ""))
    
    return dict(label_files)


def random_split(
    files: List[Path],
    ratios: Tuple[float, float, float],
    seed: Optional[int] = None
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Randomly split files into train, validation, and test sets.
    
    Args:
        files: List of file paths to split
        ratios: Tuple of (train_ratio, val_ratio, test_ratio), should sum to 1.0
        seed: Optional random seed for reproducibility
        
    Returns:
        Tuple of (train_files, val_files, test_files)
        
    Examples:
        >>> files = [Path(f"file{i}.png") for i in range(100)]
        >>> train, val, test = random_split(files, (0.75, 0.15, 0.10), seed=42)
        >>> len(train), len(val), len(test)
        (75, 15, 10)
    """
    if not files:
        return [], [], []
    
    train_ratio, val_ratio, test_ratio = ratios
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Create a copy and shuffle
    files_copy = files.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(files_copy)
    
    # Calculate split points
    n = len(files_copy)
    train_end = max(1, int(n * train_ratio))  # Ensure at least 1 if possible
    val_end = train_end + max(1, int(n * val_ratio)) if n > 1 else train_end
    
    # Handle edge case: ensure test gets at least 1 file if n >= 3
    if n >= 3 and val_end >= n:
        val_end = n - 1
    
    train_files = files_copy[:train_end]
    val_files = files_copy[train_end:val_end]
    test_files = files_copy[val_end:]
    
    return train_files, val_files, test_files


def get_file_hash(file_path: Path) -> str:
    """
    Calculate MD5 hash of a file for duplicate detection.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hex string of MD5 hash
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def copy_files(
    files: List[Path],
    target_dir: Path,
    label: str,
    dry_run: bool = False,
    check_duplicates: bool = True
) -> Tuple[int, int]:
    """
    Copy files to target directory with collision handling.
    
    Args:
        files: List of file paths to copy
        target_dir: Base target directory (will create label subdirectory)
        label: Label for the subdirectory
        dry_run: If True, only log actions without copying
        check_duplicates: If True, skip files that already exist with same content
        
    Returns:
        Tuple of (successful_copies, skipped_files)
    """
    # Create target directory
    label_dir = target_dir / label
    
    if not dry_run:
        label_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"[DRY-RUN] Would create directory: {label_dir}")
    
    successful = 0
    skipped = 0
    
    for file_path in files:
        target_path = label_dir / file_path.name
        
        # Check for duplicates
        if check_duplicates and target_path.exists():
            if get_file_hash(file_path) == get_file_hash(target_path):
                logger.debug(f"Skipping duplicate file: {file_path.name}")
                skipped += 1
                continue
        
        # Handle filename collision by adding suffix
        if target_path.exists():
            base_name = target_path.stem
            suffix = target_path.suffix
            counter = 1
            while target_path.exists():
                target_path = label_dir / f"{base_name}_{counter}{suffix}"
                counter += 1
            logger.debug(f"Resolved filename collision: {file_path.name} -> {target_path.name}")
        
        # Copy file
        if dry_run:
            logger.info(f"[DRY-RUN] Would copy: {file_path} -> {target_path}")
            successful += 1
        else:
            try:
                shutil.copy2(file_path, target_path)
                logger.debug(f"Copied: {file_path.name} -> {target_path}")
                successful += 1
            except Exception as e:
                logger.error(f"Failed to copy {file_path}: {e}")
    
    return successful, skipped


def run_import(
    source_dir: Path,
    target_dir: Path,
    ratios: Tuple[float, float, float] = (0.75, 0.15, 0.10),
    seed: Optional[int] = None,
    dry_run: bool = False,
    check_duplicates: bool = True
) -> Dict[str, any]:
    """
    Main orchestration function to import real data.
    
    Args:
        source_dir: Path to real_data/ directory
        target_dir: Path to data/ directory
        ratios: Tuple of (train, val, test) ratios
        seed: Optional random seed for reproducibility
        dry_run: If True, preview actions without copying
        check_duplicates: If True, skip duplicate files
        
    Returns:
        Dictionary with statistics about the import process
    """
    logger.info("=" * 60)
    logger.info("Starting Real Data Import")
    logger.info("=" * 60)
    logger.info(f"Source: {source_dir}")
    logger.info(f"Target: {target_dir}")
    logger.info(f"Ratios: train={ratios[0]:.0%}, val={ratios[1]:.0%}, test={ratios[2]:.0%}")
    if seed is not None:
        logger.info(f"Random seed: {seed}")
    if dry_run:
        logger.info("DRY-RUN MODE: No files will be copied")
    logger.info("=" * 60)
    
    # Validate directories
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    if not dry_run and not target_dir.exists():
        logger.warning(f"Target directory does not exist, creating: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan source directory
    logger.info("Scanning source directory...")
    label_files = scan_real_data(source_dir)
    
    if not label_files:
        logger.warning("No valid files found to import!")
        return {"total_files": 0, "labels": {}}
    
    # Statistics
    stats = {
        "total_files": sum(len(files) for files in label_files.values()),
        "total_labels": len(label_files),
        "labels": {},
        "summary": {"train": 0, "val": 0, "test": 0, "skipped": 0}
    }
    
    # Process each label
    logger.info(f"\nProcessing {len(label_files)} labels...")
    
    for label in sorted(label_files.keys()):
        files = label_files[label]
        logger.info(f"\nLabel '{label}': {len(files)} files")
        
        # Split files
        train_files, val_files, test_files = random_split(files, ratios, seed)
        
        # Copy files to each split
        train_ok, train_skip = copy_files(
            train_files, target_dir / "train", label, dry_run, check_duplicates
        )
        val_ok, val_skip = copy_files(
            val_files, target_dir / "val", label, dry_run, check_duplicates
        )
        test_ok, test_skip = copy_files(
            test_files, target_dir / "test", label, dry_run, check_duplicates
        )
        
        # Update statistics
        label_stats = {
            "total": len(files),
            "train": train_ok,
            "val": val_ok,
            "test": test_ok,
            "skipped": train_skip + val_skip + test_skip
        }
        stats["labels"][label] = label_stats
        stats["summary"]["train"] += train_ok
        stats["summary"]["val"] += val_ok
        stats["summary"]["test"] += test_ok
        stats["summary"]["skipped"] += train_skip + val_skip + test_skip
        
        logger.info(f"  → train: {train_ok}, val: {val_ok}, test: {test_ok}, skipped: {train_skip + val_skip + test_skip}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Import Summary")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {stats['total_files']}")
    logger.info(f"Total labels: {stats['total_labels']}")
    logger.info(f"Train: {stats['summary']['train']}")
    logger.info(f"Val: {stats['summary']['val']}")
    logger.info(f"Test: {stats['summary']['test']}")
    logger.info(f"Skipped (duplicates): {stats['summary']['skipped']}")
    logger.info("=" * 60)
    
    return stats


def setup_logging(log_file: Optional[Path] = None, verbose: bool = False):
    """
    Configure logging for the importer.
    
    Args:
        log_file: Optional path to log file
        verbose: If True, set DEBUG level; otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always debug in file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


def main():
    """Command-line interface for data importer."""
    parser = argparse.ArgumentParser(
        description="Import real game screenshot data into training dataset structure"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("real_data"),
        help="Source directory containing real_data images (default: real_data/)"
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data"),
        help="Target base directory for train/val/test splits (default: data/)"
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.75,0.15,0.10",
        help="Train/val/test split ratios as comma-separated (default: 0.75,0.15,0.10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without copying files"
    )
    parser.add_argument(
        "--no-duplicate-check",
        action="store_true",
        help="Disable duplicate file detection (faster but may create duplicates)"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file, args.verbose)
    
    # Parse ratios
    try:
        ratios = tuple(float(x) for x in args.ratios.split(','))
        if len(ratios) != 3:
            raise ValueError("Must provide exactly 3 ratios")
    except Exception as e:
        logger.error(f"Invalid ratios format: {e}")
        return 1
    
    # Run import
    try:
        stats = run_import(
            source_dir=args.source,
            target_dir=args.target,
            ratios=ratios,
            seed=args.seed,
            dry_run=args.dry_run,
            check_duplicates=not args.no_duplicate_check
        )
        
        logger.info("\n✓ Import completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Import failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
