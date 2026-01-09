# Tasks: Import Real Data from Game Screenshots

## Implementation Order

### Phase 1: Core Filename Parsing
- [x] Create `src/data_importer.py` module
- [x] Implement `parse_filename()` function with regex pattern matching
  - Extract label from `valid_<label>_<timestamp>.png` format
  - Handle case-insensitive "valid" prefix
  - Return None for invalid format
- [x] Write unit tests for filename parsing
  - Test valid single-character labels (0-9, A-Z)
  - Test invalid prefixes
  - Test multi-character labels
  - Test malformed filenames
- [x] Add validation for supported label characters

### Phase 2: File Discovery and Grouping
- [x] Implement `scan_real_data()` function
  - Recursively find all .png files in source directory
  - Parse each filename and group by label
  - Return dictionary mapping labels to file paths
- [x] Add error handling for unreadable directories
- [x] Write unit tests with mock file system

### Phase 3: Random Split Logic
- [x] Implement `random_split()` function
  - Accept list of files and split ratios (train, val, test)
  - Use `random.shuffle()` with optional seed parameter
  - Calculate split indices based on ratios
  - Return three separate lists
- [x] Handle edge cases:
  - Empty file list
  - List with < 3 items (ensure at least 1 in each split if possible)
  - Exact ratio calculation with integer rounding
- [x] Write unit tests verifying ratio accuracy
- [x] Test reproducibility with fixed seed

### Phase 4: File Copy Operations
- [x] Implement `copy_files()` function
  - Create target directory if missing (using `Path.mkdir(parents=True, exist_ok=True)`)
  - Copy file using `shutil.copy2()` to preserve metadata
  - Handle filename collisions with suffix
  - Verify copy success with file existence check
- [x] Add collision detection logic
- [x] Implement dry-run mode (log only, no actual copy)
- [x] Write integration tests with temporary directories

### Phase 5: Logging and Statistics
- [x] Set up logging configuration
  - Console handler for INFO and above
  - Optional file handler
  - Formatted with timestamp and level
- [x] Implement statistics tracking
  - Count total files scanned
  - Count valid vs skipped files
  - Track distribution per label and split
- [x] Create summary report function
  - Display total counts
  - Show per-label breakdown table
  - Report any errors or warnings
- [x] Test logging output format

### Phase 6: Main Orchestration
- [x] Implement `run_import()` main function
  - Validate source and target directories
  - Call scan_real_data()
  - For each label group:
    - Apply random_split()
    - Copy files to train/val/test
  - Generate and display summary
- [x] Add command-line argument parsing
  - `--source`: Source directory path (default: `real_data/`)
  - `--target`: Target base directory (default: `data/`)
  - `--ratios`: Train/val/test split ratios (default: `0.75,0.15,0.10`)
  - `--seed`: Random seed for reproducibility (optional)
  - `--dry-run`: Preview mode without copying
  - `--log-file`: Optional log file path
- [x] Create CLI entry point script

### Phase 7: Testing and Validation
- [x] Run dry-run on actual `real_data/` directory
- [x] Review logged warnings for special labels (NA, BI, etc.)
- [x] Verify split ratios are approximately correct
- [x] Test on subset of data first
- [x] Run full import and validate results
- [x] Check random sampling of copied files for correctness
- [x] Verify directory structure and file counts

### Phase 8: Documentation
- [ ] Add docstrings to all functions
- [ ] Create README section for data import usage
- [ ] Document special label handling decisions
- [ ] Add troubleshooting guide for common issues

## Dependencies
- Python standard library: `pathlib`, `shutil`, `random`, `re`, `argparse`, `logging`
- No new external dependencies required

## Validation Criteria
- All unit tests pass
- Integration test with sample data succeeds
- Dry-run output looks correct on real data
- Actual import completes without errors
- Split ratios within ±2% of target (75/15/10)
- No data loss (all valid source files copied)
- Duplicate handling works correctly
- Logging provides clear status information
