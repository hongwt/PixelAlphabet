# Design: Import Real Data from Game Screenshots

## Architecture Overview
A standalone Python script will handle the import process. It will:
1. Scan `real_data/` for PNG files
2. Parse filenames to extract labels
3. Group files by label
4. Apply random split for each label group
5. Copy files to target directories

## Component Design

### Data Importer Module
**Location**: `src/data_importer.py`

**Key Functions**:
- `parse_filename(filename: str) -> Optional[str]`: Extract label from filename
- `scan_real_data(source_dir: Path) -> Dict[str, List[Path]]`: Group files by label
- `random_split(files: List[Path], ratios: Tuple[float, float, float]) -> Tuple[List, List, List]`: Split files into train/val/test
- `copy_files(files: List[Path], target_dir: Path, label: str)`: Copy files to target directory
- `run_import(source_dir: Path, target_dir: Path, ratios: Tuple[float, float, float])`: Main orchestration function

### Label Mapping Strategy
Standard labels (0-9, A-Z) map directly to existing directories:
- `valid_5_xxx.png` → `data/train/5/` (or val/test)
- `valid_F_xxx.png` → `data/train/F/`

Special labels require clarification:
- `NA` - Needs investigation (invalid character? empty background?)
- `BI`, `D-M`, `OD`, `OU`, `PH` - Multi-character labels, may need special handling
- `2.` - Appears to be a typo

**Decision**: Initially skip special/invalid labels and log warnings. Future enhancement can add mapping rules.

## Data Flow
```
real_data/
  valid_1_xxx.png
  valid_A_xxx.png
      ↓
  [Parse & Group by Label]
      ↓
  {
    '1': [file1, file2, ...],
    'A': [file3, file4, ...]
  }
      ↓
  [Random Split per Label]
      ↓
  train (75%), val (15%), test (10%)
      ↓
  [Copy to Target]
      ↓
data/
  train/1/, val/1/, test/1/
  train/A/, val/A/, test/A/
```

## Error Handling
- **Invalid filename format**: Log warning, skip file
- **Unknown label**: Log warning, skip file (future: add to unmapped category)
- **Missing target directory**: Create if needed
- **File copy error**: Log error, continue with next file
- **Duplicate filename in target**: Add suffix to avoid overwrite

## Configuration
- Source directory: `real_data/`
- Target base directory: `data/`
- Split ratios: (0.75, 0.15, 0.10)
- Random seed: Configurable for reproducibility (default: None for true randomness)

## Performance Considerations
- File count: ~3000+ images (estimated from directory listing)
- Operation: File system copy (relatively fast)
- Parallelization: Not needed for initial version
- Memory: Minimal (only file paths in memory)

## Testing Strategy
- Unit tests for filename parsing
- Unit tests for split logic (verify ratios)
- Integration test with sample directory structure
- Dry-run mode for validation before actual copy

## Future Enhancements
- Support for label mapping configuration file
- Validation of image format and content
- De-duplication of identical images
- Incremental import (skip already imported files)
- Support for other filename patterns
