# data-import Specification

## Purpose
TBD - created by archiving change import-real-data. Update Purpose after archive.
## Requirements
### Requirement: Filename Parsing
The system MUST parse image filenames to extract character labels following the naming convention.
- Accept filenames in format: `<prefix>_<label>_<timestamp>.png`
- First segment must be "valid" or "Valid" (case-insensitive)
- Second segment is the character label
- Valid labels: single characters 0-9, A-Z, or the special label "NA"
- Invalid or unrecognized labels should be logged and skipped
- Support case-insensitive matching for the "valid" prefix and "NA" label

#### Scenario: Standard Label Extraction
Given a file named `valid_5_1746581875.7011735.png`,
When the importer parses the filename,
Then it should extract label "5" and mark the file as valid for import.

#### Scenario: NA Label Extraction
Given a file named `valid_NA_1728653158.7906616.png`,
When the importer parses the filename,
Then it should extract label "NA" and mark the file as valid for import.

#### Scenario: Invalid Filename Format
Given a file named `invalid_A_12345.png`,
When the importer parses the filename,
Then it should log a warning about invalid prefix and skip the file.

#### Scenario: Other Multi-Character Label Handling
Given a file named `valid_BI_1728653158.7906616.png`,
When the importer parses the filename,
Then it should log a warning about unsupported multi-character label and skip the file.

### Requirement: Random Distribution
The system MUST randomly distribute files into train, validation, and test sets with configurable ratios.
- Default split ratios: 75% train, 15% validation, 10% test
- Apply split independently for each label to maintain balanced representation
- Use random selection without replacement
- Support optional random seed for reproducibility
- Handle edge cases where label has fewer than 10 files

#### Scenario: Balanced Split for Common Labels
Given 100 files with label "F",
When the importer applies the default 75/15/10 split,
Then approximately 75 files should go to train, 15 to val, and 10 to test directories.

#### Scenario: Small Label Set Handling
Given 5 files with label "Z",
When the importer applies the default split,
Then at least 1 file should go to each of train/val/test if possible, with remaining files distributed proportionally.

#### Scenario: Reproducible Split
Given the same set of files and a fixed random seed of 42,
When the importer runs twice,
Then both runs should produce identical train/val/test splits.

### Requirement: File Copy Operations
The system MUST copy image files to appropriate target directories without data loss.
- Copy files from source `real_data/` to target `data/{train,val,test}/<label>/`
- Create target directories if they don't exist
- Preserve original filename in target location
- Handle filename collisions by appending suffix (e.g., `_1`, `_2`)
- Verify each file copy succeeded before continuing
- Log any copy errors but continue processing remaining files

#### Scenario: Basic File Copy
Given a file `real_data/valid_1_12345.png` assigned to train set,
When the importer copies the file,
Then it should create `data/train/1/valid_1_12345.png` with identical content.

#### Scenario: Collision Handling
Given `data/train/5/valid_5_12345.png` already exists,
When importing another `valid_5_12345.png` to train/5/,
Then the new file should be saved as `valid_5_12345_1.png`.

#### Scenario: Missing Target Directory
Given target directory `data/val/Q/` does not exist,
When importing a file with label "Q" to validation set,
Then the importer should create the directory before copying the file.

### Requirement: Import Statistics and Logging
The system MUST provide comprehensive logging and statistics about the import process.
- Log total files discovered in source directory
- Log count of valid vs invalid/skipped files
- Report distribution counts per label (train/val/test)
- Log warnings for skipped files with reasons
- Log errors for failed copy operations
- Display summary statistics at completion
- Support both console output and optional log file

#### Scenario: Summary Statistics
Given an import run processing 500 files with labels 0-9 and A-Z,
When the import completes,
Then the console should display total processed, successful, skipped counts and per-label distribution breakdown.

#### Scenario: Warning Logs
Given a file with unsupported label "NA",
When the importer encounters this file,
Then it should log: "WARNING: Skipping file valid_NA_xxx.png - unsupported label 'NA'".

### Requirement: Idempotency and Safety
The system MUST support safe repeated execution without data duplication.
- Skip already imported files if they exist in target with same content
- Provide dry-run mode to preview actions without copying
- Validate source directory exists and is readable before processing
- Validate target directory is writable
- Report actions that would be taken in dry-run mode

#### Scenario: Dry-Run Preview
Given 100 files ready for import,
When running with `--dry-run` flag,
Then the system should log all planned copy operations but not modify any files.

#### Scenario: Duplicate Detection
Given `data/train/A/valid_A_12345.png` already exists with same content,
When importing the same source file again,
Then the system should detect the duplicate and skip the copy operation.

