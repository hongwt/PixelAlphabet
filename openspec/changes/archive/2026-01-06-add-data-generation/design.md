# Design: Training Data Generation System

**Change ID**: `add-data-generation`  
**Created**: 2026-01-06

## Architecture Overview

The data generation system follows a batch processing pipeline:

```
Input: Skill Icon Images (.png) + Fonts
         ↓
    Icon Processor (resize to 50x50)
         ↓
    Patch Extractor (random 24x24 crops)
         ↓
    Text Renderer (overlay characters with fonts)
         ↓
    Style Processor (white text + black outline)
         ↓
    Output Formatter (save 24x24 PNG)
         ↓
Output: Training Images + gt.txt
```

## Key Components

### 1. DataGenerator Class
Main orchestrator that coordinates the generation process.

**Responsibilities**:
- Load icon images from directory
- Iterate through charset and fonts
- Coordinate image processing pipeline
- Manage output file creation
- Append to ground truth file

**Key Methods**:
- `generate_dataset()`: Main entry point
- `process_icon()`: Handle single icon with all character variations
- `create_training_sample()`: Generate one image sample

### 2. ImageProcessor Module
Handles image manipulation operations.

**Functions**:
- `resize_image(img, width, height)`: Scale images maintaining aspect ratio
- `extract_patch(img, x, y, size)`: Crop random region
- `draw_text_with_outline(img, text, font, position)`: Render styled text

### 3. FontManager Module
Manages font loading and selection.

**Functions**:
- `load_system_fonts()`: Get available system fonts
- `load_custom_fonts(font_dir)`: Load TTF files from Fonts/
- `create_font(font_spec, size)`: Instantiate PIL.ImageFont

### 4. Ground Truth Writer
Handles output file management.

**Functions**:
- `append_ground_truth(filepath, image_path, label)`: Thread-safe append
- `validate_ground_truth(filepath)`: Check format consistency

## Data Flow

### Per-Icon Processing
```python
for icon_file in icon_files:
    original = load_image(icon_file)
    resized = resize_image(original, 50, 50)
    
    for char in CHARSET:
        # Random font selection
        font_idx = random.randint(0, len(fonts)-1)
        font_size = random.randint(12, 14)
        font = create_font(fonts[font_idx], font_size)
        
        # Random position
        offset_x = random.randint(0, 26)
        offset_y = random.randint(0, 26)
        patch = extract_patch(resized, offset_x, offset_y, 24)
        
        # Text rendering
        text_size = measure_text(char, font)
        text_pos = (24 - text_size.width - 3, text_size.height + 3)
        
        # Draw outline (9 positions)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                draw_text(patch, char, font, 
                         (text_pos[0]+dx, text_pos[1]+dy), 
                         color='black')
        
        # Draw main text
        draw_text(patch, char, font, text_pos, color='white')
        
        # Save at native 24x24 resolution
        output_name = f"{icon_name}_FONT{font_idx}_{(offset_x+1)*(offset_y+1)}_{char}.png"
        save_image(patch, output_dir / output_name)
        append_ground_truth(gt_file, output_name, char)
```

## Design Decisions

### D1: PIL vs OpenCV
**Decision**: Use Pillow (PIL) for all operations  
**Rationale**:
- Better font rendering support (TrueType fonts)
- Simpler API for text drawing
- Consistent with Python ecosystem
- OpenCV font rendering is limited (only supports basic fonts)

**Trade-off**: PIL is slightly slower for large batches, but quality is more important than speed for dataset generation.

### D2: Image Size Strategy
**Decision**: Store at native 24x24 resolution without upscaling  
**Rationale**:
- 24x24 is the model input size (matches training resolution)
- No information loss from resize operations
- Smaller file sizes and faster generation
- Direct use by dataset loader without preprocessing

**Alternative Considered**: Upscale to 128x32 for easier visual inspection. Rejected to maintain native resolution and avoid unnecessary processing overhead.

### D3: Output Naming Convention
**Decision**: `{icon_name}_FONT{font_idx}_{position_hash}_{character}.png`  
**Rationale**:
- Preserves source icon identity (useful for debugging)
- Font index enables analysis of font-specific accuracy
- Position hash ensures unique filenames
- Character suffix allows easy filtering

### D4: Ground Truth Format
**Decision**: Tab-separated `path\tlabel` format, one line per image  
**Rationale**:
- Matches Java implementation
- Simple parsing with `split('\t')`
- Widely supported format
- Easy to inspect manually

**Note**: Use relative paths from project root for portability.

### D5: Randomization Strategy
**Decision**: Seed randomization per icon, not globally  
**Rationale**:
- Enables reproducible generation for specific icons
- Parallel processing safe (no shared RNG state)
- Can regenerate subsets without affecting others

**Implementation**: Pass seed derived from icon filename hash.

### D6: Error Handling
**Decision**: Skip problematic images with warnings, don't fail entire batch  
**Rationale**:
- One corrupted icon shouldn't block 1000s of others
- Log errors for later investigation
- Continue processing to maximize output

**Example Errors**:
- Unsupported image formats
- Font loading failures
- Disk space issues

## Configuration

### Parameters (CLI Arguments)
```python
--input-dir: Path to skill icon directory (required)
--output-dir: Path for generated images (default: "traindata/")
--fonts-dir: Path to custom fonts (default: "Fonts/")
--gt-file: Ground truth file path (default: "gt.txt")
--charset: Characters to generate (default: "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
--font-size-range: Min-max font size (default: "12-14")
--batch-size: Icons per processing batch (default: all)
--seed: Random seed for reproducibility (optional)
```

### Font Configuration
System fonts (cross-platform):
- Arial, Times New Roman, Helvetica, Verdana, Georgia, Courier New, Tahoma

Custom fonts (from Fonts/):
- ARHei.ttf, ARIALN.TTF, ARIALNB.TTF, ARKai_C.TTF, ARKai_T.TTF
- bHEI00M.TTF, bHEI01B.TTF, bKAI00M.TTF, bLEI00D.TTF

## Testing Strategy

### Unit Tests
- `test_resize_image()`: Verify correct scaling
- `test_extract_patch()`: Check boundary conditions
- `test_text_rendering()`: Validate outline effect
- `test_font_loading()`: Handle missing fonts gracefully

### Integration Tests
- `test_generate_single_icon()`: End-to-end for one icon
- `test_ground_truth_format()`: Validate output file structure
- `test_reproducibility()`: Same seed → same output

### Manual Verification
- Visual inspection of 10-20 samples
- Check character readability
- Verify outline visibility on various backgrounds

## Performance Considerations

### Expected Throughput
- Single icon with 35 chars × 18 fonts = 630 images
- ~0.1s per image → ~63s per icon
- 100 icons → ~1.75 hours

### Optimization Opportunities
1. Parallel processing (multiprocessing per icon)
2. Batch font loading (load once, reuse)
3. In-memory ground truth buffering (flush periodically)

**Decision**: Start with single-threaded; optimize if needed based on dataset size.

## Security & Safety

- **Input Validation**: Check file extensions, max file sizes
- **Path Traversal**: Sanitize output filenames (no ../
)
- **Disk Space**: Estimate required space, warn if insufficient
- **Font Injection**: Only load fonts from trusted directories

## Future Extensions

Potential enhancements (out of scope for this change):
1. **Background augmentation**: Add noise, blur, color shifts
2. **Multi-character sequences**: Generate strings, not just single chars
3. **Quality metrics**: Auto-detect low-quality samples
4. **Web UI**: Visual dataset browser/editor
5. **Format converters**: Export to COCO, YOLO formats
