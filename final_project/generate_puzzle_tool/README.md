# Jigsaw Puzzle Generator

Generate jigsaw puzzle datasets from images with customizable piece shapes, grid sizes, and layouts.

## Quick Start

```bash
# Place images in source_images/ folder
python puzzle_generator.py
```

## Options

```bash
--square              # Square pieces instead of jigsaw
--grid-size N         # NxN grid (default: 3)
--rotation DEGREES    # Max rotation angle (default: 0)
--no-noise           # Disable noise
--tab-size 0.1-0.3   # Tab size (default: 0.20)
--max-dimension PX   # Max image size (default: 900)
--output-dir PATH    # Output folder (default: output_dataset)
--source-dir PATH    # Source folder (default: source_images)
```

## Examples

```bash
# 4x4 jigsaw with rotation
python puzzle_generator.py --grid-size 4 --rotation 30

# Square pieces, no noise
python puzzle_generator.py --square --no-noise

# Large puzzle with bigger tabs
python puzzle_generator.py --grid-size 6 --tab-size 0.25 --max-dimension 1200
```

## Output Structure

```
output_dataset/
  └── image_name/
      ├── original.png      # Full puzzle image
      ├── solution.png      # Labeled solution
      ├── scattered.png     # Scattered pieces
      ├── pieces/           # Individual piece images
      └── annotations/      # JSON metadata
```
