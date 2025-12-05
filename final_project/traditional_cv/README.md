# Jigsaw Puzzle Solver

A traditional computer vision approach for solving jigsaw puzzles from pre-separated piece images. This project uses visual edge matching to assemble puzzle pieces and generates step-by-step assembly instructions.

## 🎯 Features

- **Annotation-Based Loading**: Reads pre-separated puzzle pieces with ground truth annotations
- **Piece Type Classification**: Uses corner/edge/interior classification from annotations
- **Visual Edge Matching**: Multi-strategy matching using shape, color, texture, and gradients
- **Greedy Assembly**: Brute-force corner permutation search with greedy edge/interior placement
- **Evaluation Metrics**: Compares solution against ground truth with detailed accuracy reports
- **Visual Output**: Generates labeled images, reconstructed puzzles, and assembly instructions

## 📋 Requirements

- Python 3.8+
- OpenCV >= 4.8.0
- NumPy >= 1.24.0
- scikit-image >= 0.21.0
- SciPy >= 1.11.0
- Matplotlib >= 3.7.0
- Pillow >= 10.0.0

## 💻 Usage

### Solve a Puzzle

Run the solver on a puzzle directory containing pre-separated pieces and annotations:

```bash
python main.py --puzzle-dir data/input_images/puzzle_0001
```

### Command Line Arguments

```bash
python main.py --puzzle-dir <path> --output <output_dir>
```

**Arguments:**
- `--puzzle-dir, -p`: Path to puzzle directory (required) - must contain `pieces/` and `annotations/` subdirectories
- `--output, -o`: Output directory for results (default: `data/output`)

### Example

```bash
# Solve puzzle_0001
python main.py -p data/input_images/puzzle_0001 -o data/output

# Check results
ls data/output/puzzle_0001*
```

## 📁 Project Structure

```
traditional_cv/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/
│   ├── input_images/          # Puzzle directories
│   │   └── puzzle_XXXX/
│   │       ├── pieces/        # Individual piece images (piece_0.png, piece_1.png, ...)
│   │       ├── annotations/   # puzzle_info.json with ground truth
│   │       └── original.png   # Original puzzle image (optional)
│   └── output/                # Generated results
└── src/
    ├── __init__.py
    ├── puzzle_loader.py       # Loads pieces and annotations
    ├── feature_extractor.py   # Piece type definitions and feature containers
    ├── solver.py              # Puzzle assembly with visual matching
    ├── evaluator.py           # Evaluation against ground truth
    └── visualizer.py          # Output visualization generation
```

## 📊 Input Data Format

Each puzzle directory must contain:

### `pieces/` directory
Individual piece images named `piece_0.png`, `piece_1.png`, etc. Pieces should have white backgrounds.

### `annotations/puzzle_info.json`
```json
{
  "puzzle_name": "puzzle_0001",
  "grid_size": 3,
  "pieces": [
    {
      "piece_id": 0,
      "row": 0,
      "col": 0,
      "type": "corner",
      "bbox": [0, 0, 300, 300]
    },
    ...
  ]
}
```

**Piece types:**
- `corner`: 2 flat edges (4 pieces in corners)
- `edge`: 1 flat edge (border pieces)
- `interior`: 0 flat edges (center pieces)

## 🔬 Algorithm Overview

### 1. Data Loading
- Loads piece images from `pieces/` directory
- Parses ground truth annotations from `puzzle_info.json`
- Extracts piece masks from white backgrounds

### 2. Piece Classification
Uses ground truth piece types from annotations:
- **Corners**: Placed at grid corners (0,0), (0,n-1), (n-1,0), (n-1,n-1)
- **Edges**: Placed along grid borders
- **Interior**: Placed in center positions

### 3. Visual Edge Matching
Combines multiple similarity metrics for edge comparison:
- **Shape Similarity** (10%): Hausdorff distance on normalized contours
- **Contour Boundary Colors** (45%): Color comparison along actual piece edges
- **Edge Region Colors** (30%): Histogram comparison of regions near edges
- **Gradient Patterns** (15%): Gradient correlation along contour boundaries

### 4. Puzzle Assembly
Brute-force corner search with greedy filling:
1. Try all permutations of corner pieces at corner positions
2. For each corner arrangement, greedily place edge pieces by visual match score
3. Place interior pieces at remaining positions
4. Select the arrangement with the highest total visual match score

### 5. Evaluation
Compares solution against ground truth annotations:
- **Direct Accuracy**: Percentage of pieces in correct positions
- **Neighbor Accuracy**: Percentage of correct neighbor relationships
- **Per-Type Accuracy**: Separate metrics for corners, edges, and interior pieces

## 📊 Output Files

The solver generates several output files:

1. **`*_1_input_labeled.jpg`**: Pieces displayed in a grid with numbered labels
2. **`*_2_reconstructed.jpg`**: Assembled puzzle based on solver's solution
3. **`*_3_step_XX.jpg`**: Step-by-step assembly visualization (first 10 steps)
4. **`*_4_instructions.txt`**: Text file with human-readable assembly instructions

### Sample Output (Instructions)
```
PUZZLE ASSEMBLY INSTRUCTIONS
==================================================

Step 1:
  - Take piece #0
  - Place at position: Row 0, Column 0
  - Reason: Corner piece

Step 2:
  - Take piece #2
  - Place at position: Row 0, Column 2
  - Reason: Corner piece
...
```

## 📈 Dataset

The project includes 45 pre-processed puzzles (`puzzle_0001` to `puzzle_0045`) with:
- Pre-separated piece images (3x3 grids = 9 pieces each)
- Ground truth annotations in JSON format
- Original puzzle images

## 🛠️ Technical Details

**Supported Configurations:**
- ✅ Square puzzles (currently optimized for 3x3)
- ✅ Pieces with white backgrounds
- ✅ Pre-separated piece images
- ✅ Ground truth annotations

**Current Limitations:**
- ⚠️ Fixed 3x3 grid size (hardcoded corner/edge/interior positions)
- ⚠️ Requires ground truth piece type annotations
- ⚠️ No automatic piece segmentation from full images
- ⚠️ No rotation handling

## 🎓 Course Project

This project was developed as a graduate-level computer vision final project for CS5330. Key learning areas:

- **Image Processing**: Mask extraction, contour analysis, morphological operations
- **Feature Engineering**: Multi-modal feature comparison (shape, color, texture, gradient)
- **Algorithm Design**: Brute-force search with greedy optimization
- **Evaluation**: Metrics design and ground truth comparison

### Potential Extensions

- Support for larger grid sizes (NxN)
- Automatic piece type detection from edge analysis
- Rotation handling for arbitrarily oriented pieces
- Piece segmentation from scattered piece photographs

## 📚 Module Reference

### `puzzle_loader.py`
- `PuzzleDataLoader`: Loads pieces and annotations from puzzle directories
- `PuzzlePiece`: Data class representing a single piece with image, mask, and annotations
- `PuzzleInfo`: Metadata container with ground truth positions

### `feature_extractor.py`
- `PieceType`: Enum for CORNER, EDGE, INTERIOR classification
- `PieceFeatures`: Container linking piece IDs to their types
- `create_features_from_annotations()`: Creates features from ground truth

### `solver.py`
- `PuzzleGrid`: Grid data structure for piece placement
- `AssemblyStep`: Records each placement decision
- `PuzzleSolver`: Main solver with visual matching algorithms

### `evaluator.py`
- `PuzzleEvaluator`: Compares solutions against ground truth
- `EvaluationMetrics`: Container for accuracy metrics

### `visualizer.py`
- `PuzzleVisualizer`: Generates all output visualizations

## 👤 Author

CS5330 Final Project - Graduate Computer Vision Course

---

**Happy Puzzle Solving! 🧩**
