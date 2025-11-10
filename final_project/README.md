# Jigsaw Puzzle Solver

A computer vision-based solution for solving real jigsaw puzzles from photographs. This project detects puzzle pieces from an image, analyzes their features, matches compatible pieces, and generates step-by-step assembly instructions.

## 🎯 Features

- **Piece Detection**: Automatically detects and segments individual puzzle pieces from a photograph
- **Feature Extraction**: Analyzes edge shapes, colors, and textures
- **Smart Matching**: Uses multiple algorithms (shape, color, texture) to find compatible pieces
- **Intelligent Assembly**: Greedy algorithm with constraint propagation
- **Visual Output**: Generates annotated images and step-by-step assembly instructions

## 📋 Requirements

- Python 3.8+
- OpenCV
- NumPy
- scikit-image
- SciPy
- Matplotlib

## 🚀 Installation

1. Clone the repository:
```bash
cd /Users/nakura/Documents/CS5330/final_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Quick Demo

Run the built-in demo to see the system in action:

```bash
python main.py --demo
```

This creates a sample puzzle and solves it automatically.

### Solve Your Own Puzzle

1. Take a photo of your puzzle pieces on a white/uniform background
2. Ensure pieces are **not overlapping**
3. Run the solver:

```bash
python main.py --input path/to/your/puzzle.jpg
```

### Advanced Options

```bash
python main.py --input puzzle.jpg --rows 4 --cols 5 --output results/
```

**Arguments:**
- `--input, -i`: Path to input image (required unless using --demo)
- `--output, -o`: Output directory (default: data/output)
- `--rows, -r`: Number of puzzle rows (auto-detected if not provided)
- `--cols, -c`: Number of puzzle columns (auto-detected if not provided)
- `--demo`: Create and solve a demo puzzle

## 📁 Project Structure

```
final_project/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/
│   ├── input_images/      # Place your puzzle photos here
│   └── output/            # Generated results
└── src/
    ├── piece_detector.py      # Piece detection and segmentation
    ├── feature_extractor.py   # Feature extraction from pieces
    ├── piece_matcher.py       # Piece matching algorithms
    ├── solver.py             # Puzzle assembly logic
    └── visualizer.py         # Visualization and output generation
```

## 🔬 Algorithm Overview

### 1. Piece Detection
- Background subtraction using adaptive thresholding
- Contour detection with morphological operations
- Individual piece extraction with masks

### 2. Feature Extraction
- **Edge Classification**: Identifies flat edges (border pieces) vs. interlocking edges (tabs/sockets)
- **Color Profiles**: Extracts color information along each edge
- **Shape Descriptors**: Analyzes edge contours for geometric matching
- **Piece Classification**: Categorizes pieces as corners, edges, or interior

### 3. Piece Matching
Combines multiple similarity metrics:
- **Shape Similarity** (30%): Contour matching using Hausdorff distance
- **Color Similarity** (50%): Color continuity across edges
- **Texture Similarity** (20%): Gradient pattern correlation

### 4. Puzzle Assembly
Greedy algorithm with constraint propagation:
1. Place corner pieces (2 flat edges)
2. Build border (1 flat edge)
3. Fill interior using highest-confidence matches
4. Backtrack if necessary

## 📊 Output Files

The solver generates several output files:

1. **`*_1_input_labeled.jpg`**: Original image with numbered pieces
2. **`*_2_reconstructed.jpg`**: Assembled puzzle
3. **`*_3_step_XX.jpg`**: Step-by-step assembly visualization
4. **`*_4_instructions.txt`**: Text file with assembly instructions

## 🎓 For Students & Researchers

This project was developed as a graduate-level computer vision final project. Key learning areas:

- **Image Processing**: Segmentation, contour analysis, morphological operations
- **Feature Engineering**: Multi-modal feature extraction and normalization
- **Optimization**: Constraint satisfaction, greedy algorithms
- **Computer Vision**: Real-world image analysis with varying conditions

### Customization Ideas

- Implement deep learning for edge matching
- Add rotation handling for arbitrary piece orientations
- Support for pieces with complex interlocking shapes
- Multi-puzzle detection in a single image
- Real-time solving with video input

## 🛠️ Technical Constraints

**Works Best With:**
- ✅ Non-overlapping pieces
- ✅ Uniform background (white/light solid color)
- ✅ Good lighting (minimal shadows)
- ✅ < 50 pieces (optimal performance)
- ✅ Overhead camera angle

**Challenges:**
- ⚠️ Overlapping pieces may not separate correctly
- ⚠️ Complex backgrounds reduce accuracy
- ⚠️ Shadows can affect edge detection
- ⚠️ Very similar regions (sky, solid colors) are harder to match

## 📝 Example Workflow

```bash
# 1. Create demo puzzle
python main.py --demo

# 2. Check output
ls data/output/

# 3. Solve your own puzzle (4x6 grid)
python main.py -i my_puzzle.jpg -r 4 -c 6

# 4. View results
open data/output/my_puzzle_2_reconstructed.jpg
```

## 🐛 Troubleshooting

**No pieces detected:**
- Ensure background is uniform and contrasts with pieces
- Check image quality and lighting
- Adjust `min_area` and `max_area` in `PieceDetector`

**Poor matching results:**
- Use puzzles with distinct colors and patterns
- Avoid puzzles with large uniform regions
- Ensure pieces are well-separated

**Incorrect assembly:**
- Specify correct grid dimensions with `--rows` and `--cols`
- Try puzzles with clearer visual features
- Check that all pieces are visible and not occluded

## 📚 References

This project implements techniques from:
- Image segmentation and contour analysis
- Feature extraction and descriptor matching
- Constraint satisfaction problems
- Greedy optimization with backtracking

## 📄 License

This project is for educational purposes. Feel free to use and modify for your coursework or research.

## 👤 Author

CS5330 Final Project - Graduate Computer Vision Course

## 🙏 Acknowledgments

- OpenCV community for excellent documentation
- Computer Vision course materials and resources
- Research papers on jigsaw puzzle solving algorithms

---

**Happy Puzzle Solving! 🧩**
