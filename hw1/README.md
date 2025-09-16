# Interactive Image Mosaic Generator

A sophisticated Python application that transforms images into artistic mosaics using **adaptive grid segmentation** and **advanced multi-metric tile matching**. Built with Gradio for an intuitive web interface.

## 🎨 Features

### Core Functionality
- **Adaptive Grid Segmentation**: Intelligently subdivides image regions based on complexity (variance threshold)
- **Advanced Content-Based Tile Matching**: Multi-metric scoring system combining template matching, color similarity, histogram correlation, and texture analysis
- **Multiple Tile Types**: Four distinct rendering approaches for diverse artistic effects
- **Color Quantization**: K-means clustering for stylized color reduction
- **Quality Metrics**: MSE and SSIM scoring for mosaic fidelity assessment
- **Interactive Web Interface**: Real-time parameter adjustment with Gradio

### Tile Rendering Options

1. **Solid Colors** (`solid`)
   - Simple colored squares using cell average colors
   - Clean, minimalist aesthetic

2. **Geometric Patterns** (`pattern`)
   - Horizontal, vertical, and diagonal gradients
   - Circle and cross patterns
   - HSV-based color tinting for natural blending

3. **Mini-Images** (`mini_image`)
   - Downsampled versions of original cell content
   - Pixelated effect preserving local image details

4. **Image Tiles** (`image_tiles`)
   - Uses external image collection as mosaic tiles
   - **Multi-metric matching algorithm** for optimal tile selection:
        | Metric | Weight | Purpose | Implementation |
        |--------|--------|---------|----------------|
        | **Template Matching** | 40% | Structural similarity | Normalized correlation coefficient |
        | **Color Similarity** | 30% | Average color matching | Euclidean distance in RGB space |
        | **Histogram Correlation** | 20% | Color distribution | 3D histogram comparison |
        | **Texture Analysis** | 10% | Surface pattern matching | Standard deviation comparison |
   - Combined Score = 0.4×Template + 0.3×Color + 0.2×Histogram + 0.1×Texture
   - Load from `tile_images/`

## 🛠 Technical Implementation

### Algorithm Overview
```
Input Image → Color Quantization (optional) → Adaptive Segmentation → Advanced Tile Matching → Mosaic Assembly
```

### Key Components

- **Variance-Based Segmentation**: Recursively subdivides regions with high pixel variance
- **Multi-Metric Tile Matching**:
  - Template matching for structural similarity
  - Euclidean distance for color matching
  - Histogram correlation for color distribution
  - Standard deviation comparison for texture analysis
- **HSV Color Space**: Enhanced color tinting and brightness adjustment
- **Multi-Scale Processing**: Automatic image resizing for performance optimization
- **Intelligent Fallback System**: Graceful degradation when optimal matches unavailable

### Dependencies
```python
gradio>=3.0.0
numpy
opencv-python
pillow
scikit-learn
scikit-image
```
Install via:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
hw1/
├── mosaic_generator.py     # Main application file
├── examples/               # Sample input images
│   ├── sphynx_cat_0.png
│   ├── sphynx_cat_1.jpg
│   ├── cat_sakura_cut_female.png
│   └── cat_sakura_cut_male.png
├── tile_images/           # External tile image collection
│   ├── README.md          # Tile usage instructions
│   └── *.jpg/png/jpeg     # 33+ diverse tile images
└── .gradio/              # Auto-generated Gradio cache
```

## 🚀 Usage

### Quick Start
```bash
cd hw1
pip install -r requirements.txt
python mosaic_generator.py
```

Access the web interface at `http://localhost:7860`

### Parameter Guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| **Base Grid Size** | 8-64 | Initial tile size (larger = fewer tiles) |
| **Variance Threshold** | 0-100 | Subdivision sensitivity (lower = more detail) |
| **Color Quantization** | 2-32 colors | Color palette reduction for artistic effect |
| **Quality Priority** | Speed/Quality | Processing resolution (400px vs 800px max) |

### Workflow
1. Upload an image
2. Adjust parameters for desired artistic effect
3. Select tile type and pattern (if applicable)
4. Generate mosaic
5. Compare results and iterate

## 🎯 Performance Characteristics

### Complexity Analysis
- **Time**: O(n log n) for segmentation, O(m×k) for advanced tile matching where k = tile collection size
- **Space**: O(m + k) where m = number of segments, k = number of tile images
- **Typical segment counts**: 20-200 depending on threshold and image complexity

### Quality Metrics
- **MSE**: Pixel-level difference (lower = more accurate)
- **SSIM**: Perceptual similarity (higher = more visually similar)
- **Segment Count**: Granularity indicator
- **Multi-Metric Score**: Combined weighted similarity assessment (0-1 scale)

## 🎨 Creative Applications

### Artistic Styles
- **Photo-realistic**: Low variance threshold + mini-image tiles
- **Abstract**: High variance threshold + geometric patterns
- **Collage Effect**: Image tiles with diverse source collection + advanced matching
- **Posterization**: Color quantization + solid tiles
- **Professional Quality**: Image tiles with large, diverse tile collection

### Optimal Settings by Use Case
- **Detail Preservation**: Threshold 10-20, mini-image tiles
- **Stylistic Effect**: Threshold 40-60, pattern tiles
- **Performance**: Speed mode, solid tiles
- **Premium Quality**: Image tiles with 50+ diverse tile images, quality mode
- **Experimentation**: Image tiles with varied collection + different thresholds

## 📊 Example Results

The application processes various image types effectively:
- **Portraits**: Preserves facial features with adaptive segmentation
- **Landscapes**: Balances detail in complex areas with simplification in uniform regions
- **Abstract Art**: Creates interesting interpretations with geometric patterns

## 📈 Future Enhancements

Potential improvements could include:
- GPU acceleration for large images
- Additional pattern types and tile shapes
- Batch processing capabilities
- Export options (high resolution, video sequences)
- Advanced color matching algorithms
