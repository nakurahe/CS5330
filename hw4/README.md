---
title: Spatial_Photo_Effect_Generator
app_file: dpt.py
sdk: gradio
sdk_version: 5.49.1
---

# 🎨 Spatial Photo Effect Generator

![Example Preview](https://github.com/nakurahe/CS5330/blob/main/hw4/spatial_photo_1.gif)

Create stunning parallax animations with depth-of-field effects using AI-powered depth estimation!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-5.8.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📖 Overview

This project implements a complete spatial photo effect pipeline that transforms static images into dynamic, depth-aware animations. Using Intel's Dense Prediction Transformer (DPT) for depth estimation, the system automatically separates foreground and background, applies realistic bokeh effects, and generates smooth parallax animations. Check out the demo [here](https://huggingface.co/spaces/jiahuanhe/Spatial_Photo_Effect_Generator)!

**Key Features:**
- 🤖 AI-powered depth estimation using Intel DPT-Hybrid
- 🎯 Automatic foreground/background segmentation
- 🖼️ Intelligent background inpainting
- 🎬 Smooth parallax animation with customizable motion
- 📷 Realistic depth-of-field bokeh effects
- 🎨 Custom bokeh shapes (circular, hexagonal, heart)
- 🔍 Dynamic zoom effects
- 🌐 User-friendly Gradio web interface

## 🏗️ Architecture

### Pipeline Overview

```
Input Image
    ↓
┌─────────────────────┐
│  Depth Estimation   │  ← Intel DPT-Hybrid Model
│  (Intel DPT)        │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Foreground/        │  ← Otsu Thresholding
│  Background         │    + Morphological Ops
│  Separation         │    + Connected Components
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Background         │  ← Telea + Navier-Stokes
│  Inpainting         │    + Bilateral Filtering
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Parallax Motion    │  ← Affine Transformations
│  Synthesis          │    + Smoothstep Easing
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Depth-of-Field     │  ← Depth-Based Blur
│  Bokeh Effects      │    + Custom Kernels
└─────────────────────┘
    ↓
Animated GIF Output
```

### Technical Components

#### Part 1: Depth Estimation
- **Model**: Intel DPT-Hybrid-MiDaS (~400MB)
- **Method**: Dense Prediction Transformer
- **Output**: Normalized depth map (0-1 range)
- **GPU**: Automatically uses CUDA if available

#### Part 2: Foreground/Background Separation
- **Thresholding**: Otsu's automatic threshold selection
- **Cleaning**: Morphological opening (5×5 ellipse) + closing (11×11 ellipse)
- **Selection**: Connected components analysis (largest component = foreground)
- **Smoothing**: Gaussian blur (21×21) for soft edges

#### Part 3: Background Inpainting
- **Dual Method**: Telea (Fast Marching) + Navier-Stokes algorithms
- **Blending**: 50/50 weighted average
- **Smoothing**: Bilateral filter (d=9, σ_color=75, σ_space=75)
- **Edge Coverage**: 7×7 ellipse dilation (2 iterations)

#### Part 4: Parallax Motion Synthesis
- **Frames**: 15-60 configurable frames
- **Easing**: Smoothstep function (t² × (3 - 2t))
- **Motion**: Affine transformations with different shifts
  - Foreground: ±10 pixels (adjustable)
  - Background: ±3 pixels (adjustable)
- **Styles**: Back-and-forth or continuous loop

#### Part 5: Depth-of-Field Bokeh
- **Aperture Range**: f/1.4 to f/5.6
- **Blur Levels**: 5-tier progressive blur (1, 7, 13, 19, 31 pixels)
- **Method**: Per-pixel depth-weighted blending
- **Shapes**: Circular (Gaussian), Hexagonal, Heart
- **Strength**: 0.5× to 2.0× multiplier

#### Part 6: Web Interface (Gradio)
- **Framework**: Gradio 5.8.0
- **Design**: Responsive, tabbed layout
- **Features**: Real-time preview, parameter sliders, examples
- **Export**: Automatic GIF optimization (<5MB)

## 🎮 Usage Guide

### Basic Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Parallax Strength** | 1.0-3.0 | 1.5 | Controls motion difference between layers |
| **Aperture (f-stop)** | f/1.4-f/5.6 | f/2.8 | Camera aperture (lower = more blur) |
| **Animation Frames** | 15-60 | 30 | Number of frames (more = smoother) |
| **Animation Style** | - | Back & Forth | Motion pattern type |

### Advanced Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Bokeh Shape** | - | Circular | Shape of out-of-focus blur |
| **Bokeh Strength** | 0.5-2.0 | 1.0 | Intensity of blur effect |
| **Dynamic Zoom** | On/Off | Off | Zoom in/out during animation |

### Tips for Best Results

✅ **Use portrait photos** with clear subject in foreground  
✅ **Good lighting** improves depth estimation accuracy  
✅ **Simple backgrounds** produce cleaner results  
✅ **Higher parallax** creates more dramatic effects  
✅ **Lower f-stop** (f/1.4) creates stronger bokeh  

❌ Avoid cluttered scenes with similar depth levels  
❌ Avoid very low resolution images (<400px)  
❌ Avoid images with transparent/missing regions
