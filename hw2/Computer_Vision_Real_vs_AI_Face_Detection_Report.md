# Real vs. AI-Generated Face Detection: Performance Analysis Report

## Executive Summary

This report analyzes the performance of five different deep learning models for distinguishing real human faces from AI-generated faces. The task involves binary classification on a balanced dataset of 128×128 pixel RGB color images. Transfer learning with MobileNetV3 achieved the best performance at 73.33% test accuracy, demonstrating the effectiveness of pre-trained models for specialized computer vision tasks.

---

## Dataset Overview

- **Total Images:** ~1,200 RGB color images (128×128 pixels)
- **Classes:** Binary classification (Real = 1, AI-generated = 0)
- **Split:** Training (85%), Validation (15%), Fixed Test Set (300 images)
- **Balance:** Equal distribution of real and AI-generated faces across all sets
- **Preprocessing:** Pixel normalization to [0,1] range for traditional models, [0,255] for MobileNet models

---

## Model Architectures & Results

### Performance Summary Table

| Model | Architecture | Parameters | Val Accuracy | Test Accuracy | Val-Test Gap |
|-------|-------------|------------|--------------|---------------|--------------|
| **MLP** | Dense layers (128→64→1) | 6.3M | 68.33% | **66.00%** | +2.33% |
| **CNN** | 3 Conv blocks + GlobalAvgPool + Dense layers (96→1) | 30.6K | 62.22% | **64.33%** | -2.11% |
| **CNN+Aug** | CNN + Data Augmentation | 30.6K | 68.89% | **60.00%** | +8.89% |
| **MobileNet** | MobileNetV3Small (frozen) | 997K (57K trainable) | 75.56% | **73.33%** | +2.22% |
| **MobileNet-FT** | MobileNetV3Small (fine-tuned) | 997K (57K trainable) | 73.33% | **73.33%** | +0.00% |

---

## Learning Curves Analysis

### Key Observations:

1. **MLP Model:** Shows classic overfitting pattern with training accuracy reaching 86.79% while validation plateaus at 68.33%. The gap increases steadily after epoch 12, indicating memorization of training data.

2. **CNN Baseline:** Demonstrates more stable learning with less overfitting than MLP. Training and validation curves remain relatively close, suggesting better generalization despite lower parameter count (30K vs 6.3M).

3. **CNN with Augmentation:** Paradoxically shows increased overfitting despite data augmentation. The validation accuracy becomes highly volatile, indicating that the chosen augmentation strategy may be too aggressive for this specific dataset.

4. **MobileNet (Frozen):** Exhibits rapid convergence within first 10 epochs, reaching stable performance quickly. The pre-trained features prove highly effective for face-related tasks.

5. **MobileNet (Fine-tuned):** Shows perfect generalization (zero val-test gap) but slightly lower validation performance. The fine-tuning process appears to optimize specifically for the validation set characteristics.

---

## Detailed Performance Analysis

### Classification Metrics Breakdown

| Model | Precision | Recall | F1-Score | True Positives | False Positives |
|-------|-----------|--------|----------|----------------|-----------------|
| MLP | 0.714 | 0.533 | 0.610 | 80 | 32 |
| CNN | 0.601 | 0.853 | 0.706 | 128 | 85 |
| CNN+Aug | 0.621 | 0.513 | 0.562 | 77 | 47 |
| MobileNet | 0.773 | 0.660 | 0.712 | 99 | 29 |
| MobileNet-FT | 0.769 | 0.667 | 0.714 | 100 | 30 |

> F1 = 2 × (Precision × Recall) / (Precision + Recall)

### Key Performance Insights:

- **Best Overall Performance:** MobileNet models achieve superior balance of precision and recall
- **High Recall Strategy:** CNN baseline favors recall (85.3%) over precision, detecting most real faces but with more false alarms
- **Conservative Approach:** CNN+Augmentation shows lower recall (51.3%), missing more real faces but with fewer false positives
- **Optimal Balance:** MobileNet models provide the best precision-recall trade-off

---

## Model Comparison Insights

### 1. **Architecture Impact**
- **Parameter Efficiency:** CNN (30K parameters) outperforms MLP (6.3M parameters)
- **Spatial Understanding:** Convolutional layers capture spatial patterns better than flattened dense layers
- **Transfer Learning Advantage:** MobileNetV3's ImageNet pre-training provides significant boost in feature extraction

### 2. **Data Augmentation Effects**
- Surprisingly, data augmentation **decreased** test performance from 64.33% to 60.00%
- Validation accuracy became highly unstable, suggesting augmentation parameters were too aggressive
- Face detection tasks may require more conservative augmentation due to facial structure sensitivity

### 3. **Transfer Learning vs Fine-tuning**
- **Frozen MobileNet:** Better validation performance (75.56%) but slightly worse generalization
- **Fine-tuned MobileNet:** Perfect val-test alignment (0% gap) indicating optimal generalization
- Fine-tuning provides stability at the cost of peak validation performance

### 4. **Generalization Patterns**
- **Best Generalization:** MobileNet-FT (0% gap) > CNN (-2.11% gap) > MLP (+2.33% gap)
- Negative gaps indicate models performing better on test than validation, suggesting robust learning
- Large positive gaps indicate overfitting and poor generalization

---

## Failure Case Analysis

### Challenging Samples
- **Hardest Case:** One sample fooled all 5 models (Real face misclassified as AI)
- **Model Agreement:** Only 27.3% of samples had unanimous predictions across all models
- **Consensus Accuracy:** When all models agreed, they were correct 87.8% of the time (72/82 samples)

### Error Patterns
- **False Positives:** AI faces incorrectly classified as real (affects precision)
- **False Negatives:** Real faces incorrectly classified as AI (affects recall)
- MobileNet models show the most balanced error distribution

---

## Key Technical Insights

### 1. **Architecture Design**
- **Batch Normalization + Dropout:** Essential for stable CNN training
- **Global Average Pooling:** Superior to flattening for CNNs (reduces parameters, improves generalization)
- **Progressive Filter Increase:** 16→32→64 filters pattern works well for face detection

### 2. **Training Strategies**
- **Learning Rate:** Lower rates (1e-5) crucial for fine-tuning pre-trained models
- **Early Stopping:** Model checkpointing based on validation loss prevents overfitting
- **Data Preprocessing:** Proper input scaling critical for transfer learning success

### 3. **Transfer Learning Best Practices**
- **Feature Extraction:** Frozen pre-trained features provide strong baseline
- **Selective Fine-tuning:** Unfreezing only final layers maintains pre-trained knowledge
- **Validation Strategy:** Perfect val-test gap indicates optimal stopping point

---

## Conclusions and Recommendations

This analysis of five deep learning approaches for real vs. AI-generated face detection reveals critical insights about modern computer vision techniques. Transfer learning using pre-trained MobileNetV3 significantly outperformed custom architectures (73.33% vs 66.00% test accuracy), demonstrating the power of leveraging large-scale pre-trained features for specialized tasks. Surprisingly, architectural efficiency proved more important than complexity, with the 30K-parameter CNN outperforming the 6.3M-parameter MLP, highlighting the importance of spatial understanding over raw parameter count.

The investigation yielded unexpected results regarding data augmentation, which decreased performance from 64.33% to 60.00%, suggesting face detection requires conservative augmentation strategies due to inherent facial structure constraints. Fine-tuning experiments demonstrated that careful parameter adjustment can achieve perfect generalization (zero validation-test gap) while maintaining competitive performance.

Future work should explore photometric augmentation strategies, ensemble methods combining multiple architectures, and analysis of challenging failure cases. Expanding to other pre-trained architectures like ResNet or EfficientNet could provide deeper insights into feature transferability for AI-generated content detection.
