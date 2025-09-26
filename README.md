# Deep Learning Assignment 1: Facial Expression Recognition

This repository contains a comprehensive implementation of multi-task learning for facial expression recognition, combining expression classification with valence/arousal regression using deep CNN architectures.

## 🎯 Project Overview

**Task**: Multi-task learning for facial expression analysis
- **Expression Classification**: 7 classes (0-6)
- **Valence Regression**: Continuous values (-1 to 1)
- **Arousal Regression**: Continuous values (-1 to 1)

**Models Evaluated**:
- ResNet-50 (34 epochs)
- EfficientNet-B0 (34 epochs) 
- ResNet-50 v2 with improvements (15 epochs)
- ResNet-50 v2 with reduced overfitting (10 epochs)

## 🏆 Results Summary

| Model | Val Accuracy | Test Accuracy | Val RMSE | Aro RMSE | Val Corr | Aro Corr |
|-------|-------------|---------------|----------|----------|----------|----------|
| ResNet-50 v2 (10e) | **48.17%** | - | 0.390 | 0.332 | 0.584 | 0.499 |
| ResNet-50 v2 (15e) | 47.50% | **47.5%** | 0.390 | 0.332 | 0.584 | 0.499 |
| ResNet-50 (34e) | 45.50% | 45.5% | 0.380 | 0.334 | 0.620 | 0.485 |
| EfficientNet-B0 (34e) | 42.83% | 42.8% | 0.392 | 0.345 | 0.582 | 0.430 |

**Key Achievements**:
- ✅ Best validation accuracy: **48.17%** (ResNet-50 v2, 10 epochs)
- ✅ Successfully reduced overfitting with improved regularization
- ✅ Strong valence regression performance (RMSE: 0.39, Correlation: 0.58)
- ✅ Comprehensive evaluation with confusion matrices, ROC/PR curves, and error analysis

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Structure
```
Dataset/
├── images/          # 3,999 facial expression images (.jpg)
└── annotations/     # 15,996 annotation files (.npy)
    ├── *_exp.npy    # Expression labels (0-6)
    ├── *_val.npy    # Valence values (-1 to 1)
    ├── *_aro.npy    # Arousal values (-1 to 1)
    └── *_lnd.npy    # Facial landmarks
```

### 3. Training Models

#### Basic Training
```bash
# ResNet-50 (34 epochs)
python -m src.train.run_training --backbone resnet --epochs 34 --batch_size 32 --pretrained --out outputs/resnet50_e34

# EfficientNet-B0 (34 epochs)
python -m src.train.run_training --backbone efficientnet --epochs 34 --batch_size 32 --pretrained --out outputs/efficientnet_e34
```

#### Improved Training (V2)
```bash
# ResNet-50 v2 with improvements (15 epochs)
python -m src.train.run_training --backbone resnet --epochs 15 --batch_size 32 --pretrained --v2_augmentations --amp --label_smoothing 0.1 --out outputs/resnet50_v2

# ResNet-50 v2 with reduced overfitting (10 epochs)
python -m src.train.run_training --backbone resnet --epochs 10 --batch_size 32 --pretrained --v2_augmentations --amp --label_smoothing 0.1 --out outputs/resnet50_v2_e10
```

### 4. Evaluation and Analysis
```bash
# Generate comprehensive evaluation
python -m src.analysis.evaluate --model outputs/resnet50_v2_e10/best_model.pth --backbone resnet

# Compare all models
python -m src.analysis.compare_models

# Single image inference
python scripts/infer_image.py --model outputs/resnet50_v2_e10/best_model.pth --backbone resnet --image Dataset/images/0.jpg
```

## 📊 Generated Artifacts

### Training Outputs
- **Metrics**: `outputs/*/metrics.csv` - Training/validation metrics per epoch
- **Checkpoints**: `outputs/*/best_model.pth` - Best model weights
- **Curves**: `outputs/*/training_curves.png` - Loss and accuracy plots

### Analysis Results
- **Confusion Matrices**: `outputs/comparison/confusion_*.png`
- **ROC/PR Curves**: `outputs/comparison/roc_pr_*.png`
- **Error Analysis**: `outputs/comparison/errors_*.txt`
- **Model Comparison**: `outputs/comparison/model_comparison.csv`

### Comprehensive Report
- **PDF Report**: `DL_Assignment1_Report.pdf` - Complete analysis and results

## 🏗️ Architecture Details

### Model Architecture
- **Backbone**: ResNet-50 or EfficientNet-B0 (pretrained on ImageNet)
- **Multi-task Heads**:
  - Expression: 7-class classification head
  - Valence: Single regression output
  - Arousal: Single regression output

### Training Configuration
- **Optimizer**: AdamW (weight_decay=1e-5)
- **Learning Rate**: 1e-4 with cosine annealing
- **Loss Functions**:
  - Expression: Cross-entropy with label smoothing (0.1)
  - Valence/Arousal: MSE loss
- **Data Augmentation**: Random crop, flip, rotation, color jitter
- **V2 Augmentations**: Random erasing, perspective distortion
- **Mixed Precision**: AMP for efficiency

### Evaluation Metrics
- **Expression**: Accuracy, F1-score, Kappa, Alpha
- **Valence/Arousal**: RMSE, MAE, R², Correlation, SAGR, CCC
- **Visualization**: Confusion matrices, ROC/PR curves

## 🔧 Key Improvements (V2)

1. **Stronger Data Augmentation**: Random erasing and perspective distortion
2. **Label Smoothing**: Better generalization (0.1 smoothing factor)
3. **AdamW Optimizer**: Improved weight decay and convergence
4. **Cosine Annealing**: Better learning rate scheduling
5. **Mixed Precision Training**: Faster training with AMP
6. **Reduced Overfitting**: Fewer epochs with better regularization

## 📁 Project Structure

```
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model architectures
│   ├── train/          # Training scripts and trainer
│   ├── utils/          # Utility functions and metrics
│   └── analysis/       # Evaluation and visualization
├── scripts/            # Standalone inference script
├── outputs/            # Training outputs and results
├── reports/            # Analysis reports
├── Dataset/            # Dataset directory
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎯 Usage Examples

### Data Inspection
```bash
# Quick data check
python -m src.data.quick_check

# Inspect annotation files
python -m src.data.data_inspect
```

### Fast Training (Development)
```bash
# Quick sanity check with limited batches
python -m src.train.run_training --backbone resnet --epochs 1 --fast --out outputs/test_run
```

### Custom Training
```bash
# Custom configuration
python -m src.train.run_training \
    --backbone resnet \
    --epochs 20 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --v2_augmentations \
    --amp \
    --label_smoothing 0.15 \
    --out outputs/custom_run
```

## 📈 Performance Analysis

### Best Practices Identified
1. **Transfer Learning**: Pretrained ImageNet weights provide strong initialization
2. **Regularization**: Label smoothing and data augmentation prevent overfitting
3. **Optimization**: AdamW with cosine annealing improves convergence
4. **Architecture**: ResNet-50 provides good balance of performance and efficiency
5. **Training Duration**: 10-15 epochs optimal for this dataset size

### Error Analysis Insights
- Model struggles with subtle expressions and ambiguous cases
- Valence regression performs better than arousal regression
- Some confusion between similar expression classes (e.g., fear vs surprise)
- Lighting and pose variations affect performance

## 🤝 Contributing

This project was developed as part of a Deep Learning course assignment. The codebase includes:
- Modular design for easy experimentation
- Comprehensive evaluation framework
- Detailed documentation and comments
- Reproducible training scripts

## 📄 License

This project is for educational purposes as part of a Deep Learning course assignment.