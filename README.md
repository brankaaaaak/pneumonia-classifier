# Pneumonia Detection Using Convolutional Neural Networks

## Project Overview

This project focuses on **automated detection of pneumonia** from chest X-ray images using **Convolutional Neural Networks (CNN)** and **Transfer Learning** with **ResNet50**.

**Goal**: Build models that accurately distinguish between **NORMAL** and **PNEUMONIA** chest X-rays, supporting medical diagnostics and reducing radiologists' workload.

---

## Dataset

- **Source**: Chest X-ray dataset (5,863 labeled images)
- **Classes**: `NORMAL`, `PNEUMONIA`
- **Dataset**: [Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Data Split
- **Training**: 70%  
- **Validation**: 15%  
- **Test**: 15%  

### Data Augmentation

Applied using `Keras.ImageDataGenerator`:
- Random rotation
- Width/height shift
- Zoom
- Brightness adjustment
- Horizontal flipping

---

## Models

Three different models were developed and compared:

### CNN (From Scratch)
### Transfer Learning – ResNet50
### Fine-Tuned ResNet50

---

## Training Details

| Component     | Setting                          |
|---------------|-----------------------------------|
| **Optimizer** | `Adam`                           |
| **Loss**      | `Binary Crossentropy`            |
| **Metrics**   | `Accuracy`, `F1-Score`, `AUC`    |
| **Callbacks** | `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau` |
| **Balancing** | Class weights applied to handle imbalance |

---
## Requirements

- Python **3.10+**
- Libraries:
  - `TensorFlow / Keras`
  - `NumPy`
  - `Matplotlib`
  - `Seaborn`
  - `scikit-learn`
