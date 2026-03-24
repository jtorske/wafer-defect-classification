# Autonomous Wafer Defect Detection

Deep learning pipeline for classifying semiconductor wafer defect patterns from wafer map images using convolutional and transformer-based models.

This project evaluates modern computer vision architectures for automated defect classification in semiconductor manufacturing.

---

## Overview

Semiconductor wafer maps contain spatial patterns that reveal manufacturing defects. Manual inspection is time-consuming and error-prone, motivating automated classification systems.

This project builds an end-to-end deep learning pipeline to classify wafer defect patterns from wafer map images using:

- Convolutional Neural Networks (**ResNet-18**)
- Vision Transformers (**ViT-Tiny**)

The system trains and evaluates models on a large wafer defect dataset and analyzes performance across defect categories.

---

## Dataset

The project uses the **Mixed-Type Wafer Defect Dataset** from Kaggle.

Dataset characteristics:

- **38,015 wafer map images**
- **8 defect pattern classes**
- Image resolution: **52 × 52**

Defect classes:

- Center
- Donut
- Edge-Loc
- Edge-Ring
- Loc
- Random
- Scratch
- Near-full

The dataset is highly **class imbalanced**, requiring additional preprocessing and training strategies.

Dataset split:

- **80% training**
- **20% testing**

---

## Data Preprocessing

Wafer maps were preprocessed and normalized before training.

Key steps:

- Resize images to **224 × 224**
- Normalize pixel values to **[0,1]**
- Convert wafer maps to **CNN-ready tensors**

Data augmentation was applied during training to improve generalization:

- Random horizontal and vertical flips
- Small rotations (±10°)
- Minor translations (≤5%)
- Gaussian blur
- Brightness and contrast jitter

These augmentations exploit the **rotational and translational invariance of wafer defect patterns**.

---

## Handling Class Imbalance

The dataset contains large class imbalance between defect types.

To mitigate this:

- Per-class **subsampling** was applied
- Maximum samples per class: **3000**
- Per-class evaluation metrics were used

This ensures balanced training and manageable training time.

---

## Models

Two deep learning architectures were implemented and compared.

### ResNet-18

A convolutional neural network used as the baseline model.

Key properties:

- Trained **from scratch**
- Adam optimizer
- Cosine learning rate scheduling
- Cross-entropy loss with label smoothing

CNNs perform well at detecting **local spatial features** such as small defect patterns.

---

### Vision Transformer (ViT-Tiny)

A transformer-based architecture fine-tuned from **ImageNet pretraining**.

Key properties:

- Fine-tuned using AdamW optimizer
- Global self-attention mechanism
- Captures long-range spatial relationships

Transformers are better suited for detecting **global wafer defect structures**.

---

## Training Configuration

Experiments were executed in **Google Colab using GPU acceleration**.

Training parameters:

Input size: 224 × 224
Batch size: 64

Epochs:
ResNet-18: 10
ViT-Tiny: 5

Max samples per class: 3000
Random seed: 42

Input size: 224 × 224
Batch size: 64

Epochs:
ResNet-18: 10
ViT-Tiny: 5

Max samples per class: 3000
Random seed: 42


Data loading and batching were implemented using **PyTorch DataLoader**.

---

## Evaluation

Models were evaluated on a held-out test set using:

- Overall accuracy
- Per-class accuracy
- Confusion matrices
- Precision / recall / F1 scores

Example ResNet-18 results:

Overall accuracy: ~98%

Per-class performance highlights:

Center: 99%
Donut: 100%
Edge-Ring: 98%
Scratch: 82%
Random: 95%


Performance was strongest on **majority defect classes**, while minority classes remained more challenging.

---

## Results

Key observations:

- **ResNet-18 achieved higher overall accuracy**
- **ViT-Tiny showed stronger generalization for global defects**
- Minority defect classes remained difficult due to limited training samples

Most classification errors occurred in:

- Rare defect categories
- Visually similar edge-related patterns

---

## Key Takeaways

- CNNs perform well on **localized defect patterns**
- Transformers better capture **global wafer structures**
- **Class imbalance** is the largest challenge in wafer defect classification

Future improvements could include:

- class-weighted loss
- focal loss
- oversampling rare defect types
- longer transformer training schedules




