# Salient Object Detection using Mini U-Net

## Project Overview
This project implements an end-to-end Salient Object Detection (SOD) system from scratch using PyTorch and a Mini U-Net architecture.

The goal of the project is to identify and segment the most visually important object in an image by generating a saliency mask.

---

## Dataset
Dataset used:
- DUTS Saliency Detection Dataset
- Source: Kaggle (`balraj98/duts-saliency-detection-dataset`)

Dataset preprocessing included:
- Image resizing to 128×128
- Normalization to range [0,1]
- Train / Validation / Test split
- Data augmentation:
  - Horizontal flip
  - Brightness and contrast adjustment

---

## Model Architecture
The final model is based on a Mini U-Net architecture with:
- Encoder-decoder structure
- Skip connections
- Batch Normalization
- Dropout
- Sigmoid output layer

Loss Function:
- Binary Cross Entropy (BCE)
- IoU Loss

Optimizer:
- Adam Optimizer (`lr = 1e-3`)

---

## Results

| Model | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| Baseline (5 epochs) | 0.5768 | 0.5938 | 0.5807 | 0.4126 |
| Improved (5 epochs) | 0.6479 | 0.5000 | 0.5606 | 0.3932 |
| Improved (15 epochs) | 0.6278 | 0.5774 | 0.5981 | 0.4295 |
| Mini U-Net (25 epochs) | 0.8766 | 0.8830 | 0.8788 | 0.7854 |

---
## Interactive Web Interface

A simple Gradio web application was implemented to provide an interactive demo of the final Mini U-Net model.

Features:
- Upload custom images
- Generate saliency masks in real time
- Visualize overlay segmentation results
- Display fast inference results

The interface allows users to test the Salient Object Detection model directly through a user-friendly web application.

---

## Demo
The notebook includes a live demo where users can:
- Upload an image
- Generate a saliency mask
- Visualize overlay results
- Measure inference time

---

## Final Conclusion
The final improved Mini U-Net model trained for 25 epochs achieved the best overall performance, with:

- Precision = 0.8766
- Recall = 0.8830
- F1-score = 0.8788
- IoU = 0.7854

The addition of skip connections, batch normalization, dropout, and data augmentation significantly improved segmentation quality and produced much clearer saliency masks compared to the baseline CNN model.

---

## Technologies Used
- Python
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- Google Colab
