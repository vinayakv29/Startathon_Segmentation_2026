# 🏜️ Team Trimax: High-Fidelity Offroad Autonomy Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NVIDIA RTX 3050](https://img.shields.io/badge/Hardware-RTX%203050-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/)

## 📝 Project Overview
This repository contains the official submission for the **GHR 2.0 Offroad Autonomy Challenge**. Team Trimax has engineered a semantic segmentation pipeline designed for the extreme demands of unstructured desert navigation. By bridging the gap between transformer-based feature extraction and real-time edge inference, our solution provides a robust, safety-critical occupancy map for autonomous UGVs.

## 🚀 Key Performance Indicators (KPIs)

| Metric | Results | Technical Context |
| :--- | :--- | :--- |
| **Global Pixel Accuracy** | **78.45%** | High-precision terrain classification. |
| **Mean IoU (mIoU)** | **0.2914** | Safety-weighted for hazard prioritization. |
| **Inference Latency** | **88ms** | ~11.29 FPS on mobile NVIDIA RTX 3050. |
| **Training Efficiency** | **10 Epochs** | Rapid convergence via "Greedy" fine-tuning. |

## 🧠 Architectural Deep-Dive

### 1. DINOv2 Backbone Integration
We utilize the **DINOv2 (ViT-S/14)** self-supervised transformer as our primary feature extractor. Unlike traditional ResNet architectures, DINOv2 provides dense semantic tokens that excel at identifying low-contrast boundaries in natural, non-linear desert environments.

### 2. ConvNeXt-Style Segmentation Head
Our custom decoder head utilizes **depth-wise separable convolutions** ($7 \times 7$ kernels) and **GELU activations**. This design captures broad spatial context while maintaining the lightweight footprint necessary for real-time operation on mobile workstations like the **HP Victus**.

### 3. Hazard-Critical Weighting (Safety-First)
To address the extreme class imbalance of desert datasets, we implemented a **Weighted Cross-Entropy Loss** strategy with a **5.0x penalty** for navigational hazards (Rocks, Logs, Bushes). This ensures the model treats a 10-pixel rock as a critical navigation failure rather than a minor statistical error.

## 📂 Repository Contents
* `train_segmentation.py`: End-to-end training pipeline with DINOv2 feature alignment.
* `test_segmentation.py`: Performance evaluation suite for 1002-sample test datasets.
* `segmentation_head.pth`: Optimized model weights following Phase 2 fine-tuning.
* `Report.pdf`: Comprehensive technical analysis and qualitative results.

## 🏁 How to Reproduce Results
1. **Environment Setup**:
   ```bash
   conda activate EDU
   pip install torch torchvision tqdm opencv-python numpy matplotlib

2. Training the Model
To replicate our high-convergence fine-tuning run:

Bash
python train_segmentation.py
3. Final Evaluation
To generate metrics and visualization masks for the 1002-image test set:

Bash
python test_segmentation.py
