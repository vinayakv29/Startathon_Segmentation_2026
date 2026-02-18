# 🏜️ Team Trimax: Offroad Autonomy Segmentation 2026

## 🛠️ Project Overview
This repository contains our optimized implementation for the GHR 2.0 Offroad Autonomy Segmentation challenge. Our solution is specifically engineered to enable safe navigation for Unmanned Ground Vehicles (UGVs) in unstructured desert environments by prioritizing the detection of high-risk obstacles such as rocks and bushes.

## 🧠 Technical Highlights
* Backbone: DINOv2 (vits14) leveraged for its superior self-supervised spatial feature extraction in natural, off-road environments.
* Architecture: Custom ConvNeXt-style segmentation head designed for high-resolution mapping on mobile hardware.
* Optimization Strategy: Implemented a Weighted Cross-Entropy Loss with a 5.0x penalty for critical hazards (Rocks, Logs, Bushes) to address extreme desert class imbalance.
* Robustness: Integrated a stochastic augmentation pipeline featuring Random Rotation (10°) and Color Jittering to simulate varied desert lighting and vehicle tilt.
* Hardware Efficiency: Optimized for the NVIDIA RTX 3050 (HP Victus), utilizing CUDA acceleration and VRAM-efficient batch processing.

## 🚀 Step-by-Step Instructions

### 1. Environment & Dependencies
We use the 'EDU' conda environment. To reproduce our results, ensure the following are installed:
1. Run env_setup/setup_env.bat to configure the environment.
2. Activate with 'conda activate EDU'.
3. Install necessary libraries:
   pip install torch torchvision tqdm opencv-python numpy matplotlib

### 2. Training the Model
To initiate the optimized training process:
1. Ensure the dataset is located at: C:\Startathon_2026\Data\Offroad_Segmentation_Training_Dataset\
2. Execute the training script:
   python train_segmentation.py
* Note: The script saves the best-performing weights to segmentation_head.pth.

### 3. Testing & Evaluation
To verify final performance and generate Mean IoU scores:
1. Ensure segmentation_head.pth is in the root directory.
2. Run the inference script:
   python test_segmentation.py
* Target Metric: Aiming for a Mean IoU of 0.45+ through our high-penalty optimization strategy.

## 📊 Notes on Interpretation
* Validation Loss: A steady decrease across epochs indicates successful model convergence.
* Mean IoU: This is our primary accuracy metric. Scores above 0.40 represent high-quality hazard detection in complex terrain.
* segmentation_head.pth: This file contains the trained weights and is required for all inference tasks.
