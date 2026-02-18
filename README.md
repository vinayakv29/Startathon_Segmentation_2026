# Team Trimax: Offroad Autonomy Segmentation 2026

## 🛠️ Project Overview
This repository contains our implementation for the Offroad Autonomy Segmentation challenge. Our goal is to enable safe navigation for UGVs in desert environments using semantic segmentation.

## 🧠 Technical Highlights
- **Backbone:** DINOv2 (vits14) for high-fidelity feature extraction.
- **Head:** custom ConvNeXt-style segmentation head.
- **Classes:** 10 categories (Trees, Rocks, Sky, etc.).
- **Platform:** Trained and tested on an HP Victus with CUDA acceleration.

## 🚀 How to Run
1. Run `env_setup/setup_env.bat` to configure the 'EDU' environment.
2. Activate with `conda activate EDU`.
3. Execute `train_segmentation.py` for training.
