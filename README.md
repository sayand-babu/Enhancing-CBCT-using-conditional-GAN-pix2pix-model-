# Enhancing CBCT using 3D Conditional GAN (Pix2Pix / Vox2Vox)

A PyTorch framework for 3D medical image-to-image translation that enhances Cone Beam Computed Tomography (CBCT) volumes into Fan-Beam Computed Tomography (CT)-like volumes using a 3D Conditional Generative Adversarial Network (cGAN / Vox2Vox).

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Requirements and Installation](#requirements-and-installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage Guide](#usage-guide)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Inference and Volume Synthesis](#3-inference-and-volume-synthesis)
  - [4. Evaluation and Quantitative Metrics](#4-evaluation-and-quantitative-metrics)
- [Evaluation Metrics and Results](#evaluation-metrics-and-results)
- [Memory Optimization and Hardware Notes](#memory-optimization-and-hardware-notes)
- [References and Acknowledgments](#references-and-acknowledgments)

---

## Overview

Cone Beam Computed Tomography (CBCT) is widely used in image-guided radiation therapy (IGRT), dental imaging, and surgical planning due to low radiation dose and fast acquisition. However, CBCT scans frequently exhibit artifacts, high scatter radiation, noise, and low soft-tissue contrast compared to standard planning CT scans.

This repository implements a 3D conditional GAN architecture (Vox2Vox / 3D Pix2Pix) to learn an end-to-end volumetric mapping from raw/simulated 3D CBCT scans (source domain B) to paired ground-truth planning CT scans (target domain A).

```
[ Input CBCT Volume (3D) ] ---> [ 3D U-Net Generator ] ---> [ Synthesized CT (3D) ]
                                                                  |
                                                                  v
[ Condition (CBCT) + Real/Fake CT ] ---> [ 3D PatchGAN Discriminator ] ---> [ Real / Fake Score ]
```

---

## Key Features

- **Volumetric 3D Translation**: Directly processes 3D NIfTI volumes (`.nii` / `.nii.gz`) preserving spatial continuity across axial, sagittal, and coronal planes.
- **3D U-Net Generator**: Incorporates residual/convolutional blocks, group normalization, and dynamic spatial center-cropping for skip connections.
- **3D PatchGAN Discriminator**: Operates on concatenated pairs `(CBCT, CT)` to penalize structural and high-frequency discrepancies.
- **Medical Image Augmentations**: Built-in 3D geometric transformations (random flips across LR, AP, IS axes) and intensity transforms (bias field, blur, noise, gamma) using TorchIO.
- **Memory Efficiency**: Includes PyTorch Automatic Mixed Precision (AMP), gradient checkpointing (`torch.utils.checkpoint`), and sub-volume downscaling support.
- **Quantitative Evaluation Suite**: Computes 3D SSIM, PSNR, MAE, and MSE across generated volumes, saving summaries to Excel (`.xlsx`) and trend plots (`.png`).

---

## Model Architecture

### Generator
- **Structure**: 3D U-Net encoder-decoder network with skip connections.
- **Convolutions**: 3D Conv with Group Normalization (`nn.GroupNorm`) and LeakyReLU activations ($\alpha = 0.2$).
- **Downsampling**: MaxPool3D ($2 \times 2 \times 2$).
- **Upsampling**: ConvTranspose3D (stride 2) paired with spatial center-cropping to match skip connection tensor shapes.
- **Objective**: Combined Adversarial Loss ($\mathcal{L}_{cGAN}$) and weighted L1 Reconstruction Loss ($\lambda \mathcal{L}_{L1}$ with $\lambda = 20$).

### Discriminator
- **Structure**: 3D PatchGAN classifier.
- **Input**: 2-channel concatenated 3D volume `[Condition CBCT, Target CT / Fake CT]`.
- **Loss**: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`).

---

## Project Structure

```text
├── dataloader.py          # PyTorch Dataset implementation with TorchIO augmentations
├── model.py               # 3D Generator, 3D Discriminator, and GAN wrapper classes
├── parser.py              # Command-line argument definitions and hyperparameter defaults
├── train.py               # Model training script with mixed-precision and visual logging
├── test.py / transfer.py  # Inference scripts for synthesizing CT volumes from CBCT scans
├── centercrop.ipynb       # Jupyter notebook for volume center-cropping and alignment
├── evaluation.ipynb       # Evaluation script for computing SSIM, PSNR, MAE, and MSE
├── imports.ipynb          # Environment verification notebook
├── model_01/              # Saved model weights (generator01.pth) and metric outputs
├── model_02/              # Experiment checkpoints and loss plots
├── .gitignore             # Git ignore patterns
└── README.md              # Documentation
```

---

## Requirements and Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (Recommended: 8 GB+ VRAM)
- PyTorch matching your CUDA runtime

### Setup Environment

```bash
# Install PyTorch (adjust CUDA version if necessary)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core medical imaging and data processing libraries
pip install SimpleITK nibabel torchio scikit-image numpy pandas matplotlib openpyxl tqdm
```

---

## Dataset Preparation

The pipeline processes paired 3D volumes in NIfTI format (`.nii` or `.nii.gz`).

### Recommended Directory Structure:
```text
croped_dataset/
├── TRAINCBCTSIMULATED/      # Source Domain B (CBCT scans, e.g., REC-001.nii)
├── TRAINCTAlignedToCBCT/    # Target Domain A (Ground truth CT scans, e.g., volume-001.nii)
├── TESTCBCTSTIMULATED/      # Test Source Domain B (CBCT scans for inference)
└── TESTCTAlignedToCBCT/     # Test Target Domain A (CT scans for evaluation)
```

The `splitterA` and `splitterB` arguments in `parser.py` allow extracting corresponding sample indices from filenames (for example: `REC-1.nii` with splitter `REC-` maps to index `1`).

---

## Usage Guide

### 1. Data Preprocessing
If paired volumes have inconsistent fields of view (FOV) or bounding boxes, run `centercrop.ipynb` to center-crop volumes to consistent dimensions prior to training.

---

### 2. Model Training

Start model training using `train.py`:

```bash
python train.py \
  -pathA "../croped_dataset/TRAINCTAlignedToCBCT" \
  -pathB "../croped_dataset/TRAINCBCTSIMULATED" \
  -extensionA "nii" \
  -extensionB "nii" \
  -splitterA "volume-" \
  -splitterB "REC-" \
  -dimensions 366 288 364 \
  -epochs 25 \
  -batchSize 1 \
  -learnRate 0.0002 \
  -name "generator01"
```

#### Common Arguments:
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-pathA` | `str` | `../croped_dataset/TRAINCTAlignedToCBCT` | Path to target ground-truth CTs |
| `-pathB` | `str` | `../croped_dataset/TRAINCBCTSIMULATED` | Path to input condition CBCTs |
| `-dimensions` | `int list` | `366 288 364` | Volume dimensions `[Depth, Height, Width]` |
| `--filterN` | `int list` | `32 64 128 256` | Number of filters per layer in the generator |
| `-epochs` | `int` | `10` | Total training epochs |
| `-learnRate` | `float` | `0.0002` | Learning rate for Adam optimizers |
| `-downscale` | `flag` | `False` | Downsample volumes by a factor of 2 during loading |
| `-augRngThreshold` | `float` | `0.5` | Probability of applying 3D data augmentations |
| `-name` | `str` | `generator01` | Filename prefix for saved generator checkpoints |

---

### 3. Inference and Volume Synthesis

To generate enhanced CT volumes from unseen CBCT scans:

```bash
python test.py \
  -pathB "../croped_dataset/TESTCBCTSTIMULATED" \
  -extensionB "nii" \
  -splitterB "REC-" \
  -pathGenerator "model_01/generator01.pth" \
  -pathOutput "model_01/Generated_Test_CT"
```

The output volumes are saved as standard `.nii` files and can be visualized in 3D medical viewers such as 3D Slicer or ITK-SNAP.

---

### 4. Evaluation and Quantitative Metrics

Run `evaluation.ipynb` to evaluate the synthesized CT volumes against the real ground-truth CT volumes:
1. Calculates **SSIM**, **PSNR**, **MAE**, and **MSE** on a per-volume basis.
2. Exports all results to `similarity_metrics.xlsx`.
3. Plots metric distributions and trends across samples (`metric_trends.png`).

```python
# Metric calculation per volume pair
metrics = compute_3d_metrics(real_ct_volume, generated_ct_volume)
# Output: {'SSIM': 0.9421, 'PSNR': 31.45, 'MAE': 0.0124, 'MSE': 0.0008}
```

---

## Evaluation Metrics and Results

The enhanced CT volumes are evaluated using standard quantitative image metrics:

- **Structural Similarity Index Measure (SSIM)**: Evaluates structural, luminance, and contrast similarity in 3D.
- **Peak Signal-to-Noise Ratio (PSNR)**: Measures reconstruction fidelity and noise reduction in dB.
- **Mean Absolute Error (MAE) & Mean Squared Error (MSE)**: Measures voxel-level intensity differences.

Loss curves and metrics across training iterations are saved inside the respective model checkpoint directories (`model_01/`, `model_02/`).

---

## Memory Optimization and Hardware Notes

Training 3D convolutional networks on volumetric medical data requires substantial VRAM. The following optimizations are enabled in this repository:
1. **Mixed Precision (AMP)**: Uses `torch.cuda.amp.autocast()` and `GradScaler()` to reduce memory footprint.
2. **Gradient Checkpointing**: Recomputes intermediate encoder activations on-the-fly during backpropagation via `torch.utils.checkpoint`.
3. **Group Normalization**: Employs `nn.GroupNorm` to ensure stable convergence with small batch sizes (`batch_size=1`).
4. **Memory Allocator Configuration**: Configured with `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"` to mitigate fragmentation.
5. **Downscaling Flag**: Set `-downscale` if training on memory-constrained GPUs.

---

## References and Acknowledgments

- PyTorch: https://pytorch.org/
- SimpleITK: https://simpleitk.org/
- TorchIO (Medical Image Augmentation): https://torchio.readthedocs.io/
- Pix2Pix / Vox2Vox conditional GAN architecture for medical image synthesis.
