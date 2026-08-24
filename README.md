# Enhancing CBCT using 3D Conditional GAN (Pix2Pix / Vox2Vox)

A deep learning framework for 3D medical image-to-image translation that enhances **Cone Beam Computed Tomography (CBCT)** images into high-quality **Fan-Beam Computed Tomography (CT)**-like volumes using a 3D Conditional Generative Adversarial Network (cGAN / Vox2Vox).

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Requirements & Installation](#-requirements--installation)
- [Dataset Preparation](#-dataset-preparation)
- [Usage Guide](#-usage-guide)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Training the Model](#2-training-the-model)
  - [3. Inference / Volume Synthesis](#3-inference--volume-synthesis)
  - [4. Evaluation & Quantitative Metrics](#4-evaluation--quantitative-metrics)
- [Evaluation Metrics & Results](#-evaluation-metrics--results)
- [Memory Optimization & GPU Setup](#-memory-optimization--gpu-setup)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🔬 Overview

Cone Beam Computed Tomography (**CBCT**) is widely used in image-guided radiation therapy (IGRT), dental imaging, and surgical planning due to its low radiation dose and fast acquisition. However, CBCT often suffers from:
- Increased scatter radiation and noise
- Lower soft-tissue contrast
- Artifacts and Hounsfield Unit (HU) inaccuracies

This project addresses these challenges by implementing a **3D Conditional GAN (Pix2Pix / Vox2Vox)** in PyTorch. The model learns a non-linear mapping from raw/simulated 3D CBCT scans (source domain $B$) to paired ground-truth planning CT scans (target domain $A$).

```
[ Input CBCT Volume (3D) ] ──▶ [ 3D U-Net Generator ] ──▶ [ Synthesized Enhanced CT (3D) ]
                                                                 │
                                                                 ▼
[ Condition (CBCT) + Real/Fake CT ] ──▶ [ 3D PatchGAN Discriminator ] ──▶ [ Real / Fake Score ]
```

---

## ✨ Key Features

- **End-to-End 3D Translation:** Operates directly on volumetric NIfTI (`.nii` / `.nii.gz`) data preserving cross-slice 3D spatial correlations.
- **3D U-Net Generator with Skip Connections:** Incorporates dynamic center-cropping to handle spatial dimension alignments seamlessly.
- **3D PatchGAN Discriminator:** Evaluates paired condition-image volumes $(CBCT, CT)$ and penalizes high-frequency structural errors.
- **Medical Image Augmentations with TorchIO:** 3D geometric transforms (random flips across LR, AP, IS axes) and intensity transforms (bias fields, blur, noise, gamma).
- **GPU Memory Optimization:**
  - PyTorch Mixed Precision (`torch.cuda.amp.autocast` + `GradScaler`)
  - Activation / Gradient Checkpointing (`torch.utils.checkpoint`)
  - Dynamic tensor matching and sub-volume downscaling support
- **Automated Metric Evaluation:** Generates 3D SSIM, PSNR, MAE, and MSE scores, exporting summaries to Excel (`.xlsx`) and visual trend plots (`.png`).

---

## 🏗️ Model Architecture

### Generator
- **Type:** 3D U-Net encoder-decoder with skip connections.
- **Building Blocks:** 3D Convolution $\rightarrow$ Group Normalization (`nn.GroupNorm`) $\rightarrow$ LeakyReLU ($0.2$).
- **Downsampling:** MaxPool3D ($2\times2\times2$).
- **Upsampling:** ConvTranspose3D (stride 2) + dynamic spatial center-cropping of skip connections.
- **Loss:** Composite loss combining Adversarial Loss ($\mathcal{L}_{cGAN}$) with weighted L1 Reconstruction Loss ($\lambda \mathcal{L}_{L1}$ where $\lambda = 20$).

### Discriminator
- **Type:** 3D PatchGAN Classifier.
- **Input:** 2-channel concatenated 3D volume `(Input CBCT, Target CT / Synthetic CT)`.
- **Loss:** Binary Cross Entropy with Logits (`BCEWithLogitsLoss`).

---

## 📂 Project Structure

```text
├── dataloader.py          # PyTorch Dataset & TorchIO 3D augmentation pipeline
├── model.py               # 3D Generator, 3D Discriminator, and helper blocks
├── parser.py              # CLI Argument parser for training & architecture config
├── train.py               # Main training script with mixed-precision & loss logging
├── test.py / transfer.py  # Inference scripts for synthesizing CT volumes from CBCT
├── centercrop.ipynb       # Notebook for preprocessing & center-cropping NIfTI volumes
├── evaluation.ipynb       # Quantitative evaluation (SSIM, PSNR, MAE, MSE) & plotting
├── imports.ipynb          # Environment validation notebook
├── model_01/              # Model artifacts, checkpoints (generator01.pth) & metrics
├── model_02/              # Additional training experiment runs & loss curves
├── .gitignore             # Ignored checkpoint files and temporary caches
└── README.md              # Project documentation
```

---

## ⚙️ Requirements & Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (Recommended $\ge 8\text{ GB}$ VRAM)
- PyTorch with CUDA support

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install SimpleITK nibabel torchio scikit-image numpy pandas matplotlib openpyxl tqdm
```

---

## 📊 Dataset Preparation

The pipeline expects paired 3D NIfTI volumes (`.nii` or `.nii.gz`).

### Recommended Directory Layout:
```text
croped_dataset/
├── TRAINCBCTSIMULATED/      # Source Domain B (CBCT volumes, e.g., REC-001.nii)
├── TRAINCTAlignedToCBCT/    # Target Domain A (CT volumes, e.g., volume-001.nii)
├── TESTCBCTSTIMULATED/      # Test Source Domain B (CBCT volumes for inference)
└── TESTCTAlignedToCBCT/     # Test Target Domain A (Ground truth for evaluation)
```

> **Note on Splitters:** The `splitterA` and `splitterB` arguments in `parser.py` are used to extract matching numeric volume IDs (e.g. `REC-1.nii` $\rightarrow$ `splitter='REC-'` $\rightarrow$ ID `1`).

---

## 🚀 Usage Guide

### 1. Data Preprocessing
If paired volumes have differing field of views (FOV) or dimensions, use `centercrop.ipynb` to crop the larger volumes to match the target dimensions.

---

### 2. Training the Model

Run `train.py` with custom arguments or defaults:

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

#### Key Arguments:
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-pathA` | `str` | `../croped_dataset/TRAINCTAlignedToCBCT` | Path to target ground truth CTs |
| `-pathB` | `str` | `../croped_dataset/TRAINCBCTSIMULATED` | Path to input condition CBCTs |
| `-dimensions` | `int list` | `366 288 364` | 3D Volume shape `(D, H, W)` |
| `--filterN` | `int list` | `32 64 128 256` | Generator layer filter counts |
| `-epochs` | `int` | `10` | Number of training epochs |
| `-learnRate` | `float` | `0.0002` | Adam optimizer learning rate |
| `-downscale` | `flag` | `False` | Downsample volumes by $2\times$ to fit small GPUs |
| `-augRngThreshold` | `float` | `0.5` | Probability threshold for 3D augmentations |
| `-name` | `str` | `generator01` | Saved model weights filename prefix |

---

### 3. Inference / Volume Synthesis

Generate enhanced synthetic CT scans from input CBCT scans using `test.py` (or `transfer.py`):

```bash
python test.py \
  -pathB "../croped_dataset/TESTCBCTSTIMULATED" \
  -extensionB "nii" \
  -splitterB "REC-" \
  -pathGenerator "model_01/generator01.pth" \
  -pathOutput "model_01/Generated_Test_CT"
```

Synthesized volumes are saved directly as `.nii` files ready for 3D Slicer, ITK-SNAP, or downstream contouring/planning tools.

---

### 4. Evaluation & Quantitative Metrics

Open and run `evaluation.ipynb` to compare generated CT volumes with real ground truth CT volumes:
1. Computes **SSIM**, **PSNR**, **MAE**, and **MSE** across all 3D volumes.
2. Exports all results to an Excel spreadsheet (`similarity_metrics.xlsx`).
3. Plots metric trends per sample (`metric_trends.png`).

```python
# Metrics computed per volume pair
metrics = compute_3d_metrics(real_ct_volume, generated_ct_volume)
# Returns: {'SSIM': 0.9421, 'PSNR': 31.45, 'MAE': 0.0124, 'MSE': 0.0008}
```

---

## 📈 Evaluation Metrics & Results

The enhanced CT volumes are quantitatively evaluated against target CT scans using standard image quality metrics:

- **Structural Similarity Index (SSIM):** Measures luminance, contrast, and structural similarity in 3D.
- **Peak Signal-to-Noise Ratio (PSNR):** Evaluates overall fidelity and noise suppression in decibels (dB).
- **Mean Absolute Error (MAE) & Mean Squared Error (MSE):** Measures voxel-level intensity differences.

Loss curves during training and per-subject metric distributions are saved in each experiment's folder (`model_01/`, `model_02/`).

---

## 🧠 Memory Optimization & GPU Setup

Volumetric 3D networks require substantial GPU memory. This repository implements several strategies to fit large 3D scans:
1. **PyTorch Mixed Precision (AMP):** Utilizes `torch.cuda.amp.autocast()` and `GradScaler()` to halve activation memory.
2. **Gradient Checkpointing:** Recomputes encoder and bottleneck activations during the backward pass using `torch.utils.checkpoint`.
3. **Group Normalization:** Uses `nn.GroupNorm` instead of BatchNorm to remain stable with a batch size of 1.
4. **CUDA Memory Fragmentation Control:** Configured with `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"`.
5. **Optional Downscaling:** Use `-downscale` in `parser.py` if running on GPUs with $<8\text{ GB}$ VRAM.

---

## 📄 License & Acknowledgments

- Built with [PyTorch](https://pytorch.org/), [SimpleITK](https://simpleitk.org/), and [TorchIO](https://torchio.readthedocs.io/).
- Inspired by the Pix2Pix / Vox2Vox conditional GAN architecture for medical image synthesis.
