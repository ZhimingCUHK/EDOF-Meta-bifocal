# EDOF-Meta-Bifocal

**End-to-End Optimization for Extended Depth of Field using Meta-Bifocal Optics**

This repository implements a differentiable end-to-end imaging pipeline that jointly optimizes optical parameters (Zernike coefficients) and a deep learning reconstruction network (MIMO-UNet) to achieve Extended Depth of Field (EDOF) using polarization-based bifocal imaging.

## Key Features

*   **Differentiable Optics Simulation**: Physics-based PSF generation using Zernike polynomials.
*   **Bifocal Polarization Design**: Simulates dual-focus system with independent PSFs for 0° and 90° polarization states.
*   **Depth-Dependent Blur**: Layer-based depth rendering engine that generates physically accurate defocus blur based on scene depth.
*   **End-to-End Joint Optimization**: Simultaneously learns optimal phase masks (Zernike coefficients) and neural network weights.
*   **MIMO-UNet Architecture**: Multi-Input Multi-Output U-Net for high-quality image restoration.
*   **Real-time Monitoring**: TensorBoard integration for tracking Loss, PSNR/SSIM metrics, and **visualizing PSF shape evolution**.

## Project Structure

```
EDOF-Meta-bifocal/
├── src/
│   ├── data/
│   │   └── dataloader.py        # SceneFlow dataset loader
│   ├── layers/
│   │   ├── optics.py            # Differentiable optics (Zernike → PSF)
│   │   └── imageformation.py    # Depth-layered blurring logic
│   ├── models/
│   │   └── pipeline.py          # BlurGenerationPipeline & Global Configuration
│   ├── network/
│   │   └── MIMOUNET.py          # Image restoration network
│   └── train.py                 # Main training script (Training & Evaluation)
├── checkpoints_mimo/            # (Created at runtime) Model weights & Logs
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU
- [SceneFlow Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (FlyingThings3D subset)

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/EDOF-Meta-bifocal.git
    cd EDOF-Meta-bifocal
    ```

2.  Install dependencies:
    ```bash
    pip install torch torchvision numpy matplotlib scikit-image tqdm tensorboard
    ```

3.  **Data Preparation**:
    Ensure the SceneFlow dataset is located at your data root (e.g., `/home/LionelZ/Data`). The structure should be standard SceneFlow format.

## Configuration

Configuration is currently handled via Python dictionaries within the source files:

1.  **Optical & Simulation Config**:
    *   File: `src/models/pipeline.py`
    *   Parameters: `f_mm`, `f_number`, `num_layers`, `pupil_grid_size`, etc.
    *   **Important**: Update `data_root` in the `CONFIG` dictionary to point to your dataset.

2.  **Training Config**:
    *   File: `src/train.py`
    *   Parameters: `epochs`, `batch_size`, `learning_rate`, `train_optics` (Joint optimization flag).

## Usage

### Training

To start the end-to-end training process:

```bash
python src/train.py
```

The script will:
1.  Initialize the Optical Simulator and MIMO-UNet.
2.  Load the SceneFlow dataset.
3.  Perform joint optimization (if `train_optics=True`).
4.  Evaluate on the validation set after every epoch.

### Monitoring

Use TensorBoard to visualize training progress, including the **evolving shape of the Point Spread Functions (PSF)**:

```bash
tensorboard --logdir checkpoints_mimo/logs
```

**What to look for in TensorBoard:**
*   **Scalars**:
    *   `Loss/epoch_avg`: Training loss curve.
    *   `Metrics/Val_PSNR` & `Metrics/Val_SSIM`: Restoration quality on validation set.
*   **Images**:
    *   `Images/`: Visual comparison of Sharp GT, Blurred Inputs (0°/90°), and Prediction.
    *   `PSF_Shape/`: **Real-time visualization of the learned PSFs** at different depths.

## Technical Details

### Optical Model
- **PSF Generation**: Differentiable Fourier optics model.
- **Bifocal Strategy**: Two distinct focal distances (`focus_dist_0` and `focus_dist_90`) are simulated to cover a larger depth range.
- **Optimization**: The Zernike coefficients of the phase mask are learnable parameters.

### Network Architecture
- **Input**: Stacked tensor `[B, 6, H, W]` containing the 0° blurred RGB image and 90° blurred RGB image.
- **Output**: Restored RGB image `[B, 3, H, W]`.
- **Loss**: Multi-scale Content Loss (L1) combined with perceptual metrics.

---
*This project explores the intersection of Computational Photography, Optics, and Deep Learning.*
