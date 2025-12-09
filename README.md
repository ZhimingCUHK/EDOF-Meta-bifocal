# EDOF-Meta-Bifocal

**End-to-End Optimization for Extended Depth of Field using Meta-Bifocal Optics**

This repository implements a differentiable end-to-end imaging pipeline that jointly optimizes optical parameters (Zernike coefficients) and a deep learning reconstruction network (PolarMIMO-UNet) to achieve Extended Depth of Field (EDOF) using polarization-based bifocal imaging.

## Key Features

*   **Differentiable Optics Simulation**: Physics-based PSF generation using Zernike polynomials with wavelength-dependent chromatic modeling (RGB channels).
*   **Bifocal Polarization Design**: Dual-focus system with independent focal planes for 0° and 90° polarization states, enabling depth-adaptive imaging.
*   **Polarization Sensor Modeling**: Realistic IMX250MYR-like sensor simulation with quad Bayer pattern and polarization mosaic.
*   **Layer-based Depth Rendering**: Diopter-space depth layering with physically accurate occlusion handling using the Over Operator.
*   **End-to-End Joint Optimization**: Simultaneously learns optimal phase masks (Zernike coefficients) and neural network weights.
*   **PolarMIMO-UNet Architecture**: Custom multi-scale U-Net adapted for RAW polarization data reconstruction.

## Project Structure

```
EDOF-Meta-bifocal/
├── config.yaml                      # System configuration (optics/sensor/training)
├── main.py                          # Main training script with end-to-end optimization
├── image_formation_pipeline.ipynb   # Interactive demo notebook
├── src/
│   ├── layers/
│   │   ├── diffoptics.py            # Differentiable optics (Zernike → PSF)
│   │   ├── polarsensor.py           # IMX250MYR polarization sensor simulator
│   │   └── imageformation.py        # Depth-layered blurring with Over compositing
│   ├── models/
│   │   └── pipeline.py              # Complete imaging pipeline (Optics + Sensor)
│   ├── network/
│   │   ├── polar_mimo_unet.py       # PolarMIMO-UNet for RAW reconstruction
│   │   └── layers.py                # Network building blocks
│   └── data/
│       └── dataloader.py            # SceneFlow dataset loader with depth preprocessing
├── experiments/
│   └── train_run/
│       ├── checkpoints/              # Saved model weights
│       ├── logs/                     # TensorBoard logs
│       └── visualizations/           # Training/validation visualizations
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- [SceneFlow Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/EDOF-Meta-bifocal.git
    cd EDOF-Meta-bifocal
    ```

2.  Install dependencies:
    ```bash
    pip install torch torchvision numpy matplotlib pyyaml scikit-image tqdm tensorboard
    pip install poppy astropy opencv-python
    ```

3.  Download and prepare the SceneFlow dataset:
    ```bash
    # Download FlyingThings3D_subset to /home/YourUsername/Data/
    # Expected structure:
    # /home/YourUsername/Data/
    # └── FlyingThings3D_subset/
    #     ├── train/
    #     │   ├── image_clean/right/
    #     │   └── disparity/right/
    #     └── val/
    #         ├── image_clean/right/
    #         └── disparity/right/
    ```

## Configuration

Edit `config.yaml` to customize system parameters:

```yaml
optics:
  f_mm: 25.0                  # Focal length (mm)
  f_number: 2.8               # F-number
  pixel_size_um: 3.45         # Pixel pitch (μm)
  fov_pixels: 32              # PSF field of view
  n_zernike: 15               # Number of Zernike modes
  pupil_grid_size: 256        # Pupil simulation resolution

sensor:
  height: 256                 # Sensor height (pixels)
  width: 256                  # Sensor width (pixels)
  noise_sigma: 0.01           # Readout noise level

layers:
  num_layers: 12              # Depth discretization layers
  min_dist: 0.2               # Near focus distance (m)
  max_dist: 20.0              # Far focus distance (m)

training:
  batch_size: 4
  learning_rate: 0.001        # Network learning rate
  optics_learning_rate: 1e-5  # Zernike coefficient learning rate
  epochs: 100
```

## Usage

### Training

Start end-to-end training (optics + network):

```bash
python main.py
```

**Important:** Update the data path in `main.py` line 244:
```python
data_root = '/path/to/your/Data'  # Change this to your SceneFlow location
```

The training process:
1. Loads RGB images and depth maps from SceneFlow dataset
2. Generates bifocal PSF banks using learnable Zernike coefficients
3. Simulates depth-dependent optical blur with layer compositing
4. Creates polarization sensor RAW output (mosaic pattern)
5. Reconstructs RGB image using PolarMIMO-UNet
6. Computes multi-scale L1 loss against ground truth
7. Backpropagates to update both optical parameters and network weights

### Monitoring Training

Use TensorBoard to visualize training progress:

```bash
tensorboard --logdir experiments/train_run/logs
```

Metrics logged:
- Training/Validation Loss
- PSNR and SSIM
- Zernike coefficient evolution
- Visual comparisons (GT, RAW input, reconstruction)

### Inference

Load a trained checkpoint for inference:

```python
import torch
from src.models.pipeline import ImageFormationPipeline
from src.network.polar_mimo_unet import build_net

# Load pipeline
pipeline = ImageFormationPipeline(
    optics_config=config['optics'],
    sensor_config=config['sensor'],
    layer_config=config['layers']
)

# Load trained model
model = build_net('PolarMIMOUNet', in_channel=1)
checkpoint = torch.load('experiments/train_run/checkpoints/checkpoint_epoch_100.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
raw_output, _, _ = pipeline(rgb_gt, depth_map)
reconstruction = model(raw_output)[2]  # Get full-resolution output
```

## Results

Training outputs are saved in `experiments/train_run/`:
- **Checkpoints**: Model weights saved every 5 epochs
- **Visualizations**: Comparison images (GT / Depth / RAW / Reconstruction)
- **Logs**: TensorBoard events with metrics and parameter tracking

## Technical Details

### Optical Model
- **PSF Generation**: Zernike polynomial-based wavefront error modeling
- **Chromatic Aberration**: Wavelength-dependent PSF for R/G/B channels (630/550/470 nm)
- **Defocus**: Physically accurate diopter-based defocus calculation
- **Bifocal Design**: Independent phase masks for two polarization states

### Sensor Model
- **Polarization Pattern**: Quad-pixel layout (0°/45°/90°/135°)
- **Color Filter Array**: Quad Bayer mosaic structure
- **Noise Model**: Gaussian readout noise simulation

### Network Architecture
- **Input**: Single-channel RAW mosaic [B, 1, H, W]
- **Output**: RGB image [B, 3, H/2, W/2] (half resolution due to Bayer demosaicing)
- **Architecture**: MIMO-UNet with multi-scale supervision
- **Design**: Pixel-unshuffle preprocessing to preserve mosaic structure

---
*This project explores the intersection of Computational Photography, Optics, and Deep Learning for next-generation imaging systems.*
