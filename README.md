# EDOF-Meta-Bifocal

**End-to-End Optimization for Extended Depth of Field using Meta-Bifocal Optics**

This repository implements a differentiable end-to-end imaging pipeline that jointly optimizes optical parameters (Zernike coefficients) and a deep learning reconstruction network (MIMO-UNet) to achieve Extended Depth of Field (EDOF).

## 🚀 Key Features

*   **Differentiable Optics Simulation**: Simulates Point Spread Functions (PSF) based on Zernike polynomials and depth information.
*   **Bifocal Polarization Design**: Models a system with two distinct focal planes (Near/Far) controlled by polarization states (0° and 90°).
*   **End-to-End Joint Optimization**: Simultaneously learns the optimal optical phase mask (Zernike coefficients) and the image restoration network weights.
*   **Physics-based Sensor Simulation**: Includes realistic sensor modeling with Bayer/Polarization mosaic patterns and noise injection.
*   **MIMO-UNet Architecture**: Utilizes a Multi-Input Multi-Output U-Net for robust multi-scale image restoration.

## 📂 Project Structure

```
EDOF-Meta-bifocal/
├── config.yaml             # Configuration for optics, sensor, and training
├── main.py                 # Main training script (End-to-End optimization)
├── src/
│   ├── layers/
│   │   ├── diffoptics.py   # Differentiable optics layer (Zernike -> PSF)
│   │   ├── polarsensor.py  # Polarization sensor simulation
│   │   └── imageformation.py # Depth-dependent blurring
│   ├── models/
│   │   └── pipeline.py     # Assembles Optics + Sensor into a pipeline
│   ├── network/
│   │   └── mimo_unet.py    # MIMO-UNet reconstruction network
│   └── data/
│       └── dataloader.py   # SceneFlow dataset loader
```

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/EDOF-Meta-bifocal.git
    cd EDOF-Meta-bifocal
    ```

2.  Install dependencies:
    ```bash
    pip install torch torchvision numpy matplotlib pyyaml poppy astropy opencv-python tqdm
    ```

## 🏃‍♂️ Usage

### 1. Configuration
Modify `config.yaml` to set up your optical parameters, sensor specs, and training hyperparameters.

### 2. Data Preparation
The project currently supports the **SceneFlow** dataset. Ensure your data path is correctly set in `main.py` or `config.yaml`.

### 3. Training
Run the main script to start the end-to-end training. This will optimize both the Zernike coefficients (optics) and the UNet (software).

```bash
python main.py
```

The script will:
1.  Load RGB images and Depth maps.
2.  Simulate the optical blur and sensor capture process.
3.  Feed the simulated RAW image into MIMO-UNet.
4.  Compute Multi-Scale Loss against the Ground Truth.
5.  Backpropagate gradients to update both the Network weights and Optical parameters.

## 📊 Visualization
Training progress and validation results (GT vs. Simulated Input vs. Reconstructed Output) are saved in `experiments/train_run/visualizations`.

---
*This project explores the intersection of Computational Photography and Deep Learning.*
