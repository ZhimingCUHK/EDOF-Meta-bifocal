from tabnanny import check
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.layers.optics import DifferentiableOptics  # Assuming you saved the modified class to this file


def test_defocus_sweep():
    # ================= Configuration Parameters =================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # Simulate a 35mm f/2.0 lens
    f_mm = 25.0
    f_number = 2.0
    pixel_size_um = 3.45  # Common industrial camera pixel size
    fov_pixels = 512  # Output PSF size (pixels)
    pupil_grid_size = 1024  # Pupil sampling grid (higher = more accurate)
    oversample_factor = 4  # Oversampling factor (anti-aliasing)
    n_zernike = 10  # Zernike polynomial order

    # Instantiate model
    optics = DifferentiableOptics(
        f_mm=f_mm,
        f_number=f_number,
        pixel_size_um=pixel_size_um,
        fov_pixels=fov_pixels,
        n_zernike=n_zernike,
        pupil_grid_size=pupil_grid_size,
        oversample_factor=oversample_factor).to(device)

    # ================= Test Scenario =================
    # Set camera fixed focus at different distances
    focus_dist_0_channel = 0.25
    focus_dist_90_channel = 2.5

    test_depths = [2.5, 2.05, 1.6, 1.15, 0.7, 0.25]

    results_0 = []
    results_90 = []

    print("\n--- Starting Sweep ---")
    print(f"Focus Distance at 0_channel : {focus_dist_0_channel} m")
    print(f"Focus Distance at 90_channel: {focus_dist_90_channel} m\n")

    with torch.no_grad():

        for d_obj_val in test_depths:
            # Prepare input Tensor
            d_obj = torch.tensor([d_obj_val], device=device).float()
            d_focus_0 = torch.tensor([focus_dist_0_channel],
                                     device=device).float()
            d_focus_90 = torch.tensor([focus_dist_90_channel],
                                      device=device).float()

            # Forward pass
            psf_0, psf_90 = optics(d_obj, d_focus_0, d_focus_90)
            res_0_rgb = psf_0.squeeze(0).permute(1, 2, 0).cpu().numpy()
            res_90_rgb = psf_90.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Store results (Green Channel index=1 for display)
            # psf shape is (Batch, 3, H, W)

            results_0.append(res_0_rgb)
            results_90.append(res_90_rgb)

            # Calculate peak energy and check energy conservation
            peak = psf_0.max().item()
            total_energy = psf_0.sum().item()
            print(
                f"Dist: {d_obj_val:.3f}m | Peak: {peak:.6f} | Sum: {total_energy:.4f}"
            )

    # ================= Visualization =================
    plot_psfs_rgb(test_depths, results_0, results_90, fov_pixels)


def plot_psfs_rgb(depths, psfs_0, psfs_90, fov_pixels):
    """
    Plot RGB PSF images and R/G/B channel cross-section profiles
    """
    n = len(depths)
    # Increase figure height for better detail observation
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9), constrained_layout=True)

    center = fov_pixels // 2

    for i in range(n):
        # Get data (H, W, 3)
        img_0 = psfs_0[i]
        img_90 = psfs_90[i]

        # --- [Modification 2] Normalized display ---
        # PSF values are very small, direct imshow may appear all black.
        # We normalize each image to 0-1 by its max value to see structure and color ratios
        vis_0 = img_0 / img_0.max()
        vis_90 = img_90 / img_90.max()

        # 1. First row: Channel 0 RGB
        axes[0, i].imshow(vis_0)  # No need for cmap='gray'
        axes[0, i].set_title(f"D_obj = {depths[i]}m\n(Chan 0 RGB)")
        axes[0, i].axis('off')

        # 2. Second row: Channel 90 RGB
        axes[1, i].imshow(vis_90)
        axes[1, i].set_title(f"(Chan 90 RGB)")
        axes[1, i].axis('off')

        # 3. Third row: Cross-section profiles (plot R, G, B separately)
        # Extract middle row: shape (W, 3)
        profile = img_0[center, :, :]

        # --- [Modification 3] Plot by channel ---
        # Red channel (Index 0)
        axes[2, i].plot(profile[:, 0], color='red', label='R', linewidth=1)
        # Green channel (Index 1)
        axes[2, i].plot(profile[:, 1], color='green', label='G', linewidth=1)
        # Blue channel (Index 2)
        axes[2, i].plot(profile[:, 2], color='blue', label='B', linewidth=1)

        axes[2, i].set_title("RGB Cross-section")
        axes[2, i].grid(True, alpha=0.3)

        # Only show legend in first column to avoid clutter
        if i == 0:
            axes[2, i].legend()

    plt.suptitle("RGB PSF Analysis: Chromatic Aberration & Defocus",
                 fontsize=16)
    plt.show()


def check_psf_health(psf_tensor):
    """
    Check if the PSF tensor is valid
    """
    energy = psf_tensor.sum().item()
    print(f"PSF total energy: {energy:.6f}")
    if energy < 3:
        print("Warning: PSF energy is suspiciously low!")

    edge_val = psf_tensor[0,:,:].sum() + psf_tensor[-1,:,:].sum() + \
                + psf_tensor[:,0,:].sum() + psf_tensor[:,-1.:].sum()
    if edge_val > 1e-3:
        print("Warning: PSF has significant energy at the edges!")
    else:
        print("PSF edge energy check passed.")


if __name__ == "__main__":
    test_defocus_sweep()