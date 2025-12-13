import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# Import your Pipeline
from src.models.pipeline import ImageFormationPipeline

def create_synthetic_scene(H=512, W=512):
    """
    Create a synthetic scene specifically for testing EDOF.
    - RGB: High-frequency grid pattern for observing blur.
    - Depth: Center is near (0.8m), surroundings are far (20m).
    """
    # 1. Generate RGB (high-frequency grid)
    # White background
    img_np = np.ones((H, W, 3), dtype=np.float32)
    
    # Draw black grid lines (every 16 pixels)
    grid_size = 16
    for i in range(0, H, grid_size):
        cv2.line(img_np, (0, i), (W, i), (0, 0, 0), 1)
    for j in range(0, W, grid_size):
        cv2.line(img_np, (j, 0), (j, H), (0, 0, 0), 1)
        
    # Write "NEAR" in the center
    cv2.putText(img_np, "NEAR", (W//2 - 30, H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 1), 2)
    # Write "FAR" in the corner
    cv2.putText(img_np, "FAR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (1, 0, 0), 2)

    # 2. Generate Depth Map
    # Default all to background (20m)
    depth_np = np.ones((H, W), dtype=np.float32) * 20.0
    
    # Set center area to near (0.8m) -> matches Channel 0 focus
    # Previously was 1.5m, exactly between 0.8m and 6.0m diopter, causing equal blur on both sides!
    center_h, center_w = H // 2, W // 2
    size = 60
    depth_np[center_h-size:center_h+size, center_w-size:center_w+size] = 0.8

    # Convert to Tensor
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() # [1, 3, H, W]
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float() # [1, 1, H, W]
    
    return img_tensor, depth_tensor

def test_pipeline():
    print("--- Starting Pipeline Visual Test ---")
    
    # 1. Configuration (using our optimized high-performance parameters)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    optics_config = {
        'f_mm': 25.0,
        'f_number': 2.2,       # Large aperture, ensure obvious bokeh
        'pixel_size_um': 3.45,
        'fov_pixels': 512,      # 64 is enough to see square bokeh clearly
        'n_zernike': 15,
        'pupil_grid_size': 1024,
        'oversample_factor': 4
    }
    
    layer_config = {
        'num_layers': 10,
        'min_dist': 1.5,
        'max_dist': 4.0
    }

    # 2. Initialize Pipeline
    pipeline = ImageFormationPipeline(
        optics_config=optics_config, 
        layer_config=layer_config
    ).to(device)
    pipeline.eval()

    # 3. Generate synthetic data
    print("Generating synthetic checkerboard scene...")
    rgb, depth = create_synthetic_scene(H=512, W=512)
    rgb = rgb.to(device)
    depth = depth.to(device)

    # 4. Run simulation
    print(f"Running simulation...")
    print(f"Focus Settings -> Near: {pipeline.d_focus_0:.2f}m, Far: {pipeline.d_focus_90:.2f}m")
    
    with torch.no_grad():
        raw, img_near, img_far = pipeline(rgb, depth)

    # 5. Visualization comparison
    print("Plotting results...")
    
    # Convert to Numpy for plotting
    def to_np(t):
        # First move data to CPU and detach
        x = t.detach().cpu()
    
        # If 4D tensor [B, C, H, W], we need to remove Batch dimension
        if x.dim() == 4:
            x = x[0]  # Take first image from batch -> [C, H, W]
            x = x.permute(1, 2, 0)  # Convert to numpy image format -> [H, W, C]
    
        # Finally squeeze. If single channel [H, W, 1] becomes [H, W]; if RGB [H, W, 3] stays unchanged
        return x.squeeze().numpy()

    img_in = to_np(rgb)
    depth_in = to_np(depth)
    out_near = to_np(img_near)
    out_far = to_np(img_far)

    plt.figure(figsize=(15, 5))

    # Subplot 1: Original depth
    plt.subplot(1, 4, 1)
    plt.title("Input Depth Map\n(Blue=Near, Yellow=Far)")
    plt.imshow(depth_in, cmap='plasma_r') # plasma_r makes near (small values) bright, far dark, or vice versa
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # Subplot 2: Original image
    plt.subplot(1, 4, 2)
    plt.title("Input Sharp Image")
    plt.imshow(np.clip(img_in, 0, 1))
    plt.axis('off')

    # Subplot 3: Near focus channel output (0 degree)
    # Expected: Center "NEAR" should be sharp, surrounding "FAR" and grid should be blurred
    plt.subplot(1, 4, 3)
    plt.title(f"Channel 0 (Near Focus: {pipeline.d_focus_0}m)\nTarget: Center Clear")
    plt.imshow(np.clip(out_near, 0, 1))
    plt.axis('off')

    # Subplot 4: Far focus channel output (90 degree)
    # Expected: Surrounding "FAR" should be sharp, center "NEAR" should be blurred
    plt.subplot(1, 4, 4)
    plt.title(f"Channel 90 (Far Focus: {pipeline.d_focus_90}m)\nTarget: Edge Clear")
    plt.imshow(np.clip(out_far, 0, 1))
    plt.axis('off')

    plt.tight_layout()
    save_path = 'test_pipeline_result.png'
    plt.savefig(save_path)
    print(f"Test finished! Result saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    test_pipeline()