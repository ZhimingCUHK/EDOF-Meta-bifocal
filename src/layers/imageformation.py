import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F


def get_layer_masks(depth_map_meters,
                    num_layers=12,
                    min_dist=0.2,
                    max_dist=20.0):
    """
    Convert physical depth map to K layer Masks (based on uniform diopter slicing)
    depth_map_meters: [B, 1, H, W]
    Returns: layers_alpha [B, K, 1, H, W]
    """
    device = depth_map_meters.device

    # 1. Convert to diopter (1/d)
    diopter_map = 1.0 / (depth_map_meters + 1e-8)
    max_diopter = 1.0 / min_dist  # 5.0
    min_diopter = 1.0 / max_dist  # 0.05

    # 2. Normalize diopter to [0, K]
    # 0 represents farthest (20m), K represents nearest (0.2m)
    norm_diopter = (diopter_map - min_diopter) / (max_diopter - min_diopter)
    norm_diopter = torch.clamp(norm_diopter, 0.0, 1.0 - 1e-6)
    scaled_diopter = norm_diopter * num_layers

    # 3. Generate Mask for each layer
    # layer_indices: [1, K, 1, 1, 1]
    layer_indices = torch.arange(num_layers,
                                 device=device).view(1, num_layers, 1, 1, 1)

    # Compare: which layer the pixel belongs to
    # diff: [B, K, 1, H, W]
    diff = scaled_diopter.unsqueeze(1) - layer_indices

    # Hard Mask: Select interval [0, 1)
    mask = (diff >= 0) & (diff < 1)

    # Flip order! Ensure k=0 is nearest (foreground), k=N is farthest (background), or vice versa
    # The logic here is 0 corresponds to farthest.
    # To match "near occludes far" compositing logic, usually expect index 0 to be foreground.
    # So we flip it:
    return torch.flip(mask.float(), dims=[1])

def render_blurred_image_v2(sharp_img, layer_masks, psf_bank):
    """
    Rewritten rendering function based on image_process.py logic.
    Uses normalization and Over Operator to handle occlusion, solving background black edge problem.
    
    Args:
        sharp_img: [B, 3, H, W]
        layer_masks: [B, K, 1, H, W] (assumed order: 0=foreground/nearest, K-1=background/farthest)
        psf_bank: [K, 3, H, W]
    """
    B, K, _, H, W = layer_masks.shape
    device = sharp_img.device
    eps = 1e-6  # Small value to prevent division by zero

    # --- 1. Prepare cumulative Alpha for normalization (Cumsum Alpha) ---
    # Logic: Calculate the sum of current layer and all layers behind it.
    # In image_process.py: reverse -> cumsum -> reverse.
    # Here masks are already [near -> far], so we need to flip to [far -> near] first for cumsum
    masks_flipped = torch.flip(layer_masks, dims=[1])
    cumsum_flipped = torch.cumsum(masks_flipped, dim=1)
    cumsum_alpha = torch.flip(cumsum_flipped, dims=[1])  # [B, K, 1, H, W]

    # Collect results for each layer after processing
    norm_vol_list = []
    norm_alpha_list = []

    # --- 2. Layer-by-layer blurring and normalization ---
    for k in range(K):
        # Get current layer data
        mask = layer_masks[:, k]          # [B, 1, H, W]
        mask_cum = cumsum_alpha[:, k]     # [B, 1, H, W] (source for normalization denominator)
        psf = psf_bank[k]                 # [3, H, W]

        # Premultiply Alpha: Extract current layer color
        # Corresponds to volume = layered_depth * img in image_process.py
        volume = sharp_img * mask

        # Prepare convolution weights
        pad_h, pad_w = psf.shape[1] // 2, psf.shape[2] // 2
        psf_weight = psf.unsqueeze(1)  # [3, 1, H, W]

        # Execute blurring (convolution)
        # Corresponds to img_psf_conv in image_process.py
        # 1. Blur color volume
        b_vol = F.conv2d(volume, psf_weight, padding=(pad_h, pad_w), groups=3)
        # 2. Blur current layer Alpha
        mask_expanded = mask.repeat(1, 3, 1, 1)
        b_alpha = F.conv2d(mask_expanded, psf_weight, padding=(pad_h, pad_w), groups=3)
        # 3. Blur cumulative Alpha (for normalization)
        mask_cum_expanded = mask_cum.repeat(1, 3, 1, 1)
        b_cumsum = F.conv2d(mask_cum_expanded, psf_weight, padding=(pad_h, pad_w), groups=3)

        # Crop size (in case padding causes size change)
        if b_vol.shape[-2:] != (H, W):
            b_vol = b_vol[..., :H, :W]
            b_alpha = b_alpha[..., :H, :W]
            b_cumsum = b_cumsum[..., :H, :W]

        # --- Key step: Energy normalization ---
        # Corresponds to image_process.py: blurred_volume / (blurred_cumsum_alpha + eps)
        # This step corrects brightness dimming caused by PSF diffusion
        # Increased eps to 1e-3 to prevent gradient explosion when b_cumsum is small
        safe_eps = 1e-3
        b_vol_norm = b_vol / (b_cumsum + safe_eps)
        b_alpha_norm = b_alpha / (b_cumsum + safe_eps)

        # Clamp range
        b_alpha_norm = torch.clamp(b_alpha_norm, 0.0, 1.0)

        norm_vol_list.append(b_vol_norm)
        norm_alpha_list.append(b_alpha_norm)

    # Stack results [B, K, 3, H, W]
    stack_vol = torch.stack(norm_vol_list, dim=1)
    stack_alpha = torch.stack(norm_alpha_list, dim=1)

    # --- 3. Compositing (Over Operator) ---
    # Corresponds to over_op and reduce_sum in image_process.py

    # Calculate transmittance: Proportion of light remaining after passing through all previous layers
    # Formula: T_i = (1-a_0) * (1-a_1) * ... * (1-a_{i-1})
    # Use cumprod (cumulative product)
    one_minus_alpha = 1.0 - stack_alpha
    transmittance = torch.cumprod(one_minus_alpha, dim=1)

    # Construct weights: Layer 0 weight is 1, layer k weight is transmittance[k-1]
    ones = torch.ones_like(stack_alpha[:, :1])
    weights = torch.cat([ones, transmittance[:, :-1]], dim=1)  # [B, K, 3, H, W]

    # Weighted sum to get final image
    # Captimg = Sum( Weight_k * Layer_Color_k )
    final_image = torch.sum(weights * stack_vol * stack_alpha, dim=1)
    # Note: If stack_vol is already premultiplied by alpha (in b_vol step), may not need to multiply stack_alpha again.
    # But according to image_process.py logic, it uses over_alpha * blurred_volume in final reduce_sum.
    # And its blurred_volume is normalized color.
    # Here stack_vol is (Color * Mask)_blurred / Cumsum_blurred.
    # Usually recommended final output: torch.sum(weights * stack_vol, dim=1)
    # Because stack_vol already implicitly contains the visibility distribution of that layer.

    final_image = torch.sum(weights * stack_vol, dim=1)

    return final_image
