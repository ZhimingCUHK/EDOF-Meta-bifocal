import torch
import torch.nn as nn
from src.layers.diffoptics import DifferentiableOptics
from src.layers.imageformation import get_layer_masks, render_blurred_image_v2
from src.layers.polarsensor import IMX250MYR_SENSOR

class ImageFormationPipeline(nn.Module):
    def __init__(self, 
                 optics_config=None, 
                 sensor_config=None, 
                 layer_config=None):
        """
        End-to-end Image Formation Pipeline for EDOF-Meta-Bifocal system.
        
        Args:
            optics_config (dict): Parameters for DifferentiableOptics
            sensor_config (dict): Parameters for IMX250MYR_SENSOR
            layer_config (dict): Parameters for depth layering (num_layers, min_dist, max_dist)
        """
        super().__init__()
        
        # Default configurations
        if optics_config is None:
            optics_config = {}
        if sensor_config is None:
            sensor_config = {'height': 512, 'width': 512}
        if layer_config is None:
            layer_config = {'num_layers': 12, 'min_dist': 0.2, 'max_dist': 20.0}

        # 1. Initialize Optics Layer
        self.optics = DifferentiableOptics(**optics_config)
        
        # 2. Initialize Sensor Layer
        self.sensor = IMX250MYR_SENSOR(**sensor_config)
        
        # 3. Layer Settings
        self.num_layers = layer_config.get('num_layers', 12)
        self.min_dist = layer_config.get('min_dist', 0.2)
        self.max_dist = layer_config.get('max_dist', 20.0)
        
        # 4. Auto-calculate Bifocal Focus Distances
        # Strategy: 
        #   - 0 deg polarization -> Near Focus (Layer 1, 2nd nearest)
        #   - 90 deg polarization -> Far Focus (Layer N-2, 2nd farthest)
        self.d_focus_0, self.d_focus_90 = self._calculate_bifocal_distances()
        
        print(f"[Pipeline] Initialized.")
        print(f"  - Focus 0 deg (Near): {self.d_focus_0:.4f} m")
        print(f"  - Focus 90 deg (Far): {self.d_focus_90:.4f} m")

    def _calculate_bifocal_distances(self):
        """
        Auto-calculate focus distances aligned with layer centers.
        """
        min_diopter = 1.0 / self.max_dist
        max_diopter = 1.0 / self.min_dist
        
        # Near Focus: Layer 1 (2nd nearest)
        # Original index logic: k_orig = num_layers - 1 - k_output
        # We want output index k=1 -> k_orig = num_layers - 2
        k_near = self.num_layers - 2
        norm_val_near = (k_near + 0.5) / self.num_layers
        diopter_near = min_diopter + norm_val_near * (max_diopter - min_diopter)
        d_near = 1.0 / diopter_near
        
        # Far Focus: Layer N-2 (2nd farthest)
        # We want output index k=num_layers-2 -> k_orig = 1
        k_far = 1
        norm_val_far = (k_far + 0.5) / self.num_layers
        diopter_far = min_diopter + norm_val_far * (max_diopter - min_diopter)
        d_far = 1.0 / diopter_far
        
        return d_near, d_far

    def generate_psf_banks(self):
        """
        Generate PSF banks for all layers for both polarizations.
        Returns:
            psf_bank_0: [K, 3, H_psf, W_psf]
            psf_bank_90: [K, 3, H_psf, W_psf]
        """
        psf_list_0 = []
        psf_list_90 = []
        
        min_diopter = 1.0 / self.max_dist
        max_diopter = 1.0 / self.min_dist
        
        # Iterate through layers to generate PSFs
        # Note: The loop order must match get_layer_masks output (Nearest -> Farthest)
        for k in range(self.num_layers):
            # Calculate object distance for current layer center
            # get_layer_masks returns [Foreground(Near) -> Background(Far)]
            # So k=0 is Nearest.
            # In diopter space (linear), High Diopter is Near.
            # We map k=0 -> High Diopter (Index N-1 in linear space)
            k_orig = self.num_layers - 1 - k
            
            norm_val = (k_orig + 0.5) / self.num_layers
            diopter = min_diopter + norm_val * (max_diopter - min_diopter)
            d_obj = 1.0 / diopter
            
            # Generate PSF pair for this depth
            # d_obj is scalar here, but optics expects tensor or float
            # We pass float, optics handles it.
            psf_0, psf_90 = self.optics(d_obj, self.d_focus_0, self.d_focus_90)
            
            psf_list_0.append(psf_0)
            psf_list_90.append(psf_90)
            
        psf_bank_0 = torch.stack(psf_list_0)   # [K, 3, H, W]
        psf_bank_90 = torch.stack(psf_list_90) # [K, 3, H, W]
        
        return psf_bank_0, psf_bank_90

    def forward(self, sharp_img, depth_map):
        """
        Args:
            sharp_img: [B, 3, H, W] RGB image (0-1)
            depth_map: [B, 1, H, W] Depth map in meters
        Returns:
            raw_sensor_output: [B, 1, H, W] Mosaic RAW image
            img_blurred_0: [B, 3, H, W] Simulated 0-deg image
            img_blurred_90: [B, 3, H, W] Simulated 90-deg image
        """
        # 1. Generate PSF Banks (Differentiable)
        psf_bank_0, psf_bank_90 = self.generate_psf_banks()
        
        # 2. Calculate Layer Masks
        layer_masks = get_layer_masks(
            depth_map, 
            num_layers=self.num_layers, 
            min_dist=self.min_dist, 
            max_dist=self.max_dist
        )
        
        # 3. Render Blurred Images (Layered Rendering)
        img_blurred_0 = render_blurred_image_v2(sharp_img, layer_masks, psf_bank_0)
        img_blurred_90 = render_blurred_image_v2(sharp_img, layer_masks, psf_bank_90)
        
        # 4. Simulate Sensor (Polarization + Mosaic + Noise)
        raw_output = self.sensor(img_blurred_0, img_blurred_90)
        
        return raw_output, img_blurred_0, img_blurred_90
