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
        super().__init__()
        
        if optics_config is None: optics_config = {}
        if sensor_config is None: sensor_config = {'height': 512, 'width': 512}
        
        # === 修改点 1: 更新默认深度范围 ===
        # 配合 Dataloader: 0.5m 起始，25.0m 结束 (之后归为背景层)
        if layer_config is None:
            layer_config = {'num_layers': 12, 'min_dist': 0.5, 'max_dist': 25.0}

        self.optics = DifferentiableOptics(**optics_config)
        self.sensor = IMX250MYR_SENSOR(**sensor_config)
        
        self.num_layers = layer_config.get('num_layers', 12)
        self.min_dist = layer_config.get('min_dist', 0.5)
        self.max_dist = layer_config.get('max_dist', 25.0)
        
        # === 修改点 2: 手动指定 SceneFlow 的最佳双焦 ===
        self.d_near, self.d_far = self._calculate_bifocal_distances()
        
        # === FIX: 绑定变量名，确保 generate_psf_banks 能正确调用 ===
        self.d_focus_0 = self.d_near   # Channel 0 (Near)
        self.d_focus_90 = self.d_far   # Channel 90 (Far)
        
        print(f"[Pipeline] Initialized.")
        print(f"  - EDOF Range: {self.min_dist}m to {self.max_dist}m (+Infinity)")
        print(f"  - Focus 0 deg (Near): {self.d_focus_0:.2f} m")
        print(f"  - Focus 90 deg (Far): {self.d_focus_90:.2f} m")

    def _calculate_bifocal_distances(self):
        """
        Manual override for SceneFlow dataset.
        """
        # 近焦 0.8m
        d_near = 0.8
        # 远焦 6.0m
        d_far = 20.0
        return d_near, d_far

    def generate_psf_banks(self):
        """
        Generate PSFs for K layers.
        Order must match get_layer_masks: Index 0 (Near) -> Index K-1 (Far)
        """
        psf_list_0 = []
        psf_list_90 = []
        
        max_diopter = 1.0 / self.min_dist # Near
        min_diopter = 1.0 / self.max_dist # Far
        
        for k in range(self.num_layers):
            # Calculate center diopter for layer k
            # k=0 -> Near, k=K-1 -> Far
            norm_val = k / (self.num_layers - 1)
            diopter = max_diopter - norm_val * (max_diopter - min_diopter)
            d_obj = 1.0 / diopter
            
            psf_0, psf_90 = self.optics(d_obj, self.d_focus_0, self.d_focus_90)
            psf_list_0.append(psf_0)
            psf_list_90.append(psf_90)
            
        psf_bank_0 = torch.stack(psf_list_0)
        psf_bank_90 = torch.stack(psf_list_90)
        
        return psf_bank_0, psf_bank_90

    def forward(self, sharp_img, depth_map):
        psf_bank_0, psf_bank_90 = self.generate_psf_banks()

        if not hasattr(self,'has_saved_debug'):
            import torchvision
            import os
            os.makedirs('debug_psfs',exist_ok=True)
            
            torchvision.utils.save_image(psf_bank_0[0], 'debug_psfs/ch0_layer0_near.png', normalize=True)
            torchvision.utils.save_image(psf_bank_0[-1], 'debug_psfs/ch0_layer11_far.png', normalize=True)

            torchvision.utils.save_image(psf_bank_90[0], 'debug_psfs/ch90_layer0_near.png', normalize=True)
            torchvision.utils.save_image(psf_bank_90[-1], 'debug_psfs/ch90_layer11_far.png', normalize=True)

            print("[Pipeline] Saved debug PSF images.")
            self.has_saved_debug = True
        
        # depth_map 可能包含 900m 的值，get_layer_masks 会处理它
        layer_masks = get_layer_masks(
            depth_map, 
            num_layers=self.num_layers, 
            min_dist=self.min_dist, 
            max_dist=self.max_dist
        )
        
        img_blurred_0 = render_blurred_image_v2(sharp_img, layer_masks, psf_bank_0)
        img_blurred_90 = render_blurred_image_v2(sharp_img, layer_masks, psf_bank_90)
        
        raw_output = self.sensor(img_blurred_0, img_blurred_90)
        
        return raw_output, img_blurred_0, img_blurred_90