import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- 关键修改：添加项目根目录到 sys.path 以便导入 src 模块 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入你提供的模块
from src.data.dataloader import SceneFlowDataset
# 确保 optics.py, imageformation.py 在同一目录
from src.layers.optics import DifferentiableOptics
from src.layers.imageformation import get_layer_masks, render_blurred_image_v2

# --- 全局配置 ---
CONFIG = {
    # 路径配置
    "data_root": "/home/LionelZ/Data",  # <--- 修改为实际路径
    "batch_size": 1, 
    "image_size": (256, 256),          

    # 光学参数 (必须与 optics.py 匹配)
    "f_mm": 25.0,
    "f_number": 2.4,
    "pixel_size_um": 3.45,             
    "fov_pixels": 512,                 # 原始 PSF 视场
    "n_zernike": 10,
    "pupil_grid_size": 1024,
    
    # 渲染参数
    "num_layers": 10,                   
    "min_depth": 0.2,                  
    "max_depth": 2.0,                  
    "focus_dist_0": 0.3,
    "focus_dist_90": 1.2,                 
    "layer_temp": 50,                
}

class BlurGenerationPipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 初始化光学模型
        # 这是一个 nn.Module，包含可学习参数 (Zernike coeffs)
        self.optics = DifferentiableOptics(
            f_mm=config["f_mm"],
            f_number=config["f_number"],
            pixel_size_um=config["pixel_size_um"],
            fov_pixels=config["fov_pixels"],
            n_zernike=config["n_zernike"],
            pupil_grid_size=config["pupil_grid_size"]
        )

    def generate_psf_bank(self, num_layers, min_depth, max_depth, focus_dist_0, focus_dist_90):
        """
        预计算每一层的 PSF (PSF Bank)
        【关键修改】：这里严格调用 self.optics.forward
        """
        device = self.optics.zernike_0.device
        
        # 1. 计算每一层的物理深度中心 (Depth Centers)
        # 逻辑与 imageformation.py 保持一致 (Diopter 空间插值)
        max_diopter = 1.0 / min_depth
        min_diopter = 1.0 / max_depth
        
        layer_indices = torch.arange(num_layers, device=device).float()
        layer_diopters = max_diopter - (layer_indices / (num_layers - 1)) * (max_diopter - min_diopter)
        layer_depths = 1.0 / layer_diopters  # [K]
        
        # 2. 准备对焦距离输入
        focus_dist_t_0 = torch.tensor(self.config["focus_dist_0"], device=device).float()
        focus_dist_t_90 = torch.tensor(self.config["focus_dist_90"], device=device).float()
        
        psf_bank_0 = []
        psf_bank_90 = []
        
        # 3. 逐层前向传播
        for d in layer_depths:
            # 准备 d_obj (Batch=1)
            # optics.forward 需要 d_obj 为 Tensor 或 float
            d_obj = d.view(1) 
            
            # ---  optics.forward ---
            # 它返回 (psf_0, psf_90)
            # psf_0 的形状通常是 [3, H_down, W_down] (因为内部做了 sample_psf type=0)
            psf_0, psf_90 = self.optics(d_obj, focus_dist_t_0, focus_dist_t_90)
            
            # 我们选用 psf_0 作为主 PSF (如果你需要双目/相位模拟，可以用 psf_90)
            # optics.forward 输出如果带 Batch 维度需要 squeeze
            if psf_0.dim() == 4: # [B, 3, H, W]
                psf_bank_0.append(psf_0.squeeze(0))
            else:
                psf_bank_0.append(psf_0)

            if psf_90.dim() == 4: # [B, 3, H, W]
                psf_bank_90.append(psf_90.squeeze(0))
            else:
                psf_bank_90.append(psf_90)
            
        psf_bank_0 = torch.stack(psf_bank_0) # [K, 3, Hk, Wk]
        psf_bank_90 = torch.stack(psf_bank_90) # [K, 3, Hk, Wk]
        return psf_bank_0, psf_bank_90

    def forward(self, sharp_image, depth_map):
        """
        Args:
            sharp_image: [B, 3, H, W] (0-1)
            depth_map: [B, 1, H, W] (Meters)
        """
        # 1. 生成层权重 Mask
        layer_masks = get_layer_masks(
            depth_map,
            num_layers=self.config["num_layers"],
            min_dist=self.config["min_depth"],
            max_dist=self.config["max_depth"],
            temperature=self.config["layer_temp"]
        )
        
        # 2. 生成 PSF Bank (严格通过 optics.forward)
        psf_bank_0, psf_bank_90 = self.generate_psf_bank(
            self.config["num_layers"], 
            self.config["min_depth"], 
            self.config["max_depth"],
            self.config["focus_dist_0"],
            self.config["focus_dist_90"]
        )
        
        # 3. 渲染
        blurred_image_0 = render_blurred_image_v2(
            sharp_image, 
            layer_masks, 
            psf_bank_0
        )

        blurred_image_90 = render_blurred_image_v2(
            sharp_image, 
            layer_masks, 
            psf_bank_90
        )
        
        return blurred_image_0, blurred_image_90, layer_masks, psf_bank_0, psf_bank_90

# --- 测试脚本 ---
def run_demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 1. 准备数据 (添加了 Dummy Data 回退机制)
    sharp_img = None
    depth_map = None

    try:
        dataset = SceneFlowDataset(
            data_root=CONFIG["data_root"],
            dataset_type='train',
            image_size=CONFIG["image_size"],
            use_random_crop=True
        )
        if len(dataset) > 0:
            sample = dataset[0]
            sharp_img = sample['image'].unsqueeze(0).to(device)  # [1, 3, H, W]
            depth_map = sample['depthmap'].unsqueeze(0).to(device) # [1, 1, H, W]
            print("Loaded data from SceneFlowDataset.")
        else:
            print("Dataset is empty.")
    except Exception as e:
        print(f"Dataset Init Failed or path not found: {e}")

    # 如果数据加载失败，使用随机生成的测试数据
    if sharp_img is None:
        print("Using dummy data (Random Image + Gradient Depth) for testing...")
        H, W = CONFIG["image_size"]
        # 随机图像
        sharp_img = torch.rand(1, 3, H, W).to(device)
        # 渐变深度图 (从 min_depth 到 max_depth)
        y, x = torch.meshgrid(
            torch.linspace(0, 1, H), 
            torch.linspace(0, 1, W), 
            indexing='ij'
        )
        # 创建一个从左到右深度变化的图，范围覆盖 min_depth 到 max_depth
        min_d = CONFIG["min_depth"]
        max_d = CONFIG["max_depth"]
        depth_val = min_d + (max_d - min_d) * x
        depth_map = depth_val.unsqueeze(0).unsqueeze(0).to(device)
    
    # 2. 初始化 Pipeline
    pipeline = BlurGenerationPipeline(CONFIG).to(device)
    
    # 3. 前向传播
    # 这里我们模拟一次反向传播的需求，所以不加 torch.no_grad()
    # (仅演示，实际 inference 时可加)
    blurred_img_0, blurred_img_90, masks, psfs_0, psfs_90 = pipeline(sharp_img, depth_map)

    print("Forward Pass Successful.")
    print(f"Sharp Image: {sharp_img.shape}")
    print(f"PSF Bank 0: {psfs_0.shape}")
    print(f"Blurred Image 0: {blurred_img_0.shape}")
    print(f"Blurred Image 90: {blurred_img_90.shape}")

    # --- 可视化 ---
    sharp_np = sharp_img[0].permute(1, 2, 0).detach().cpu().numpy()
    blurred_np_0 = blurred_img_0[0].permute(1, 2, 0).detach().cpu().numpy()
    blurred_np_90 = blurred_img_90[0].permute(1, 2, 0).detach().cpu().numpy()
    depth_np = depth_map[0, 0].detach().cpu().numpy()
    
    blurred_np_0 = np.clip(blurred_np_0, 0, 1)
    blurred_np_90 = np.clip(blurred_np_90, 0, 1)

    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 5, 1)
    plt.title("Sharp Image")
    plt.imshow(sharp_np)
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.title("Depth Map")
    plt.imshow(depth_np, cmap='plasma')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.title(f"Blurred 0 deg\n(Focus @ {CONFIG['focus_dist_0']}m)")
    plt.imshow(blurred_np_0)
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.title(f"Blurred 90 deg\n(Focus @ {CONFIG['focus_dist_90']}m)")
    plt.imshow(blurred_np_90)
    plt.axis('off')

    plt.subplot(1, 5, 5)
    # 显示第0层 PSF (RGB 混合)
    # 确保索引不越界
    if psfs_0.shape[0] > 0:
        # [3, H, W] -> [H, W, 3]
        psf_display = psfs_0[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # 简单的归一化以便显示 (避免全黑或过曝)
        if psf_display.max() > 0:
            psf_display = psf_display / psf_display.max()
            
        # Gamma 校正让暗部细节更明显 (可选)
        psf_display = np.power(np.clip(psf_display, 0, 1), 1/2.2)
        
        plt.title(f"PSF 0 Layer 0 (RGB)\nShape: {psf_display.shape[:2]}")
        plt.imshow(psf_display)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()