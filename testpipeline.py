import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# 引入你的 Pipeline
from src.models.pipeline import ImageFormationPipeline

def create_synthetic_scene(H=512, W=512):
    """
    创建一个专门用于测试 EDOF 的合成场景。
    - RGB: 高频网格图 (Grid)，用于观察模糊。
    - Depth: 中间是近景 (0.8m)，四周是远景 (20m)。
    """
    # 1. 生成 RGB (高频网格)
    # 白色背景
    img_np = np.ones((H, W, 3), dtype=np.float32)
    
    # 画黑色网格线 (每 16 像素一条)
    grid_size = 16
    for i in range(0, H, grid_size):
        cv2.line(img_np, (0, i), (W, i), (0, 0, 0), 1)
    for j in range(0, W, grid_size):
        cv2.line(img_np, (j, 0), (j, H), (0, 0, 0), 1)
        
    # 在中间写个字 "NEAR"
    cv2.putText(img_np, "NEAR", (W//2 - 30, H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 1), 2)
    # 在角落写个字 "FAR"
    cv2.putText(img_np, "FAR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (1, 0, 0), 2)

    # 2. 生成 Depth Map
    # 默认全是背景 (20m)
    depth_np = np.ones((H, W), dtype=np.float32) * 20.0
    
    # 中间区域设置为近景 (0.8m) -> 对应 Channel 0 的焦点
    # 之前是 1.5m，正好处于 0.8m 和 6.0m 的中间屈光度，导致两边模糊程度一样！
    center_h, center_w = H // 2, W // 2
    size = 60
    depth_np[center_h-size:center_h+size, center_w-size:center_w+size] = 0.8

    # 转换为 Tensor
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() # [1, 3, H, W]
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float() # [1, 1, H, W]
    
    return img_tensor, depth_tensor

def test_pipeline():
    print("--- Starting Pipeline Visual Test ---")
    
    # 1. 配置 (使用我们优化后的高性能参数)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    optics_config = {
        'f_mm': 25.0,
        'f_number': 2.2,       # 大光圈，确保虚化明显
        'pixel_size_um': 3.45,
        'fov_pixels': 128,      # 64 足够看清正方形光斑
        'n_zernike': 15,
        'pupil_grid_size': 256
    }
    
    layer_config = {
        'num_layers': 30,
        'min_dist': 0.8,
        'max_dist': 20.0
    }

    # 2. 初始化 Pipeline
    pipeline = ImageFormationPipeline(
        optics_config=optics_config, 
        layer_config=layer_config
    ).to(device)
    pipeline.eval()

    # 3. 生成合成数据
    print("Generating synthetic checkerboard scene...")
    rgb, depth = create_synthetic_scene(H=512, W=512)
    rgb = rgb.to(device)
    depth = depth.to(device)

    # 4. 运行仿真
    print(f"Running simulation...")
    print(f"Focus Settings -> Near: {pipeline.d_focus_0:.2f}m, Far: {pipeline.d_focus_90:.2f}m")
    
    with torch.no_grad():
        raw, img_near, img_far = pipeline(rgb, depth)

    # 5. 可视化对比
    print("Plotting results...")
    
    # 转 Numpy 方便画图
    def to_np(t):
    # 先把数据移到 CPU 并 detach
        x = t.detach().cpu()
    
    # 如果是 4D 张量 [B, C, H, W]，通常我们需要去掉 Batch 维度
        if x.dim() == 4:
            x = x[0]  # 取 Batch 中的第一张图 -> [C, H, W]
            x = x.permute(1, 2, 0)  # 转为 numpy 图片格式 -> [H, W, C]
    
    # 最后再 squeeze。如果是单通道 [H, W, 1] 会变成 [H, W]；如果是 RGB [H, W, 3] 则保持不变
        return x.squeeze().numpy()

    img_in = to_np(rgb)
    depth_in = to_np(depth)
    out_near = to_np(img_near)
    out_far = to_np(img_far)

    plt.figure(figsize=(15, 5))

    # 子图 1: 原始深度
    plt.subplot(1, 4, 1)
    plt.title("Input Depth Map\n(Blue=Near, Yellow=Far)")
    plt.imshow(depth_in, cmap='plasma_r') # plasma_r 让近处(小数值)偏亮色，远处偏深色，或者反之，看个人喜好
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # 子图 2: 原始图像
    plt.subplot(1, 4, 2)
    plt.title("Input Sharp Image")
    plt.imshow(np.clip(img_in, 0, 1))
    plt.axis('off')

    # 子图 3: 近焦通道输出 (0度)
    # 预期：中间的 "NEAR" 应该清晰，四周的 "FAR" 和网格应该模糊
    plt.subplot(1, 4, 3)
    plt.title(f"Channel 0 (Near Focus: {pipeline.d_focus_0}m)\nTarget: Center Clear")
    plt.imshow(np.clip(out_near, 0, 1))
    plt.axis('off')

    # 子图 4: 远焦通道输出 (90度)
    # 预期：四周的 "FAR" 应该清晰，中间的 "NEAR" 应该模糊
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