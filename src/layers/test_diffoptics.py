import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 尝试导入您的模块，自动处理可能的类名拼写问题
try:
    # 假设您已经修正了拼写
    from diffoptics import DifferentiableOptics as DiffOptics
    print("导入成功: DifferentiableOptics")
except ImportError:
    try:
        # 如果还是旧的拼写
        from diffoptics import DifferentableOptics as DiffOptics
        print("导入成功: DifferentableOptics (注意：建议修正类名拼写)")
    except ImportError:
        print("错误: 找不到 diffoptics.py，请确保文件在当前目录下。")
        exit()

def test_psf_generation():
    print("\n=== 开始测试 PSF 生成 ===")
    
    # 1. 初始化模型
    # 假设这是一个 f/2.0 的镜头，焦距 25mm，像元 3.45um
    model = DiffOptics(
        f_mm=25.0, 
        f_number=2.0, 
        pixel_size_um=3.45, 
        fov_pixels=64,  # 输出 64x64 的 PSF以便观察细节
        n_zernike=15
    )
    
    # 将模型移至 GPU (如果可用)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"运行设备: {device}")

    # 2. 准备输入数据
    # 场景 1: 完全对焦 (Should look sharp)
    # 0度光路: 物体在 1.0m，对焦也在 1.0m
    # 90度光路: 物体在 1.0m，对焦在 0.5m (故意离焦)
    d_obj = torch.tensor([1.0], device=device)
    focus_0 = 1.0
    focus_90 = 0.5
    
    # 3. 前向传播
    psf_0, psf_90 = model(d_obj, focus_0, focus_90)
    
    # 4. 验证输出形状
    print(f"\n[检查形状]")
    print(f"PSF 0 (Focus) Shape: {psf_0.shape}")   # 期望: [3, 64, 64] 或 [1, 3, 64, 64]
    print(f"PSF 90 (Defocus) Shape: {psf_90.shape}")

    # 5. 验证能量守恒 (Sum should be close to 1.0)
    energy_0 = psf_0.sum(dim=(-1, -2))
    energy_90 = psf_90.sum(dim=(-1, -2))
    print(f"\n[检查能量守恒 (Sum ~ 1.0)]")
    print(f"PSF 0 Energy (RGB): {energy_0.detach().cpu().numpy()}") 
    print(f"PSF 90 Energy (RGB): {energy_90.detach().cpu().numpy()}")

    # 6. 可视化对比
    # 将 Tensor 转为 Numpy 用于绘图
    p0_np = psf_0.detach().cpu().squeeze().permute(1, 2, 0).numpy() # [H, W, 3]
    p90_np = psf_90.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    
    # 简单的归一化以便显示 (虽然已经是 Normalized，但为了显示亮度)
    p0_vis = p0_np / p0_np.max()
    p90_vis = p90_np / p90_np.max()

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("0-deg PSF (In Focus)\nSharp Dot")
    plt.imshow(p0_vis)
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("90-deg PSF (Defocused)\nBlurry Disk")
    plt.imshow(p90_vis)
    plt.colorbar()
    
    plt.savefig("test_psf_output.png")
    print("\n[可视化]")
    print("已保存对比图到 test_psf_output.png，请查看。")
    plt.close()

def test_differentiability():
    print("\n=== 开始测试可微分性 (Gradient Check) ===")
    
    # 注意：这里 fov_pixels=32，所以 center=16
    model = DiffOptics(fov_pixels=32, n_zernike=15)
    
    # 确保参数需要梯度
    print(f"Zernike_0 requires grad: {model.zernike_0.requires_grad}")
    
    d_obj = torch.tensor([1.0])
    
    # 前向传播
    psf_0, _ = model(d_obj, 1.0, 0.5)
    
    # [修正] Loss 计算
    # psf_0 shape is [1, 3, 32, 32]
    # 我们需要索引 [:, :, center, center]
    center = psf_0.shape[-1] // 2
    loss = -1 * psf_0[:, :, center, center].sum() # 修正了索引错误
    
    print(f"Initial Loss: {loss.item()}")
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 检查梯度
    if model.zernike_0.grad is not None:
        grad_norm = model.zernike_0.grad.norm().item()
        print(f"Gradient calculated successfully! Norm: {grad_norm:.6f}")
    else:
        print("错误: Gradient is None.")

if __name__ == "__main__":
    test_psf_generation()
    test_differentiability()