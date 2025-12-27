import torch
import torch.nn.functional as F
import torch.fft

def get_layer_masks(depth_map_meters,
                    num_layers=6,
                    min_dist=0.5,
                    max_dist=25.0,
                    temperature=50.0):
    """
    基于 Softmax 的可微层权重生成 (Differentiable Layering)。
    解决了原代码中使用 argmin 导致的梯度截断问题。
    
    Args:
        depth_map_meters: [B, 1, H, W] 物理深度图 (米)
        num_layers: int, 深度层数 (K)
        min_dist: float, 最近对焦平面 (米)
        max_dist: float, 最远对焦平面 (米)
        temperature: float, Softmax 温度系数. 
                     值越大越接近 One-hot (硬切分), 值越小越模糊 (软混合).
                     建议范围 10.0 - 100.0.
    
    Returns:
        soft_masks: [B, K, 1, H, W] 每一层的权重图, sum(dim=1) == 1.0
    """
    device = depth_map_meters.device
    B, _, H, W = depth_map_meters.shape
    
    # 1. 转换到 屈光度空间 (Diopter space = 1/d)
    # 物理上，焦距的变化与 1/d 线性相关，因此在 Diopter 空间线性分层最合理
    # +1e-8 防止除零
    input_diopter = 1.0 / (depth_map_meters + 1e-8)
    
    max_diopter = 1.0 / min_dist  # 近处 (e.g. 1/0.5 = 2.0 D)
    min_diopter = 1.0 / max_dist  # 远处 (e.g. 1/25 = 0.04 D)

    # 2. 生成 K 个层的中心屈光度 (Layer Centers)
    # 顺序: Index 0 = Near (Foreground), Index K-1 = Far (Background)
    # 符合 Front-to-Back 渲染顺序
    layer_indices = torch.arange(num_layers, device=device).float()
    
    # 线性插值生成中心点
    # D_k = D_max - k * (D_max - D_min) / (K-1)
    layer_centers = max_diopter - (layer_indices / (num_layers - 1)) * (max_diopter - min_diopter)
    
    # Reshape for broadcasting: [1, K, 1, 1, 1] vs [B, 1, 1, H, W]
    layer_centers = layer_centers.view(1, num_layers, 1, 1, 1)
    input_diopter_expanded = input_diopter.unsqueeze(1) # [B, 1, 1, H, W]

    # 3. 计算 L1 距离 (Distance to Layer Centers)
    dist_to_layers = torch.abs(input_diopter_expanded - layer_centers)

    # 4. Soft Assignment via Softmax (关键修改!)
    # 使用负距离进行 Softmax: 距离越近 -> 值越大
    # dist * temp 如果太大可能会导致 exp 溢出，但在 Diopter 空间通常数值较小 (<2.0)，比较安全
    soft_masks = F.softmax(-dist_to_layers * temperature, dim=1)
    
    return soft_masks

def fft_conv2d(img, psf, pad_mode='reflect'):
    """
    使用 FFT 实现的快速卷积，包含 Padding 以减少边界伪影。
    
    Args:
        img: [B, C, H, W]
        psf: [C, H_k, W_k] or [1, C, H_k, W_k]
    Returns:
        res: [B, C, H, W]
    """
    B, C, H, W = img.shape
    device = img.device
    
    # 确保 PSF 维度正确
    if psf.dim() == 3:
        psf = psf.unsqueeze(0) # [1, C, Hk, Wk]
    
    kh, kw = psf.shape[-2:]
    
    # === FIX: Handle PSF larger than image ===
    # If PSF is too large, crop it to a reasonable size
    max_kernel_size = min(H, W)  # PSF should not exceed image size
    
    if kh > max_kernel_size or kw > max_kernel_size:
        # Crop PSF from center
        crop_h = min(kh, max_kernel_size)
        crop_w = min(kw, max_kernel_size)
        
        start_h = (kh - crop_h) // 2
        start_w = (kw - crop_w) // 2
        
        psf = psf[..., start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Renormalize PSF to preserve energy
        psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        
        kh, kw = crop_h, crop_w
    
    # 1. 边界填充 (Padding)
    # 填充量为 Kernel 半径，但不超过图像尺寸的一半
    pad_h = min(kh // 2, H // 2)
    pad_w = min(kw // 2, W // 2)
    
    # 使用 reflection padding 效果通常比 zero padding 好
    img_padded = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode=pad_mode)
    
    # 新的 H, W
    H_pad, W_pad = img_padded.shape[-2:]
    
    # 2. 频域转换
    # rfft2 利用实数对称性加速
    img_freq = torch.fft.rfft2(img_padded, s=(H_pad, W_pad))
    psf_freq = torch.fft.rfft2(psf, s=(H_pad, W_pad))
    
    # 3. 频域相乘
    res_freq = img_freq * psf_freq
    
    # 4. 逆变换
    res = torch.fft.irfft2(res_freq, s=(H_pad, W_pad))
    
    # 5. 相位校正 (Circular Shift)
    roll_h = -kh // 2
    roll_w = -kw // 2
    res = torch.roll(res, shifts=(roll_h, roll_w), dims=(-2, -1))
    
    # 6. Crop 回原始尺寸
    res = res[..., pad_h:pad_h+H, pad_w:pad_w+W]
    
    return res

def render_blurred_image_v2(sharp_img, layer_masks, psf_bank):
    """
    基于层的渲染 (Layer-based Rendering) - Front-to-Back Compositing.
    支持 Soft Masks.
    包含亮度归一化修复，防止离焦区域变黑。
    """
    B, K, _, H, W = layer_masks.shape
    
    # 初始化累积图像 (Output Color) 和 剩余透过率 (Transmittance)
    output_image = torch.zeros_like(sharp_img)
    # 新增: 累积权重图，用于记录能量分布
    output_weight = torch.zeros_like(sharp_img) 
    
    transmittance = torch.ones((B, 1, H, W), device=sharp_img.device)
    
    # 循环: 从近 (k=0) 到 远 (k=K-1)
    for k in range(K):
        # 1. 获取当前层数据
        mask = layer_masks[:, k]      # [B, 1, H, W]
        psf = psf_bank[k]             # [3, Hk, Wk]

        # 2. 准备当前层的颜色贡献
        layer_color = sharp_img * mask 
        
        # 3. FFT 卷积
        # 对颜色分量进行模糊
        b_color = fft_conv2d(layer_color, psf)
        
        # 对 Alpha 通道 (Mask) 进行模糊
        # 这代表了如果图像是全白的，模糊后该层贡献了多少亮度
        b_alpha = fft_conv2d(mask.repeat(1, 3, 1, 1), psf)
        
        # 限制 alpha 范围
        b_alpha = torch.clamp(b_alpha, 0.0, 1.0)
        
        # 计算单通道 alpha 用于遮挡计算
        b_alpha_1ch = b_alpha.mean(dim=1, keepdim=True)

        # 4. Front-to-Back 合成
        # 颜色累积
        contribution = b_color * transmittance
        output_image = output_image + contribution
        
        # --- 新增: 权重累积 ---
        # 我们同时也累积"纯亮度"的贡献。
        # 如果前景散开了，b_alpha 会变小，weight_contribution 也会变小。
        # 背景因为被遮挡(mask=0)，贡献也是0。
        # 最终 output_weight 在前景区域会小于 1.0。
        weight_contribution = b_alpha * transmittance
        output_weight = output_weight + weight_contribution
        
        # 更新透过率
        transmittance = transmittance * (1.0 - b_alpha_1ch)
        
    # 5. 归一化 (关键步骤)
    # 将累积的颜色除以累积的权重，补回损失的亮度
    # +1e-6 防止除零
    final_image = output_image / (output_weight + 1e-6)
    
    return final_image