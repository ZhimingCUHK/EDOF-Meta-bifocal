import torch
import torch.nn.functional as F
import torch.fft

def get_layer_masks(depth_map_meters,
                    num_layers=12,
                    min_dist=0.5,
                    max_dist=25.0):
    """
    Robust Layering using Nearest Neighbor (Argmin).
    Handles background (depth > max_dist) correctly by assigning it to the last layer.
    
    Order: Index 0 = Nearest (Foreground), Index K-1 = Farthest (Background)
    """
    device = depth_map_meters.device
    
    # 1. 转换到屈光度空间 (Diopter = 1/d)
    # 物理规律：距离越近，屈光度越大；距离越远，屈光度越小。
    input_diopter = 1.0 / (depth_map_meters + 1e-8)
    
    max_diopter = 1.0 / min_dist  # 对应近处边界 (e.g. 2.0 D)
    min_diopter = 1.0 / max_dist  # 对应远处边界 (e.g. 0.04 D)

    # 2. 生成 K 个层的中心屈光度 (Layer Centers)
    # 线性插值生成每个层的中心屈光度值
    # k=0 -> Near (Max Diopter), k=K-1 -> Far (Min Diopter)
    layer_indices = torch.arange(num_layers, device=device).float()
    
    # Formula: D_k = D_max - (k / (K-1)) * (D_max - D_min)
    layer_centers = max_diopter - (layer_indices / (num_layers - 1)) * (max_diopter - min_diopter)
    
    # Reshape for broadcasting: [1, K, 1, 1, 1] vs [B, 1, 1, H, W]
    layer_centers = layer_centers.view(1, num_layers, 1, 1, 1)
    input_diopter_expanded = input_diopter.unsqueeze(1) # [B, 1, 1, H, W]

    # 3. Hard Assignment via Argmin (Nearest Neighbor)
    # 计算输入像素与所有层中心的距离，找到最近的层
    # 这能自动处理超远距离 (e.g. 900m -> D~0 -> Closest to D_min -> Index K-1)
    dist_to_layers = torch.abs(input_diopter_expanded - layer_centers)
    mask_indices = torch.argmin(dist_to_layers, dim=1, keepdim=True) # [B, 1, 1, H, W]

    # 4. 生成 One-hot Mask
    masks = torch.zeros_like(dist_to_layers)
    masks.scatter_(1, mask_indices, 1.0)
    
    return masks

def fft_conv2d(img, psf):
    """
    使用 FFT 实现卷积，大幅加速大核计算。
    Args:
        img: [B, C, H, W]
        psf: [C, H_k, W_k] 
    Returns:
        res: [B, C, H, W]
    """
    B, C, H, W = img.shape
    device = img.device
    
    # PSF 尺寸
    kh, kw = psf.shape[-2:]
    
    # 1. 预处理 PSF (维度对齐)
    # psf: [C, H_k, W_k] -> [1, C, H_k, W_k]
    if psf.dim() == 3:
        psf = psf.unsqueeze(0)
    
    # 2. 计算 Padding 后的 FFT 尺寸
    # 为了避免循环卷积带来的边缘伪影，理论上应该 Pad 到 H+kh-1。
    # 但在 Deep Optics 训练中，为了速度和显存，通常直接 Pad 到 H, W (Circular Convolution)。
    # 只要边缘不是核心区域，这带来的误差可以忽略。
    fft_h, fft_w = H, W

    # 3. 频域转换 (RFFT 利用实数对称性加速)
    # s=(H, W) 会自动对输入进行 Padding (右侧补0)
    img_freq = torch.fft.rfft2(img, s=(fft_h, fft_w)) # [B, C, H, W/2+1]
    psf_freq = torch.fft.rfft2(psf, s=(fft_h, fft_w)) # [1, C, H, W/2+1]
    
    # 4. 频域卷积 (点乘)
    res_freq = img_freq * psf_freq
    
    # 5. 逆变换回空域
    res = torch.fft.irfft2(res_freq, s=(fft_h, fft_w))
    
    # 6. 相位校正 (Circular Shift)
    # 因为 rfft2 默认是以左上角 (0,0) 为原点进行 Padding 的。
    # 而我们的 PSF 能量中心在 (kh/2, kw/2)。
    # 直接卷积会导致结果向右下偏移 (kh/2, kw/2)。
    # 我们需要把结果 Roll 回来，让中心对齐。
    roll_h = -kh // 2
    roll_w = -kw // 2
    res = torch.roll(res, shifts=(roll_h, roll_w), dims=(-2, -1))
    
    # 确保尺寸严格匹配 (以防 fft size 变化)
    res = res[..., :H, :W]
    
    return res

def render_blurred_image_v2(sharp_img, layer_masks, psf_bank):
    """
    基于 FFT 卷积的分层渲染函数 (Standard Front-to-Back Compositing)。
    修复了之前版本中多重 Alpha 乘法和归一化错误的问题。
    
    Args:
        sharp_img: [B, 3, H, W] - 原始清晰图像
        layer_masks: [B, K, 1, H, W] - 每一层的 Mask (One-hot)
        psf_bank: [K, 3, Hk, Wk] - 每一层的 PSF
        
    Returns:
        final_image: [B, 3, H, W] - 渲染后的模糊图像
    """
    B, K, _, H, W = layer_masks.shape
    
    # 初始化累积图像和透射率
    # output_image: 最终图像
    # transmittance: 剩余光线透过率 (初始为 1.0，即全透)
    output_image = torch.zeros_like(sharp_img)
    transmittance = torch.ones((B, 1, H, W), device=sharp_img.device)
    
    # 循环：从近 (k=0) 到远 (k=K-1)
    for k in range(K):
        # 1. 获取当前层数据
        mask = layer_masks[:, k]      # [B, 1, H, W]
        psf = psf_bank[k]             # [3, Hk, Wk]

        # 2. 准备卷积输入 (Premultiplied Color)
        # 这一层原本的颜色贡献 = 图像 * Mask
        layer_color = sharp_img * mask 
        
        # 3. FFT 卷积
        # 模糊后的颜色 (Blurred Premultiplied Color)
        # 模糊后的 Alpha (Blurred Alpha)
        b_color = fft_conv2d(layer_color, psf)
        b_alpha = fft_conv2d(mask.repeat(1, 3, 1, 1), psf)
        
        # 限制 Alpha 范围，防止数值误差导致 > 1
        b_alpha = torch.clamp(b_alpha, 0.0, 1.0)

        # 4. Front-to-Back 合成
        # 当前层贡献 = 模糊颜色 * 当前剩余透射率
        output_image = output_image + b_color * transmittance
        
        # 更新透射率
        # 光线穿过当前层后，被遮挡了 b_alpha 部分
        # T_new = T_old * (1 - alpha)
        transmittance = transmittance * (1.0 - b_alpha)
        
    return output_image