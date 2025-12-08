import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- 您的原始代码 (修正了 __init_masks 调用) ---
class IMX250MYR_SENSOR(nn.Module):
    def __init__(self, height, width, noise_sigma=0.01):
        super().__init__()
        self.height = height
        self.width = width
        self.noise_sigma = noise_sigma

        self.register_buffer('mask_0', torch.zeros(1, 1, height, width))
        self.register_buffer('mask_90', torch.zeros(1, 1, height, width))
        self.register_buffer('mask_mix', torch.zeros(1, 1, height, width))

        self.register_buffer('bayer_r', torch.zeros(1, 3, height, width))
        self.register_buffer('bayer_g', torch.zeros(1, 3, height, width))
        self.register_buffer('bayer_b', torch.zeros(1, 3, height, width))

        # 修正: 您的代码中这里是 __init_masks (双下划线), 但定义是 _init_masks
        self._init_masks() 

    def _init_masks(self):
        # polarization masks
        self.mask_0[:, :, 1::2, 1::2] = 1.0  # 0 degree (Row odd, Col odd)
        self.mask_90[:, :, 0::2, 0::2] = 1.0  # 90 degree (Row even, Col even)
        
        # 修正/优化: 45/135 degree 位于剩余位置
        # 原始代码:
        self.mask_mix[:, :, 0::2, 1::2] = 1.0  # (Row even, Col odd)
        self.mask_mix[:, :, 1::2, 0::2] = 1.0  # (Row odd, Col even)

        # Quad bayer masks
        # Red channel
        self.bayer_r[:, 0, 0::4, 0::4] = 1.0
        self.bayer_r[:, 0, 0::4, 1::4] = 1.0
        self.bayer_r[:, 0, 1::4, 0::4] = 1.0
        self.bayer_r[:, 0, 1::4, 1::4] = 1.0

        # Blue channel
        self.bayer_b[:, 2, 2::4, 2::4] = 1.0
        self.bayer_b[:, 2, 2::4, 3::4] = 1.0
        self.bayer_b[:, 2, 3::4, 2::4] = 1.0
        self.bayer_b[:, 2, 3::4, 3::4] = 1.0

        # Green channel
        self.bayer_g[:, 1, 0::4, 2::4] = 1.0
        self.bayer_g[:, 1, 0::4, 3::4] = 1.0
        self.bayer_g[:, 1, 1::4, 2::4] = 1.0
        self.bayer_g[:, 1, 1::4, 3::4] = 1.0
        self.bayer_g[:, 1, 2::4, 0::4] = 1.0
        self.bayer_g[:, 1, 2::4, 1::4] = 1.0
        self.bayer_g[:, 1, 3::4, 0::4] = 1.0
        self.bayer_g[:, 1, 3::4, 1::4] = 1.0

    def forward(self, img_0, img_90):
        img_mix = (img_0 + img_90) / 2.0

        # calculate sensor response
        # Using broadcasting to select the specific color channel at the specific bayer location
        val_90 = (img_90 * self.bayer_r).sum(dim=1, keepdim=True) + \
                 (img_90 * self.bayer_g).sum(dim=1, keepdim=True ) + \
                 (img_90 * self.bayer_b).sum(dim=1, keepdim=True )

        val_0 = (img_0 * self.bayer_r).sum(dim=1, keepdim=True) + \
                 (img_0 * self.bayer_g).sum(dim=1, keepdim=True) + \
                 (img_0 * self.bayer_b).sum(dim=1, keepdim=True)

        val_mix = (img_mix * self.bayer_r).sum(dim=1, keepdim=True) + \
                  (img_mix * self.bayer_g).sum(dim=1, keepdim=True) + \
                  (img_mix * self.bayer_b).sum(dim=1, keepdim=True)

        # combine the raw
        raw = val_0 * self.mask_0 + val_90 * self.mask_90 + val_mix * self.mask_mix

        if self.training and self.noise_sigma > 0.0:
            noise = torch.randn_like(raw) * self.noise_sigma
            raw_out = torch.clamp(raw + noise, 0.0, 1.0)
        else:
            raw_out = raw

        return raw_out

# --- 测试脚本 ---

def test_sensor_logic():
    print("=== 开始测试传感器逻辑 ===")
    
    H, W = 16, 16 # 使用较小的尺寸方便观察
    sensor = IMX250MYR_SENSOR(H, W)
    sensor.eval() # 关闭 Dropout/BatchNorm (虽然这里没有) 并停止加噪以便测试逻辑

    # 1. 形状测试
    print("1. 测试输出形状...")
    dummy_0 = torch.rand(2, 3, H, W)
    dummy_90 = torch.rand(2, 3, H, W)
    out = sensor(dummy_0, dummy_90)
    assert out.shape == (2, 1, H, W), f"Shape mismtach: Expected {(2, 1, H, W)}, got {out.shape}"
    print("   [通过] 输出形状正确。")

    # 2. 偏振掩膜测试
    # 策略：输入全白的 0度光，全黑的 90度光。
    # 预期：0度像素值为 1，90度像素值为 0，混合像素值为 0.5
    print("2. 测试偏振掩膜分布...")
    white_img = torch.ones(1, 3, H, W)
    black_img = torch.zeros(1, 3, H, W)
    
    raw_pol_test = sensor(white_img, black_img)
    
    # 验证 0度 mask 位置 (val=1.0)
    assert torch.allclose(raw_pol_test * sensor.mask_0, sensor.mask_0), "Mask 0 degree error"
    # 验证 90度 mask 位置 (val=0.0)
    assert torch.sum(raw_pol_test * sensor.mask_90) == 0, "Mask 90 degree error"
    # 验证 Mix mask 位置 (val=0.5)
    assert torch.allclose(raw_pol_test * sensor.mask_mix, sensor.mask_mix * 0.5), "Mask Mix error"
    print("   [通过] 偏振掩膜逻辑正确。")

    # 3. Quad Bayer 颜色测试
    # 策略：输入全红图像，检查 RAW 图中哪些位置有值
    print("3. 测试 Quad Bayer 颜色分布...")
    red_img = torch.zeros(1, 3, H, W); red_img[:, 0, :, :] = 1.0
    
    # 输入相同的红图给 0 和 90，消除偏振影响，只看颜色
    raw_red_test = sensor(red_img, red_img) 
    
    # 检查 Red Mask 区域是否为 1
    red_pixels = raw_red_test * sensor.bayer_r[:, 0:1, :, :] # bayer_r 是 [1,3,H,W] 广播过来
    assert torch.sum(red_pixels) == torch.sum(sensor.bayer_r[:, 0, :, :]), "Red channel pixels missing"
    
    # 检查非红区域是否为 0
    non_red_pixels = raw_red_test * (1 - sensor.bayer_r[:, 0:1, :, :])
    assert torch.sum(non_red_pixels) == 0, "Color leaked into non-red pixels"
    print("   [通过] 颜色滤光片逻辑正确。")

    # 4. 可视化
    print("4. 生成可视化图表...")
    visualize_patterns(sensor)

def visualize_patterns(sensor):
    """可视化掩膜图案"""
    plt.figure(figsize=(12, 5))

    # 1. 显示偏振模式
    # 0 deg = Red, 90 deg = Blue, Mix = Green
    pol_vis = torch.zeros(sensor.height, sensor.width, 3)
    pol_vis[:, :, 0] = sensor.mask_0.squeeze()   # Red channel for 0 deg
    pol_vis[:, :, 2] = sensor.mask_90.squeeze()  # Blue channel for 90 deg
    pol_vis[:, :, 1] = sensor.mask_mix.squeeze() # Green channel for Mix
    
    plt.subplot(1, 2, 1)
    plt.imshow(pol_vis.numpy())
    plt.title("Polarization Pattern\nR=0, B=90, G=Mix")
    plt.axis('off')

    # 2. 显示 Quad Bayer 模式
    bayer_vis = torch.zeros(sensor.height, sensor.width, 3)
    bayer_vis[:, :, 0] = sensor.bayer_r[0, 0, :, :]
    bayer_vis[:, :, 1] = sensor.bayer_g[0, 1, :, :]
    bayer_vis[:, :, 2] = sensor.bayer_b[0, 2, :, :]

    plt.subplot(1, 2, 2)
    plt.imshow(bayer_vis.numpy())
    plt.title("Quad Bayer Pattern\n(2x2 blocks)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("   [完成] 图表已生成。")

if __name__ == "__main__":
    test_sensor_logic()