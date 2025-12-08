import torch
import torch.nn as nn


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

        self._init_masks()

    def _init_masks(self):
        # polarization masks
        self.mask_0[:, :, 1::2, 1::2] = 1.0  # 0 degree
        self.mask_90[:, :, 0::2, 0::2] = 1.0  # 90 degree
        self.mask_mix[:, :, 0::2, 1::2] = 1.0  # 45 degree
        self.mask_mix[:, :, 1::2, 0::2] = 1.0  # 135 degree

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