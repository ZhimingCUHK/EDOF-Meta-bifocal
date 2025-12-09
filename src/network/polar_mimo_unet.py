import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        # The interpolate here is for processing Feature Maps and must be preserved
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)

class SCM(nn.Module):
    def __init__(self, out_plane, in_channel=3):
        super(SCM, self).__init__()
        self.in_channel = in_channel
        self.out_plane = out_plane
        self.main = nn.Sequential(
            BasicConv(in_channel, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-in_channel, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        main_out = self.main(x)
        cat_out = torch.cat([x, main_out], dim=1)
        return self.conv(cat_out)

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class PolarMIMOUNet(nn.Module):
    """
    MIMO-UNet specifically adapted for polarized/Bayer RAW input
    Input: [B, 1, H, W] RAW Data
    Output: [B, 3, H/2, W/2] RGB Data (Scale 1, 2, 3)
    """
    def __init__(self, num_res=8, in_channel=1):
        super(PolarMIMOUNet, self).__init__()

        base_channel = 32
        
        # Define multi-scale RAW input channels (calculated via PixelUnshuffle)
        # Input (Raw): 1 ch
        # Scale 1 (Input to Encoder): Unshuffle(2) -> 1 * 2*2 = 4 ch
        # Scale 2: Unshuffle(Scale1, 2) -> 4 * 2*2 = 16 ch
        # Scale 3: Unshuffle(Scale2, 2) -> 16 * 2*2 = 64 ch
        c1, c2, c3 = 4, 16, 64

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            # Modification 1: Input layer accepts 4 channels (Packed RAW)
            BasicConv(c1, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            # Modification 2: Output layer outputs 3 channels (RGB)
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        # Modification 3: SCM1 accepts deepest Scale 3 input (64 channels)
        self.SCM1 = SCM(base_channel * 4, in_channel=c3)
        
        self.FAM2 = FAM(base_channel * 2)
        # Modification 4: SCM2 accepts middle Scale 2 input (16 channels)
        self.SCM2 = SCM(base_channel * 2, in_channel=c2)

    def pixel_unshuffle(self, x, downscale_factor):
        """
        Space-to-Depth conversion for processing RAW data
        """
        b, c, h, w = x.shape
        out_channel = c * (downscale_factor ** 2)
        new_h = h // downscale_factor
        new_w = w // downscale_factor
        x = x.view(b, c, new_h, downscale_factor, new_w, downscale_factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, out_channel, new_h, new_w)
        return x

    def forward(self, x):
        # x: [Batch, 1, H, W] RAW Input
        
        # === Core modification: Generate multi-scale inputs ===
        # Use Unshuffle instead of Interpolate to protect Bayer/Polar structure
        
        # Scale 1 (Network Input): [B, 4, H/2, W/2]
        x_1 = self.pixel_unshuffle(x, 2) 
        
        # Scale 2 (Mid): [B, 16, H/4, W/4]
        x_2 = self.pixel_unshuffle(x_1, 2)
        
        # Scale 3 (Small): [B, 64, H/8, W/8]
        x_4 = self.pixel_unshuffle(x_2, 2)

        # SCM feature extraction
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        # Encoder (Main Path)
        x_ = self.feat_extract[0](x_1) # Input is x_1 (4ch)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        # AFF feature fusion (interpolate can be used here since we're in feature domain)
        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        # Decoder & Output
        # Scale 3 Output
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        # Modification 5: Remove + x_4 (Raw cannot be added to RGB)
        outputs.append(z_) 

        # Scale 2 Output
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_)

        # Scale 1 Output (Full Res RGB)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z)

        # outputs[0]: H/8 resolution RGB
        # outputs[1]: H/4 resolution RGB
        # outputs[2]: H/2 resolution RGB (final result)
        return outputs

def build_net(model_name, in_channel=1):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMO-UNetPlus":
        raise ModelError('MIMO-UNetPlus implementation for Polar Raw is not ready yet. Use PolarMIMOUNet.')
    elif model_name == "MIMO-UNet":
        return PolarMIMOUNet(in_channel=in_channel)
    elif model_name == "PolarMIMOUNet": # Explicit call
        return PolarMIMOUNet(in_channel=in_channel)
        
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNet (Polar version).')