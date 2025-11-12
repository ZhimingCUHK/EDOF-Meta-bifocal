import torch
import math
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple


def img_psf_conv_torch_bchw(img: Tensor, psf: Tensor):
    """ 
    Using FFT to convolve image with PSF
    Args:
    img(Tensor): [B,C,H,W] input image
    psf(Tensor): [K,C,h,w] PSF kernels
    Returns:
    convolved_img(Tensor): [B,C,H,W] convolved image
    """
    B, C, H, W = img.shape
    K, C_psf, h, w = psf.shape
    assert C == C_psf, "Channel number of img and psf must be the same"

    device = img.device
    dtype = img.dtype

    H2, W2 = 2 * H, 2 * W  # zero-padding to avoid circular convolution
    img_pad = F.pad(img, (0, W, 0, H), mode='constant', value=0)

    # calculating the otf
    otf = psf2otf_torch_bchw(psf.to(device=device, dtype=dtype), H2, W2)

    # FFT the image
    img_f = torch.fft.fft2(img_pad, dim=(-2, -1))

    # the image(after fft) times the otf
    # img_f:[B,1,C,H2,W2]
    # otf:[1,K,C,H2,W2]
    # Y_F:[B,K,C,H2,W2]
    Y_F = img_f.unsqueeze(1) * otf.unsqueeze(0)

    y_big = torch.fft.ifft(Y_F, dim=(-2, -1)).real

    blurred_img = y_big[..., :H, :W]

    return blurred_img

def psf2otf_torch_bchw(psf: Tensor, out_h: int, out_w: int):
    """ 
    (B,C,H,W)
    Make PSF [K,C,h,w] to OTF [K,C,out_h,out_w]
    The center of PSF should be at (h // 2, w // 2)
    """
    K, C, H, W = psf.shape
    device = psf.device
    dtype = psf.dtype

    # pad the psf to out_h, out_w
    pad_top = (out_h - H) // 2
    pad_bottom = out_h - H - pad_top
    pad_left = (out_w - W) // 2
    pad_right = out_w - W - pad_left

    psf_padded = F.psf(psf, (pad_left, pad_right, pad_top, pad_bottom),
                       'constant', 0)

    # fftshift the psf
    psf_shifted = torch.roll(psf_padded,
                             shifts=(-H // 2, -W // 2),
                             dims=(2, 3))

    # compute the OTF
    otf = torch.fft.fft2(psf_shifted, dim=(2, 3))

    return otf


def matting_torch_bchw(depthmap: Tensor, n_depths: int, binary: bool):
    """ 
    (B,C,H,W)
    Args:
    depthmap(Tensor):[B,1,H,W]
    n_depths(int):K the number of depths
    binary(bool):True -> one-hot
    Returns:
    layered_depth(Tensor):[B,1,K,H,W]
    """
    x = depthmap
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError("depthmap should be with shape [B,1,H,W]")
    B,_,H,W = x.shape
    device,dtype = x.device, x.dtype
    x = torch.clamp(x,min=1e-8,max=1.0)

    # (K,1,1)
    d = torch.arange(1,n_depths+1,device=device,dtype=dtype).view(-1,1,1)

    # (1,H,W)
    x_scaled = (x * float(n_depths)).squeeze(1)

    # using broadcasting to get layered depth
    diff = x_scaled.unsqueeze(0) - d.unsqueeze(1)  # (K,H,W)

    if binary:
        logi = (diff >= 0.) & (diff < 1.)
        alpha = torch.where(logi,torch.ones_like(diff),torch.zeros_like(diff))
    else:
        raise NotImplementedError("Only binary=True is implemented.")

    return alpha.unsqueeze(1) # (B,K,1,H,W)

def over_op_torch_bchw(alpha:Tensor):
    """
    (B,C,H,W)

    Args:
    alpha(Tensor):[B,K,1,H,W]
    
    Returns:
    T_before(Tensor):[B,K,1,H,W]
    """
    alpha_permuted = alpha.permute(0,2,3,4,1)

    one_minus = (1 - alpha_permuted).clamp(0, 1)
    T_after = torch.cumprod(one_minus, dim=-1)

    ones_slice = torch.ones_like(alpha_permuted[:,:,:,:,0:1])
    T_shift = T_after[:,:,:,:,:-1]
    T_before = torch.cat([ones_slice,T_shift],dim=-1)

    return T_before.permute(0,4,1,2,3)


def capture_img_torch_bchw(img: Tensor, depthmap: Tensor, psfs: Tensor):
    """
    (B,C,H,W)
    simulate the captured image by sensor
    Args:
        img:Tensor:the clean image [B,C,H,W]
        depthmap:Tensor:the depthmap [B,1,H,W]
        psfs:Tensor:the PSF kernels [K,C,h,w]
    Returns:
        Tensor:the captured image [B,C,H,W]
    """
    eps = 1e-3
    B, C, H, W = img.shape
    K = psfs.shape[0]
    device = img.device
    dtype = img.dtype

    # depth mapping [B,1,H,W] -> [B,K,1,H,W]
    layered_alpha = matting_torch_bchw(depthmap, K, binary=True)

    # rgb mapping [B,C,H,W] -> [B,K,C,H,W]
    volume = img.unsqueeze(1) * layered_alpha

    # conv in different layers
    blurred_volume = torch.zeros_like(volume)
    blurred_alpha = torch.zeros_like(layered_alpha)

    psf_alpha = psfs[:, 0:1, :, :]  # [K,1,h,w]

    for k in range(K):
        vol_k = volume[:, k, :, :]
        psf_k = psfs[k:k + 1, :, :, :]
        blurred_k = img_psf_conv_torch_bchw(vol_k, psf_k).squeeze(1)
        blurred_volume[:, k, :, :] = blurred_k

        # convolve alpha
        alpha_k = layered_alpha[:, k, :, :]
        psf_k = psf_alpha[k:k + 1, :, :, :]
        blurred_alpha_k = img_psf_conv_torch_bchw(alpha_k, psf_k).squeeze(1)
        blurred_alpha[:, k, :, :] = blurred_alpha_k

    # alpha cumulative product
    cumsum_alpha_permuted = torch.flip(torch.cumsum(torch.flip(
        layered_alpha.permute(0, 2, 3, 4, 1), dims=[-1]),
                                                    dim=-1),
                                       dims=[-1])
    cumsum_alpha = cumsum_alpha_permuted.permute(0, 5, 1, 2, 3)

    E = torch.zeros_like(blurred_volume)
    for k in range(K):
        ca_k = cumsum_alpha[:, k, :, :]
        psf_k = psf_alpha[k:k + 1, :, :, :]
        E_k = img_psf_conv_torch_bchw(ca_k, psf_k).squeeze(1)
        E[:, k, :, :] = E_k

    C_tilde = blurred_volume / (E + eps)
    A_tilde = blurred_alpha.repeat(1, 1, C, 1, 1) / (E + eps)

    T_before = over_op_torch_bchw(A_tilde[:, :, 0:1, :, :])

    captimg = (C_tilde * T_before).sum(dim=1)

    return captimg

def sensor_noise_torch(x: Tensor,
                       a_poisson: float,
                       b_sqrt: float,
                       clip: Tuple[float, float] = (1e-6, 1.0),
                       poisson_max: float = 100,
                       sample_poisson: bool = False):
    """ 
    Args:
    x:captured image by sensor
    a_poisson:float
    b_sqrt:float
    return:
    as same as x's shape(added poisson noise and readout noise)
    """
    device = x.device
    dtype = x.dtype
    low, high = clip

    # -- Shot Noise (Poisson) --
    if a_poisson > 0.0:
        x_clamped = x.clamp(min=low, max=poisson_max)

        if sample_poisson:
            with torch.no_grad():
                lam = (x_clamped / float(a_poisson)).to(dtype=dtype,
                                                        device=device)
                counts = torch.poisson(lam)
            shot = counts * float(a_poisson)
        else:
            # Poisson(λ) ~ λ + sqrt(λ) * N(0,1)
            noise_shot = torch.randn_like(x_clamped) * torch.sqrt(
                (x_clamped * float(a_poisson)).clamp_min(0.0))
            shot = x_clamped + noise_shot
        y = shot
    else:
        y = x

    # -- Readout Noise (Gaussian) --
    if b_sqrt > 0.0:
        y = y + torch.randn_like(y) * float(b_sqrt)

    # -- clipping --
    y = y.clamp(min=low, max=high)
    return y
