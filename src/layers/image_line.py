import torch
from torch import Tensor
from typing import Tuple

def img_psf_conv_torch(img: Tensor,
                       psf: Tensor,
                       otf: Tensor=None,
                       adjoint: bool = False,
                       return_otf: bool = False):
    """
    Return:
    y(blur images):[B,K,H,W,C]
    otf:psf-->otf,as big as after padding img
    """
    added_B = False
    if img.dim() == 3:
        img = img[None, :, :, :]
        added_B = True
    B, H, W, C = img.shape

    added_K = False
    if psf.dim() == 3:
        psf = psf[None,:,:,:]
        added_K = True
    K,h,w,C_psf = psf.shape
    assert C_psf == C

    device = img.device
    fdtype = img.real.dtype if img.is_complex() else img.dtype

    H2,W2 = 2*H,2*W
    img_pad = torch.zeros((B,H2,W2,C),device=device,dtype=fdtype)
    img_pad[:,:H,:W,:] = img

    #prepare OTF
    if otf is None:
        otf = psf2otf_torch(psf.to(device=device,dtype=fdtype),H2,W2)
    if otf.dim() == 3:
        otf = otf.unsqueeze(0)

    IMG_F = torch.fft.fft2(img_pad,dim=(1,2))
    OTF_F = otf.to(device=IMG_F.device,dtype=IMG_F.dtype)
    if adjoint:
        OTF_F = torch.conj(OTF_F)

    # for broadcasting
    Y_F = IMG_F[:,None,:,:,:] * OTF_F[None,:,:,:,:]

    y_big = torch.fft.ifft2(Y_F,dim=(2,3)).real

    blurred_img = y_big[:,:,:H,:W,:]

    if added_K:
        blurred_img = blurred_img[:,0,:,:,:]
    if added_B:
        blurred_img = blurred_img[0,:,:,:,:]

    if return_otf:
        return blurred_img,otf
    return blurred_img

def psf2otf_torch(psf: Tensor, out_h: int, out_w: int):
    added_K = False
    if psf.dim() == 3:
        psf = psf[None, :, :, :]
        added_K = True

    K, H, W, C = psf.shape
    device = psf.device
    dtype = psf.dtype

    def ifftshift2d(x):
        return torch.roll(torch.roll(x, shifts=(-H // 2), dims=1),
                          shifts=(-W // 2),
                          dims=2)

    psf_shifted = ifftshift2d(psf)

    otf_pad = torch.zeros((K, out_h, out_w, C), device=device, dtype=dtype)
    otf_pad[:, :H, :W, :] = psf_shifted

    otf = torch.fft.fft2(otf_pad, dim=(1, 2))

    if added_K:
        otf = otf[0]

    return otf


def matting_torch(depthmap: Tensor,
                  n_depths: int,
                  binary: bool,
                  eps: float = 1e-8):
    """ 
    Args:
    Input:
    depthmap:
        [B,1,1,H,W] or [B,1,H,W] or [1,H,W]
    n_depths:K depth layers
    binary:True -> one-hot
    Output:
    alpha:[B,1,K,H,W]
    """
    x = depthmap
    if x.dim() == 3:
        x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif x.dim() == 4:
        if x.shape[1] != 1:
            x = x.unsqueeze(1)
        x = x.unsqueeze(2)
    elif x.dim() == 5:
        pass
    else:
        raise ValueError("depthmap should be 3D/4D/5D tensor")

    device, dtype = x.device, x.dtype

    x = torch.clamp(x, min=eps, max=1)

    d = torch.arange(1, n_depths + 1, device=device,
                     dtype=dtype).view(1, 1, -1, 1, 1)

    x_scaled = x * float(n_depths)
    diff = d - x_scaled

    alpha = torch.zeros_like(diff)

    if binary:
        logi = (diff >= 0.) & (diff < 1.)
        alpha = torch.where(logi, torch.ones_like(diff),
                            torch.zeros_like(diff))
    else:
        mask_left = (diff > -1.) & (diff <= 0.)
        alpha[mask_left] = diff[mask_left] + 1.0
        mask_right = (diff > 0.) & (diff <= 1.)
        alpha[mask_right] = 1.0
    return alpha


def depthmap_to_layer_depth_torch(depthmap: Tensor, n_depths: int,
                                  binary: False):
    layered_depth = matting_torch(depthmap, n_depths, binary=binary)
    return layered_depth


def over_op_torch(alpha: Tensor):
    one_minus = (1 - alpha).clamp(0, 1)
    T_after = torch.cumprod(one_minus, dim=2)
    ones_slice = torch.ones_like(alpha[:, :, 0:1, :, :])
    T_shift = T_after[:, :, :-1, :, :]
    T_before = torch.cat([ones_slice, T_shift], dim=2)
    return T_before


def capture_img_torch(img: Tensor, depthmap: Tensor, psfs: Tensor,
                      scene_distances: Tensor):  # generate sensor img
    occlusion = True
    eps = 1e-3
    B, H, W, C = img.shape
    K = psfs.shape[0]
    device = img.device
    dtype = img.dtype
    depthmap = depthmap.to(device=device, dtype=dtype)
    psfs = psfs.to(device=device, dtype=dtype)

    # depth mapping
    layered_alpha = matting_torch(depthmap, len(scene_distances), binary=True)
    layered_alpha_rgb = layered_alpha.repeat(1, C, 1, 1, 1)

    # mapping rgb
    img_k = img.unsqueeze(1).repeat(1, K, 1, 1, 1)
    volume = layered_alpha_rgb.permute(0, 2, 3, 4, 1) * img_k
    scale = volume.max()
    volume = volume / (scale + 1e-12)

    # conv in different layers
    blurred_volume = torch.zeros_like(volume)
    blurred_alpha_rgb = torch.zeros_like(volume)

    for k in range(K):
        vol_k = volume[:, k]
        layered_alpha_k_rgb = layered_alpha_rgb[:, :, k].permute(0, 2, 3, 1)
        psf_k = psfs[k:k + 1]

        blurred_k = img_psf_conv_torch(vol_k, psf_k)[:, 0]
        blurred_alpha_k = img_psf_conv_torch(layered_alpha_k_rgb, psf_k)[:, 0]

        blurred_volume[:, k] = blurred_k
        blurred_alpha_rgb[:, k] = blurred_alpha_k

    cumsum_alpha = torch.flip(torch.cumsum(torch.flip(layered_alpha, dims=[2]),
                                           dim=2),
                              dims=[2])
    E = torch.zeros((B, K, H, W, C), device=device, dtype=dtype)
    for k in range(K):
        ca_k = cumsum_alpha[:, 0, k]
        ca_k_c = ca_k.unsqueeze(-1).repeat(1, 1, 1, C)
        psf_k = psfs[k:k + 1]
        E_k = img_psf_conv_torch(ca_k_c, psf_k)[:, 0]
        E[:, k] = E_k

    C_tilde = blurred_volume / (E + eps)
    A_tilde = blurred_alpha_rgb / (E + eps)

    T_before = over_op_torch(A_tilde)

    captimg = (C_tilde * T_before).sum(dim=1)
    captimg = captimg * (scale + 1e-12)
    volume = volume * (scale + 1e-12)

    if not occlusion:
        sensor_stack = torch.zeros_like(volume)
        for k in range(K):
            captimg = sensor_stack.sum(dim=1)
            captimg = captimg * (scale + 1e-12)
            volume = volume * (scale + 1e-12)
            return captimg, volume

    return captimg, volume

def sensor_noise_torch(x:Tensor,a_poisson:float,b_sqrt:float,clip:Tuple[float,float] = (1e-6,1.0),poisson_max:float=100,sample_poisson:bool = False):
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
    low,high = clip

    # -- Shot Noise (Poisson) --
    if a_poisson > 0.0:
        x_clamped = x.clamp(min=low,max=poisson_max)

        if sample_poisson:
            with torch.no_grad():
                lam = (x_clamped/float(a_poisson)).to(dtype=dtype,device=device)
                counts = torch.poisson(lam)
            shot = counts*float(a_poisson)
        else:
            # Poisson(λ) ~ λ + sqrt(λ) * N(0,1)
            noise_shot = torch.randn_like(x_clamped) * torch.sqrt((x_clamped*float(a_poisson)).clamp_min(0.0))
            shot = x_clamped + noise_shot
        y = shot
    else:
        y = x

    # -- Readout Noise (Gaussian) --
    if b_sqrt > 0.0:
        y = y + torch.randn_like(y)*float(b_sqrt)

    # -- clipping --
    y = y.clamp(min=low,max=high)
    return y
    