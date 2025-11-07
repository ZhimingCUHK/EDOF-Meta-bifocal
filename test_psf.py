import numpy as np
import torch
from torch import Tensor


def define_input_fields_torch(x_mesh: Tensor, y_mesh: Tensor,
                              scene_distances: Tensor, wave_lengths: Tensor):
    """ 
    x_mesh,y_mesh:[H,W]
    scene_distances:[T]
    wave_lengths:[C]
    return :field [T,H,W,C]
    """
    dev = x_mesh.device
    #read the data
    x = x_mesh.to(device=dev, dtype=torch.float32)
    y = y_mesh.to(device=dev, dtype=torch.float32)
    z = scene_distances.to(device=dev, dtype=torch.float32)
    z = z[:, None, None, None]
    lambdas = wave_lengths.to(device=dev, dtype=torch.float32)
    lambdas = lambdas[None, None, None, :]

    #compute the radius
    r2 = x**2 + y**2
    r2 = r2[None,:,:,None]

    #compute the lambdas and broadcast
    k = (2 * torch.pi) / lambdas


    #calculate the distance
    rho = torch.sqrt(r2 + z**2)

    #calculate the phase
    phi = k * rho

    field = torch.exp(1j * phi).to(torch.complex64)

    return field


def point_source_layer_torch(x_mesh: Tensor, y_mesh: Tensor,
                             scene_distances: Tensor, wave_lengths: Tensor,
                             n_sample_depth: int, step: int):
    """ 
    Turn to the shape [K,M,M,C]
    E = exp(j*(2pi/lambda)*sqrt(x**2+y**2+z**2))
    k = n_sample_depth
    """
    dev = x_mesh.device
    # read the data
    x = x_mesh.to(device=dev, dtype=torch.float32)
    y = y_mesh.to(device=dev, dtype=torch.float32)
    z_list = scene_distances.to(device=dev, dtype=torch.float32).flatten()
    T = z_list.numel()
    k = int(n_sample_depth)
    num_batches = max(T // k, 1)
    b = step % num_batches
    start, stop = b * k, b * k + k
    z_batch = z_list[start:stop].view(-1, 1, 1, 1)

    r2 = x**2 + y**2
    r2 = r2[None, :, :, None]
    lambdas = wave_lengths.to(device=dev, dtype=torch.float32)
    lambdas = lambdas[None, None, None, :]
    k_l = (2 * torch.pi) / lambdas

    rho = torch.sqrt(r2 + z_batch**2)
    phi = k_l * rho
    output_field = torch.exp(1j * phi).to(torch.complex64)

    return output_field

def lens_layer_torch(input_field: Tensor, x_mesh: Tensor, y_mesh: Tensor,
               focal_length: float, wave_lengths: Tensor, model:str):

    dev = input_field.device
    x = x_mesh.to(device=dev,dtype=torch.float32)
    y = y_mesh.to(device=dev,dtype=torch.float32)
    r2 = x**2 + y**2
    r2 = r2[None, :, :, None]
    wave_lengths = wave_lengths.to(device=dev,dtype=torch.float32)
    k = (2 * torch.pi) / wave_lengths[None, None, None, :]
    focal_length = torch.as_tensor(float(focal_length),device=dev,dtype=torch.float32)

    if model == 'paraxial':  # 近轴近似
        phi_xy = -(r2 / (2 * focal_length))
        phi = k * phi_xy
    elif model == 'exact':
        rho = torch.sqrt(r2 + focal_length**2)
        phi_xy = (focal_length - rho)
        phi = k * phi_xy

    lens_phase = torch.exp(1j * phi).to(dtype=torch.complex64)

    output_field = lens_phase * input_field

    return output_field


def zernike_layer_height_torch(input_field: Tensor, coeffs: Tensor,
                               zernike_volume: Tensor, wave_lengths: Tensor,
                               refractive_idcs: Tensor, bound_val: float):
    """ 
    Return:
    height_map:[1,H,W,1](float64)
    output_field:[K,H,W,C](complex64)
    """
    dev = input_field.device
    K, H, W, C = input_field.shape

    coeffs = coeffs.to(dev=dev, dtype=torch.float32)
    Z = zernike_volume.to(dev=dev, dtype=torch.float32)
    wave_lengths = wave_lengths.to(dev=dev, dtype=torch.float32)
    refractive_idcs = refractive_idcs.to(dev=dev, dtype=torch.float32)

    alpha = coeffs * float(bound_val)
    height_map_hw = torch.einsum('t,thw->hw', alpha, Z)
    height_map = height_map_hw[None, :, :, None]

    k = (2.0 * torch.pi) / wave_lengths[None, None, None, :]
    n_minus1 = (refractive_idcs - 1)[None, None, None, :]
    phi = k * n_minus1 * height_map

    phase = torch.exp(1j * phi).to(torch.complex64)
    field = input_field.to(torch.complex64)
    output_field = phase * field

    return output_field,height_map


def phase_from_height_map_torch(height_map: Tensor, wave_lengths: Tensor,
                          refractive_idcs: Tensor):
    """ 
    Assume we have a x_mesh,y_mesh,which size is M X M,and height_map is still M X M
    """
    dev = height_map.device

    refractive_idcs = refractive_idcs.to(device=dev, dtype=torch.float32)
    n_minus1 = (refractive_idcs - 1)[None, None, None, :]

    lambdas = wave_lengths.to(device=dev, dtype=torch.float32)
    k = (2 * torch.pi) / lambdas[None, None, None, :]

    phi_delay = height_map * k * n_minus1

    phase_shifts = torch.exp(1j * phi_delay).to(dtype=torch.complex64)

    return phase_shifts


def aperture_layer_torch(input_field: Tensor, D: float, pixel_size: float, center=None):
    assert input_field.ndim == 4
    _, H, W, _ = input_field.shape
    dev = input_field.device
    fdtype = input_field.dtype

    yy = torch.arange(-H // 2, H - H // 2, device=dev, dtype=torch.float32)
    xx = torch.arange(-W // 2, W - W // 2, device=dev, dtype=torch.float32)
    y, x = torch.meshgrid(yy, xx, indexing="ij")

    # 
    if center is None:
        cx = torch.tensor(0.0, device=dev, dtype=torch.float32)
        cy = torch.tensor(0.0, device=dev, dtype=torch.float32)
    else:
        cx = torch.tensor(float(center[0]), device=dev, dtype=torch.float32)
        cy = torch.tensor(float(center[1]), device=dev, dtype=torch.float32)

    r = torch.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy))[None, :, :, None]

    radius_px = (D * 0.5) / float(pixel_size)
    radius_px = min(radius_px, min(H, W) / 2.0 - 1.0)
    radius = torch.tensor(radius_px, device=dev, dtype=r.dtype)

    mask = (r <= radius).to(torch.float32)
    mask = mask.to(fdtype)  

    output_field = input_field * mask
    return output_field



def fresnel_propogation_layer_torch(input_field: Tensor, wave_lengths: Tensor,
                                    distance: float, pixel_size: float):
    """ 
    Args:
    input_field:[K,H,W,C]
    wave_lengths:[C]
    distance:different distance
    pixel_size:float
    return:psf [K,H,W,C]
    """
    K, H, W, C = input_field.shape
    device = input_field.device
    fx = torch.fft.fftfreq(W, d=pixel_size).to(device,dtype=torch.float32)
    fy = torch.fft.fftfreq(H, d=pixel_size).to(device,dtype=torch.float32)
    FX, FY = torch.meshgrid(fy, fx, indexing='ij')

    FX = FX[None, :, :, None]
    FY = FY[None, :, :, None]

    squared_sum = FX**2 + FY**2
    wave_lengths = wave_lengths.to(device,dtype=torch.float32)[None, None, None, :]

    H_kernel = torch.exp(-1j * torch.pi * wave_lengths * distance *
                         squared_sum)

    # pass in the frequency space
    input_fft = torch.fft.fft2(input_field, dim=(1, 2))
    output_field = input_fft * H_kernel
    output_field = torch.fft.ifft2(output_field, dim=(1, 2))

    # get psf
    psf = torch.abs(output_field)**2
    psf = psf / (psf.sum(dim=(1, 2), keepdim=True) + 1e-8)

    return psf


if __name__ == '__main__':


    # test_bifocal_fresnel.py
    import math
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # ========== 可调参数 ==========
    H = W = 256                 # 瞳面采样数（像素）
    pixel_size = 6e-6           # 采样间距（米/像素）——与 Fresnel 的 d 参数一致
    D = 10e-3                   # 物理口径直径（米），例如 10 mm
    focal_length = 50e-3        # 理想薄透镜焦距（米），例如 50 mm
    use_lens = True             # 是否乘上理想透镜相位
    model = 'exact'          # 'paraxial' 或 'exact'

    # 物距（点光源到瞳面距离，米）；可以放多层
    scene_distances = torch.tensor([0.30], dtype=torch.float32)  # 30 cm
    # 波长（米）；可多通道
    wave_lengths = torch.tensor([550e-9], dtype=torch.float32)   # 单通道绿光

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========== 构建坐标网格（米） ==========
    # 注意：坐标以瞳面中心为原点，分辨率= pixel_size
    yy = torch.arange(-H//2, H - H//2, device=device, dtype=torch.float32) * pixel_size
    xx = torch.arange(-W//2, W - W//2, device=device, dtype=torch.float32) * pixel_size
    y_mesh, x_mesh = torch.meshgrid(yy, xx, indexing="ij")  # [H,W]

    # ========== 生成点光源在瞳面的入射复场 ==========
    # 返回 [T,H,W,C] 的复振幅
    field = define_input_fields_torch(
        x_mesh, y_mesh,
        scene_distances.to(device),
        wave_lengths.to(device)
    )  # complex64, [T,H,W,C]

    # ========== 乘理想薄透镜相位（可选） ==========
    if use_lens:
        field = lens_layer_torch(
            input_field=field, x_mesh=x_mesh, y_mesh=y_mesh,
            focal_length=focal_length, wave_lengths=wave_lengths.to(device),
            model=model
        )  # [T,H,W,C] complex64

    # ========== aperture ==========
    field = aperture_layer_torch(
        input_field=field, D=D, pixel_size=pixel_size, center=None
    )  # [T,H,W,C]

    # ========== （像距 s' 按薄透镜公式求） ==========
    # 1/f = 1/z + 1/s'  -> s' = 1 / (1/f - 1/z)
    # 这里对每个 z 求一个对应的像距；波长通道共用相同距离
    z_list = scene_distances.to(device)
    s_img = []
    for z in z_list:
        if use_lens:
            # 防止 z≈f 造成除零；若 z<f，会得到负 s'（虚像），这里仍可数值传播
            denom = (1.0 / focal_length) - (1.0 / float(z))
            s_prime = 1.0 / denom
        else:
            # no lenses，可直接传播到一个固定距离（例如焦平面处）
            s_prime = focal_length
        s_img.append(float(s_prime))
    s_img = torch.tensor(s_img, device=device, dtype=torch.float32)  # [T]

    #  PSF in every depth（也可以一次性把距离 broadcast 到 [T,1,1,1] 做掉）
    psf_list = []
    for t in range(field.shape[0]):  # T
        psf_t = fresnel_propogation_layer_torch(
            input_field=field[t:t+1, ...],                 # [1,H,W,C]
            wave_lengths=wave_lengths.to(device),          # [C]
            distance=float(s_img[t]),                      # 该深度对应像距
            pixel_size=pixel_size
        )  # [1,H,W,C]
        psf_list.append(psf_t)
    psf = torch.cat(psf_list, dim=0)  # [T,H,W,C]

    # ========== 一些检查 ==========
    with torch.no_grad():
        energy = psf.sum(dim=(1,2,3))  # 每个 T 的能量
        print("PSF energy per depth (should be ~1):", energy.detach().cpu().numpy())
        print("PSF shape:", tuple(psf.shape))  # [T,H,W,C]

    # ========== visualize（只画 T=0, C=0 的切片） ==========
    psf_vis = psf[0, :, :, 0].detach().cpu().numpy()
    psf_vis /= psf_vis.max() + 1e-8

    plt.figure(figsize=(4,4))
    plt.imshow(psf_vis, cmap='gray', origin='lower')
    plt.title(f'PSF (z={float(scene_distances[0])*1e3:.0f} mm, λ={wave_lengths[0]*1e9:.0f} nm)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
