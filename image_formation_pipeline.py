import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import math
import poppy.zernike  # Requires: pip install poppy
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple

# ==============================================================================
# 1. Get Scene Flow Dataset
# ==============================================================================


class SceneFlowDataset(Dataset):
    """ 
    The dataset loads images and their corresponding depth maps from the Scene Flow dataset.
    Returns:
        img_gt (Tensor): Ground truth image tensor of shape (3, H, W)
        img_input (Tensor): Input image tensor of shape (3, H, W) (same as img_gt here)
        depth_map (Tensor): Depth map tensor of shape (1, H, W)
    """

    def __init__(self,
                 data_root,
                 dataset_type='train',
                 image_size=256,
                 use_random_crop=True):
        """ 
        Initialize the dataset.
        Parameters:
            data_root (str): Root directory of the Scene Flow dataset.
            dataset_type (str): 'train' or 'val' to specify the dataset split.
            image_size (int): Size of the cropped images (image_size x image_size).
            use_random_crop (bool): Whether to use random cropping or center cropping.
        """
        super().__init__()
        self.image_size = image_size
        self.use_random_crop = use_random_crop

        # define image and disparity directories based on dataset type
        if dataset_type == 'train':
            self.img_dir = os.path.join(
                data_root, 'FlyingThings3D_subset/train/image_clean/right')
            self.disp_dir = os.path.join(
                data_root, 'FlyingThings3D_subset/train/disparity/right')
        elif dataset_type == 'val':
            self.img_dir = os.path.join(
                data_root, 'FlyingThings3D_subset/val/image_clean/right')
            self.disp_dir = os.path.join(
                data_root, 'FlyingThings3D_subset/val/disparity/right')
        else:
            raise ValueError("dataset_type must be 'train' or 'val'")

        # get list of file IDs
        self.file_ids = [
            f[:-4] for f in os.listdir(self.img_dir) if f.endswith('.png')
        ]
        if not self.file_ids:
            raise RuntimeError(f'No image files found in {self.img_dir}')

        # Define image transformations
        if self.use_random_crop:
            self.transform = transforms.RandomCrop(
                (self.image_size, self.image_size))
        else:
            self.transform = transforms.CenterCrop(
                (self.image_size, self.image_size))

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        """ 
        Get and return a data item
        """
        # Get file ID
        file_id = self.file_ids[index]

        # Load image and depth map
        img_path = os.path.join(self.img_dir, file_id + '.png')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f'Failed to load image: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Convert

        # Get depth map
        disp_path = os.path.join(self.disp_dir, file_id + '.pfm')
        disparity = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        if disparity is None:
            raise IOError(f'Failed to load disparity map: {disp_path}')

        # Ensure disparity map is single channel
        if disparity.ndim == 3:
            disparity = disparity[:, :, 0]

        # Convert disparity to depth
        depthmap = disparity.astype(np.float32)
        depthmap -= depthmap.min()
        depthmap /= (depthmap.max() + 1e-8)
        depthmap = 1.0 - depthmap

        # Convert to pytorch tensor (C,H,W)
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        depthmap_tensor = torch.from_numpy(depthmap).float().unsqueeze(0)

        # Apply cropping
        stacked = torch.cat((image_tensor, depthmap_tensor), dim=0)
        stacked = torch.cat((image_tensor, depthmap_tensor), dim=0)

        try:
            stacked_cropped = self.transform(stacked)
        except ValueError as e:
            print(f'Error during cropping, file ID {file_id}: {e}')
            return (torch.zeros(3, self.image_size, self.image_size),
                    torch.zeros(3, self.image_size, self.image_size),
                    torch.zeros(1, self.image_size, self.image_size))

        img_gt = stacked_cropped[0:3, :, :]
        depth_map = stacked_cropped[3:4, :, :]

        return img_gt, img_gt.clone(), depth_map


# ==============================================================================
# Part 2: PSF Generation (from optics_fft_demo.ipynb)
# ==============================================================================

def make_xy_mesh(H:int,W:int,device=None,dtype=torch.float32):
    # pixel centers
    ys = torch.arange(H,device=device,dtype=dtype) - (H-1)/2.0
    xs = torch.arange(W,device=device,dtype=dtype) - (W-1)/2.0
    yy,xx = torch.meshgrid(ys,xs,indexing='ij')  # [H,W]

    if H > 1:
        yy = yy / ((H-1)/2.0)  # normalize to [-1,1]
    else:
        yy = torch.zeros_like(yy)
    if W > 1:
        xx = xx / ((W-1)/2.0)  # normalize to [-1,1]
    else:
        xx = torch.zeros_like(xx)
    return xx,yy  # [H,W]

def circular_pupil(xx:torch.Tensor,yy:torch.Tensor,NA:float=1.0):
    # NA is relative to the maximum radius (1.0)
    r2 = xx**2 + yy**2
    r_max = torch.sqrt(r2.max())
    if r_max == 0:
        pupil = torch.ones_like(xx,dtype=xx.dtype,device=xx.device)
    else:
        pupil = (r2 <= (NA * r_max)**2).to(dtype=xx.dtype)
    return pupil.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

def zernike_basis_with_poppy(xx:torch.Tensor,yy:torch.Tensor,nm_list):
    """ 
    Using poppy.zernike to generate Zernike basis functions.
    Parameters:
        xx: [H,W] grid x-coordinates (normalized units [-1,1])
        yy: [H,W] grid y-coordinates (normalized units [-1,1])
        nm_list: list of (n,m) tuples for Zernike polynomials
    Returns:
        basis: [len(nm_list), H, W] Zernike basis function tensor
    """
    H,W = xx.shape
    device = xx.device
    dtype = xx.dtype

    # Convert to numpy for poppy usage
    xx_np = xx.cpu().numpy()
    yy_np = yy.cpu().numpy()

    # Calculate radius and angle
    rho_np = np.sqrt(xx_np**2 + yy_np**2)
    theta_np = np.arctan2(yy_np, xx_np)

    mask_np = (rho_np <= 1.0)

    Zs_list_np = []
    for (n,m) in nm_list:

        Z_np = poppy.zernike.zernike(n, m, rho=rho_np, theta=theta_np)
        Z_np = Z_np * mask_np  # Mask areas outside the unit circle
        Z_np = np.nan_to_num(Z_np,nan=0.0)  # Replace NaN with 0
        Zs_list_np.append(Z_np)

    Z_stack_np = np.stack(Zs_list_np, axis=0)  # [num_basis, H, W]

    Z_stack_torch = torch.from_numpy(Z_stack_np).to(device=device,dtype=dtype)
    return Z_stack_torch # [num_basis, H, W]

def zernike_phase(coeffs:torch.Tensor,Z:torch.Tensor):
    """ 
    coeffs: [num_basis]
    Z: [num_basis, H, W]
    Returns:
        phase: [1,1,H,W]
    """
    assert coeffs.ndim == 1 and Z.ndim == 3 and coeffs.shape[0] == Z.shape[0]
    phi = torch.einsum('t, thw -> hw', coeffs, Z)  # [H,W]
    return phi.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

def defocus_phase(defocus_amounts:torch.Tensor,xx:torch.Tensor,yy:torch.Tensor):
    """
    defocus_amounts:[K] 
    Returns:
        phase: [K,1,H,W] 
    """
    rho2 = (xx**2 + yy**2).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return defocus_amounts.view(-1,1,1,1) * rho2  # [K,1,H,W]

def mixed_phase(phi_zernike:torch.Tensor,phi_defocus:torch.Tensor,wavelengths:torch.Tensor,scale_by_wavelength:bool=True):
    """
    phi_zernike: [1,1,H,W]
    phi_defocus: [K,1,H,W]
    wavelengths: [C]
    Returns:
        phase: [K,C,H,W]
    """
    K,_,H,W = phi_defocus.shape
    C = wavelengths.numel()
    phi = phi_zernike.expand(K,1,H,W) + phi_defocus  # [K,1,H,W]
    phi = phi.expand(K,C,H,W)  # [K,C,H,W]
    if scale_by_wavelength:
        two_pi_over_lambda = (2*math.pi / wavelengths).view(1,C,1,1)  # [1,C,1,1]
        phi = phi * two_pi_over_lambda  # [K,C,H,W]
    return phi  # [K,C,H,W]

def psf_from_phase(pupil_amp:torch.Tensor,phi:torch.Tensor,do_fftshift:bool=False):
    """ 
    pupil_amp: [1,1,H,W]
    phi: [K,C,H,W]
    Returns:
        psf: [K,C,H,W]
    """
    K,C,H,W = phi.shape
    phi = torch.nan_to_num(phi,nan=0.0,posinf=0.0,neginf=0.0)
    mask = (pupil_amp > 0).to(phi.dtype)
    phi = phi * mask  # Set phase to zero where pupil is zero
    pupil = pupil_amp.expand(K,C,H,W) # Bug fix: was expand(K,1,H,W)
    U = pupil.to(phi.dtype) * torch.exp(1j * phi)  # [K,C,H,W]

    U = torch.fft.fft2(U, dim=(-2,-1),norm='ortho')  # [K,C,H,W]
    if do_fftshift:
        U = torch.fft.fftshift(U, dim=(-2,-1))
    psf = (U.conj() * U).real  # [K,C,H,W]
    psf = psf / psf.sum(dim=(-2,-1),keepdim=True).clamp_min(1e-12)  # Normalize energy
    return psf  # [K,C,H,W]

def zernike_defocus_psf(H:int,W:int,
                        nm_list,
                        coeffs,
                        defocus_amounts:torch.Tensor,
                        wavelengths:torch.Tensor,
                        NA:float=1.0,
                        scale_by_wavelength:bool=True,
                        do_fftshift:bool=True,
                        device=None,
                        dtype=torch.float32):
    """
    Returns:
        psf: [K,C,H,W]
    """
    # Grid
    xx,yy = make_xy_mesh(H,W,device=device,dtype=dtype)  # [H,W]

    # Pupil
    pupil_amp = circular_pupil(xx,yy,NA=NA)  # [1,1,H,W]

    # Zernike basis functions
    Z = zernike_basis_with_poppy(xx,yy,nm_list)  # [num_basis,H,W]

    # Zernike phase
    coeffs_tensor = torch.as_tensor(coeffs,device=device,dtype=dtype)
    phi_zernike = zernike_phase(coeffs_tensor,Z)  # [1,1,H,W]

    # Defocus phase
    defocus_tensor = torch.as_tensor(defocus_amounts,device=device,dtype=dtype)
    phi_defocus = defocus_phase(defocus_tensor,xx,yy)  # [K,1,H,W]

    # Mixed phase
    wave_tensor = torch.as_tensor(wavelengths,device=device,dtype=dtype)
    phi = mixed_phase(phi_zernike,phi_defocus,wave_tensor,scale_by_wavelength=scale_by_wavelength)  # [K,C,H,W]

    # PSF
    psf = psf_from_phase(pupil_amp,phi,do_fftshift=do_fftshift)  # [K,C,H,W]

    return psf  # [K,C,H,W]


# ==============================================================================
# Part 3: Image Formation Pipeline (from image_line.py)
# ==============================================================================

def img_psf_conv_torch_bchw(img, psf):
    B,C,H,W = img.shape
    K,Cp,h,w = psf.shape
    assert C==Cp
    # Delta kernel shortcut
    if (h,w)==(1,1) and torch.allclose(psf[...,0,0],
                                       torch.ones_like(psf[...,0,0])):
        return img.unsqueeze(1).expand(B,K,C,H,W).contiguous()
    H2,W2 = 2*H, 2*W
    IMG = torch.fft.fft2(F.pad(img, (0,W,0,H)), dim=(-2,-1))  # [B,C,H2,W2]
    OTF = psf2otf_torch_bchw(psf.to(img.dtype), H2, W2)       # [K,C,H2,W2]
    Y   = IMG.unsqueeze(1) * OTF.unsqueeze(0)                 # [B,K,C,H2,W2]
    y   = torch.fft.ifft2(Y, dim=(-2,-1)).real
    return y[..., :H, :W]                                     # [B,K,C,H,W]


def psf2otf_torch_bchw(psf: Tensor, out_h: int, out_w: int):
    """ 
    (K,C,H,W)
    Convert PSF [K,C,h,w] to OTF [K,C,out_h,out_w]
    PSF center should be at (h // 2, w // 2)
    """
    K, C, h, w = psf.shape
    device = psf.device
    dtype = psf.dtype

    Z = torch.zeros((K, C, out_h, out_w), device=device, dtype=dtype)
    Z[:, :, :h, :w] = psf
    Z = torch.roll(Z, shifts=(-(h // 2), -(w // 2)), dims=(-2, -1))
    otf = torch.fft.fft2(Z, dim=(-2, -1))
    return otf

def matting_torch_bchw(depthmap: Tensor, n_depths: int, binary: bool):
    """ 
    (B,C,H,W)
    Parameters:
    depthmap(Tensor):[B,1,H,W]
    n_depths(int):K, number of depth layers
    binary(bool):True -> one-hot encoding
    Returns:
    layered_depth(Tensor):[B,K,1,H,W]
    """
    x = depthmap
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError("depthmap should have shape [B,1,H,W]")
    B,_,H,W = x.shape
    device,dtype = x.device, x.dtype
    x = torch.clamp(x,min=1e-8,max=1.0 - 1e-6)

    # (K,1,1)
    d = torch.arange(1,n_depths+1,device=device,dtype=dtype).view(-1,1,1)

    # (B,H,W)
    x_scaled = (x * float(n_depths)).squeeze(1)

    # Use broadcasting to get layered depth
    # x_scaled.unsqueeze(0) -> [1,B,H,W]
    # d.unsqueeze(1)        -> [K,1,1,1]
    diff = x_scaled.unsqueeze(0) - d.unsqueeze(1)  # (K,B,H,W)

    if binary:
        # Bug fix: Adjust binary matting logic, select layer K
        # where d-1 <= x_scaled < d
        # or k-1 <= x_scaled < k (if k is 1-indexed)
        # x_scaled - d >= -1 and x_scaled - d < 0
        logi = (diff >= -1.0) & (diff < 0.0)
        alpha = torch.where(logi,torch.ones_like(diff),torch.zeros_like(diff))
    else:
        raise NotImplementedError("Only binary=True is implemented")

    # Bug fix: Return shape should be [B,K,1,H,W]
    # alpha is [K,B,H,W]
    # permute(1,0,2,3) -> [B,K,H,W]
    # unsqueeze(2)     -> [B,K,1,H,W]
    return alpha.permute(1, 0, 2, 3).unsqueeze(2)

def normalize_psf(psf, eps=1e-8):
    # psf: [K,C,h,w]
    s = psf.sum(dim=(-1,-2), keepdim=True)
    # Avoid division by zero
    s = s.clamp_min(eps)
    return psf / s


def over_op_torch_bchw(alpha:Tensor):
    """
    (B,C,H,W)
    Alpha compositing 'over' operator.
    Parameters:
    alpha(Tensor):[B,K,1,H,W]
    
    Returns:
    T_before(Tensor):[B,K,1,H,W] (transmittance up to layer k)
    """
    # [B, 1, H, W, K]
    alpha_permuted = alpha.permute(0,2,3,4,1)

    one_minus = (1 - alpha_permuted).clamp(0, 1)
    # T_after: transmittance after passing through layer k
    T_after = torch.cumprod(one_minus, dim=-1)

    # T_before (transmittance before hitting layer k) is the shift of T_after
    ones_slice = torch.ones_like(alpha_permuted[:,:,:,:,0:1])
    T_shift = T_after[:,:,:,:,:-1]
    T_before = torch.cat([ones_slice,T_shift],dim=-1) # [B, 1, H, W, K]

    # [B, K, 1, H, W]
    return T_before.permute(0,4,1,2,3)


def capture_img_torch_bchw(img: Tensor, depthmap: Tensor, psfs: Tensor):
    """
    (B,C,H,W)
    Simulate captured image
    Parameters:
        img:Tensor: Clear image [B,C,H,W]
        depthmap:Tensor: Depth map [B,1,H,W]
        psfs:Tensor: PSF kernels [K,C,h,w]
    Returns:
        Tensor: Captured image [B,C,H,W]
    """
    eps = 1e-3
    B, C, H, W = img.shape
    K, C_psf, h, w = psfs.shape
    assert C == C_psf, "Image and PSF channels do not match"
    device = img.device
    dtype = img.dtype

    # Depth mapping [B,1,H,W] -> [B,K,1,H,W]
    layered_alpha = matting_torch_bchw(depthmap, K, binary=True)

    # RGB mapping [B,C,H,W] -> [B,K,C,H,W]
    volume = img.unsqueeze(1) * layered_alpha

    # Convolution for different layers
    blurred_volume = torch.zeros_like(volume, device=device, dtype=dtype)
    blurred_alpha = torch.zeros_like(layered_alpha, device=device, dtype=dtype)

    psf_alpha = psfs[:, 0:1, :, :]  # [K,1,h,w] (use first channel of PSF for alpha)

    for k in range(K):
        vol_k = volume[:, k, :, :]  # [B,C,H,W]
        psf_k = psfs[k:k + 1, :, :, :] # [1,C,h,w]
        # img_psf_conv_torch_bchw returns [B, 1, C, H, W]
        blurred_k = img_psf_conv_torch_bchw(vol_k, psf_k).squeeze(1) # [B,C,H,W]
        blurred_volume[:, k, :, :] = blurred_k

        # Convolve alpha
        alpha_k = layered_alpha[:, k, :, :] # [B,1,H,W]
        psf_a_k = psf_alpha[k:k + 1, :, :, :] # [1,1,h,w]
        blurred_alpha_k = img_psf_conv_torch_bchw(alpha_k, psf_a_k).squeeze(1) # [B,1,H,W]
        blurred_alpha[:, k, :, :] = blurred_alpha_k

    # --- Alpha Compositing ---
    # This implements the "Divisive Flow" model
    # From "Depth-of-Field Rendering with Multilayer Physically-Based..." (Martín et al. 2021)

    # cumsum_alpha_permuted: [B, 1, H, W, K]
    cumsum_alpha_permuted = torch.flip(torch.cumsum(torch.flip(
        layered_alpha.permute(0, 2, 3, 4, 1), dims=[-1]),
                                                    dim=-1),
                                       dims=[-1])

    # Bug fix: Tensor has 5 dimensions, index 5 is out of range. Should be 4.
    # cumsum_alpha: [B, K, 1, H, W]
    cumsum_alpha = cumsum_alpha_permuted.permute(0, 4, 1, 2, 3)

    E = torch.zeros_like(blurred_volume, device=device, dtype=dtype) # [B,K,C,H,W]
    for k in range(K):
        ca_k = cumsum_alpha[:, k, :, :] # [B,1,H,W]
        psf_a_k = psf_alpha[k:k + 1, :, :, :] # [1,1,h,w]
        E_k = img_psf_conv_torch_bchw(ca_k, psf_a_k).squeeze(1) # [B,1,H,W]
        E[:, k, :, :] = E_k.repeat(1,C,1,1) # Repeat for C channels

    C_tilde = blurred_volume / (E + eps)
    A_tilde = blurred_alpha.repeat(1, 1, C, 1, 1) / (E + eps)

    # T_before: [B, K, 1, H, W]
    T_before = over_op_torch_bchw(A_tilde[:, :, 0:1, :, :]) # Transmittance

    # Repeat T_before for all color channels
    T_before_rgb = T_before.repeat(1, 1, C, 1, 1)

    # Sum along K layers
    captimg = (C_tilde * T_before_rgb).sum(dim=1) # [B,C,H,W]

    return captimg

def sensor_noise_torch(x: Tensor,
                       a_poisson: float,
                       b_sqrt: float,
                       clip: Tuple[float, float] = (1e-6, 1.0),
                       poisson_max: float = 100,
                       sample_poisson: bool = False):
    """ 
    Parameters:
    x: Captured image
    a_poisson:float
    b_sqrt:float
    Returns:
    Same shape as x (with Poisson noise and read noise added)
    """
    device = x.device
    dtype = x.dtype
    low, high = clip

    # -- Shot noise (Poisson) --
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

    # -- Read noise (Gaussian) --
    if b_sqrt > 0.0:
        y = y + torch.randn_like(y) * float(b_sqrt)

    # -- Clipping --
    y = y.clamp(min=low, max=high)
    return y


# ==============================================================================
# 4. Main pipeline implementation
# ==============================================================================

if __name__ == '__main__':

    # pipeline parameters

    data_root = '/home/LionelZ/Data/'

    image_size = 256
    batch_size = 1  # use for demo
    K = 10  # the number of depth layers / PSFs
    H_psf = W_psf = 10  # PSF size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Using the device: {device}")
    print(f"Loading Scene Flow dataset from {data_root} ...")

    try:
        # --- get data batch ---
        train_dataset = SceneFlowDataset(
            data_root=data_root,
            dataset_type='train',
            image_size=image_size,
            use_random_crop=False)  # Center crop for demo

        if len(train_dataset) == 0:
            raise RuntimeError('Training dataset is empty.')

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)  # num_workers=0 for simplicity

        print(f'Dataset loaded with {len(train_dataset)} samples.')
        print('Fetching the first batch...')

        # Get single batch
        img_gt_batch, _, depth_map_batch = next(iter(train_loader))

        img_gt = img_gt_batch.to(device=device, dtype=dtype)
        depth_map = depth_map_batch.to(device=device, dtype=dtype)

        B, C, H, W = img_gt.shape
        print(
            f'Data batch loaded: Image [B={B},C={C},H={H},W={W}], Depth [B={B},1,H={H},W={W}]'
        )

        # --- 3. Generate PSFs ---
        print(f'Generating {K} PSFs of size {H_psf}x{W_psf}...')
        nm_list = [
            (2, 2),  # Astigmatism
            (2, -2),  # Oblique Astigmatism
            (3, 1),  # Coma
            (3, -1),  # Oblique Coma
            (4, 0),  # Spherical Aberration
            (3, 3),  # Trefoil
            (3, -3),  # Oblique Trefoil
            (4, 2),  # Secondary Astigmatism
            (4, -2),  # ...
            (5, 1),  # Secondary Coma
        ]  # Aberrations
        coeffs = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                  0.05]  # Zernike coefficients

        # K defocus levels from -100 to 100
        defocus = torch.linspace(-100,
                                 100,
                                 steps=K,
                                 device=device,
                                 dtype=dtype)

        # Wavelengths (R, G, B) - to simulate achromatic (or chromatic) PSF
        # Bug fix: PSF code generates [K, 1, H, W], but image is [B, 3, H, W].
        # We will use a single wavelength (monochrome) and repeat it.
        wavelengths = torch.tensor([550e-9], device=device,
                                   dtype=dtype)  # Green

        psfs_mono = zernike_defocus_psf(
            H_psf,
            W_psf,
            nm_list,
            coeffs,
            defocus,
            wavelengths,
            NA=1,
            scale_by_wavelength=False,
            do_fftshift=True,
            device=device,
            dtype=dtype)  # Shape: [K, 1, H_psf, W_psf]

        psfs_mono = normalize_psf(psfs_mono)  # Normalize PSF

        # Bug fix: Repeat monochrome PSF for 3 RGB channels
        psfs = psfs_mono.repeat(1, C, 1, 1)  # Shape: [K, 3, H_psf, W_psf]
        print(f'PSF generated. Shape: {psfs.shape}')

        # --- 4. Render blurred image ---
        print('Rendering blurred image (simulating capture)...')
        # This function applies matting, layered convolution, and alpha compositing
        blurred_img = capture_img_torch_bchw(img_gt, depth_map, psfs)
        print('Rendering complete.')

        # --- 5. Add sensor noise ---
        print('Adding sensor noise...')
        noisy_blurred_img = sensor_noise_torch(
            blurred_img,
            a_poisson=0.01,  # Shot noise level
            b_sqrt=0.02)  # Read noise level
        print('Noise added.')

        # --- 6. Visualization ---
        print('Displaying results...')

        # Move data to CPU and convert to numpy for matplotlib
        img_np = img_gt[0].permute(1, 2, 0).cpu().numpy()
        depth_np = depth_map[0].squeeze().cpu().numpy()
        blurred_np = noisy_blurred_img[0].permute(1, 2, 0).cpu().numpy()

        # Clip values for display (noise may exceed [0,1] range)
        blurred_np_clipped = np.clip(blurred_np, 0, 1)

        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.title('Ground Truth Image')
        plt.imshow(img_np)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Depth Map')
        plt.imshow(depth_np, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Captured Image (Blurred + Noise)')
        plt.imshow(blurred_np_clipped)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"\n--- Error ---")
        print(f"Dataset not found: {e}")
        print(f"Please check the 'data_root' path ({data_root})")
        print(
            "and ensure the dataset is located in the 'FlyingThings3D_subset' subdirectory.\n"
        )
    except ImportError as e:
        print(f"\n--- Error ---")
        print(f"Required library not found: {e}")
        print(
            "Please ensure 'poppy' (pip install poppy), 'opencv-python' and 'matplotlib' are installed.\n"
        )
    except Exception as e:
        print(f"\n--- Unexpected Error ---")
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':

    # pipeline parameters

    
    data_root = '/home/LionelZ/Data/'

    image_size = 256
    batch_size = 1   # use for demo
    K = 10           # the number of depth layers / PSFs
    H_psf = W_psf = 31 # PSF size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Using the device: {device}")
    print(f"Loading Scene Flow dataset from {data_root} ...")

    try:
        # --- get data batch ---
        train_dataset = SceneFlowDataset(data_root=data_root,
                                         dataset_type='train',
                                         image_size=image_size,
                                         use_random_crop=False) # Center crop for demo

        if len(train_dataset) == 0:
            raise RuntimeError('Training dataset is empty.')

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0) # num_workers=0 for simplicity

        print(f'Dataset loaded with {len(train_dataset)} samples.')
        print('Fetching the first batch...')

        # Get single batch
        img_gt_batch, _, depth_map_batch = next(iter(train_loader))

        img_gt = img_gt_batch.to(device=device, dtype=dtype)
        depth_map = depth_map_batch.to(device=device, dtype=dtype)

        B, C, H, W = img_gt.shape
        print(f'Data batch loaded: Image [B={B},C={C},H={H},W={W}], Depth [B={B},1,H={H},W={W}]')

        # --- 3. Generate PSFs ---
        print(f'Generating {K} PSFs of size {H_psf}x{W_psf}...')
        nm_list = [
        (2, 2),   # Astigmatism
        (2, -2),  # Oblique Astigmatism
        (3, 1),   # Coma
        (3, -1),  # Oblique Coma
        (4, 0),   # Spherical Aberration
        (3, 3),   # Trefoil
        (3, -3),  # Oblique Trefoil
        (4, 2),   # Secondary Astigmatism
        (4, -2),  # ...
        (5, 1),   # Secondary Coma
    ]   # Aberrations
        coeffs = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]            # Zernike coefficients 

        # K defocus levels from -100 to 100
        defocus = torch.linspace(-100,100, steps=K, device=device, dtype=dtype)

        # Wavelengths (R, G, B) - to simulate achromatic (or chromatic) PSF
        # Bug fix: PSF code generates [K, 1, H, W], but image is [B, 3, H, W].
        # We will use a single wavelength (monochrome) and repeat it.
        wavelengths = torch.tensor([550e-9], device=device, dtype=dtype) # Green

        psfs_mono = zernike_defocus_psf(
            H_psf, W_psf, nm_list, coeffs, defocus, wavelengths,
            NA=1, scale_by_wavelength=False, do_fftshift=True,
            device=device, dtype=dtype
        ) # Shape: [K, 1, H_psf, W_psf]

        psfs_mono = normalize_psf(psfs_mono)  # Normalize PSF

        # Bug fix: Repeat monochrome PSF for 3 RGB channels
        psfs = psfs_mono.repeat(1, C, 1, 1) # Shape: [K, 3, H_psf, W_psf]
        print(f'PSF generated. Shape: {psfs.shape}')

        # --- 4. Render blurred image ---
        print('Rendering blurred image (simulating capture)...')
        # This function applies matting, layered convolution, and alpha compositing
        blurred_img = capture_img_torch_bchw(img_gt, depth_map, psfs)
        print('Rendering complete.')

        # --- 5. Add sensor noise ---
        print('Adding sensor noise...')
        noisy_blurred_img = sensor_noise_torch(blurred_img,
                                               a_poisson=0.01, # Shot noise level
                                               b_sqrt=0.02)   # Read noise level
        print('Noise added.')

        # --- 6. Visualization ---
        print('Displaying results...')

        # Move data to CPU and convert to numpy for matplotlib
        img_np = img_gt[0].permute(1, 2, 0).cpu().numpy()
        depth_np = depth_map[0].squeeze().cpu().numpy()
        blurred_np = noisy_blurred_img[0].permute(1, 2, 0).cpu().numpy()

        # Clip values for display (noise may exceed [0,1] range)
        blurred_np_clipped = np.clip(blurred_np, 0, 1)

        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.title('Ground Truth Image')
        plt.imshow(img_np)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Depth Map')
        plt.imshow(depth_np, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Captured Image (Blurred + Noise)')
        plt.imshow(blurred_np_clipped)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"\n--- Error ---")
        print(f"Dataset not found: {e}")
        print(f"Please check the 'data_root' path ({data_root})")
        print("and ensure the dataset is located in the 'FlyingThings3D_subset' subdirectory.\n")
    except ImportError as e:
        print(f"\n--- Error ---")
        print(f"Required library not found: {e}")
        print("Please ensure 'poppy' (pip install poppy), 'opencv-python' and 'matplotlib' are installed.\n")
    except Exception as e:
        print(f"\n--- Unexpected Error ---")
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()