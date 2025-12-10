import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import poppy

class DifferentiableOptics(nn.Module):
    def __init__(self,
                 f_mm=25.0,
                 f_number=2.0,
                 pixel_size_um=3.45,
                 fov_pixels=32,
                 n_zernike=15,
                 pupil_grid_size=256):
        super().__init__()

        self.f_mm = f_mm
        self.f_number = f_number
        self.pixel_size = pixel_size_um * 1e-6
        self.fov_pixels = fov_pixels
        self.pupil_grid_size = pupil_grid_size
        self.n_zernike = n_zernike
        self.pupil_diam = (f_mm * 1e-3) / f_number

        # Learnable Zernike coefficients
        # 我们定义全部参数，但在 forward 中会屏蔽前几项
        self.zernike_0 = nn.Parameter(torch.zeros(n_zernike))
        self.zernike_90 = nn.Parameter(torch.zeros(n_zernike))

        # === 核心修改 1: 混合初始化 (Hybrid Initialization) ===
        # 方案 2 (CPM/Coma) + 方案 4 (Complementary Astigmatism)
        # 目的：强制 0度/90度 通道在 纹理(XY) 和 深度(Z) 上都互补
        with torch.no_grad():
            self.zernike_0.zero_()
            self.zernike_90.zero_()
            
            # 强度系数 (Empirical values)
            # 恢复为 0.3 以引入像差差异，解决"模糊程度一样"的问题
            val_astig = 0.3 * 1e-6
            val_coma  = 0.3 * 1e-6

            # --- Channel 0 (0 deg) ---
            # 垂直特征增强
            # Index 5 (Noll 6): Vertical Astigmatism
            # Index 6 (Noll 7): Vertical Coma
            self.zernike_0[5] = val_astig
            self.zernike_0[6] = val_coma

            # --- Channel 90 (90 deg) ---
            # 水平特征增强 (正交)
            # Index 5 (Noll 6): Negative -> Horizontal Astigmatism
            # Index 7 (Noll 8): Horizontal Coma
            self.zernike_90[5] = -val_astig
            self.zernike_90[7] = val_coma
            
        print("[Optics] Initialized: Hybrid Strategy (Frozen Defocus, Comp Astigmatism + Coma).")

        # Pre-compute basis
        basis_numpy, aperture_numpy = self._precompute_zernike_basis_with_poppy(n_zernike, pupil_grid_size)
        self.register_buffer('zernike_basis', torch.from_numpy(basis_numpy).float())
        self.register_buffer('aperture', torch.from_numpy(aperture_numpy).float())
        self.register_buffer('wavelengths', torch.tensor([630e-9, 550e-9, 470e-9]).view(3, 1, 1))

    def _precompute_zernike_basis_with_poppy(self, n_modes, pupil_grid_size):
        x_lin = np.linspace(-1, 1, pupil_grid_size)
        y_lin = np.linspace(-1, 1, pupil_grid_size)
        y, x = np.meshgrid(y_lin, x_lin, indexing='ij')
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        aperture = (r <= 1.0).astype(np.float32)

        basis = []
        for i in range(1, n_modes + 1):
            z = poppy.zernike.zernike1(i, rho=r, theta=theta, outside=0.0)
            basis.append(z)
        return np.stack(basis), aperture

    def calc_defocus_wfe(self, d_obj, d_focus):
        if isinstance(d_obj, torch.Tensor):
            inv_d_obj = 1.0 / (d_obj + 1e-8)
        else:
            inv_d_obj = 0.0 if d_obj == float('inf') else 1.0 / d_obj
        inv_d_focus = 1.0 / d_focus
        f_m = self.f_mm * 1e-3
        w_pv = (f_m**2) / (8.0 * (self.f_number**2)) * (inv_d_focus - inv_d_obj)
        return w_pv

    def _compute_psf(self, coeffs):
        # 1. OPD
        if coeffs.dim() == 1:
            opd = torch.einsum('n,nij->ij', coeffs, self.zernike_basis)
            opd = opd.unsqueeze(0)
        else:
            opd = torch.einsum('bn,nij->bij', coeffs, self.zernike_basis)
        
        # 2. Phase
        phase = 2.0 * np.pi * opd.unsqueeze(1) / self.wavelengths
        pupil = self.aperture * torch.exp(1j * phase)

        # 3. FFT
        field = torch.fft.fft2(torch.fft.ifftshift(pupil, dim=(-2, -1)))
        field = torch.fft.fftshift(field, dim=(-2, -1))
        psf_raw = torch.abs(field) ** 2

        # 4. Resampling
        f_m = self.f_mm * 1e-3
        psf_channels = []
        for i in range(3):
            zoom = (self.pixel_size * self.fov_pixels * self.pupil_diam) / \
                   (self.wavelengths[i,0,0] * f_m * self.pupil_grid_size)
            s = float(zoom)
            grid_1d = torch.linspace(-s, s, self.fov_pixels, device=psf_raw.device)
            grid_y, grid_x = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
            grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

            if psf_raw.dim() == 4:
                grid = grid.expand(psf_raw.shape[0], -1, -1, -1)
                psf_in = psf_raw[:, i:i+1, :, :]
            else:
                psf_in = psf_raw[i:i+1, :, :].unsqueeze(0)
            
            sampled = F.grid_sample(psf_in, grid, mode='bilinear', align_corners=False)
            psf_channels.append(sampled)

        psf_final = torch.cat(psf_channels, dim=1)
        # Energy Normalization
        psf_final = psf_final / (psf_final.sum(dim=(-2, -1), keepdim=True) + 1e-8)

        if coeffs.dim() == 1:
            return psf_final.squeeze(0)
        return psf_final

    def forward(self, d_obj, current_focus_dist_0, current_focus_dist_90):
        """
        Forward pass with Zernike index freezing.
        """
        # --- Channel 0 ---
        wfe_0 = self.calc_defocus_wfe(d_obj, current_focus_dist_0)
        
        # === 核心修改 2: 冻结前 4 项 (0,1,2,3) ===
        # 0: Piston, 1: Tilt X, 2: Tilt Y, 3: Defocus
        # 克隆参数，切断前 4 项的梯度回传
        coeffs_0 = self.zernike_0.clone()
        # 强制清零 (Masking out)
        coeffs_0[:4] = 0.0 
        
        # 将物理离焦 WFE 加到 Index 3
        if isinstance(wfe_0, torch.Tensor) and wfe_0.dim() > 0:
            # Batch mode
            coeffs_0 = coeffs_0.unsqueeze(0).repeat(wfe_0.shape[0], 1)
            coeffs_0[:, 3] = coeffs_0[:, 3] + wfe_0
        else:
            # Scalar mode
            coeffs_0[3] = coeffs_0[3] + wfe_0

        psf_0 = self._compute_psf(coeffs_0)

        # --- Channel 90 ---
        wfe_90 = self.calc_defocus_wfe(d_obj, current_focus_dist_90)
        
        coeffs_90 = self.zernike_90.clone()
        coeffs_90[:4] = 0.0 # Freeze
        
        if isinstance(wfe_90, torch.Tensor) and wfe_90.dim() > 0:
            coeffs_90 = coeffs_90.unsqueeze(0).repeat(wfe_90.shape[0], 1)
            coeffs_90[:, 3] = coeffs_90[:, 3] + wfe_90
        else:
            coeffs_90[3] = coeffs_90[3] + wfe_90

        psf_90 = self._compute_psf(coeffs_90)

        return psf_0, psf_90