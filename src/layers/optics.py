from turtle import st
from altair import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import poppy


class DifferentiableOptics(nn.Module):

    def __init__(self,
                 f_mm,
                 f_number,
                 pixel_size_um,
                 fov_pixels,
                 n_zernike,
                 pupil_grid_size,
                 oversample_factor=4):
        super().__init__()

        self.f_mm = f_mm
        self.f_number = f_number
        self.pixel_size = pixel_size_um * 1e-6
        self.fov_pixels = fov_pixels
        self.pupil_grid_size = pupil_grid_size
        self.n_zernike = n_zernike
        self.oversample_factor = oversample_factor

        # 光瞳直径 D = f / F# (单位: 米)
        self.pupil_diam = (f_mm * 1e-3) / f_number
        self.pupil_radius = self.pupil_diam / 2.0

        self.zernike_0 = nn.Parameter(torch.zeros(n_zernike))
        self.zernike_90 = nn.Parameter(torch.zeros(n_zernike))

        # --- Nyquist Check (保留上一版的检查) ---
        min_lambda = 470e-9
        print(
            f"[Optics Check] Diffraction Spot Size approx: {1.22*min_lambda*f_number*1e6:.2f} um"
        )

        # --- Initialize Zernike  (保留上一版的逻辑) ---
        with torch.no_grad():
            self.zernike_0.zero_()
            self.zernike_90.zero_()

            strength = 2 * 1e-6    # 0.3 micron
            
            if n_zernike > 6:
                self.zernike_0[8] = strength
                self.zernike_0[9] = strength
            
            if n_zernike > 7:
                self.zernike_90[8] = strength 
                self.zernike_90[9] = strength
        # --- 预计算 ---
        basis_numpy, aperture_numpy, r2_phys = self._precompute_grids(
            n_zernike, pupil_grid_size)

        self.register_buffer('zernike_basis',
                             torch.from_numpy(basis_numpy).float())
        self.register_buffer('aperture',
                             torch.from_numpy(aperture_numpy).float())
        # r2_phys: 物理半径的平方 (meter^2)，用于计算球面波相位
        self.register_buffer('r2_phys', torch.from_numpy(r2_phys).float())

        # 波长: (1, 3, 1, 1) 用于广播
        self.register_buffer(
            'wavelengths',
            torch.tensor([630e-9, 550e-9, 470e-9]).view(1, 3, 1, 1))

    def _precompute_grids(self, n_modes, pupil_grid_size):
        """
        预计算 Zernike 基底以及物理坐标网格
        """
        # 1. 归一化坐标 [-1, 1] 用于 poppy 计算 Zernike
        x_lin = np.linspace(-1, 1, pupil_grid_size)
        y_lin = np.linspace(-1, 1, pupil_grid_size)
        y, x = np.meshgrid(y_lin, x_lin, indexing='ij')
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        aperture = (r <= 1.0).astype(np.float32)

        # 2. 物理坐标 [meters] 用于计算 Defocus Phase
        # r 是归一化的 (0~1)，物理半径 = r * (Diameter / 2)
        r_phys = r * self.pupil_radius
        r2_phys = r_phys**2

        basis = []
        for i in range(1, n_modes + 1):
            z = poppy.zernike.zernike1(i, rho=r, theta=theta, outside=0.0)
            basis.append(z)

        return np.stack(basis), aperture, r2_phys

    def get_defocus_phase(self, d_obj, d_focus):
        """
        计算物理离焦相位 (修正版)
        基于广义光瞳函数: Phase Error ~ (1/d_obj - 1/d_focus)
        该公式模拟镜头"对焦在 d_focus 处"时的波前误差。
        """
        # k: (1, 3, 1, 1)
        k = 2 * np.pi / self.wavelengths

        # r2: (1, 1, Grid, Grid)
        # 确保 r2_phys 在 __init__ 中已经正确注册
        r2 = self.r2_phys.unsqueeze(0).unsqueeze(0)

        # 处理 d_obj (Batch, 1)
        if isinstance(d_obj, torch.Tensor):
            d_obj = d_obj.view(-1, 1, 1, 1)
            inv_d_obj = 1.0 / (d_obj + 1e-8)
        else:
            inv_d_obj = 0.0 if d_obj == float('inf') else 1.0 / d_obj

        # 处理 d_focus (Batch, 1)
        if isinstance(d_focus, torch.Tensor):
            d_focus = d_focus.view(-1, 1, 1, 1)
            inv_d_focus = 1.0 / (d_focus + 1e-8)
        else:
            inv_d_focus = 1.0 / d_focus

        # 公式推导:
        # 理想波前 (Target): 从 d_focus 发出的球面波，经过透镜后应变成平面波(或汇聚波)。
        # 实际波前 (Actual): 从 d_obj 发出的球面波。
        # 相位差 = k * r^2 / 2 * (实际曲率 - 目标曲率)
        # 当 d_obj == d_focus 时，括号内为0，相位差为0，也就是完美成像。

        total_phase = (k * r2 / 2.0) * (inv_d_obj - inv_d_focus)

        return total_phase

    def sample_psf(self, psf, sample_type):
        squeeze_b = False
        if psf.dim() == 3:   # (C,H,W)
            psf = psf.unsqueeze(0)  # -> (1,C,H,W)
            squeeze_b = True

        if sample_type == 0:
            out = psf[:, :, ::2, ::2]
        elif sample_type == 1:
            out = psf[:, :, ::2, 1::2]
        elif sample_type == 2:
            out = psf[:, :, 1::2, ::2]
        elif sample_type == 3:
            out = psf[:, :, 1::2, 1::2]
        else:
            raise ValueError("sample_type must be 0..3")

        return out.squeeze(0) if squeeze_b else out

    def _compute_psf(self, coeffs, extra_phase=None):
        """
        计算 PSF
        coeffs: Zernike系数 (Batch, N)
        extra_phase: 额外的物理相位 (Batch, 3, Grid, Grid)，例如离焦相位
        """
        # 1. Zernike OPD (Optical Path Difference)
        # opd: (Batch, Grid, Grid)
        if coeffs.dim() == 1:
            coeffs = coeffs.unsqueeze(0)
        opd = torch.einsum('bn,nij->bij', coeffs, self.zernike_basis)

        # 2. Convert OPD to Phase
        # phase_zernike: (Batch, 3, Grid, Grid)
        phase_zernike = 2.0 * np.pi * opd.unsqueeze(1) / self.wavelengths

        # 3. Combine with Extra Phase (Defocus)
        total_phase = phase_zernike
        if extra_phase is not None:
            # extra_phase 应该已经是 (B, 3, G, G) 或者能广播
            total_phase = total_phase + extra_phase

        # 4. Pupil Function
        pupil = self.aperture * torch.exp(1j * total_phase)

        # 5. FFT (Fraunhofer Diffraction)
        field = torch.fft.fft2(torch.fft.ifftshift(pupil, dim=(-2, -1)))
        field = torch.fft.fftshift(field, dim=(-2, -1))
        psf_raw = torch.abs(field)**2

        # 6. High-Res Sampling & Binning (Oversampling -> AvgPool)
        f_m = self.f_mm * 1e-3
        high_res_fov = self.fov_pixels * self.oversample_factor
        psf_channels = []

        for i in range(3):
            # 计算缩放因子 s (Sensor Size / FFT Window Size)
            fft_total_width = (self.wavelengths[0, i, 0, 0] * f_m *
                               self.pupil_grid_size) / self.pupil_diam
            sensor_total_width = self.pixel_size * self.fov_pixels
            s = sensor_total_width / fft_total_width

            grid_1d = torch.linspace(-s,
                                     s,
                                     high_res_fov,
                                     device=psf_raw.device)
            grid_y, grid_x = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
            grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
            grid = grid.expand(psf_raw.shape[0], -1, -1, -1)

            psf_in = psf_raw[:, i:i + 1, :, :]
            sampled_high_res = F.grid_sample(psf_in,
                                             grid,
                                             mode='bicubic',
                                             align_corners=False)
            psf_channels.append(sampled_high_res)

        psf_high_res = torch.cat(psf_channels, dim=1)
        psf_final = F.avg_pool2d(psf_high_res,
                                 kernel_size=self.oversample_factor,
                                 stride=self.oversample_factor)

        # Energy Normalization
        psf_final = psf_final / (psf_final.sum(dim=(-2, -1), keepdim=True) +
                                 1e-8)

        if coeffs.shape[0] == 1:
            return psf_final.squeeze(0)
        return psf_final

    def forward(self, d_obj, current_focus_dist_0, current_focus_dist_90):
        """
        Forward Pass
        注意：现在通过 explicit phase map 来处理离焦，
        而不是修改 Zernike 的 Index 3 (Defocus) 系数。
        """

        # --- Channel 0 ---
        # 1. 计算物理离焦相位
        phase_defocus_0 = self.get_defocus_phase(d_obj, current_focus_dist_0)

        # 2. 准备 Zernike 系数 (冻结前4项: Piston, Tilt, Defocus)
        coeffs_0 = self.zernike_0.clone()
        coeffs_0[:4] = 0.0  # 强制将 Zernike Defocus 置零，因为我们已经用上面的 phase 模拟了

        if phase_defocus_0.ndim > 3 and coeffs_0.ndim == 1:
            # 如果 phase 是 Batch 模式，扩展 coeffs
            coeffs_0 = coeffs_0.unsqueeze(0).repeat(phase_defocus_0.shape[0],
                                                    1)

        # 3. 计算 PSF (传入 Zernike + 物理 Defocus)
        psf_0 = self._compute_psf(coeffs_0, extra_phase=phase_defocus_0)

        psf_0 = self.sample_psf(psf_0, sample_type=0)

        psf_0 = psf_0 / (psf_0.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        # --- Channel 90 ---
        phase_defocus_90 = self.get_defocus_phase(d_obj, current_focus_dist_90)

        coeffs_90 = self.zernike_90.clone()
        coeffs_90[:4] = 0.0

        if phase_defocus_90.ndim > 3 and coeffs_90.ndim == 1:
            coeffs_90 = coeffs_90.unsqueeze(0).repeat(
                phase_defocus_90.shape[0], 1)

        psf_90 = self._compute_psf(coeffs_90, extra_phase=phase_defocus_90)
        psf_90 = self.sample_psf(psf_90, sample_type=3)
        psf_90 = psf_90 / (psf_90.sum(dim=(-2, -1), keepdim=True) + 1e-8)

        return psf_0, psf_90
