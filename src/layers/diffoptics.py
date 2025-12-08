import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import poppy
import astropy.units as u


class DifferentiableOptics(nn.Module):

    def __init__(self,
                 f_mm=25.0,
                 f_number=2.0,
                 pixel_size_um=3.45,
                 fov_pixels=32,
                 n_zernike=15,
                 pupil_grid_size=256):
        """
        Args:
            f_mm: focal length in mm
            f_number: f-number of the lens
            pixel_size_um: pixel size in micrometers
            fov_pixels: field of view in pixels (assumed square)
            n_zernike: number of Zernike polynomials to use (optimizable)
            pupil_grid_size: size of the pupil grid for simulation
        """
        super().__init__()


        self.f_mm = f_mm
        self.f_number = f_number
        self.pixel_size = pixel_size_um * 1e-6  # in meters
        self.fov_pixels = fov_pixels
        self.pupil_grid_size = pupil_grid_size
        self.n_zernike = n_zernike
        self.pupil_diam = (f_mm * 1e-3) / f_number  # in meters

        self.zernike_0 = nn.Parameter(
            torch.zeros(n_zernike))  # Optimizable Zernike coefficients in 0
        self.zernike_90 = nn.Parameter(
            torch.zeros(n_zernike))  # Optimizable Zernike coefficients in 90

        print(
            f"Pre-computing Zernike basis (Grid:{pupil_grid_size}x{pupil_grid_size})..."
        )
        basis_numpy, aperture_numpy = self._precompute_zernike_basis_with_poppy(
            n_zernike, pupil_grid_size)

        self.register_buffer('zernike_basis',
                             torch.from_numpy(basis_numpy).float())
        self.register_buffer('aperture',
                             torch.from_numpy(aperture_numpy).float())

        self.register_buffer('wavelengths',
                             torch.tensor([630e-9, 550e-9, 470e-9]).view(
                                 3, 1, 1))  # RGB wavelengths in meters

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
        """
        Calculate defocus wavefront error
        Args:
            d_obj: object distance in meters
            d_focus: focus distance in meters
        Returns:
            defocus_wfe: defocus wavefront error in meters
        """
        # Physics Formula
        # W = (f^2 / 8*N^2) * (1/d_focus - 1/d_obj)
        if isinstance(d_obj, torch.Tensor):
            inv_d_obj = 1.0 / (d_obj + 1e-8)
        else:
            inv_d_obj = 0.0 if d_obj == float('inf') else 1.0 / d_obj

        inv_d_focus = 1.0 / d_focus

        f_m = self.f_mm * 1e-3  # focal length in meters
        w_pv = (f_m**2) / (8.0 *
                           (self.f_number**2)) * (inv_d_focus - inv_d_obj)

        return w_pv

    def _compute_psf(self, coeffs):
        """
        Compute PSF from Zernike coefficients with correct physical scaling
        """
        # 1. Compute OPD
        if coeffs.dim() == 1:
            opd = torch.einsum('n,nij->ij', coeffs, self.zernike_basis)
            opd = opd.unsqueeze(0)
        else:
            opd = torch.einsum('bn,nij->bij', coeffs, self.zernike_basis)
        
        # 2. Phase calculation
        phase = 2.0 * np.pi * opd.unsqueeze(1) / self.wavelengths

        # 3. Pupil Function
        pupil = self.aperture * torch.exp(1j * phase)

        # 4. FFT to get PSF
        field = torch.fft.fft2(torch.fft.ifftshift(pupil, dim=(-2, -1)))
        field = torch.fft.fftshift(field, dim=(-2, -1))
        psf_raw = torch.abs(field) ** 2  # 此时是 FFT 坐标系下的强度


        # 5. Physical Resampling
        f_m = self.f_mm * 1e-3
        
        psf_channels = []
        for i in range(3):
            # Calculate Zoom Factor
            zoom = (self.pixel_size * self.fov_pixels * self.pupil_diam) / \
                   (self.wavelengths[i,0,0] * f_m * self.pupil_grid_size)
            
            s = float(zoom)
            
            # Create Grid
            grid_1d = torch.linspace(-s, s, self.fov_pixels, device=psf_raw.device)
            grid_y, grid_x = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
            grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

            if psf_raw.dim() == 4:
                grid = grid.expand(psf_raw.shape[0], -1, -1, -1)
                psf_in = psf_raw[:, i:i+1, :, :]
            else:
                psf_in = psf_raw[i:i+1, :, :].unsqueeze(0)
            
            # Sample
            sampled_psf = F.grid_sample(psf_in, grid, mode='bilinear', align_corners=False)
            psf_channels.append(sampled_psf)

        psf_final = torch.cat(psf_channels, dim=1) # [B, 3, FOV, FOV]

        # 6. Final Energy Normalization (Moved Here!)
        psf_final = psf_final / (psf_final.sum(dim=(-2, -1), keepdim=True) + 1e-8)

        if coeffs.dim() == 1:
            return psf_final.squeeze(0)
            
        return psf_final

    def forward(self, d_obj, current_focus_dist_0, current_focus_dist_90):
        wfe_0 = self.calc_defocus_wfe(d_obj, current_focus_dist_0)
        coeffs_0 = self.zernike_0.clone()

        if isinstance(wfe_0, torch.Tensor) and wfe_0.dim() > 0:
            coeffs_0 = coeffs_0.unsqueeze(0).repeat(wfe_0.shape[0], 1).clone()
            coeffs_0[:, 3] = coeffs_0[:, 3] + wfe_0
        else:
            coeffs_0[3] = coeffs_0[3] + wfe_0

        psf_0 = self._compute_psf(coeffs_0)

        wfe_90 = self.calc_defocus_wfe(d_obj, current_focus_dist_90)
        coeffs_90 = self.zernike_90.clone()

        if isinstance(wfe_90, torch.Tensor) and wfe_90.dim() > 0:
            coeffs_90 = coeffs_90.unsqueeze(0).repeat(wfe_90.shape[0],
                                                      1).clone()
            coeffs_90[:, 3] = coeffs_90[:, 3] + wfe_90
        else:
            coeffs_90[3] = coeffs_90[3] + wfe_90

        psf_90 = self._compute_psf(coeffs_90)

        return psf_0, psf_90
