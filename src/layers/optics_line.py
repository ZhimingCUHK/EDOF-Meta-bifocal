import abc
import tensorflow as tf
import numpy as np
from numpy.fft import ifftshift
from scipy import interpolate
import fractions
import poppy
import scipy.io as sio
import torch
from torch import Tensor
from layers.convolution import convolve_with_psf_fft,convolve_with_psf_fft

################################ Materials #####################################
def get_refractive_idcs(wavelengths):
    _wavelengths = wavelengths * 1e-6
    _refractive_idcs = np.sqrt(1 + 0.6961663*_wavelengths**2/(_wavelengths**2-0.0684043**2) +
                                  0.4079426*_wavelengths**2/(_wavelengths**2-0.1162414**2) +
                                  0.8974794*_wavelengths**2/(_wavelengths**2-9.896161**2))
    return _refractive_idcs

def get_response_curve(wavelengths,file_name):
    _wavelengths = wavelengths * 1e-9
    tmp = np.loadtxt(file_name)
    f_linear = interpolate.interp1d(tmp[:,0],tmp[:,1])
    qutm_efficience = f_linear(_wavelengths)
    return qutm_efficience

def myprint(name,idea):
    print(name)
    print(idea)

################################ Useful functions #####################################
def gaussian_noise(image,stddev=0.001):
    dtype = image.dtype
    return image+tf.random.normal(image.shape,0.0,stddev,dtype=dtype)

def random_uniform(image,interval=[0.001,0.02]):
    return image + tf.random.uniform(minval=interval[0],maxval=interval[1],shape=[])

def fspecial(shape=(3,3),sigma=0.5):
    """
    2D Gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss - 1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def zoom(image_batch,zoom_fraction):
    """
    Get central crop of batch
    """
    images = tf.unstack(image_batch,axis=0)
    crops = []
    for image in images:
        crop = tf.image.central_crop(image,zoom_fraction)
        crops.append(crop)
    return tf.stack(crops,axis=0)

def transp_fft2d(a_tensor,dtype=tf.complex64):
    """
    Takes images of shape [batch_size,x,y,channels] and transposes them
    correctly for tensorflows fft2d to work
    """
    # Tensorflow's fft2d only supports complex64 dtype
    a_tensor = tf.cast(a_tensor,dtype=tf.complex64)
    # Transpose the tensor to [batch_size,channels,x,y]
    a_tensor_transp = tf.transpose(a_tensor,[0,3,1,2])
    a_fft2d = tf.fft2d(a_tensor_transp)
    a_fft2d = tf.transpose(a_fft2d,[0,2,3,1])
    return a_fft2d

def transp_ifft2d(a_tensor,dtype=tf.complex64):
    a_tensor = tf.cast(a_tensor,dtype=tf.complex64)
    a_tensor = tf.cast(a_tensor,dtype=tf.complex64)
    a_ifft2d_transp = tf.ifft2d(a_tensor)
    # Transpose back to [batch_size,x,y,channels]
    a_ifft2d = tf.transpose(a_ifft2d_transp,[0,2,3,1])
    return a_ifft2d

def compl_exp_tf(phase,dtype=tf.complex64,name='complex_exp'):
    """
    Complex exponent via euler's formula,since cuda doesn't have a GPU kernel for that
    Casts to *dtype*
    """
    phase = tf.cast(phase,dtype=tf.float64)
    return tf.add(tf.cast(tf.cos(phase),dtype=dtype),
                  1.j * tf.cast(tf.sin(phase),dtype=dtype),
                  name=name)

def laplacian_filter_tf(img_batch):
    """
    Laplacian filter
    Also considers diagnoals.
    """
    laplacian_filter = tf.constant([[1,1,1],[1,-8,1],[1,1,1]],dtype=tf.float32)
    laplacian_filter = tf.reshape(laplacian_filter,[3,3,1,1])
    filter_input = tf.cast(img_batch,dtype=tf.float32)
    filtered_batch = tf.nn.convolution(filter_input,filter=laplacian_filter,padding='SAME',data_format='NHWC')
    return filtered_batch

def laplace_l1_regularizer(scale):
    if np.allclose(scale,0.0):
        print("Scale of zero disables the laplace_l1_regularizer")

    def laplace_l1(a_tensor):
        with tf.name_scope('laplace_l1_regularizer'):
            laplace_filtered = laplacian_filter_tf(a_tensor)
            laplace_filtered = laplace_filtered[:,1:-1,1:-1,:]
            attach_summaries("Laplace_filtered",tf.abs(laplace_filtered),image=True,log_image=True)
            return scale * tf.reduce_mean(tf.abs(laplace_filtered))
    return laplace_l1

def laplace_l2_regularizer(scale):
    if np.allclose(scale,0.0):
        print("Scale of zero disables the laplace_l2_regularizer")

    def laplace_l2(a_tensor):
        with tf.name_scope('laplace_l2_regularizer'):
            laplace_filtered = laplacian_filter_tf(a_tensor)
            laplace_filtered = laplace_filtered[:,1:-1,1:-1,:]
            attach_summaries("Laplace_filtered",tf.abs(laplace_filtered),image=True,log_image=True)
            return scale * tf.reduce_mean(tf.square(laplace_filtered))

    return laplace_l2

def phaseshifts_from_phase_map(phase_map,wave_lengths,refractive_idcs):
    """
    Calculates the phase shifts created by a phase map with certain
    refractive index for light with specific wave length
    """
    # refractive index difference
    delta_N = refractive_idcs.reshape([1,1,1,-1])-1
    # wave number
    # wave_coef = wave_lengths/wave_lengths[1]
    wave_number = 2. * np.pi / wave_lengths
    wave_number = wave_number.reshape([1,1,1,-1])
    # phase delay indiced by height field
    phi = wave_number * phase_map
    phase_shifts = compl_exp_tf(phi)
    return phase_shifts

# def phaseshifts_from_height_map(height_map,wave_lengths,refractive_idcs):
#     """
#     Calculates the phase shifts created by a height map with certain
#     refractive index for light with specific wave length
#     """
#     # refractive index difference
#     delta_N = refractive_idcs - 1
#     # wave number
#     wave_nos = 2 * np.pi / wave_lengths
#     wave_nos = wave_nos.reshape([1,1,1,-1])
#     # phase delay indiced by height field
#     phi = wave_nos * height_map
#     phase_shifts = compl_exp_tf(phi)
#     return phase_shifts

def get_one_phase_shift(wave_length,refractive_index):
    """
    Calculates the thickness (in meter) of a phase shift of 2 pi
    """
    # refractive index difference
    delta_N = refractive_index - 1
    # wave number
    wave_nos = 2 * np.pi / wave_length
    two_pi_thickness = (2. * np.pi) / (wave_nos * delta_N)
    return two_pi_thickness

def attach_summaries(name,var,image=False,log_image=False):
    if image:
        tf.summary.image(name,var,max_outputs=3)
    if log_image and image:
        tf.summary.image(name + '_log',tf.log(var+1e-12),max_outputs=3)
    tf.summary.scalar(name)
    tf.summary.scalar(name)


################################ Optics source field #####################################
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


def planewave(x_mesh: np.ndarray, y_mesh: np.ndarray,
              wave_lengths: np.ndarray):
    x = np.asarray(x_mesh, dtype=np.float64)
    y = np.asarray(y_mesh, dtype=np.float64)
    lambdas = np.asarray(wave_lengths, dtype=np.float64)

    r2 = x**2 + y**2
    r2 = r2[None, :, :, None]
    k = (2 * np.pi / lambdas)[None, None, None, :]

    phi = np.zeros_like[r2]

    field = np.exp(1j * phi)[None, :, :, :].astype(np.complex64, copy=False)

    return field

################################ lens and optical elements #####################################
def zernike_layer_fab(mode:str,input_field:Tensor,phase_map_far:Tensor,phase_map_near:Tensor): #制造好的zernike_layer,用于微调？
    if mode == 'far':
        phi = phase_map_far
    elif mode == 'near':
        phi = phase_map_near
    zernike_phase = torch.exp(1j*phi).to(torch.complex64)
    input_field = input_field.to(torch.complex64)
    zernike_field = torch.multiply(input_field,zernike_phase)
    return zernike_field

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



