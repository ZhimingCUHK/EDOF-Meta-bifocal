import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
try:
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
except ImportError:
    print("Warning: scikit-image not installed. Metrics will be 0.")
    compare_ssim = None
    compare_psnr = None

from src.models.pipeline import ImageFormationPipeline
from src.data.dataloader import SceneFlowDataset
from src.network.mimo_unet import build_net

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def tensor_to_image(tensor):
    """Convert tensor [C, H, W] to numpy [H, W, C] for plotting"""
    img = tensor.detach().cpu().numpy()
    if img.shape[0] == 1:
        img = img.squeeze(0)
    else:
        img = img.transpose(1, 2, 0)
    return np.clip(img, 0, 1)

def calculate_metrics(img1, img2):
    """
    Calculate PSNR and SSIM for a pair of images.
    img1, img2: [C, H, W] tensors, range [0, 1]
    """
    if compare_psnr is None or compare_ssim is None:
        return 0.0, 0.0
        
    img1_np = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.detach().cpu().numpy().transpose(1, 2, 0)
    
    psnr = compare_psnr(img1_np, img2_np, data_range=1.0)
    # Handle channel_axis for newer skimage versions, or multichannel for older
    try:
        ssim = compare_ssim(img1_np, img2_np, data_range=1.0, channel_axis=2)
    except TypeError:
        ssim = compare_ssim(img1_np, img2_np, data_range=1.0, multichannel=True)
        
    return psnr, ssim

class MultiScaleLoss(nn.Module):
    def __init__(self):
        super(MultiScaleLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, outputs, label):
        # outputs: [pred_1/4, pred_1/2, pred_full]
        # label: [B, 3, H, W]
        
        # Downsample label to match output scales
        label_2 = F.interpolate(label, scale_factor=0.5, mode='bilinear', align_corners=False)
        label_4 = F.interpolate(label_2, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        loss_4 = self.criterion(outputs[0], label_4)
        loss_2 = self.criterion(outputs[1], label_2)
        loss_1 = self.criterion(outputs[2], label)
        
        # Weighted sum (weights can be tuned, here we use equal or standard weights)
        # MIMO-UNet paper often uses: 
        loss = loss_4 + loss_2 + loss_1
        return loss

def save_checkpoint(model, optimizer, epoch, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

def validate_and_visualize(pipeline, model, val_loader, device, epoch, output_dir, writer=None):
    model.eval()
    pipeline.eval() # Pipeline usually doesn't have BN/Dropout but good practice
    
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for i, (rgb_gt, _, depth_map) in enumerate(val_loader):
            rgb_gt = rgb_gt.to(device)
            depth_map = depth_map.to(device)
            
            # 1. Simulate
            raw_output, img_near, img_far = pipeline(rgb_gt, depth_map)
            
            # 2. Reconstruct
            preds = model(raw_output)
            pred_img = preds[2] # Full resolution output
            
            # Calculate metrics
            for b in range(rgb_gt.shape[0]):
                p, s = calculate_metrics(rgb_gt[b], pred_img[b])
                psnr_list.append(p)
                ssim_list.append(s)

            # 3. Visualize (Only for the first batch)
            if i == 0:
                B = rgb_gt.shape[0]
                idx = 0 # Show first image in batch
                
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                
                # Row 1: Inputs
                axes[0,0].imshow(tensor_to_image(rgb_gt[idx]))
                axes[0,0].set_title("GT Sharp Image")
                axes[0,0].axis('off')
                
                depth_vis = depth_map[idx].detach().cpu().squeeze().numpy()
                im = axes[0,1].imshow(depth_vis, cmap='plasma')
                axes[0,1].set_title("Depth Map")
                axes[0,1].axis('off')
                plt.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)
                
                raw_vis = raw_output[idx].detach().cpu().squeeze().numpy()
                axes[0,2].imshow(raw_vis, cmap='gray')
                axes[0,2].set_title("Simulated Raw Input")
                axes[0,2].axis('off')
                
                # Row 2: Intermediate & Output
                axes[1,0].imshow(tensor_to_image(img_near[idx]))
                axes[1,0].set_title("Optical Blur (Near Focus)")
                axes[1,0].axis('off')
                
                axes[1,1].imshow(tensor_to_image(img_far[idx]))
                axes[1,1].set_title("Optical Blur (Far Focus)")
                axes[1,1].axis('off')
                
                axes[1,2].imshow(tensor_to_image(pred_img[idx]))
                axes[1,2].set_title(f"Reconstructed (Epoch {epoch})")
                axes[1,2].axis('off')
                
                plt.tight_layout()
                vis_path = os.path.join(output_dir, f'val_epoch_{epoch:03d}.png')
                plt.savefig(vis_path)
                
                if writer:
                    writer.add_figure('Val/Visualization', fig, epoch)
                    writer.add_image('Val/GT', rgb_gt[idx], epoch)
                    writer.add_image('Val/Recon', pred_img[idx], epoch)
                    writer.add_image('Val/Raw', raw_output[idx], epoch)
                
                plt.close()
            
    avg_psnr = np.mean(psnr_list) if psnr_list else 0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0
    
    print(f"Validation Epoch {epoch}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
    
    if writer:
        writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        writer.add_scalar('Val/SSIM', avg_ssim, epoch)

def main():
    # 1. Load Configuration
    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # Create output directories
    exp_dir = 'experiments/train_run'
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    vis_dir = os.path.join(exp_dir, 'visualizations')
    log_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)

    # 2. Initialize Pipeline (Simulation)
    print("Initializing Image Formation Pipeline...")
    pipeline = ImageFormationPipeline(
        optics_config=config['optics'],
        sensor_config=config['sensor'],
        layer_config=config['layers']
    ).to(device)
    
    # Enable gradients for pipeline parameters for end-to-end optimization
    for param in pipeline.parameters():
        param.requires_grad = True

    # 3. Initialize Network (Reconstruction)
    print("Initializing MIMO-UNet...")
    # Input is 1-channel Raw, Output is 3-channel RGB
    model = build_net('MIMO-UNet', in_channel=1).to(device)

    # 4. Data Loading
    data_root = '/home/LionelZ/Data'
    batch_size = config['training']['batch_size']
    
    print(f"Loading SceneFlow dataset from {data_root}...")
    try:
        train_dataset = SceneFlowDataset(
            data_root=data_root,
            dataset_type='train',
            image_size=(config['sensor']['height'], config['sensor']['width']),
            use_random_crop=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Use a small subset for validation or the same dataset if val not available
        val_dataset = SceneFlowDataset(
            data_root=data_root,
            dataset_type='val', # Assuming val split exists, otherwise use train
            image_size=(config['sensor']['height'], config['sensor']['width']),
            use_random_crop=True
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 5. Setup Training
    criterion = MultiScaleLoss().to(device)
    
    # Joint optimization: Network + Optics
    # Usually optics parameters need a smaller learning rate
    optics_lr = config['training'].get('optics_learning_rate', 1e-5)
    print(f"Optimizer setup: Network LR = {config['training']['learning_rate']}, Optics LR = {optics_lr}")

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': config['training']['learning_rate']},
        {'params': pipeline.parameters(), 'lr': optics_lr} 
    ])
    
    num_epochs = config['training']['epochs']
    start_epoch = 0

    print("Starting training (End-to-End Optimization)...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        pipeline.train() # Ensure pipeline is in train mode
        
        epoch_loss = 0
        start_time = time.time()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (rgb_gt, _, depth_map) in enumerate(pbar):
            rgb_gt = rgb_gt.to(device)
            depth_map = depth_map.to(device)
            
            optimizer.zero_grad()

            # --- Forward Pass ---
            
            # 1. Simulation (Gradient ENABLED)
            # raw_output: [B, 1, H, W]
            raw_output, _, _ = pipeline(rgb_gt, depth_map)
            
            # 2. Reconstruction
            preds = model(raw_output) # Returns list of [pred_1/4, pred_1/2, pred_full]
            
            # 3. Loss
            loss = criterion(preds, rgb_gt)
            
            # --- Backward Pass ---
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log batch loss
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            
        avg_loss = epoch_loss / len(train_loader)
        duration = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {duration:.2f}s. Avg Loss: {avg_loss:.6f}")
        
        writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        
        # Log Zernike Coefficients
        z0 = pipeline.optics.zernike_0.detach().cpu().numpy()
        z90 = pipeline.optics.zernike_90.detach().cpu().numpy()
        
        for idx, val in enumerate(z0):
            writer.add_scalar(f'Optics/Zernike_0_Mode_{idx}', val, epoch)
        for idx, val in enumerate(z90):
            writer.add_scalar(f'Optics/Zernike_90_Mode_{idx}', val, epoch)

        # Validation & Saving
        if (epoch + 1) % 1 == 0: # Visualize every epoch
            validate_and_visualize(pipeline, model, val_loader, device, epoch+1, vis_dir, writer)
            
        if (epoch + 1) % 5 == 0: # Save checkpoint every 5 epochs
            save_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch+1, save_path)
            print(f"Checkpoint saved to {save_path}")

    print("Training finished.")
    writer.close()

if __name__ == "__main__":
    main()
