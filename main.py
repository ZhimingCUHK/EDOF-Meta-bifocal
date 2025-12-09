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
# 确保你的 mimo_unet.py 里已经注册了 'PolarMIMOUNet'
from src.network.polar_mimo_unet import build_net 

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
    try:
        ssim = compare_ssim(img1_np, img2_np, data_range=1.0, channel_axis=2)
    except TypeError:
        ssim = compare_ssim(img1_np, img2_np, data_range=1.0, multichannel=True)
        
    return psnr, ssim

# ================= 修改点 1: MultiScaleLoss 适配 H/2 输出 =================
class MultiScaleLoss(nn.Module):
    def __init__(self):
        super(MultiScaleLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, outputs, label):
        """
        outputs: list of predictions from PolarMIMOUNet
                 outputs[0]: Small (H/8)
                 outputs[1]: Mid   (H/4)
                 outputs[2]: Large (H/2) <- Main Output
        label: GT image [B, 3, H, W] (Full Resolution)
        """
        # 1. 将 GT 下采样到 H/2，作为主输出的 Target
        label_half = F.interpolate(label, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # 2. 将 GT 继续下采样到 H/4 和 H/8
        label_quarter = F.interpolate(label_half, scale_factor=0.5, mode='bilinear', align_corners=False)
        label_eighth = F.interpolate(label_quarter, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # 3. 计算 Loss (注意 outputs 的顺序与 Label 的对应关系)
        # 假设 outputs 顺序是 [Small, Mid, Large]
        loss_small = self.criterion(outputs[0], label_eighth)
        loss_mid   = self.criterion(outputs[1], label_quarter)
        loss_large = self.criterion(outputs[2], label_half)
        
        loss = loss_small + loss_mid + loss_large
        return loss

def save_checkpoint(model, optimizer, epoch, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

# ================= 修改点 2: 验证时对齐尺寸 =================
def validate_and_visualize(pipeline, model, val_loader, device, epoch, output_dir, writer=None):
    model.eval()
    pipeline.eval()
    
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for i, (rgb_gt, _, depth_map) in enumerate(val_loader):
            rgb_gt = rgb_gt.to(device)
            depth_map = depth_map.to(device)
            
            # 1. Simulate (RAW is Full Res HxW)
            raw_output, img_near, img_far = pipeline(rgb_gt, depth_map)
            
            # 2. Reconstruct (Output is H/2 x W/2)
            preds = model(raw_output)
            pred_img = preds[2] 
            
            # 3. Resize GT to match Prediction (H/2) for metrics
            gt_half = F.interpolate(rgb_gt, scale_factor=0.5, mode='bilinear', align_corners=False)

            # Calculate metrics
            for b in range(rgb_gt.shape[0]):
                # 使用缩小后的 GT 计算
                p, s = calculate_metrics(gt_half[b], pred_img[b])
                psnr_list.append(p)
                ssim_list.append(s)

            # 3. Visualize
            if i == 0:
                B = rgb_gt.shape[0]
                idx = 0
                
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                
                # Row 1: Inputs
                axes[0,0].imshow(tensor_to_image(rgb_gt[idx]))
                axes[0,0].set_title("GT Sharp (Full Res)")
                axes[0,0].axis('off')
                
                depth_vis = depth_map[idx].detach().cpu().squeeze().numpy()
                im = axes[0,1].imshow(depth_vis, cmap='plasma')
                axes[0,1].set_title("Depth Map")
                axes[0,1].axis('off')
                plt.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)
                
                raw_vis = raw_output[idx].detach().cpu().squeeze().numpy()
                axes[0,2].imshow(raw_vis, cmap='gray')
                axes[0,2].set_title("Simulated Raw Input (Mosaic)")
                axes[0,2].axis('off')
                
                # Row 2: Outputs
                axes[1,0].imshow(tensor_to_image(img_near[idx]))
                axes[1,0].set_title("Optical Blur (Near)")
                axes[1,0].axis('off')
                
                axes[1,1].imshow(tensor_to_image(img_far[idx]))
                axes[1,1].set_title("Optical Blur (Far)")
                axes[1,1].axis('off')
                
                # Visualization of Result (Half Res)
                axes[1,2].imshow(tensor_to_image(pred_img[idx]))
                axes[1,2].set_title(f"Reconstructed (Half Res) Ep:{epoch}")
                axes[1,2].axis('off')
                
                plt.tight_layout()
                vis_path = os.path.join(output_dir, f'val_epoch_{epoch:03d}.png')
                plt.savefig(vis_path)
                
                if writer:
                    writer.add_figure('Val/Visualization', fig, epoch)
                    writer.add_image('Val/GT_Full', rgb_gt[idx], epoch)
                    writer.add_image('Val/Recon_Half', pred_img[idx], epoch)
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
    
    exp_dir = 'experiments/train_run'
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    vis_dir = os.path.join(exp_dir, 'visualizations')
    log_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    # 2. Initialize Pipeline
    print("Initializing Image Formation Pipeline...")
    pipeline = ImageFormationPipeline(
        optics_config=config['optics'],
        sensor_config=config['sensor'],
        layer_config=config['layers']
    ).to(device)
    
    for param in pipeline.parameters():
        param.requires_grad = True

    # ================= 修改点 3: 初始化 PolarMIMOUNet =================
    print("Initializing PolarMIMOUNet...")
    # 使用修改后的类名（在 build_net 中注册的名字）
    # in_channel=1 因为输入是单通道 RAW
    try:
        model = build_net('PolarMIMOUNet', in_channel=1).to(device)
    except:
        print("Warning: 'PolarMIMOUNet' not found in build_net, trying 'MIMO-UNet'...")
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
        
        val_dataset = SceneFlowDataset(
            data_root=data_root,
            dataset_type='val',
            image_size=(config['sensor']['height'], config['sensor']['width']),
            use_random_crop=True
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 5. Setup Training
    criterion = MultiScaleLoss().to(device)
    
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
        pipeline.train()
        
        epoch_loss = 0
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (rgb_gt, _, depth_map) in enumerate(pbar):
            rgb_gt = rgb_gt.to(device)
            depth_map = depth_map.to(device)
            
            optimizer.zero_grad()

            # --- Forward Pass ---
            raw_output, _, _ = pipeline(rgb_gt, depth_map)
            preds = model(raw_output) 
            
            # Loss Function 会自动处理分辨率匹配
            loss = criterion(preds, rgb_gt)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is {loss.item()} at step {i}. Skipping.")
                optimizer.zero_grad()
                continue

            # --- Backward Pass ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            
        avg_loss = epoch_loss / len(train_loader)
        duration = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {duration:.2f}s. Avg Loss: {avg_loss:.6f}")
        
        writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        
        # Log Optics Params
        z0 = pipeline.optics.zernike_0.detach().cpu().numpy()
        z90 = pipeline.optics.zernike_90.detach().cpu().numpy()
        for idx, val in enumerate(z0):
            writer.add_scalar(f'Optics/Zernike_0_Mode_{idx}', val, epoch)
        for idx, val in enumerate(z90):
            writer.add_scalar(f'Optics/Zernike_90_Mode_{idx}', val, epoch)

        if (epoch + 1) % 1 == 0:
            validate_and_visualize(pipeline, model, val_loader, device, epoch+1, vis_dir, writer)
            
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch+1, save_path)
            print(f"Checkpoint saved to {save_path}")

    print("Training finished.")
    writer.close()

if __name__ == "__main__":
    main()