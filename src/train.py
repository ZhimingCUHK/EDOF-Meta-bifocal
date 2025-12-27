import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # <--- 新增导入
import torchvision.utils as vutils # 用于生成网格图像
import numpy as np
import os
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# 尝试导入评估指标库
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError:
    print("Warning: scikit-image not installed. PSNR/SSIM calculation might fail.")
    psnr_metric = None
    ssim_metric = None

# --- 1. 路径设置 (确保能导入 src) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. 导入你的模块 ---
from src.models.pipeline import BlurGenerationPipeline, CONFIG
from src.data.dataloader import SceneFlowDataset
from src.network.MIMOUNET import MIMOUNet

# --- 新增: Zernike 正则化 Loss ---
class ZernikeRegLoss(nn.Module):
    def __init__(self, weight=0.05):
        super().__init__()
        self.weight = weight
        
    def forward(self, simulator):
        # 获取 Zernike 系数
        # 这里的系数单位是米 (m)
        z0 = simulator.optics.zernike_0
        z90 = simulator.optics.zernike_90
        
        # 我们希望约束系数在微米级别，避免过大
        # 将单位转换为微米 (um) 后计算平方和，这样数值梯度更健康
        # 忽略前4项 (Piston, Tilt, Defocus)，因为它们在 forward 中被强制置零或由物理模型控制
        # 但为了防止参数本身漂移，对整体做约束也是可以的，或者切片 [4:]
        
        loss_0 = torch.mean((z0[4:] * 1e6) ** 2)
        loss_90 = torch.mean((z90[4:] * 1e6) ** 2)
        
        return self.weight * (loss_0 + loss_90)

# --- 新增: 计算 PSNR 和 SSIM 的辅助函数 ---
def compute_metrics(pred, target):
    """
    计算 Batch 的平均 PSNR 和 SSIM
    pred, target: [B, 3, H, W] tensors, range [0, 1]
    """
    if psnr_metric is None or ssim_metric is None:
        return 0.0, 0.0

    # 转为 Numpy [B, H, W, 3]
    pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    target_np = target.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    psnr_val = 0
    ssim_val = 0
    batch_size = pred_np.shape[0]
    
    for i in range(batch_size):
        p = np.clip(pred_np[i], 0, 1)
        t = np.clip(target_np[i], 0, 1)
        
        psnr_val += psnr_metric(t, p, data_range=1.0)
        # win_size 必须是奇数且小于图像尺寸，channel_axis=2 表示 RGB 通道在最后
        ssim_val += ssim_metric(t, p, data_range=1.0, channel_axis=2, win_size=11)
        
    return psnr_val / batch_size, ssim_val / batch_size

# --- 新增: 记录 PSF Bank 到 TensorBoard ---
def log_psf_bank(writer, simulator, epoch, config):
    """
    提取并可视化每一层的 PSF
    """
    with torch.no_grad():
        # 生成所有层的 PSF
        psf_0, psf_90 = simulator.generate_psf_bank(
            config["num_layers"], 
            config["min_depth"], 
            config["max_depth"],
            config["focus_dist_0"],
            config["focus_dist_90"]
        )
        # psf_0: [K, 3, H, W]
        
        # 归一化以便可视化 (每张 PSF 独立归一化，看清形状)
        def normalize_psf(psf):
            flat = psf.view(psf.shape[0], -1)
            max_val = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            return psf / (max_val + 1e-8)

        vis_psf_0 = normalize_psf(psf_0)
        vis_psf_90 = normalize_psf(psf_90)
        
        # 制作网格 (nrow=5 表示每行显示5个层)
        grid_0 = vutils.make_grid(vis_psf_0, nrow=5, normalize=False, pad_value=1)
        grid_90 = vutils.make_grid(vis_psf_90, nrow=5, normalize=False, pad_value=1)
        
        writer.add_image("PSF_Shape/0_deg_Layers", grid_0, epoch)
        writer.add_image("PSF_Shape/90_deg_Layers", grid_90, epoch)

# --- 新增: 测试集评估函数 ---
def evaluate(model, simulator, dataloader, device):
    model.eval()
    # 注意：如果 simulator 包含 BN 等层，这里也应该 eval，但通常 optics 只有参数
    # simulator.eval() 
    
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Testing"):
            sharp_img = sample['image'].to(device)
            depth_map = sample['depthmap'].to(device)
            
            # 1. 使用训练好的光学模型生成模糊图
            blur_0, blur_90, _, _, _ = simulator(sharp_img, depth_map)
            
            # 2. 网络恢复
            input_stack = torch.cat([blur_0, blur_90], dim=1)
            outputs = model(input_stack)
            final_pred = outputs[-1] # 取最高分辨率输出
            
            # 3. 计算指标
            p, s = compute_metrics(final_pred, sharp_img)
            total_psnr += p
            total_ssim += s
            count += 1
            
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_psnr, avg_ssim

# --- 3. 定义多尺度 Loss ---
class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, outputs, target):
        # MIMO-UNet 返回 [out_4, out_2, out_1] (从小到大)
        # 我们需要对 target 进行下采样来匹配
        loss = 0
        weights = [0.1, 0.2, 1.0] # 给予全分辨率输出最大的权重
        
        for i, out in enumerate(outputs):
            # 动态调整 target 大小以匹配 output
            if out.shape[-2:] != target.shape[-2:]:
                t = torch.nn.functional.interpolate(target, size=out.shape[-2:], mode='bilinear', align_corners=False)
            else:
                t = target
            
            loss += weights[i] * self.l1(out, t)
            
        return loss

# --- 4. 训练配置 ---
TRAIN_CONFIG = {
    "epochs": 20,
    "batch_size": 8,          # <--- 新增: 设置批量大小
    "learning_rate": 1e-4,
    "train_optics": False,    # <--- 锁住光学参数
    "zernike_reg_weight": 0.0, # 此时不需要正则化
    "save_dir": "./checkpoints_stage1",
    "num_workers": 4,
    "log_interval": 100,
    # ... 其他配置
}

# --- 5. 辅助函数: 保存验证图片 ---
def save_sample_images(epoch, sharp, blur_0, blur_90, pred, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # 取 Batch 中的第一张图
    s = sharp[0].permute(1, 2, 0).detach().cpu().numpy()
    b0 = blur_0[0].permute(1, 2, 0).detach().cpu().numpy()
    b90 = blur_90[0].permute(1, 2, 0).detach().cpu().numpy()
    p = pred[0].permute(1, 2, 0).detach().cpu().numpy()
    
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1); plt.title("Sharp GT"); plt.imshow(np.clip(s, 0, 1)); plt.axis('off')
    plt.subplot(1, 4, 2); plt.title("Blur 0 deg"); plt.imshow(np.clip(b0, 0, 1)); plt.axis('off')
    plt.subplot(1, 4, 3); plt.title("Blur 90 deg"); plt.imshow(np.clip(b90, 0, 1)); plt.axis('off')
    plt.subplot(1, 4, 4); plt.title(f"Pred (Ep {epoch})"); plt.imshow(np.clip(p, 0, 1)); plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}_val.png"))
    plt.close()

# --- 6. 主训练循环 ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on {device}...")
    
    # 初始化 TensorBoard Writer
    log_dir = os.path.join(TRAIN_CONFIG["save_dir"], "logs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    
    # A. 初始化数据集
    # 注意：这里我们只加载 Sharp 和 Depth，模糊图在 GPU 上实时生成
    try:
        train_dataset = SceneFlowDataset(
            data_root=CONFIG["data_root"],
            dataset_type='train',
            image_size=CONFIG["image_size"],
            use_random_crop=True
        )
        print(f"Dataset loaded: {len(train_dataset)} samples.")
        
        # --- 新增: 加载测试集 ---
        # 修改: dataset_type 改为 'val'，因为 dataloader 不支持 'test'
        test_dataset = SceneFlowDataset(
            data_root=CONFIG["data_root"],
            dataset_type='val', 
            image_size=CONFIG["image_size"],
            use_random_crop=False # 测试时不随机裁剪
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        print(f"Test Dataset loaded: {len(test_dataset)} samples.")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your data path in pipeline.py CONFIG.")
        return

    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True
    )

    # B. 初始化模型
    # 1. 物理仿真器 (Simulator)
    simulator = BlurGenerationPipeline(CONFIG).to(device)
    
    # 2. 图像恢复网络 (Restoration Network)
    # 使用真正的 MIMO-UNet
    model = MIMOUNet().to(device)

    # C. 优化器设置
    # 分组设置参数，给予不同的学习率
    params = [
        {
            'params': model.parameters(), 
            'lr': TRAIN_CONFIG["learning_rate"]
        }
    ]
    
    if TRAIN_CONFIG["train_optics"]:
        params.append({
            'params': simulator.parameters(), 
            'lr': TRAIN_CONFIG["learning_rate"] * 0.1  # <--- 关键：光学参数学习率设为网络的 10%
        })

    optimizer = optim.AdamW(params) # 移除全局 lr，使用参数组中的 lr
    criterion = ContentLoss() # 使用多尺度 Loss
    criterion_reg = ZernikeRegLoss(weight=TRAIN_CONFIG["zernike_reg_weight"]) # <--- 初始化正则化 Loss

    # D. 循环
    os.makedirs(TRAIN_CONFIG["save_dir"], exist_ok=True)
    
    global_step = 0  # <--- 用于记录总步数

    for epoch in range(1, TRAIN_CONFIG["epochs"] + 1):
        model.train()
        if TRAIN_CONFIG["train_optics"]:
            simulator.train()
        
        # --- 新增: 每个 Epoch 开始时记录一次 PSF 形状 ---
        if TRAIN_CONFIG["train_optics"] or epoch == 1:
            log_psf_bank(writer, simulator, epoch, CONFIG)
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{TRAIN_CONFIG['epochs']}")
        
        for batch_idx, sample in enumerate(progress_bar):
            global_step += 1 # <--- 步数自增

            # 1. 获取数据并移至 GPU
            sharp_img = sample['image'].to(device)      # [B, 3, H, W]
            depth_map = sample['depthmap'].to(device)   # [B, 1, H, W]
            
            # 2. 物理仿真生成模糊图 (On-the-fly Simulation)
            # 如果不训练光学，使用 no_grad 节省显存和计算
            if TRAIN_CONFIG["train_optics"]:
                blur_0, blur_90, _, _, _ = simulator(sharp_img, depth_map)
            else:
                with torch.no_grad():
                    blur_0, blur_90, _, _, _ = simulator(sharp_img, depth_map)
            
            # 3. 准备网络输入 (Stacking)
            # [B, 3, H, W] + [B, 3, H, W] -> [B, 6, H, W]
            input_stack = torch.cat([blur_0, blur_90], dim=1)
            
            # 4. 网络前向传播
            outputs = model(input_stack)
            
            # 5. 计算 Loss (多尺度)
            content_loss = criterion(outputs, sharp_img)
            
            # 计算正则化 Loss
            reg_loss = torch.tensor(0.0, device=device)
            if TRAIN_CONFIG["train_optics"]:
                reg_loss = criterion_reg(simulator)
            
            loss = content_loss + reg_loss
            
            # 6. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), reg=reg_loss.item())

            # --- TensorBoard Logging ---
            if global_step % TRAIN_CONFIG["log_interval"] == 0:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)
                writer.add_scalar("Loss/content_step", content_loss.item(), global_step)
                writer.add_scalar("Loss/reg_step", reg_loss.item(), global_step)

            # 简单的可视化保存 (每个 Epoch 保存第一批次)
            if batch_idx == 0:
                # 取全分辨率输出 (列表最后一个)
                final_pred = outputs[-1]
                save_sample_images(epoch, sharp_img, blur_0, blur_90, final_pred, TRAIN_CONFIG["save_dir"])
                
                # 将图像写入 TensorBoard
                # input_stack 是 [B, 6, H, W]，我们需要拆分显示
                # 取 Batch 中的前 4 张图显示 (如果 batch_size < 4 则取全部)
                n_vis = min(4, sharp_img.shape[0])
                
                writer.add_images("Images/1_Sharp_GT", sharp_img[:n_vis], epoch)
                writer.add_images("Images/2_Blur_0deg", input_stack[:n_vis, :3, :, :], epoch)
                writer.add_images("Images/3_Blur_90deg", input_stack[:n_vis, 3:, :, :], epoch)
                writer.add_images("Images/4_Prediction", final_pred[:n_vis], epoch)

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch) # <--- 记录 Epoch 平均 Loss

        # --- 新增: 每个 Epoch 结束后进行测试评估 ---
        val_psnr,val_ssim = evaluate(model, simulator, test_loader, device)
        print(f"Epoch {epoch} Validation -> PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f}")

        writer.add_scalar("Metrics/Val_PSNR", val_psnr, epoch)
        writer.add_scalar("Metrics/Val_SSIM", val_ssim, epoch)
        
        # 保存模型权重
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optics_state_dict': simulator.state_dict(), # 保存光学参数以防万一
            }, os.path.join(TRAIN_CONFIG["save_dir"], f"checkpoint_ep{epoch}.pth"))

    # --- 新增: 训练结束后进行测试 ---
    print("\nTraining finished.")
    writer.close() # <--- 关闭 Writer

if __name__ == "__main__":
    train()