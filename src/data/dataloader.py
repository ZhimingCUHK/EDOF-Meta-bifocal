import os
import cv2
import numpy as np
import torch
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def read_pfm(file):
    """
    专门读取 SceneFlow 数据集的 .pfm 格式视差图
    """
    with open(file, 'rb') as f:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(rb'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data) # PFM 存储是倒置的，需要上下翻转
        
        return data, scale

class SceneFlowDataset(Dataset):
    """
    Scene Flow Dataset Loader (Enhanced for Background Preservation)
    
    Returns:
        (img_gt, img_gt_clone, depth_map)
        - img_gt: [3, H, W] RGB (0-1)
        - img_gt_clone: Clone of img_gt
        - depth_map: [1, H, W] Physical Depth in Meters.
          Range: [0.5, 1000.0]
          Values > 20.0 are preserved to represent background.
    """
    def __init__(self, data_root, dataset_type='train', image_size=(256, 256), use_random_crop=False):
        super().__init__()
        self.image_size = image_size
        self.use_random_crop = use_random_crop

        if dataset_type == 'train':
            self.img_dir = os.path.join(data_root, 'FlyingThings3D_subset/train/image_clean/right')
            self.depth_dir = os.path.join(data_root, 'FlyingThings3D_subset/train/disparity/right')
        elif dataset_type == 'val':
            self.img_dir = os.path.join(data_root, 'FlyingThings3D_subset/val/image_clean/right')
            self.depth_dir = os.path.join(data_root, 'FlyingThings3D_subset/val/disparity/right')
        else:
            raise ValueError("dataset_type must be 'train' or 'val'")
        
        if not os.path.exists(self.img_dir):
            print(f"Warning: Image directory not found: {self.img_dir}")
            self.file_ids = []
        else:
            self.file_ids = [f[:-4] for f in os.listdir(self.img_dir) if f.endswith('.png')]
            print(f"Loaded {len(self.file_ids)} images from {dataset_type} dataset.")

        if self.use_random_crop:
            self.transform = transforms.RandomCrop((self.image_size[0], self.image_size[1]))
        else:
            self.transform = transforms.CenterCrop((self.image_size[0], self.image_size[1]))

    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, index):
        file_id = self.file_ids[index]

        # 1. Read RGB Image
        img_path = os.path.join(self.img_dir, file_id + '.png')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f'Warning: Failed to read image {img_path}.')
            return self._return_empty()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # 2. Read Disparity (.pfm)
        disp_path = os.path.join(self.depth_dir, file_id + '.pfm')
        
        try:
            disparity, _ = read_pfm(disp_path)
            disparity = disparity.astype(np.float32)
        except Exception:
            disparity = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
            if disparity is None:
                print(f'Warning: Failed to read disparity {disp_path}.')
                return self._return_empty()
        
        if disparity.ndim == 3:
            disparity = disparity[:, :, 0]

        # =========================================================
        # 物理一致性转换 (Physical Consistency) + 背景保留
        # =========================================================
        
        # FlyingThings3D constant: f * B ~= 1050.0
        F_TIMES_B = 1050.0
        SCNEN_SCALE = 0.05
        F_TIMES_B_NEW = F_TIMES_B * SCNEN_SCALE
        
        # 定义截断范围
        # MIN_DEPTH: 0.5m (太近的物体对于 EDOF 来说是无效干扰，建议过滤)
        # MAX_DEPTH: 1000.0m (保留背景信息，不要在这里截断到 20m)
        MIN_DEPTH_M = 0.5
        MAX_DEPTH_M = 1000.0 

        mask_valid = disparity > 0
        depth_meters = np.zeros_like(disparity, dtype=np.float32)
        
        # Z = f * B / d
        depth_meters[mask_valid] = F_TIMES_B_NEW / (disparity[mask_valid] + 1e-6)
        
        # 处理无效视差（无穷远）
        depth_meters[~mask_valid] = MAX_DEPTH_M
        
        # 截断: 
        # < 0.5m -> 0.5m (防止除零或过大模糊)
        # > 1000m -> 1000m (统一视为无穷远背景)
        depth_meters = np.clip(depth_meters, MIN_DEPTH_M, MAX_DEPTH_M)

        # =========================================================

        # To Tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        depthmap_tensor = torch.from_numpy(depth_meters).unsqueeze(0).float()

        # Stack for consistent cropping
        stacked = torch.cat([image_tensor, depthmap_tensor], dim=0)

        try:
            cropped = self.transform(stacked)
        except Exception as e:
            print(f'Error during cropping {file_id}: {e}')
            return self._return_empty()

        img_gt = cropped[0:3, :, :]
        depth_map = cropped[3:4, :, :] # Values can be > 20.0

        return img_gt, img_gt.clone(), depth_map
    
    def _return_empty(self):
        return (torch.zeros(3, self.image_size[0], self.image_size[1]),
                torch.zeros(3, self.image_size[0], self.image_size[1]),
                torch.zeros(1, self.image_size[0], self.image_size[1]))

if __name__ == "__main__":
    # Test block
    import matplotlib.pyplot as plt
    TEST_DATA_ROOT = '/home/LionelZ/Data' 
    
    try:
        dataset = SceneFlowDataset(
            data_root=TEST_DATA_ROOT, 
            dataset_type='train',
            image_size=(512, 512),
            use_random_crop=True
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        images, _, depths = next(iter(loader))
        
        print(f"Depth Stats: Min={depths.min().item():.2f}m, Max={depths.max().item():.2f}m")
        # 如果 Max 接近 1000，说明背景保留成功
        
    except Exception as e:
        print(f"Test failed: {e}")