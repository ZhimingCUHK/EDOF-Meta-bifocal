import dis
import os
import cv2
from matplotlib.pylab import f
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class SceneFlowDataset(Dataset):
    """
    Read the Scene Flow dataset

    Returns:
        (img_gt,img_gt_clone,depth_map)
        - img_gt: Ground truth image Tensor [3,H,W]
        - img_gt_clone: Clone of ground truth image Tensor [3,H,W]
        - depth_map: Depth map Tensor [1,H,W]
    """
    def __init__(self,data_root,dataset_type='train',image_size=(256, 256),use_random_crop=False):
        """ 
        Args:
            data_root (str): Root directory of the dataset
            dataset_type (str, optional): 'train' or 'val'. Defaults to 'train'.
            image_size (tuple, optional): Size of the images. Defaults to (256, 256).
            use_random_crop (bool, optional): Whether to use random cropping. Defaults to False.
        """
        super().__init__()
        self.image_size = image_size
        self.use_random_crop = use_random_crop

        # define the dataset paths
        if dataset_type == 'train':
            self.img_dir = os.path.join(data_root,'FlyingThings3D_subset/train/image_clean/right')
            self.depth_dir = os.path.join(data_root,'FlyingThings3D_subset/train/disparity/right')
        elif dataset_type == 'val':
            self.img_dir = os.path.join(data_root,'FlyingThings3D_subset/val/image_clean/right')
            self.depth_dir = os.path.join(data_root,'FlyingThings3D_subset/val/disparity/right')
        else:
            raise ValueError("dataset_type must be 'train' or 'val'")
        
        # check if paths exist
        if not os.path.exists(self.img_dir):
            print(f"Warning: Image directory not found: {self.img_dir}")
            self.file_ids = []
        else:
            self.file_ids = [f[:-4] for f in os.listdir(self.img_dir) if f.endswith('.png')]
            print(f"Loaded {len(self.file_ids)} images from {dataset_type} dataset.")

        if not self.file_ids:
            raise RuntimeError("No images found in the specified directory.")
        
        if self.use_random_crop:
            self.transform = transforms.RandomCrop((self.image_size[0], self.image_size[1]))
        else:
            self.transform = transforms.CenterCrop((self.image_size[0], self.image_size[1]))

    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, index):
        """ 
        get and return a data sample
        """
        file_id = self.file_ids[index]

        # get the RGB image
        img_path = os.path.join(self.img_dir, file_id + '.png')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f'Warning: Failed to read image {img_path}. Returning a zero tensor.')
            return self._return_empty()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]

        disp_path = os.path.join(self.depth_dir, file_id + '.pfm')
        disparity = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)

        if disparity is None:
            print(f'Warning: Failed to read disparity map {disp_path}. Returning a zero tensor.')
            return self._return_empty()
        
        if disparity.ndim == 3:
            disparity = disparity[:, :, 0]  # Use only one channel if it's multi-channel

        # The logic of transform the disparity map to depth map

        min_dist = 0.2
        max_dist = 20.0

        disp_min = disparity.min()
        disp_max = disparity.max()

        if disp_max - disp_min < 1e-6:
            depth_meters = np.full_like(disparity, max_dist,dtype=np.float32)
        else:
            disp_norm = (disparity - disp_min) / (disp_max - disp_min)

            min_diopter = 1.0 / max_dist
            max_diopter = 1.0 / min_dist

            current_diopter = min_diopter + disp_norm * (max_diopter - min_diopter)

            depth_meters = 1.0 / (current_diopter + 1e-6)

        # Change to Tensor and crop
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # [3,H,W]
        depthmap_tensor = torch.from_numpy(depth_meters).unsqueeze(0).float()  # [1,H,W]

        stacked = torch.cat([image_tensor, depthmap_tensor], dim=0)  # [4,H,W]

        try:
            cropped = self.transform(stacked)
        except Exception as e:
            print(f'Error during cropping,file_id {file_id}: {e}')
            return self._return_empty()

        img_gt = cropped[0:3, :, :]
        depth_map = cropped[3:4, :, :]

        return img_gt, img_gt.clone(), depth_map
    
    def _return_empty(self):
        return (torch.zeros(3,self.image_size[0],self.image_size[1]),
                torch.zeros(3,self.image_size[0],self.image_size[1]),
                torch.zeros(1,self.image_size[0],self.image_size[1]))
    

# Unit Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    TEST_DATA_ROOT = '/home/LionelZ/Data' 
    
    print(f"--- Starting Unit Test ---")
    print(f"Dataset Root: {TEST_DATA_ROOT}")

    try:
        dataset = SceneFlowDataset(
            data_root=TEST_DATA_ROOT,
            dataset_type='train', # or 'val'
            image_size=(256, 256),
            use_random_crop=True
        )
    except Exception as e:
        print(f"Init failed: {e}")
        exit()

    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # get the first Batch
        print("Loading one batch...")
        try:
            # __getitem__ returns: img_gt, img_clone, depth_gt
            images, _, depths = next(iter(loader))
            
            print(f"\nBatch Data Shapes:")
            print(f"RGB Images: {images.shape}  (Batch, C, H, W)")
            print(f"Depth Maps: {depths.shape}  (Batch, 1, H, W)")
            
            
            idx = 0
            rgb_vis = images[idx].permute(1, 2, 0).numpy()
            depth_vis = depths[idx].squeeze(0).numpy()
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.title("RGB Image (Ground Truth)")
            plt.imshow(np.clip(rgb_vis, 0, 1))
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title("Depth Map (Meters)")
            plt.imshow(depth_vis, cmap='plasma')
            plt.colorbar(label='Depth (m)')
            plt.axis('off')
            
            plt.tight_layout()
            print("\nDisplaying visualization... (Close window to exit)")
            plt.show()
            
            print("Test finished successfully.")
            
        except Exception as e:
            print(f"Error getting batch: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Dataset is empty. Please check your data path.")
