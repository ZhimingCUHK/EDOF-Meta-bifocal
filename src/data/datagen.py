import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class SceneFlowDataset(Dataset):
    """ 
    Used to read the Scene Flow dataset for training and testing.
    Returns:
        (img_gt,img_gt_clone,,depth_map)
        - img_gt: Ground truth image tensor [C, H, W]
        - img_gt_clone: Clone of the ground truth image tensor [C, H, W], will use later for network input
        - depth_map: Depth map tensor [1, H, W]
    """

    def __init__(self,
                 data_root,
                 dataset_type='train',
                 image_size=256,
                 use_random_crop=False):
        """ 
        Args:
            data_root(str): Root directory of the dataset
            dataset_type(str): 'train' or 'test'
            image_size(int): Size of the cropped images
            use_random_crop(bool): Whether to use random cropping for data augmentation
        """
        super().__init__()
        self.image_size = image_size
        self.use_random_crop = use_random_crop

        # Definite the dataset path
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

        # get all image ID without suffix
        self.file_ids = [
            f[:-4] for f in os.listdir(self.img_dir) if f.endswith('.png')
        ]
        if not self.file_ids:
            raise RuntimeError(f'No image files found in {self.img_dir}')

        # define image transform
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
        get and return one data item
        """
        # get file ID
        file_id = self.file_ids[index]

        # load image and depth map
        img_path = os.path.join(self.img_dir, file_id + '.png')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f'Failed to load image: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Convert

        # get the depth map
        disp_path = os.path.join(self.disp_dir, file_id + '.pfm')
        disparity = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        if disparity is None:
            raise IOError(f'Failed to load disparity map: {disp_path}')

        # Ensure the disparity map is in 1-channel format
        if disparity.ndim == 3:
            disparity = disparity[:, :, 0]

        # Disparity to depth conversion
        depthmap = disparity.astype(np.float32)
        depthmap -= depthmap.min()
        depthmap /= (depthmap.max() + 1e-8)
        depthmap = 1.0 - depthmap

        # Transform to pytorch tensor (C,H,W)
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        depthmap_tensor = torch.from_numpy(depthmap).float().unsqueeze(0)

        # Apply cropping
        stacked = torch.cat((image_tensor, depthmap_tensor), dim=0)

        try:
            stacked_cropped = self.transform(stacked)
        except ValueError as e:
            print(f'Error during cropping, the file ID is {file_id}: {e}')
            return (torch.zeros(3, self.image_size, self.image_size),
                    torch.zeros(3, self.image_size, self.image_size),
                    torch.zeros(1, self.image_size, self.image_size))

        img_gt = stacked_cropped[0:3, :, :]
        depth_map = stacked_cropped[3:4, :, :]

        return img_gt, img_gt.clone(), depth_map



if __name__ == '__main__':

    # Change the data_root to your dataset path
    data_root = '/home/LionelZ/Data/'

    image_size = 256
    batch_size = 4

    print(f"Loading Scene Flow dataset from {data_root} ...")

    # Example Usage
    try:
        train_dataset = SceneFlowDataset(data_root=data_root,
                                         dataset_type='train',
                                         image_size=image_size,
                                         use_random_crop=True)
        if len(train_dataset) == 0:
            raise RuntimeError('Training dataset is empty.')
        else:
            print(
                f'Training dataset loaded with {len(train_dataset)} samples.')

            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=4)
            print('Getting the first batch ...')
            for img_gt_batch, _, depth_map_batch in train_loader:
                print(f'Image batch shape: {img_gt_batch.shape}')
                print(f'Depth map batch shape: {depth_map_batch.shape}')

                img_to_show = img_gt_batch[0].permute(1, 2, 0).numpy()
                depth_to_show = depth_map_batch[0].squeeze().numpy()

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title('Ground Truth Image')
                plt.imshow(img_to_show)
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title('Depth Map')
                plt.imshow(depth_to_show, cmap='gray')
                plt.axis('off')

                break
    except FileNotFoundError as e:
        print(f"\n---Wrong---")
        print(f"Dataset not found: {e}")
        print(
            "Please check the data_root path and ensure the dataset is correctly placed.\n"
        )
    except ImportError as e:
        print(f"\n---Wrong---")
        print(f"Required library not found: {e}")
        print(
            "Please ensure all required libraries are installed correctly.\n")

