import os
import re
from typing import final
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt

def read_pfm(file):
    with open(file,'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        
        dim_batch = re.match(rb'^(\d+)\s(\d+)\s$',f.readline())
        if dim_batch:
            width, height = map(int,dim_batch.groups())
        else:
            raise Exception('Malformed PFM header.')
        
        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        data = np.fromfile(f,endian + 'f')
        shape = (height,width,3) if color else (height,width)

        data = np.reshape(data,shape)
        data = np.flipud(data)

        return data,scale
    
class SceneFlowDataset(Dataset):
    
    def __init__(self,data_root,dataset_type='train',image_size=(540,960),padding=0,is_training=True,use_random_crop=False):
        """ 
        image_size: (height,width)
        padding: int
        """
        super().__init__()
        self.is_training = is_training
        self.image_size = image_size
        self.padding = padding
        self.use_random_crop = use_random_crop

        if dataset_type == 'train':
            self.img_dir = os.path.join(data_root,'FlyingThings3D_subset/train/image_clean/right')
            self.depth_dir = os.path.join(data_root,'FlyingThings3D_subset/train/disparity/right')
        elif dataset_type == 'val':
            self.img_dir = os.path.join(data_root,'FlyingThings3D_subset/val/image_clean/right')
            self.depth_dir = os.path.join(data_root,'FlyingThings3D_subset/val/disparity/right')
        else:
            raise ValueError("dataset_type must be 'train' or 'val'")
        
        self.file_ids = []
        if not os.path.exists(self.img_dir):
            print(f"Warning: Image directory not found: {self.img_dir}")
        else:
            candidates = [f[:-4] for f in os.listdir(self.img_dir) if f.endswith('.png')]
            for file_id in candidates:
                if os.path.exists(os.path.join(self.depth_dir,file_id + '.pfm')):
                    self.file_ids.append(file_id)

            print(f"Loaded {len(self.file_ids)} valid pairs from {dataset_type} dataset.")
       
        if self.use_random_crop:
            self.transform = T.RandomCrop((self.image_size[0],self.image_size[1]))
        else:
            self.transform = T.CenterCrop((self.image_size[0],self.image_size[1]))

    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, index):
        # Read the image
        file_id = self.file_ids[index]

        img_path = os.path.join(self.img_dir,file_id + '.png')
        image = cv2.imread(img_path,cv2.IMREAD_COLOR)
        
        if image is None:
            print(f'Warning: Failed to read image: {img_path}')
            return self._return_empty()
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) / 255.0

        disp_path = os.path.join(self.depth_dir,file_id + '.pfm')

        try:
            disparity,_ = read_pfm(disp_path)
            disparity = disparity.astype(np.float32)
        except Exception:
            disparity = cv2.imread(disp_path,cv2.IMREAD_UNCHANGED)
            if disparity is None:
                print(f'Warning: Failed to read disparity: {disp_path}')
                return self._return_empty()
            
        if disparity.ndim == 3:
            disparity = disparity[:,:,0]

        # Padding
        if self.padding > 0:
            image = np.pad(image,((self.padding,self.padding),(self.padding,self.padding),(0,0)),mode='reflect')
            disparity = np.pad(disparity,((self.padding,self.padding),(self.padding,self.padding)),mode='reflect')

        
        # Numpy to tensor
        image_tensor = torch.from_numpy(image).permute(2,0,1).float()
        disp_tensor = torch.from_numpy(disparity).unsqueeze(0).float()

        # disparity to depth
        depthmap = disp_tensor.clone()
        depthmap -= depthmap.min()
        if depthmap.max() > 0:
            depthmap /= depthmap.max()

        depthmap = 1.0 - depthmap

        combined = torch.cat([image_tensor,depthmap],dim=0)

        combined_cropped = self.transform(combined)

        image_tensor = combined_cropped[:3,:,:]
        depthmap = combined_cropped[3:,:,:]

        depthmap = F.gaussian_blur(depthmap,kernel_size=[5,5],sigma=[0.8,0.8])

        min_depth = 0.2
        max_depth = 2.0

        min_val = 1.0 / max_depth
        max_val = 1.0 / min_depth

        real_disp = max_val - (max_val - min_val) * depthmap
        final_depth_map = 1.0 / (real_disp + 1e-8)

        depth_conf = torch.ones_like(final_depth_map)

        sample = {
            'id':file_id,
            'image':image_tensor,
            'depthmap':final_depth_map,
            'depth_conf':depth_conf
        }

        return sample
    
    def _return_empty(self):
        H,W = self.image_size
        return {
            'id':'error',
            'image':torch.zeros(3,H,W),
            'depthmap':torch.zeros(1,H,W),
            'depth_conf':torch.zeros(1,H,W) 
        }
    

if __name__ == '__main__':

    TEST_DATA_ROOT = '/home/LionelZ/Data'

    try:
        dataset = SceneFlowDataset(
            data_root=TEST_DATA_ROOT,
            dataset_type='train',
            image_size=(540,960),
            padding=10,
            is_training=True,
            use_random_crop=True
        )

        if len(dataset) > 0:
            loader = DataLoader(dataset,batch_size=4,shuffle=True)
            batch = next(iter(loader))

            images = batch['image']
            depths = batch['depthmap']
            ids = batch['id']

            print("\n--- DataLoader Test Successful ---")
            print(f"Batch Shape:{images.shape}")
            print(f"Depth Shape:{depths.shape}")
            print(f"Depth Range: Min={depths.min():.4f},Max={depths.max():.4f}")
            print(f"(Ref Logic:0.0 is Near, 1.0 is Far)")

            fig,axes = plt.subplots(2,4,figsize=(16,8))

            for i in range(4):
                rgb = images[i].permute(1,2,0).numpy()
                axes[0,i].imshow(rgb)
                axes[0,i].set_title(f"ID:{ids[i]}\nRGB")
                axes[0,i].axis('off')

                d = depths[i,0].numpy()
                im = axes[1,i].imshow(d,cmap='gray',vmin=0.0,vmax=1.0)
                axes[1,i].set_title("Depth Map")
                axes[1,i].axis('off')

            plt.suptitle("SceneFlowDataset Sample Visualization",fontsize=16)
            plt.tight_layout()
            plt.show()
            print("[Success] Visualization Complete.\n")

        else:
            print("[Error] No data found.")

        
    except Exception as e:
        print(f"[Error] DataLoader Test Failed: {e}\n")
        import traceback
        traceback.print_exc()


























































