import os
import os.path as osp
import pdb

import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np

class PMDDataset(Dataset):
    def __init__(self, data_path: str, scale: int, train_mode: bool = True, dilation: int = 5) -> None:
        
        self.image_dir = osp.join(data_path, 'image')
        self.mask_dir = osp.join(data_path, 'mask')
        
        if train_mode:
            self.edge_dir = osp.join(data_path, 'edge')
            self.train_mode = train_mode
        
        self.images = os.listdir(self.image_dir)
        
        self.image_transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((scale, scale)),
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((scale, scale))
        ])
        self.dilate = dilation
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        image_meta = {}
        
        img_path = osp.join(self.image_dir, self.images[index])
        mask_path = osp.join(self.mask_dir, self.images[index].replace('.jpg', '.png'))
        image = Image.open(img_path).convert("RGB")
        hsv = image.convert("HSV")
        mask = Image.open(mask_path).convert("1") # bit
        
        # transformation images
        rgb_image = self.image_transform(image)
        sv_image = self.image_transform(hsv)[1:]
        mask_image = self.mask_transform(mask)
        
        if self.train_mode:
            # edge making 
            edge_canny = cv2.Canny(np.array(mask_image.squeeze(0), dtype=np.uint8) * 255, 0, 255)
            edge_canny[edge_canny < 127] = 0
            edge_canny[edge_canny > 127] = 1
            edge = cv2.dilate(edge_canny, np.ones((self.dilate,self.dilate)))
        else:
            edge = np.zeros(mask_image.shape[1], mask_image.shape[2])
            
        edge_image = torch.tensor(np.expand_dims(edge, 0), dtype=torch.float32)
            
        # image meta file making 
        w, h = image.size
        image_meta['filename_image'] = img_path
        image_meta['filename_mask'] = mask_path
        image_meta['height'] = h
        image_meta['width'] = w
        
        return rgb_image, sv_image, mask_image, edge_image, image_meta
