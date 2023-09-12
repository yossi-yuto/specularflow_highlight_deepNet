import os
from PIL import Image
import cv2
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch
import pdb

from utils import img_transform, mask_transform, hsv_transform

class SSF_SH_Dataset(Dataset):

    def __init__(self, image_list, mask_dir, edge_dilate = 1, resize=(416,416)):

        self.mask_dir = mask_dir

        self.transform_image = img_transform(resize)
        self.transform_mask = mask_transform(resize)

        self.trasform_to_tensor = transforms.ToTensor()
        self.dilate = edge_dilate
        self.images = image_list
        self.resize = resize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path).replace('.jpg','.png'))
        
        original_image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        hsv_img = original_image.convert("HSV")
        hsv = transforms.functional.to_tensor(hsv_img)
        hsv = transforms.functional.resize(hsv, self.resize) 
        
        image = self.transform_image(original_image)
        mask = self.transform_mask(mask)

        # エッジ画像（配列の生成）
        edge_canny = cv2.Canny(np.array(mask.squeeze(0), dtype=np.uint8) * 255, 0, 255)
        edge_canny[edge_canny < 127] = 0
        edge_canny[edge_canny > 127] = 1
        edge = cv2.dilate(edge_canny, np.ones((self.dilate,self.dilate)))
        
        edge = torch.tensor(np.expand_dims(edge, 0).astype(np.float32))
        original_image = np.asarray(original_image.resize(self.resize)).copy()
        original_image.flags.writeable = True
        
        return image, mask, edge, self.trasform_to_tensor(original_image), hsv[1:,:,:]

