import os
import random
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VGGFacesDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.label_to_number = {}
        
        self.data = []
        for i, label in enumerate(listdir(root)):
            self.label_to_number[i] = label
            label_path = os.path.join(root, label)
            for img in listdir(label_path):
                img_path = os.path.join(label_path, img)
                self.data.append((i, img_path))

        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, path = self.data[idx]
        img = self.get_image(path)
        
        return label, img
    
    def get_image(self, img_path):
        '''Returns Pytorch.Tensor of image'''
        img = Image.open(img_path)      

        if self.transform:
            img = self.transform(img)
       
        if not torch.is_tensor(img):        
            img = transforms.ToTensor()(img)
        
        return img