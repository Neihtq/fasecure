import os
import random
import torch
import glob

from os import listdir
from PIL import Image
from itertools import compress

from torch.utils.data import Dataset
from torchvision import transforms
    
    
class LFWDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.labels = []
        for label in listdir(root):
            img_path = os.path.join(root, label)
            if len(listdir(img_path)) > 1:
                self.labels.append(label)
        self.transform = transform
        
    def __len__(self):
        # returns amount of classes
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        folder = os.path.join(self.root, label)
        img_path = os.path.join(folder, listdir(folder)[0])
        anchor = self.get_image(img_path)
        img_path = os.path.join(folder, listdir(folder)[1])
        positive = self.get_image(img_path)
        negative = self.get_negative(idx)
        
        return label, anchor, positive, negative
        
    def get_image(self, img_path):
        img = Image.open(img_path)
       
        if self.transform:
            img = self.transform(img)
       
        if not torch.is_tensor(img):        
            img = transforms.ToTensor()(img)
        
        return img
    
    def get_negative(self, idx):
        include = [n for n in range(0, len(self.labels)) if n != idx]
        i = random.choice(include)
        label = self.labels[i]
        folder = os.path.join(self.root, label)
        img_path = os.path.join(folder, listdir(folder)[0])
    
        return self.get_image(img_path)


class LFWEvaluationDataset(Dataset):
    def __init__(self, root, transform=None, cropped_faces=False):
        self.root = root
        self.cropped_faces = cropped_faces
        self.labels = []
        self.mask = []
        self.transform = transform
        for label in listdir(root):
            img_path = os.path.join(root, label)
            if len(listdir(img_path)) > 5:
                for imgs_per_folder in range(len(listdir(img_path))):
                    self.labels.append(label)
                    self.mask.append(True)
            else:
                for imgs_per_folder in range(len(listdir(img_path))):
                    self.mask.append(False)

        self.image_filenames = glob.glob(os.path.join(root, "**/*.jpg"), recursive=True)
        # Use mask to filter classes with less than certain amount of images
        self.image_filenames = list(compress(self.image_filenames, self.mask))

    def __len__(self):
        # returns amount of classes
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.image_filenames[idx]
        
        if self.cropped_faces:
            img = self.get_image(img)

        return label, img
        
    def get_image(self, img_path):
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor