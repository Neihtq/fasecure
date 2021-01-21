import csv
import os
import random
import torch
import glob
import pathlib
import numpy as np
import pandas as pd

from random import shuffle
from os import listdir
from os.path import dirname, abspath
from PIL import Image
from itertools import compress

from torch.utils.data import Dataset
from torchvision import transforms
    
from models.FaceNet import FaceNet

ABSOLUTE_DIR = dirname(abspath(__file__))
MODEL_DIR = os.path.join(ABSOLUTE_DIR, '..', 'models', 'FaceNetOnLFW.pth')


class ImageDataset(Dataset):
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
        shuffle(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, img_path = self.data[idx]
        img = self.get_image(img_path)

        return label, img
        
    def get_image(self, img_path):
        '''Returns Pytorch.Tensor of image'''
        img_path = pathlib.Path(img_path)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
       
        if not torch.is_tensor(img):        
            img = transforms.ToTensor()(img)
        
        return img.float()
    

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