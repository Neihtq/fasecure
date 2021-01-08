import csv
import os
import random
import torch
import glob
import numpy as np
import pandas as pd

from os import listdir
from os.path import dirname, abspath
from PIL import Image
from itertools import compress

from torch.utils.data import Dataset
from torchvision import transforms
    
from models.FaceNet import FaceNet

from constants import FACE_FEATURES

ABSOLUTE_DIR = dirname(abspath(__file__))
MODEL_DIR = os.path.join(ABSOLUTE_DIR, '..', 'models', 'FaceNetOnLFW.pth')

class LFWDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.labels = []
        for label in listdir(root):
            img_path = os.path.join(root, label)
            if len(listdir(img_path)) > 1:
                self.labels.append(label)

        self.labels = self.labels[0:50]
        self.transform = transform


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        self.embeddings_info = pd.read_csv('embeddings.csv')
        with open('embeddings.npy', 'rb') as f:
            self.embeddings_np = np.load(f)

        label = self.labels[idx]
        folder = os.path.join(self.root, label)
        
        anchor_embedding_index = self.embeddings_info[label][0]
        anchor_embedding_index = int(anchor_embedding_index.split(',')[0][1:])
        anchor_embedding_np = self.embeddings_np[anchor_embedding_index]
        img_path = os.path.join(folder, listdir(folder)[0])
        anchor = self.get_image(img_path)
  
        positive = self.get_positive(label, anchor)
        negative = self.get_negative(label, anchor)
        
        return label, anchor, positive, negative
        
    def get_image(self, img_path):
        '''Returns Pytorch.Tensor of image'''
        img = Image.open(img_path)      

        if self.transform:
            img = self.transform(img)
       
        if not torch.is_tensor(img):        
            img = transforms.ToTensor()(img)
        
        return img
    
    #def get_negative(self, idx):
    #    include = [n for n in range(0, len(self.labels)) if n != idx]
    #    i = random.choice(include)
    #    label = self.labels[i]
    #    folder = os.path.join(self.root, label)
    #    img_path = os.path.join(folder, listdir(folder)[0])
    
    #    return self.get_image(img_path)

    def get_positive(self, label, anchor):
        difference = 0
        prev = difference
        curr_path = ""
        for info in self.embeddings_info[label]:
            if pd.notnull(info):
                index, img_path = info[0], info[1]
                embedding = self.embeddings_np[index]

                distance = np.linalg.norm(anchor - embedding)
                difference = max(distance, difference)
                if difference != prev:
                    curr_path = img_path
        
        return self.get_image(curr_path)

    def get_negative(self, label, anchor):
        difference = 0
        prev = difference
        curr_path = ""
        for l in self.labels:
            if l != label:
                for info in self.embeddings[l]:
                    if pd.notnull(info):
                        index, img_path = info[0], info[0]
                        embedding = self.embeddings_np[index]

                        distance = np.linalg.norm(self.embeddings[label][0][0] - embedding[0])
                        difference = min(distance, difference)
                        if difference != prev:
                            curr_path = img_path
        
        return self.get_image(curr_path)


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